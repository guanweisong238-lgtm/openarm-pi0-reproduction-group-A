"""A1 抽屉多形态与多高度参数化轨迹采集脚本（位于 openarm-a1/scripts/a1_simulation/）。

复用固定版本上游 IK/路径规划，以 30 Hz 零阶保持位置目标，物理子步仍为 2 ms；
重力补偿每个物理步重算。--mode contact 禁用抓取绑定，用于真实接触采集。
支持 5 种抽屉外形预设、5 种材质配色、宽/深/高独立缩放及 X/Y/Z 三维偏移。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
A2_ROOT = ROOT.parent / "openarm-pi0-reproduction-group-A"
A2_PYTHON = A2_ROOT / ".venv/bin/python"

# 若当前环境缺少 scipy（ContactDrawerController IK 所需），自动切换到同机 A2 虚拟环境执行
if importlib.util.find_spec("scipy") is None and A2_PYTHON.exists() and Path(sys.executable) != A2_PYTHON:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}:{existing_pp}" if existing_pp else str(ROOT)
    os.execve(str(A2_PYTHON), [str(A2_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.drawer_scene_generator import (
    APPEARANCE_PRESETS,
    SHAPE_PRESETS,
    build_scene_overrides_dict,
    compile_drawer_scene,
    resolve_drawer_variation,
)
from simulation.drawer_simulation import UPSTREAM_COMMIT, finger_handle_contacts, resolve_upstream_source

ACTUATORS = [
    name
    for side in ("right", "left")
    for name in (*(f"{side}_joint{i}_ctrl" for i in range(1, 8)), f"{side}_finger1_ctrl")
]
CAMERAS = {
    "ceiling": "camera_ceiling",
    "wrist_left": "camera_wrist_left",
    "wrist_right": "camera_wrist_right",
}
FPS = 30


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_preview(directory: Path) -> None:
    """将已保存的三路原始 RGB 拼成诊断视频，并保存拉出最远一帧。"""
    import numpy as np

    with np.load(directory / "episode.npz", allow_pickle=False) as episode:
        if "images_ceiling" not in episode:
            return
        views = [episode[f"images_{key}"] for key in CAMERAS]
        peak = int(np.argmin(episode["drawer"]))
        if importlib.util.find_spec("cv2") is not None:
            import cv2

            writer = cv2.VideoWriter(
                str(directory / "preview.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                (960, 240),
            )
            if not writer.isOpened():
                raise RuntimeError("无法创建诊断视频")
            try:
                for index in range(len(views[0])):
                    rgb = np.concatenate([view[index] for view in views], axis=1)
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    for slot, label in enumerate(CAMERAS):
                        cv2.putText(
                            frame, label, (320 * slot + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                        )
                    cv2.putText(
                        frame,
                        f"t={episode['timestamp'][index]:.2f}s  open={-1000*episode['drawer'][index]:.1f}mm",
                        (8, 233),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                    )
                    writer.write(frame)
                    if index == peak:
                        cv2.imwrite(str(directory / "peak_open.png"), frame)
            finally:
                writer.release()
        else:
            from PIL import Image

            rgb = np.concatenate([view[peak] for view in views], axis=1)
            Image.fromarray(rgb).save(directory / "peak_open.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None, help="openarm-control 上游目录（默认自动探测）")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("assisted", "contact"), default="contact")
    parser.add_argument("--expert", choices=("upstream", "contact"), default="contact")
    parser.add_argument(
        "--variant-preset",
        choices=tuple(SHAPE_PRESETS.keys()),
        default="standard",
        help="抽屉外形预设家族",
    )
    parser.add_argument(
        "--appearance-preset",
        choices=tuple(APPEARANCE_PRESETS.keys()),
        default="classic_wood_blue",
        help="抽屉与柜体材质配色预设",
    )
    parser.add_argument("--handle-radius", type=float, default=None, help="调整把手半径（米），范围 [0.005, 0.02]")
    parser.add_argument("--drawer-offset-x", type=float, default=0.0, help="抽屉柜沿世界 X 轴偏移（米）")
    parser.add_argument("--drawer-offset-y", type=float, default=0.0, help="抽屉柜沿世界 Y 轴横移（米）")
    parser.add_argument("--drawer-offset-z", type=float, default=0.0, help="抽屉柜沿世界 Z 轴高度偏移（米）")
    parser.add_argument("--width-scale", type=float, default=None, help="抽屉与柜体宽度（Y向）缩放系数")
    parser.add_argument("--depth-scale", type=float, default=None, help="抽屉与柜体深度（X向）缩放系数")
    parser.add_argument("--height-scale", type=float, default=None, help="抽屉与柜体高度（Z向）缩放系数")
    parser.add_argument("--handle-span-scale", type=float, default=None, help="把手横杆长度（Y向）缩放系数")
    parser.add_argument("--handle-standoff-scale", type=float, default=None, help="把手前探距离（X向）缩放系数")
    parser.add_argument(
        "--panel-style",
        choices=("inset", "framed", "raised_center", "twin_rail", "bevel_top"),
        default=None,
        help="抽屉面板样式覆盖",
    )
    parser.add_argument("--contact-pitch", type=float, default=15.0)
    parser.add_argument(
        "--approach-offset",
        type=float,
        nargs=3,
        default=[-0.10, 0.0, 0.055],
        metavar=("X", "Y", "Z"),
        help="接近路点相对把手的世界 XYZ 偏移（米）",
    )
    parser.add_argument(
        "--initial-right-jitter",
        type=float,
        default=0.0,
        help="记录前 reset 右臂 7 关节的均匀扰动半幅（rad）",
    )
    parser.add_argument("--seed", type=int, default=0, help="初始右臂扰动随机种子")
    parser.add_argument("--pull-distance", type=float, default=0.09)
    parser.add_argument("--free-pitch", action="store_true")
    parser.add_argument("--return-drawer", action="store_true")
    parser.add_argument(
        "--wrist-view",
        choices=("source", "side"),
        default="side",
        help="side 使用固定的腕部侧向虚拟相机，避开掌部遮挡",
    )
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--export-lerobot", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.initial_right_jitter <= 0.05 or args.seed < 0:
        parser.error("初始关节扰动应在 [0, 0.05] rad，seed 应非负")
    if (
        not -0.15 <= args.approach_offset[0] <= -0.05
        or not -0.05 <= args.approach_offset[1] <= 0.05
        or not -0.03 <= args.approach_offset[2] <= 0.12
    ):
        parser.error("接近路点超出试采范围")

    variation = resolve_drawer_variation(
        variant_preset=args.variant_preset,
        appearance_preset=args.appearance_preset,
        drawer_offset_x_m=args.drawer_offset_x,
        drawer_offset_y_m=args.drawer_offset_y,
        drawer_offset_z_m=args.drawer_offset_z,
        width_scale=args.width_scale,
        depth_scale=args.depth_scale,
        height_scale=args.height_scale,
        handle_span_scale=args.handle_span_scale,
        handle_standoff_scale=args.handle_standoff_scale,
        handle_radius_m=args.handle_radius,
        panel_style=args.panel_style,
    )

    task_prompt = "Open the drawer, then close it." if args.return_drawer else "Open the drawer."
    source_dir = resolve_upstream_source(args.source)
    args.output = args.output.resolve()
    commit = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"上游版本不匹配：{commit}")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.export_lerobot and args.no_images:
        raise ValueError("导出 LeRobot 需要三路图像")
    if args.expert == "contact" and args.mode != "contact":
        raise ValueError("真实接触专家只能使用 --mode contact")
    if args.return_drawer and args.expert != "contact":
        raise ValueError("往返示教需要使用 --expert contact")
    if not 0 < args.pull_distance <= 0.1:
        raise ValueError("拉动行程超出试验范围")

    os.environ.setdefault("MUJOCO_GL", "egl")
    for cache_candidate in (ROOT / ".cache/drawer-deps", A2_ROOT / ".cache/drawer-deps"):
        if cache_candidate.exists():
            sys.path.insert(0, str(cache_candidate))
    sys.path.insert(0, str(source_dir))

    import mujoco
    import numpy as np
    from openarm_control.articulated import ArticulatedController
    from openarm_control.config import ARTICULATED_SCENE
    from openarm_control.trajectory import quintic_polynomial

    controller_class = ArticulatedController
    if args.expert == "contact":
        from scripts.a1_simulation.contact_drawer_controller import ContactDrawerController

        controller_class = ContactDrawerController

    def compile_model():
        return compile_drawer_scene(
            ARTICULATED_SCENE,
            variation,
            wrist_view=args.wrist_view,
            shadow_size=1024,
        )

    model = compile_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)
    actuator_ids = np.array([model.actuator(name).id for name in ACTUATORS])
    joint_ids = model.actuator_trnid[actuator_ids, 0]
    qpos_ids = model.jnt_qposadr[joint_ids]
    arm_dofs = model.jnt_dofadr[joint_ids[np.array([i for i in range(16) if i % 8 != 7])]]
    drawer_id = model.joint("drawer_slide").qposadr[0]
    low, high = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if model.jnt_limited[jid]:
            low[aid] = max(low[aid], model.jnt_range[jid, 0])
            high[aid] = min(high[aid], model.jnt_range[jid, 1])

    initial_right_delta = np.zeros(7)
    if args.initial_right_jitter:
        right_qpos = qpos_ids[:7]
        original = data.qpos[right_qpos].copy()
        delta_low = np.maximum(-args.initial_right_jitter, low[actuator_ids[:7]] - original)
        delta_high = np.minimum(args.initial_right_jitter, high[actuator_ids[:7]] - original)
        if np.any(delta_low >= delta_high):
            raise ValueError("初始关节扰动与限位没有可采样交集")
        delta = np.random.default_rng(args.seed).uniform(delta_low, delta_high)
        perturbed = original + delta
        if np.any(perturbed < low[actuator_ids[:7]]) or np.any(perturbed > high[actuator_ids[:7]]):
            raise ValueError("扰动后的初始右臂位置超限，拒绝静默裁剪")
        data.qpos[right_qpos] = perturbed
        data.ctrl[actuator_ids[:7]] = perturbed
        initial_right_delta = delta
        mujoco.mj_forward(model, data)

    args.output.mkdir(parents=True)
    renderer = None if args.no_images else mujoco.Renderer(model, height=240, width=320)
    records, images, stages = [], {key: [] for key in CAMERAS}, []
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    initial_state = None
    initial_eq_data = model.eq_data.copy()
    initial_drawer = float(data.qpos[drawer_id])

    def advance(m, d, native, substeps):
        if args.mode == "contact" and np.any(d.eq_active[m.eq_type == mujoco.mjtEq.mjEQ_WELD]):
            raise RuntimeError("真实接触模式出现了活动 weld 约束")
        d.ctrl[:] = native
        for _ in range(substeps):
            mujoco.mj_forward(m, d)
            d.qfrc_applied[:] = 0
            d.qfrc_applied[arm_dofs] = d.qfrc_bias[arm_dofs]
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)

    class Recorder(controller_class):
        def _weld(self, ppc, fixture):
            if args.mode == "assisted":
                super()._weld(ppc, fixture)

        def _drive(self, active, other, path, dur, grip=None, viewer=None, dt_realtime=False):
            nonlocal initial_state
            path = np.atleast_2d(np.asarray(path, float))
            if not path.size:
                path = data.qpos[active.king.qpos_indices][None].copy()
            previous = data.qpos[active.king.qpos_indices].copy()
            points = np.vstack([previous, path]) if not np.allclose(path[0], previous) else path
            other_hold = data.qpos[other.king.qpos_indices].copy()
            count = round(dur * FPS)
            if not np.isclose(count / FPS, dur):
                raise ValueError("阶段时长必须为 30Hz 整数周期")
            stages.append(
                {
                    "first_frame": len(records),
                    "frames": count,
                    "duration": dur,
                    "gripper_target": grip,
                    "weld_active": bool(data.eq_active[model.equality("grasp_right_drawer").id]),
                }
            )
            for index in range(count):
                mujoco.mj_forward(model, data)
                if initial_state is None:
                    initial_state = np.zeros(mujoco.mj_stateSize(model, state_spec))
                    mujoco.mj_getState(model, data, initial_state, state_spec)
                requested = data.ctrl.copy()
                progress, _, _ = quintic_polynomial(index / FPS, 0.0, dur, 0.0, 1.0)
                requested[active.arm_acts] = active._sample(points, progress)
                requested[other.arm_acts] = other_hold
                if grip is not None:
                    requested[active.grip_act] = grip
                applied = np.clip(requested, low, high)
                frame = len(records)
                substeps = round((frame + 1) / FPS / model.opt.timestep) - round(frame / FPS / model.opt.timestep)
                records.append(
                    {
                        "state": data.qpos[qpos_ids].copy(),
                        "qpos": data.qpos.copy(),
                        "action": applied[actuator_ids].copy(),
                        "native_action": applied.copy(),
                        "clipped": requested != applied,
                        "timestamp": data.time,
                        "drawer": float(data.qpos[drawer_id]),
                        "substeps": substeps,
                        "eq_active": data.eq_active.copy(),
                        "eq_data": model.eq_data.copy(),
                    }
                )
                if args.expert == "contact":
                    counts, forces = finger_handle_contacts(model, data)
                    records[-1].update(finger_contact_counts=counts, finger_normal_forces=forces)
                if renderer is not None:
                    for key, camera in CAMERAS.items():
                        renderer.update_scene(data, camera=camera)
                        images[key].append(renderer.render().copy())
                advance(model, data, applied, substeps)
            print(f"阶段{len(stages)}完成，帧数={len(records)}，抽屉={-data.qpos[drawer_id]*1000:.1f} mm", flush=True)
            return True

    try:
        controller = Recorder(model, data)
        if args.expert == "contact":
            controller.contact_pitch = args.contact_pitch
            controller.pull_distance = args.pull_distance
            controller.free_pitch = args.free_pitch
            controller.return_drawer = args.return_drawer
            controller.approach_offset = np.asarray(args.approach_offset, dtype=float)
        completed = controller.open_drawer()
        hold = data.qpos[controller.right.king.qpos_indices].copy()
        controller._drive(
            controller.right,
            controller.left,
            [hold],
            1.0,
            grip=float(data.ctrl[controller.right.grip_act]),
        )
    finally:
        if renderer is not None:
            renderer.close()

    arrays = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    arrays.update({f"images_{key}": np.asarray(value) for key, value in images.items() if value})
    arrays.update(
        initial_integration_state=initial_state,
        initial_eq_data=initial_eq_data,
        terminal_qpos=data.qpos.copy(),
        terminal_state=data.qpos[qpos_ids].copy(),
        terminal_timestamp=np.asarray(data.time),
        actuator_names=np.asarray(ACTUATORS),
    )
    assert np.isfinite(arrays["state"]).all() and np.isfinite(arrays["action"]).all()
    np.testing.assert_allclose(
        arrays["timestamp"], np.arange(len(records)) / FPS, atol=model.opt.timestep / 2 + 1e-8, rtol=0
    )
    np.savez_compressed(args.output / "episode.npz", **arrays)
    write_preview(args.output)

    # 独立模型只重放动作与显式环境绑定事件，不回灌任何中间关节状态
    replay_model = compile_model()
    replay_data = mujoco.MjData(replay_model)
    replay_model.eq_data[:] = initial_eq_data
    mujoco.mj_setState(replay_model, replay_data, initial_state, state_spec)
    mujoco.mj_forward(replay_model, replay_data)
    max_error = 0.0
    for r in records:
        replay_model.eq_data[:] = r["eq_data"]
        replay_data.eq_active[:] = r["eq_active"]
        max_error = max(max_error, float(np.max(np.abs(replay_data.qpos - r["qpos"]))))
        advance(replay_model, replay_data, r["native_action"], int(r["substeps"]))
    max_error = max(max_error, float(np.max(np.abs(replay_data.qpos - data.qpos))))
    final_open = initial_drawer - float(data.qpos[drawer_id])
    held_open = bool(np.all(initial_drawer - arrays["drawer"][-FPS:] >= 0.08) and final_open >= 0.08)
    open_mask = initial_drawer - arrays["drawer"] >= 0.08
    sustained_open = bool(
        len(open_mask) >= FPS
        and np.any(np.convolve(open_mask.astype(int), np.ones(FPS, dtype=int), mode="valid") == FPS)
    )
    final_closed = bool(
        abs(final_open) <= 0.005 and np.all(np.abs(initial_drawer - arrays["drawer"][-FPS:]) <= 0.005)
    )
    contact_ctrl_path = Path(__file__).resolve().parent / "contact_drawer_controller.py"
    report = {
        "source_repository": "https://github.com/Manas-arumalla/openarm-control",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contact_controller_sha256": hashlib.sha256(contact_ctrl_path.read_bytes()).hexdigest(),
        "source_commit": commit,
        "license": "Apache-2.0",
        "mujoco_version": mujoco.__version__,
        "prompt": task_prompt,
        "source_scene": str(Path(ARTICULATED_SCENE).relative_to(source_dir)),
        "scene_sha256": hashlib.sha256(Path(ARTICULATED_SCENE).read_bytes()).hexdigest(),
        "mode": args.mode,
        "fps": FPS,
        "frames": len(records),
        "seconds": float(data.time),
        "expert": args.expert,
        "grasp_diagnostics": getattr(controller, "grasp_diagnostics", []),
        "expert_parameters": (
            {
                "pitch_degrees": controller.contact_pitch,
                "open_gripper_target": controller.contact_open,
                "grasp_point_local_m": controller.right.king.tool_offset.tolist(),
                "bilateral_force_stop_N": 1.0,
                "nominal_pull_m": controller.pull_distance,
                "free_pitch": controller.free_pitch,
                "return_drawer": controller.return_drawer,
                "approach_offset_world_m": controller.approach_offset.tolist(),
                "base_elbow_clearance_m": 0.006,
            }
            if args.expert == "contact"
            else None
        ),
        "scene_overrides": build_scene_overrides_dict(variation),
        "image_shape": None if args.no_images else [240, 320, 3],
        "wrist_view": args.wrist_view,
        "shadow_map_size": 1024,
        "camera_local_poses": {
            name: {
                "position": model.cam_pos[model.camera(name).id].tolist(),
                "quaternion_wxyz": model.cam_quat[model.camera(name).id].tolist(),
                "fovy_degrees": float(model.cam_fovy[model.camera(name).id]),
            }
            for name in CAMERAS.values()
        },
        "camera_mapping": CAMERAS,
        "actuator_order": ACTUATORS,
        "action_representation": "16D absolute joint position targets, radians, right then left",
        "gravity_compensation": "qfrc_bias on 14 arm DOFs, recomputed each physical substep",
        "initialization": "ready keyframe; upstream _park sets left arm qpos/qvel to zero before recording",
        "initial_right_randomization": {
            "seed": args.seed,
            "uniform_half_width_rad": args.initial_right_jitter,
            "applied_delta_rad": initial_right_delta.tolist(),
            "sampling": "uniform over intersection of +/-half-width and joint/control limits",
            "method": "reset right qpos and ctrl before recording; no intermediate state injection",
        },
        "expert_returned_success": bool(completed),
        "upstream_returned_success": bool(completed) if args.expert == "upstream" else None,
        "final_open_m": final_open,
        "peak_open_m": float(initial_drawer - arrays["drawer"].min()),
        "success_8cm_held_1s": held_open,
        "opened_8cm_held_1s_during_episode": sustained_open,
        "final_closed_5mm_held_1s": final_closed,
        "success_open_then_close": bool(args.return_drawer and sustained_open and final_closed),
        "weld_frames": int(arrays["eq_active"][:, model.equality("grasp_right_drawer").id].sum()),
        "clipped_frames": int(np.any(arrays["clipped"], axis=1).sum()),
        "replay_max_qpos_error": max_error,
        "replay_verified": bool(max_error < 1e-6),
        "stages": stages,
        "exported_lerobot": False,
        "ready_for_current_cell_training": False,
        "limitations": [
            "仅一条试采轨迹，无独立验证集",
            "上游 pedestal 场景，与当前 CELL 场景不同",
            (
                "回放需使用记录的重力补偿协议；此模式没有抓取绑定"
                if args.mode == "contact"
                else "辅助绑定和重力补偿属于环境机制，16维动作不包含它们；回放必须复现同一机制"
            ),
        ],
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
