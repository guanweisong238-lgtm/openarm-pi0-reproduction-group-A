"""试采 OpenArm Control 抽屉轨迹；独立于现有抓方块训练配置。

复用固定版本上游 IK/路径规划，以 30 Hz 零阶保持位置目标，物理子步仍为
2 ms；重力补偿每个物理步重算。辅助绑定只作为环境诊断量保存，不作为模型
输入或动作标签。--mode contact 禁用抓取绑定，用于检查真实接触是否可用。
输出包含可独立回放的物理状态、绑定事件、三路 RGB 和可选 LeRobot v2.1。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "14ea59faad6706b9add5d7f8d386e43239a6360f"
ACTUATORS = [name for side in ("right", "left") for name in (
    *(f"{side}_joint{i}_ctrl" for i in range(1, 8)), f"{side}_finger1_ctrl")]
CAMERAS = {"ceiling": "camera_ceiling", "wrist_left": "camera_wrist_left",
           "wrist_right": "camera_wrist_right"}
FPS = 30


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_preview(directory):
    """将已保存的三路原始 RGB 拼成诊断视频，并保存拉出最远一帧。"""
    import cv2
    import numpy as np
    with np.load(directory / "episode.npz", allow_pickle=False) as episode:
        if "images_ceiling" not in episode:
            return
        views = [episode[f"images_{key}"] for key in CAMERAS]
        peak = int(np.argmin(episode["drawer"]))
        writer = cv2.VideoWriter(str(directory / "preview.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 240))
        if not writer.isOpened():
            raise RuntimeError("无法创建诊断视频")
        try:
            for index in range(len(views[0])):
                rgb = np.concatenate([view[index] for view in views], axis=1)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for slot, label in enumerate(CAMERAS):
                    cv2.putText(frame, label, (320 * slot + 8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"t={episode['timestamp'][index]:.2f}s  open={-1000*episode['drawer'][index]:.1f}mm",
                            (8, 233), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                writer.write(frame)
                if index == peak:
                    if not cv2.imwrite(str(directory / "peak_open.png"), frame):
                        raise RuntimeError("无法保存诊断截图")
        finally:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".cache/openarm-control")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("assisted", "contact"), default="assisted")
    parser.add_argument("--expert", choices=("upstream", "contact"), default="upstream")
    parser.add_argument("--handle-radius", type=float,
                        help="显式试验参数：调整把手半径（米），不修改上游文件")
    parser.add_argument("--drawer-offset-x", type=float, default=0.0,
                        help="显式试验参数：抽屉柜沿世界X轴移动的米数")
    parser.add_argument("--contact-pitch", type=float, default=15.0)
    parser.add_argument("--drawer-offset-y", type=float, default=0.0,
                        help="柜体沿世界Y横移，米；范围[-0.03,0.03]")
    parser.add_argument("--approach-offset", type=float, nargs=3, default=[-0.10, 0.0, 0.055],
                        metavar=("X", "Y", "Z"), help="接近路点相对把手的世界XYZ偏移，米")
    parser.add_argument("--initial-right-jitter", type=float, default=0.0,
                        help="记录前reset右臂7关节的均匀扰动半幅，rad；默认0保持历史轨迹")
    parser.add_argument("--seed", type=int, default=0, help="初始右臂扰动随机种子，不是模型推理噪声种子")
    parser.add_argument("--pull-distance", type=float, default=0.09)
    parser.add_argument("--free-pitch", action="store_true")
    parser.add_argument("--return-drawer", action="store_true")
    parser.add_argument("--wrist-view", choices=("source", "side"), default="source",
                        help="side使用固定的腕部侧向虚拟相机，避开掌部遮挡")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--export-lerobot", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.initial_right_jitter <= 0.05 or args.seed < 0:
        parser.error("初始关节扰动应在[0,0.05]rad，seed应非负")
    if (not -0.03 <= args.drawer_offset_y <= 0.03
            or not -0.15 <= args.approach_offset[0] <= -0.05
            or not -0.05 <= args.approach_offset[1] <= 0.05
            or not -0.03 <= args.approach_offset[2] <= 0.12):
        parser.error("柜体横移或接近路点超出试采范围")
    task_prompt = "Open the drawer, then close it." if args.return_drawer else "Open the drawer."
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"上游版本不匹配：{commit}")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.export_lerobot and args.no_images:
        raise ValueError("导出 LeRobot 需要三路图像")
    if args.expert == "contact" and args.mode != "contact":
        raise ValueError("真实接触专家只能使用--mode contact")
    if args.return_drawer and args.expert != "contact":
        raise ValueError("往返示教需要使用--expert contact")
    if args.handle_radius is not None and not 0.005 <= args.handle_radius <= 0.02:
        raise ValueError("把手半径试验范围为0.005至0.02米")
    if not -0.1 <= args.drawer_offset_x <= 0.1 or not 0 < args.pull_distance <= 0.1:
        raise ValueError("抽屉偏移和拉动行程超出试验范围")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    sys.path.insert(0, str(ROOT / ".cache/drawer-deps"))
    sys.path.insert(0, str(args.source))
    import mujoco
    import numpy as np
    from openarm_control.articulated import ArticulatedController
    from openarm_control.config import ARTICULATED_SCENE
    from openarm_control.trajectory import quintic_polynomial
    controller_class = ArticulatedController
    if args.expert == "contact":
        from contact_drawer_controller import ContactDrawerController, finger_handle_contacts
        controller_class = ContactDrawerController

    def compile_model():
        spec = mujoco.MjSpec.from_file(ARTICULATED_SCENE)
        next(body for body in spec.bodies if body.name == "cabinet_drawer").pos[0] += args.drawer_offset_x
        next(body for body in spec.bodies if body.name == "cabinet_drawer").pos[1] += args.drawer_offset_y
        if args.handle_radius is not None:
            next(geom for geom in spec.geoms if geom.name == "drawer_handle").size[0] = args.handle_radius
        # 顶部相机沿用上游 tablecam 的内外参；腕部相机保持原样。
        top = next(cam for cam in spec.cameras if cam.name == "tablecam")
        camera = spec.worldbody.add_camera(name="camera_ceiling")
        camera.pos, camera.quat, camera.fovy = top.pos, top.quat, top.fovy
        model = spec.compile()
        if args.wrist_view == "side":
            # 固定在腕部局部坐标系，不依赖抽屉真值或逐帧跟踪。
            position = np.array([0.07, 0.10, -0.04])
            target = np.array([0.0, 0.0, -0.14])
            z_axis = position - target
            z_axis /= np.linalg.norm(z_axis)
            x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, np.column_stack([x_axis, y_axis, z_axis]).ravel())
            for name in ("camera_wrist_left", "camera_wrist_right"):
                cid = model.camera(name).id
                model.cam_pos[cid], model.cam_quat[cid], model.cam_fovy[cid] = position, quaternion, 85
        model.vis.quality.shadowsize = 1024
        model.vis.global_.offwidth = max(320, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(240, model.vis.global_.offheight)
        return model

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
    # 只改变episode的初始条件；之后不注入中间状态、不向动作标签直接加噪声。
    initial_right_delta = np.zeros(7)
    if args.initial_right_jitter:
        right_qpos = qpos_ids[:7]
        original = data.qpos[right_qpos].copy()
        # ready的joint6靠近上限；在扰动窗口与物理限位的交集内采样，
        # 不先产生越界值再裁剪。靠近限位时分布自然不对称，完整记录实际delta。
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
            raise RuntimeError("真实接触模式出现了活动weld约束")
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
                raise ValueError("阶段时长必须为30Hz整数周期")
            stages.append({"first_frame": len(records), "frames": count, "duration": dur,
                           "gripper_target": grip, "weld_active": bool(data.eq_active[model.equality('grasp_right_drawer').id])})
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
                records.append({"state": data.qpos[qpos_ids].copy(), "qpos": data.qpos.copy(),
                                "action": applied[actuator_ids].copy(), "native_action": applied.copy(),
                                "clipped": requested != applied, "timestamp": data.time,
                                "drawer": float(data.qpos[drawer_id]), "substeps": substeps,
                                "eq_active": data.eq_active.copy(), "eq_data": model.eq_data.copy()})
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
        # 专家结束后保持当时的夹爪目标1秒，用实际位移判定；contact模式此时仍夹持。
        hold = data.qpos[controller.right.king.qpos_indices].copy()
        controller._drive(controller.right, controller.left, [hold], 1.0,
                          grip=float(data.ctrl[controller.right.grip_act]))
    finally:
        if renderer is not None:
            renderer.close()
    arrays = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    arrays.update({f"images_{key}": np.asarray(value) for key, value in images.items() if value})
    arrays.update(initial_integration_state=initial_state, initial_eq_data=initial_eq_data,
                  terminal_qpos=data.qpos.copy(), terminal_state=data.qpos[qpos_ids].copy(),
                  terminal_timestamp=np.asarray(data.time), actuator_names=np.asarray(ACTUATORS))
    assert np.isfinite(arrays["state"]).all() and np.isfinite(arrays["action"]).all()
    np.testing.assert_allclose(arrays["timestamp"], np.arange(len(records)) / FPS,
                               atol=model.opt.timestep / 2 + 1e-8, rtol=0)
    np.savez_compressed(args.output / "episode.npz", **arrays)
    write_preview(args.output)

    # 独立模型只重放动作与显式环境绑定事件，不回灌任何中间关节状态。
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
    # 往返任务在轨迹中间保持打开，最后应关闭；使用连续1秒窗口检查打开阶段。
    open_mask = initial_drawer - arrays["drawer"] >= 0.08
    sustained_open = bool(len(open_mask) >= FPS and np.any(
        np.convolve(open_mask.astype(int), np.ones(FPS, dtype=int), mode="valid") == FPS))
    final_closed = bool(abs(final_open) <= 0.005 and np.all(np.abs(
        initial_drawer - arrays["drawer"][-FPS:]) <= 0.005))
    report = {"source_repository": "https://github.com/Manas-arumalla/openarm-control",
              "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "contact_controller_sha256": hashlib.sha256(
                  (ROOT / "scripts/a2_pi0/contact_drawer_controller.py").read_bytes()).hexdigest(),
              "source_commit": commit, "license": "Apache-2.0", "mujoco_version": mujoco.__version__,
              "prompt": task_prompt,
              "source_scene": str(Path(ARTICULATED_SCENE).relative_to(args.source)),
              "scene_sha256": hashlib.sha256(Path(ARTICULATED_SCENE).read_bytes()).hexdigest(),
              "mode": args.mode, "fps": FPS, "frames": len(records), "seconds": float(data.time),
              "expert": args.expert, "grasp_diagnostics": getattr(controller, "grasp_diagnostics", []),
              "expert_parameters": ({"pitch_degrees": controller.contact_pitch,
                  "open_gripper_target": controller.contact_open,
                  "grasp_point_local_m": controller.right.king.tool_offset.tolist(),
                  "bilateral_force_stop_N": 1.0, "nominal_pull_m": controller.pull_distance,
                  "free_pitch": controller.free_pitch,
                  "return_drawer": controller.return_drawer,
                  "approach_offset_world_m": controller.approach_offset.tolist(),
                  "base_elbow_clearance_m": 0.006} if args.expert == "contact" else None),
              "scene_overrides": {"drawer_handle_radius_m": args.handle_radius,
                                  "drawer_offset_y_m": args.drawer_offset_y,
                                  "drawer_offset_x_m": args.drawer_offset_x},
              "image_shape": None if args.no_images else [240, 320, 3],
              "wrist_view": args.wrist_view, "shadow_map_size": 1024,
              "camera_local_poses": {name: {"position": model.cam_pos[model.camera(name).id].tolist(),
                  "quaternion_wxyz": model.cam_quat[model.camera(name).id].tolist(),
                  "fovy_degrees": float(model.cam_fovy[model.camera(name).id])} for name in CAMERAS.values()},
              "camera_mapping": CAMERAS, "actuator_order": ACTUATORS,
              "action_representation": "16D absolute joint position targets, radians, right then left",
              "gravity_compensation": "qfrc_bias on 14 arm DOFs, recomputed each physical substep",
              "initialization": "ready keyframe; upstream _park sets left arm qpos/qvel to zero before recording",
              "initial_right_randomization": {
                  "seed": args.seed, "uniform_half_width_rad": args.initial_right_jitter,
                  "applied_delta_rad": initial_right_delta.tolist(),
                  "sampling": "uniform over intersection of +/-half-width and joint/control limits",
                  "method": "reset right qpos and ctrl before recording; no intermediate state injection"},
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
              "replay_max_qpos_error": max_error, "replay_verified": bool(max_error < 1e-6),
              "stages": stages, "exported_lerobot": False,
              "ready_for_current_cell_training": False,
              "limitations": ["仅一条试采轨迹，无独立验证集", "上游 pedestal 场景，与当前 CELL 场景不同",
                              ("回放需使用记录的重力补偿协议；此模式没有抓取绑定"
                               if args.mode == "contact" else "辅助绑定和重力补偿属于环境机制，16维动作不包含它们；回放必须复现同一机制")]}
    write_json(args.output / "report.json", report)
    if args.export_lerobot:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from openarm_policy import OpenArmInputs
        features = {key: {"dtype": "float32", "shape": (16,), "names": ACTUATORS}
                    for key in ("observation.state", "action")}
        for key in CAMERAS:
            features[f"observation.images.{key}"] = {
                "dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}
        dataset_path = args.output / "lerobot_v21"
        dataset = LeRobotDataset.create(repo_id="local/openarm_drawer_trial", root=dataset_path,
            fps=FPS, robot_type="openarm_bimanual", features=features,
            use_videos=True, image_writer_threads=4)
        for index, r in enumerate(records):
            frame = {"observation.state": r["state"].astype(np.float32),
                     "action": r["action"].astype(np.float32), "task": task_prompt}
            frame.update({f"observation.images.{key}": arrays[f"images_{key}"][index] for key in CAMERAS})
            dataset.add_frame(frame)
        dataset.save_episode()
        # 无验证划分，禁止将同一条轨迹同时标成训练集和验证集。
        info_path = dataset_path / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["splits"] = {"train": "0:1"}
        write_json(info_path, info)
        loaded = LeRobotDataset(repo_id="local/openarm_drawer_trial", root=dataset_path,
            delta_timestamps={"action": [i / FPS for i in range(16)]}, video_backend="pyav")
        for index in range(len(loaded)):
            sample = loaded[index]
            result = OpenArmInputs()(sample)
            assert result["state"].shape == (16,) and result["actions"].shape == (16, 16)
            assert all(image.shape == (240, 320, 3) for image in result["image"].values())
            np.testing.assert_allclose(result["state"], arrays["state"][index], atol=1e-6)
            expected = arrays["action"][np.minimum(index + np.arange(16), len(loaded) - 1)]
            np.testing.assert_allclose(result["actions"], expected, atol=1e-6)
        report.update(exported_lerobot=True, lerobot_frames_verified=len(loaded),
                      lerobot_path=str(dataset_path.relative_to(ROOT)))
        write_json(args.output / "report.json", report)
        write_json(dataset_path / "trial_provenance.json", report)
        (dataset_path / "README.md").write_text(
            "# OpenArm drawer diagnostic trial\n\n"
            "One scripted episode; no independent validation split.\n"
            f"Mode: {args.mode}. Open phase >=8 cm held for 1 s: {sustained_open}.\n"
            f"Open-then-close success: {report['success_open_then_close']}.\n"
            "Replay requires the recorded position actions and gravity compensation; grasp welds follow the mode above.\n"
            "See trial_provenance.json and docs/drawer-data-trial.md in the project.\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
