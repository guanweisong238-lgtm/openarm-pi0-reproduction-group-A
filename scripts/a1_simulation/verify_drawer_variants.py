"""A1 抽屉多形态与多高度改动一键可控验证脚本（位于 openarm-a1/scripts/a1_simulation/）。

提供三个递进且完全可控的验证层级：
1. --mode scene  （秒级）：编译全部 5 套抽屉预设，校验底座贴合桌面（z=0.40m）、无穿模，并导出三路相机拼图与总览对比图。
2. --mode expert （约15秒）：在全部 5 套抽屉变体上运行真实接触专家（禁用 weld），校验开合行程 >=80mm 且独立回放误差 <1e-6。
3. --mode policy （使用可用权重）：加载 checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000
                  在基准抽屉与新外形/新高度变体抽屉上运行闭环推理（支持无头导出 MP4 视频或 --gui 实时可视化窗口）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
A2_ROOT = ROOT.parent / "openarm-pi0-reproduction-group-A"
A2_PYTHON = A2_ROOT / ".venv/bin/python"

if any(arg in ("policy", "all", "--gui") for arg in sys.argv[1:]):
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.drawer_scene_generator import (
    build_scene_overrides_dict,
    compile_drawer_scene,
    resolve_drawer_variation,
)
from simulation.drawer_simulation import DrawerSimulation, finger_handle_contacts, resolve_upstream_source

DEFAULT_CHECKPOINT = A2_ROOT / "checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000"
CAMERAS = ("camera_ceiling", "camera_wrist_left", "camera_wrist_right")


def _ensure_a2_python_for_mode(mode: str) -> None:
    """当执行 expert / objects_zed / policy 模式且当前环境缺少 scipy/jax 时，自动切换到 A2 .venv 执行。"""
    needs_switch = False
    if mode in ("expert", "objects_zed", "all") and importlib.util.find_spec("scipy") is None:
        needs_switch = True
    if mode in ("policy", "all") and importlib.util.find_spec("jax") is None:
        needs_switch = True
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    if mode in ("policy", "all"):
        env["MUJOCO_GL"] = "glfw"
        env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    if needs_switch and A2_PYTHON.exists() and Path(sys.executable) != A2_PYTHON:
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{ROOT}:{existing_pp}" if existing_pp else str(ROOT)
        os.execve(str(A2_PYTHON), [str(A2_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env)
    elif mode in ("policy", "all") and "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ:
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def verify_scene_mode(config_path: Path, selected_names: list[str] | None) -> dict:
    """模式 1：编译全部抽屉变体场景，验证几何约束并生成三路相机渲染对比图。"""
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    config = json.loads(config_path.read_text())
    variants = config.get("variants") or config.get("episodes") or []
    if selected_names:
        variants = [v for v in variants if v["name"] in selected_names]

    source_dir = resolve_upstream_source()
    scene_xml = source_dir / "v2/openarm_mujoco_v2/articulated_scene.xml"
    preview_dir = ROOT / "outputs/drawer_variants_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    variant_Rows_images = []
    for item in variants:
        variation = resolve_drawer_variation(overrides_dict=item)
        model = compile_drawer_scene(str(scene_xml), variation, wrist_view="side", shadow_size=1024)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
        mujoco.mj_forward(model, data)

        penetration = min((c.dist for c in data.contact), default=0.0)
        base_bottom_z = variation["derived_geometry"]["base_bottom_world_z_m"]
        assert abs(base_bottom_z - 0.40) < 1e-6, f"底座未贴合桌面: {base_bottom_z}"
        assert penetration >= -0.002, f"初始穿模过大: {penetration}"

        renderer = mujoco.Renderer(model, height=240, width=320)
        views = []
        try:
            for cam in CAMERAS:
                renderer.update_scene(data, camera=cam)
                views.append(renderer.render().copy())
        finally:
            renderer.close()

        composite = np.concatenate(views, axis=1)
        img = Image.fromarray(composite)
        draw = ImageDraw.Draw(img)
        label = (
            f"{item['name']} | preset={variation['variant_preset']} | "
            f"appearance={variation['appearance_preset']} | "
            f"pos_offset=({variation['drawer_offset_x_m']:+.2f}, {variation['drawer_offset_y_m']:+.2f}, {variation['drawer_offset_z_m']:+.2f})m"
        )
        draw.rectangle([0, 0, 960, 24], fill=(20, 20, 20))
        draw.text((8, 5), label, fill=(255, 255, 255))
        out_png = preview_dir / f"{item['name']}.png"
        img.save(out_png)
        variant_Rows_images.append( np.asarray(img) )

        row_info = {
            "name": item["name"],
            "variant_preset": variation["variant_preset"],
            "appearance_preset": variation["appearance_preset"],
            "config_sha256": variation["config_sha256"],
            "derived_geometry": variation["derived_geometry"],
            "min_contact_dist_mm": round(float(penetration * 1000), 3),
            "base_flush_with_table": True,
            "preview_image": str(out_png.relative_to(ROOT)),
        }
        rows.append(row_info)
        print(
            f"[SCENE PASS] {item['name']}: handle_pos={variation['derived_geometry']['handle_world_pos_closed_m']} "
            f"base_z={base_bottom_z:.2f}m preview={out_png.relative_to(ROOT)}",
            flush=True,
        )

    if variant_Rows_images:
        grid = np.concatenate(variant_Rows_images, axis=0)
        grid_path = preview_dir / "all_variants_comparison.png"
        Image.fromarray(grid).save(grid_path)
        print(f"[SCENE GRID] 五款抽屉总览对比图已保存至: {grid_path.relative_to(ROOT)}", flush=True)

    report = {"mode": "scene", "all_passed": True, "variants": rows}
    out_dir = ROOT / "outputs/drawer_variants_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene_verification_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def verify_expert_mode(config_path: Path, selected_names: list[str] | None, with_images: bool = False) -> dict:
    """模式 2：在各款抽屉变体上运行真实接触专家采集与独立无头回放校验。"""
    config = json.loads(config_path.read_text())
    variants = config.get("variants") or config.get("episodes") or []
    if selected_names:
        variants = [v for v in variants if v["name"] in selected_names]

    out_root = ROOT / "outputs/drawer_variants_verify/expert_trials"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for item in variants:
        variation = resolve_drawer_variation(overrides_dict=item)
        ep_dir = out_root / item["name"]
        cmd = [
            sys.executable,
            str(ROOT / "scripts/a1_simulation/collect_drawer_trial.py"),
            "--mode",
            "contact",
            "--expert",
            "contact",
            "--variant-preset",
            variation["variant_preset"],
            "--appearance-preset",
            variation["appearance_preset"],
            "--drawer-offset-x",
            str(variation["drawer_offset_x_m"]),
            "--drawer-offset-y",
            str(variation["drawer_offset_y_m"]),
            "--drawer-offset-z",
            str(variation["drawer_offset_z_m"]),
            "--pull-distance",
            str(item.get("pull_distance", 0.09)),
            "--wrist-view",
            "side",
            "--output",
            str(ep_dir),
        ]
        if not with_images:
            cmd.append("--no-images")
        subprocess.run(cmd, check=True)

        replay_cmd = [
            sys.executable,
            str(ROOT / "scripts/a1_simulation/replay_drawer_trial.py"),
            "--episode",
            str(ep_dir / "episode.npz"),
            "--headless",
        ]
        subprocess.run(replay_cmd, check=True)

        rep = json.loads((ep_dir / "report.json").read_text())
        summary = {
            "name": item["name"],
            "variant_preset": item["variant_preset"],
            "appearance_preset": item["appearance_preset"],
            "expert_returned_success": rep["expert_returned_success"],
            "success_8cm_held_1s": rep["success_8cm_held_1s"],
            "peak_open_mm": round(rep["peak_open_m"] * 1000, 2),
            "final_open_mm": round(rep["final_open_m"] * 1000, 2),
            "weld_frames": rep["weld_frames"],
            "replay_max_qpos_error": rep["replay_max_qpos_error"],
            "replay_verified": rep["replay_verified"],
        }
        assert summary["expert_returned_success"] and summary["success_8cm_held_1s"], f"专家未达标: {summary}"
        assert summary["weld_frames"] == 0, "出现了非法 weld 帧"
        assert summary["replay_verified"], "独立回放误差超限"
        results.append(summary)
        print(
            f"[EXPERT PASS] {item['name']}: peak_open={summary['peak_open_mm']}mm "
            f"weld={summary['weld_frames']} replay_err={summary['replay_max_qpos_error']:.2e}",
            flush=True,
        )

    report = {"mode": "expert", "all_passed": True, "variants": results}
    verify_dir = ROOT / "outputs/drawer_variants_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    (verify_dir / "expert_verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def _build_variant_start_episode(
    episode_dir: Path,
    reference_dir: Path,
    variation_spec: dict,
) -> Path:
    """基于参考 episode_012 的 ready 积分初态，快速构造指定抽屉变体的闭环评估起点。"""
    import mujoco
    import numpy as np

    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)

    ref_report = json.loads((reference_dir / "report.json").read_text())
    with np.load(reference_dir / "episode.npz", allow_pickle=False) as raw:
        ref_npz = {k: raw[k].copy() for k in raw.files}

    source_dir = resolve_upstream_source()
    scene_xml = source_dir / ref_report["source_scene"]
    variation = resolve_drawer_variation(overrides_dict=variation_spec)
    model = compile_drawer_scene(str(scene_xml), variation, wrist_view="side", shadow_size=1024)
    data = mujoco.MjData(model)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    model.eq_data[:] = ref_npz["initial_eq_data"]
    mujoco.mj_resetData(model, data)
    mujoco.mj_setState(model, data, ref_npz["initial_integration_state"], spec)
    mujoco.mj_forward(model, data)

    state = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    poses = {
        name: {
            "position": model.cam_pos[model.camera(name).id].tolist(),
            "quaternion_wxyz": model.cam_quat[model.camera(name).id].tolist(),
            "fovy_degrees": float(model.cam_fovy[model.camera(name).id]),
        }
        for name in CAMERAS
    }
    np.savez_compressed(
        episode_dir / "episode.npz",
        initial_integration_state=state,
        initial_eq_data=model.eq_data.copy(),
        actuator_names=ref_npz["actuator_names"],
    )
    report = {
        **ref_report,
        "mujoco_version": mujoco.__version__,
        "start_only": True,
        "frames": 0,
        "scene_overrides": build_scene_overrides_dict(variation),
        "wrist_view": "side",
        "camera_local_poses": poses,
    }
    (episode_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return episode_dir


def verify_policy_mode(
    checkpoint: Path,
    config_path: Path,
    selected_names: list[str] | None,
    seconds: float = 16.0,
    gui: bool = False,
) -> dict:
    """模式 3：使用现有模型权重在基准抽屉与变体抽屉上运行闭环推理验证。"""
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    sys.path[:0] = [
        str(A2_ROOT / "scripts/a2_pi0"),
        str(A2_ROOT / ".cache/drawer-deps"),
        str(A2_ROOT / ".cache/openarm-control"),
    ]
    import cv2
    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from drawer_start_fit_config import get_policy_config
    from live_drawer_success import update_judge
    from openpi import transforms
    from openpi.models import model as _model
    from openpi.policies import policy as _policy
    from openpi.shared import normalize

    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"未找到模型权重目录: {checkpoint}")

    protocol = json.loads((checkpoint / "input_protocol.json").read_text())
    horizon = int(protocol.get("action_horizon", 16))
    seed = int(protocol.get("fixed_noise_seed", 42))
    action_chunk = 4

    print(
        f"[POLICY] 正在加载模型权重 (CPU->GPU bfloat16 低峰值加载): {checkpoint} (backend={jax.default_backend()})...",
        flush=True,
    )
    train_cfg = get_policy_config(protocol)
    params_np = _model.restore_params(checkpoint / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)
    loaded_model = train_cfg.model.load(jax.device_put(params_np))
    del params_np
    loaded_model.eval()
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    norm_stats = normalize.load(checkpoint / "assets" / data_cfg.asset_id)
    policy = _policy.Policy(
        loaded_model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_cfg.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.model_transforms.inputs,
        ],
        output_transforms=[
            *data_cfg.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": int(protocol.get("sampling_steps", 10))},
        metadata=train_cfg.policy_metadata,
    )
    print("[POLICY] 模型权重加载完成！", flush=True)

    ref_ep = A2_ROOT / "data/processed/openarm_drawer_success_v003/episodes/episode_012"
    if not ref_ep.exists():
        raise FileNotFoundError(f"参考起点不存在: {ref_ep}")

    config = json.loads(config_path.read_text())
    variants = config.get("variants") or config.get("episodes") or []
    if selected_names:
        variants = [v for v in variants if v["name"] in selected_names]
    else:
        # 默认评测：1) 与 016 训练同分布的标准抽屉 (dx=0.10)  2) 宽矮浅橡木新变体  3) 窄高胡桃木新变体
        variants = [
            {
                "name": "v01_standard_id12_dx010",
                "variant_preset": "standard",
                "appearance_preset": "classic_wood_blue",
                "drawer_offset_x": 0.10,
                "drawer_offset_y": 0.0,
                "drawer_offset_z": 0.0,
            },
            variants[1],  # v02_wide_low_oak
            variants[2],  # v03_narrow_tall_walnut
        ]

    out_dir = ROOT / "outputs/drawer_variants_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    starts_dir = out_dir / "policy_starts"

    rollout_results = []
    for item in variants:
        ep_dir = _build_variant_start_episode(starts_dir / item["name"], ref_ep, item)
        env = DrawerSimulation(ep_dir)
        noise_rng = np.random.default_rng(seed)
        noise = noise_rng.standard_normal((horizon, 32)).astype(np.float32)
        left_hold = env.data.ctrl[env.actuator_ids[8:]].copy()
        state: dict = {}
        peak_open_m = 0.0
        total_steps = round(seconds * 30)
        video_path = out_dir / f"policy_rollout_{item['name']}.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30 / action_chunk, (960, 240)
        )

        viewer_ctx = None
        if gui:
            import mujoco.viewer

            viewer_ctx = mujoco.viewer.launch_passive(
                env.model, env.data, show_left_ui=False, show_right_ui=False
            )
            viewer_ctx.__enter__()
            viewer_ctx.cam.lookat[:] = [0.40, -0.24, 0.53]
            viewer_ctx.cam.distance = 0.90
            viewer_ctx.cam.azimuth = -140
            viewer_ctx.cam.elevation = -25

        try:
            while env.steps < total_steps:
                observation = env.observation()
                block = np.asarray(policy.infer(observation, noise=noise)["actions"])
                if protocol.get("right8_loss"):
                    block = block.copy()
                    block[:, 8:] = left_hold

                rgb_concat = np.concatenate(
                    [observation[f"observation.images.{k}"] for k in ("ceiling", "wrist_left", "wrist_right")],
                    axis=1,
                )
                frame_bgr = cv2.cvtColor(rgb_concat, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    frame_bgr,
                    f"{item['name']} t={env.steps/30:.2f}s open={state.get('opening', 0.0)*1000:.1f}mm",
                    (8, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                writer.write(frame_bgr)

                for action in block[: min(action_chunk, total_steps - env.steps)]:
                    env.step(action.copy())
                    contacts, forces = finger_handle_contacts(env.model, env.data, include_stems=True)
                    opening = -float(env.data.qpos[env.drawer_qpos])
                    peak_open_m = max(peak_open_m, opening)
                    state = update_judge(state, opening, contacts, forces, task="open")
                    if viewer_ctx is not None:
                        viewer_ctx.sync()
                        time.sleep(1 / 30)
                    if state["success"]:
                        break
                if env.steps % 40 == 0 or state.get("success"):
                    print(
                        f"  [{item['name']}] step={env.steps}/{total_steps} "
                        f"open={opening*1000:.1f}mm grasped={state.get('grasped', False)} "
                        f"success={state.get('success', False)}",
                        flush=True,
                    )
                if state.get("success"):
                    break
        finally:
            writer.release()
            if viewer_ctx is not None:
                viewer_ctx.__exit__(None, None, None)
            env.close()

        summary = {
            "name": item["name"],
            "variant_preset": env.variation["variant_preset"],
            "appearance_preset": env.variation["appearance_preset"],
            "drawer_offset_xyz_m": [
                env.variation["drawer_offset_x_m"],
                env.variation["drawer_offset_y_m"],
                env.variation["drawer_offset_z_m"],
            ],
            "steps_executed": env.steps,
            "peak_open_mm": round(peak_open_m * 1000, 2),
            "final_open_mm": round(state.get("opening", 0.0) * 1000, 2),
            "grasped": bool(state.get("grasped", False)),
            "opened_8cm_held_1s": bool(state.get("opened", False)),
            "task_open_success": bool(state.get("success", False)),
            "video": str(video_path.relative_to(ROOT)),
        }
        rollout_results.append(summary)
        print(
            f"[POLICY ROLLOUT] {item['name']}: success={summary['task_open_success']} "
            f"grasped={summary['grasped']} peak_open={summary['peak_open_mm']}mm video={summary['video']}",
            flush=True,
        )

    report = {
        "mode": "policy",
        "checkpoint": str(checkpoint),
        "rollouts": rollout_results,
    }
    (out_dir / "policy_verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def verify_objects_zed_mode(
    config_path: Path | None = None,
    selected_names: list[str] | None = None,
    with_images: bool = False,
) -> dict:
    """模式 4（objects_zed）：验证 4 种物体（方块/橡皮/胶棒/空塑料瓶）放入空心抽屉 + 头部 Stereolabs ZED Mini 双目相机。"""
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    if config_path is None or config_path.name == "drawer_variants_preset.json":
        config_path = ROOT / "configs/drawer_objects_zed_preset.json"
    config = json.loads(config_path.read_text())
    variants = config.get("variants") or []
    if selected_names:
        variants = [v for v in variants if v["name"] in selected_names]

    source_dir = resolve_upstream_source()
    scene_xml = source_dir / "v2/openarm_mujoco_v2/articulated_scene.xml"
    preview_dir = ROOT / "outputs/drawer_variants_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    out_root = ROOT / "outputs/drawer_variants_verify/objects_zed_trials"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. 渲染 ZED Mini 头部双目外观特写 + 左/右双目视野 + 顶视 4 物体同台 + 双抽屉总览图
    overview_var = resolve_drawer_variation(
        variant_preset="standard",
        appearance_preset="classic_wood_blue",
        hollow_drawer=True,
        initial_drawer_open_m=0.09,
        drawer_layout="dual_side_by_side",
        secondary_variant_preset="wide_low",
        secondary_appearance_preset="light_oak_white",
        objects=["cube", "eraser", "glue_stick", "plastic_bottle"],
        target_object="cube",
        enable_head_stereo=True,
    )
    ov_model = compile_drawer_scene(str(scene_xml), overview_var, wrist_view="side", shadow_size=1024)
    ov_data = mujoco.MjData(ov_model)
    mujoco.mj_resetDataKeyframe(ov_model, ov_data, ov_model.key("ready").id)
    mujoco.mj_forward(ov_model, ov_data)

    left_pos = ov_data.cam_xpos[ov_model.camera("camera_head_left").id]
    right_pos = ov_data.cam_xpos[ov_model.camera("camera_head_right").id]
    stereo_baseline_mm = float(np.linalg.norm(left_pos - right_pos) * 1000.0)
    assert abs(stereo_baseline_mm - 63.0) < 0.1, f"ZED Mini 双目基线偏离 63mm: {stereo_baseline_mm}"

    renderer = mujoco.Renderer(ov_model, height=240, width=320)
    try:
        # 外部特写相机观察机器人头部安装的 ZED Mini 3D 双目实体
        ext_cam = mujoco.MjvCamera()
        ext_cam.lookat[:] = [0.10, 0.0, 0.72]
        ext_cam.distance = 0.58
        ext_cam.azimuth = 148.0
        ext_cam.elevation = -16.0
        renderer.update_scene(ov_data, camera=ext_cam)
        img_zed_ext = renderer.render().copy()

        renderer.update_scene(ov_data, camera="camera_head_left")
        img_head_l = renderer.render().copy()

        renderer.update_scene(ov_data, camera="camera_head_right")
        img_head_r = renderer.render().copy()

        renderer.update_scene(ov_data, camera="camera_ceiling")
        img_ceil = renderer.render().copy()
    finally:
        renderer.close()

    header_row = np.concatenate([img_zed_ext, img_head_l, img_head_r, img_ceil], axis=1)
    header_pil = Image.fromarray(header_row)
    h_draw = ImageDraw.Draw(header_pil)
    for idx, title in enumerate(
        (
            "External View: Head ZED Mini (63mm)",
            "ZED Mini Left (camera_head_left)",
            "ZED Mini Right (camera_head_right)",
            "Overhead: 4 Objects & Dual Hollow Drawers",
        )
    ):
        h_draw.rectangle([320 * idx, 0, 320 * (idx + 1) - 1, 22], fill=(18, 22, 28))
        h_draw.text((320 * idx + 6, 4), title, fill=(255, 255, 255))

    rows_images = [np.asarray(header_pil)]
    results = []

    # 2. 对清单中的每个物体与抽屉变体执行真实接触抓取放入抽屉 + 独立回放校验
    for item in variants:
        variation = resolve_drawer_variation(overrides_dict=item)
        ep_dir = out_root / item["name"]
        cmd = [
            sys.executable,
            str(ROOT / "scripts/a1_simulation/collect_drawer_trial.py"),
            "--mode",
            "contact",
            "--expert",
            "contact",
            "--variant-preset",
            variation["variant_preset"],
            "--appearance-preset",
            variation["appearance_preset"],
            "--drawer-offset-x",
            str(variation["drawer_offset_x_m"]),
            "--drawer-offset-y",
            str(variation["drawer_offset_y_m"]),
            "--drawer-offset-z",
            str(variation["drawer_offset_z_m"]),
            "--pull-distance",
            str(item.get("pull_distance", 0.09)),
            "--initial-drawer-open",
            str(variation["initial_drawer_open_m"]),
            "--drawer-layout",
            variation["drawer_layout"],
            "--camera-preset",
            str(item.get("camera_preset", "zed_4cam")),
            "--wrist-view",
            "side",
            "--output",
            str(ep_dir),
        ]
        if variation.get("hollow_drawer"):
            cmd.append("--hollow-drawer")
        if variation["head_stereo_camera"]["enabled"]:
            cmd.append("--enable-head-stereo")
        if variation.get("objects"):
            cmd.extend(["--objects", *[o["type"] for o in variation["objects"]]])
        if variation.get("target_object"):
            cmd.extend(["--target-object", variation["target_object"]])
        if not with_images:
            cmd.append("--no-images")
        subprocess.run(cmd, check=True)

        replay_cmd = [
            sys.executable,
            str(ROOT / "scripts/a1_simulation/replay_drawer_trial.py"),
            "--episode",
            str(ep_dir / "episode.npz"),
            "--headless",
        ]
        subprocess.run(replay_cmd, check=True)

        rep = json.loads((ep_dir / "report.json").read_text())
        with np.load(ep_dir / "episode.npz", allow_pickle=False) as ep_npz:
            term_qpos = ep_npz["terminal_qpos"].copy()

        # 渲染该轨迹完成时刻的四视图（ZED 左目、ZED 右目、顶视、右腕）
        v_model = compile_drawer_scene(str(scene_xml), variation, wrist_view="side", shadow_size=1024)
        v_data = mujoco.MjData(v_model)
        v_data.qpos[:] = term_qpos
        mujoco.mj_forward(v_model, v_data)
        v_renderer = mujoco.Renderer(v_model, height=240, width=320)
        try:
            row_views = []
            for cname in ("camera_head_left", "camera_head_right", "camera_ceiling", "camera_wrist_right"):
                v_renderer.update_scene(v_data, camera=cname)
                row_views.append(v_renderer.render().copy())
        finally:
            v_renderer.close()

        row_comp = np.concatenate(row_views, axis=1)
        row_pil = Image.fromarray(row_comp)
        r_draw = ImageDraw.Draw(row_pil)
        banner = (
            f"{item['name']} | prompt=\"{rep['prompt']}\" | "
            f"inside_drawer={rep['target_object_inside_drawer']} | rel={rep['target_object_rel_to_drawer_m']}m"
        )
        r_draw.rectangle([0, 0, 1280, 22], fill=(20, 20, 20))
        r_draw.text((8, 4), banner, fill=(255, 255, 255))
        item_png = preview_dir / f"{item['name']}.png"
        row_pil.save(item_png)
        rows_images.append(np.asarray(row_pil))

        summary = {
            "name": item["name"],
            "prompt": rep["prompt"],
            "variant_preset": variation["variant_preset"],
            "appearance_preset": variation["appearance_preset"],
            "drawer_layout": variation["drawer_layout"],
            "initial_drawer_open_m": variation["initial_drawer_open_m"],
            "target_object": rep["target_object"],
            "target_object_inside_drawer": rep["target_object_inside_drawer"],
            "target_object_rel_to_drawer_m": rep["target_object_rel_to_drawer_m"],
            "expert_returned_success": rep["expert_returned_success"],
            "final_open_mm": round(rep["final_open_m"] * 1000, 2),
            "weld_frames": rep["weld_frames"],
            "replay_max_qpos_error": rep["replay_max_qpos_error"],
            "replay_verified": rep["replay_verified"],
            "preview_image": str(item_png.relative_to(ROOT)),
        }
        assert summary["expert_returned_success"] and summary["target_object_inside_drawer"], f"物体放入抽屉未达标: {summary}"
        assert summary["weld_frames"] == 0, "出现了非法 weld 帧"
        assert summary["replay_verified"], "独立回放误差超限"
        results.append(summary)
        print(
            f"[OBJECTS+ZED PASS] {item['name']}: target={summary['target_object']} "
            f"inside={summary['target_object_inside_drawer']} rel={summary['target_object_rel_to_drawer_m']} "
            f"weld={summary['weld_frames']} replay_err={summary['replay_max_qpos_error']:.2e}",
            flush=True,
        )

    overview_png = preview_dir / "zed_mini_and_objects_overview.png"
    Image.fromarray(np.concatenate(rows_images, axis=0)).save(overview_png)
    print(f"[OBJECTS+ZED OVERVIEW] ZED Mini 双目与四物体入抽屉总览图已保存至: {overview_png.relative_to(ROOT)}", flush=True)

    report = {
        "mode": "objects_zed",
        "all_passed": True,
        "zed_mini_stereo_baseline_mm": round(stereo_baseline_mm, 3),
        "overview_image": str(overview_png.relative_to(ROOT)),
        "variants": results,
    }
    verify_dir = ROOT / "outputs/drawer_variants_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    (verify_dir / "objects_zed_verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("scene", "expert", "objects_zed", "policy", "all"),
        default="scene",
        help="验证层级：scene（几何与渲染）/ expert（抽屉开合专家）/ objects_zed（四物体入抽屉+ZED双目）/ policy（模型闭环推理）/ all（全部）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/drawer_variants_preset.json",
        help="抽屉预设变体配置文件路径",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="仅验证指定名称的变体（默认验证预设清单中的全部变体）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="policy 模式使用的模型检查点路径",
    )
    parser.add_argument("--seconds", type=float, default=16.0, help="policy 模式每条闭环推理时长（秒）")
    parser.add_argument("--gui", action="store_true", help="policy 模式下弹出 MuJoCo 实时 3D 窗口")
    parser.add_argument("--with-images", action="store_true", help="expert/objects_zed 模式下同时渲染保存多路 RGB 与 MP4")
    args = parser.parse_args()

    _ensure_a2_python_for_mode(args.mode)

    if args.mode in ("scene", "all"):
        verify_scene_mode(args.config, args.variants)
    if args.mode in ("expert", "all"):
        verify_expert_mode(args.config, args.variants, with_images=args.with_images)
    if args.mode in ("objects_zed", "all"):
        verify_objects_zed_mode(args.config, args.variants, with_images=args.with_images)
    if args.mode in ("policy", "all"):
        verify_policy_mode(args.checkpoint, args.config, args.variants, seconds=args.seconds, gui=args.gui)


if __name__ == "__main__":
    main()
