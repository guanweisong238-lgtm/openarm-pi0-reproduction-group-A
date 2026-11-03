"""在独立 MuJoCo 模型中回放抽屉示教动作，支持实时 GUI 观看及无窗口（--headless）确定性校验。

仅在每轮起点恢复物理状态與抽屉变体几何（外形预设、配色、尺寸缩放、X/Y/Z 偏移）；
随后执行记录的目标、重力补偿与辅助绑定事件，不注入中间关节状态。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
A2_ROOT = ROOT.parent / "openarm-pi0-reproduction-group-A"
A2_PYTHON = A2_ROOT / ".venv/bin/python"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.drawer_scene_generator import compile_drawer_scene, resolve_drawer_variation
from simulation.drawer_simulation import UPSTREAM_COMMIT, resolve_upstream_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        type=Path,
        required=True,
        help="待回放的 episode.npz 路径或包含 episode.npz 的目录",
    )
    parser.add_argument("--source", type=Path, default=None, help="openarm-control 上游目录（默认自动探测）")
    parser.add_argument("--headless", action="store_true", help="无窗口确定性物理回放校验")
    parser.add_argument("--label", default="", help="窗口中的轨迹序号与用途标签")
    parser.add_argument(
        "--review-controls",
        action="store_true",
        help="N 结束当前轨迹，Q 退出整组审阅（返回码 20）",
    )
    args = parser.parse_args()

    episode_path = args.episode.resolve()
    if episode_path.is_dir():
        episode_path = episode_path / "episode.npz"
    directory = episode_path.parent
    provenance = json.loads((directory / "report.json").read_text())

    import mujoco
    import numpy as np

    if mujoco.__version__ != provenance["mujoco_version"] and A2_PYTHON.exists() and Path(sys.executable) != A2_PYTHON:
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{ROOT}:{existing_pp}" if existing_pp else str(ROOT)
        os.execve(str(A2_PYTHON), [str(A2_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env)

    pitch = (provenance.get("expert_parameters") or {}).get("pitch_degrees")
    episode_label = directory.name + (f" | pitch {pitch:g} deg" if pitch is not None else "")
    if args.label:
        episode_label = args.label + " | " + episode_label

    source_dir = resolve_upstream_source(args.source)
    commit = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT or commit != provenance["source_commit"]:
        raise ValueError("上游源码版本与示教不匹配")
    if mujoco.__version__ != provenance["mujoco_version"]:
        raise ValueError(f"MuJoCo 版本与示教不匹配: {mujoco.__version__} vs {provenance['mujoco_version']}")
    scene = source_dir / provenance["source_scene"]
    if hashlib.sha256(scene.read_bytes()).hexdigest() != provenance["scene_sha256"]:
        raise ValueError("场景文件与示教不匹配")

    keys = (
        "qpos",
        "action",
        "native_action",
        "substeps",
        "eq_data",
        "eq_active",
        "initial_integration_state",
        "initial_eq_data",
        "terminal_qpos",
        "actuator_names",
    )
    with np.load(episode_path, allow_pickle=False) as source:
        episode = {key: source[key] for key in keys}

    overrides = provenance.get("scene_overrides", {})
    variation = resolve_drawer_variation(overrides_dict=overrides)
    offset = variation["drawer_offset_x_m"]
    model = compile_drawer_scene(
        str(scene),
        variation,
        camera_local_poses=provenance.get("camera_local_poses"),
        shadow_size=provenance.get("shadow_map_size", 1024),
    )
    data = mujoco.MjData(model)
    policy_ids = np.array([model.actuator(str(name)).id for name in episode["actuator_names"]])
    np.testing.assert_array_equal(episode["native_action"][:, policy_ids], episode["action"])
    arm_joints = [model.joint(f"openarm_{side}_joint{i}").id for side in ("right", "left") for i in range(1, 8)]
    arm_dofs = model.jnt_dofadr[arm_joints]
    drawer_qpos = model.joint("drawer_slide").qposadr[0]
    frames = len(episode["action"])
    if frames != provenance["frames"] or not np.isfinite(episode["native_action"]).all():
        raise ValueError("示教帧数或动作不合法")

    output = ROOT / "outputs/drawer_replay" / f"replay_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True, exist_ok=False)

    def reset():
        model.eq_data[:] = episode["initial_eq_data"]
        mujoco.mj_resetData(model, data)
        mujoco.mj_setState(model, data, episode["initial_integration_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)

    def step(index):
        error = float(np.max(np.abs(data.qpos - episode["qpos"][index])))
        model.eq_data[:] = episode["eq_data"][index]
        data.eq_active[:] = episode["eq_active"][index]
        data.ctrl[:] = episode["native_action"][index]
        for _ in range(int(episode["substeps"][index])):
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = 0
            data.qfrc_applied[arm_dofs] = data.qfrc_bias[arm_dofs]
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        return error

    def finish(max_error, cycle):
        max_error = max(max_error, float(np.max(np.abs(data.qpos - episode["terminal_qpos"]))))
        report = {
            "episode": str(episode_path),
            "variant_preset": variation["variant_preset"],
            "appearance_preset": variation["appearance_preset"],
            "source_commit": commit,
            "mode": provenance["mode"],
            "cycle": cycle,
            "frames_executed": frames,
            "simulation_seconds": float(data.time),
            "max_qpos_error": max_error,
            "replay_verified": max_error < 1e-6,
            "final_open_m": -float(data.qpos[drawer_qpos]),
            "gui": not args.headless,
            "task_success_8cm_held_1s": provenance.get(
                "opened_8cm_held_1s_during_episode", provenance["success_8cm_held_1s"]
            ),
            "task_success_open_then_close": provenance.get("success_open_then_close", False),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if max_error >= 1e-6:
            raise RuntimeError("回放轨迹偏离已保存示教")

    reset()
    if args.headless:
        error = max(step(index) for index in range(frames))
        finish(error, 1)
        return

    import mujoco.viewer

    paused, restart = threading.Event(), threading.Event()
    advance_review, quit_review = threading.Event(), threading.Event()

    def on_key(key):
        if key == 32:
            paused.clear() if paused.is_set() else paused.set()
        elif key in (ord("R"), ord("r")):
            restart.set()
        elif args.review_controls and key in (ord("N"), ord("n")):
            advance_review.set()
        elif args.review_controls and key in (ord("Q"), ord("q")):
            quit_review.set()

    with mujoco.viewer.launch_passive(
        model, data, key_callback=on_key, show_left_ui=False, show_right_ui=False
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.30 + offset / 2, -0.24, 0.57 + variation["drawer_offset_z_m"] / 2]
            viewer.cam.distance = 0.95
            viewer.cam.azimuth = -140
            viewer.cam.elevation = -25
        viewer.sync()
        print(
            f"GUI_READY: {episode_label}; {output}; Space=pause, R=restart; looping {provenance['mode']} demonstration",
            flush=True,
        )
        cycle, index, max_error = 1, 0, 0.0
        resume_at = time.perf_counter() + 3.0
        while viewer.is_running():
            if advance_review.is_set() or quit_review.is_set():
                break
            began = time.perf_counter()
            with viewer.lock():
                if restart.is_set():
                    reset()
                    index, max_error = 0, 0.0
                    resume_at = began + 1.0
                    restart.clear()
                if index < frames and not paused.is_set() and began >= resume_at:
                    max_error = max(max_error, step(index))
                    index += 1
                    if index == frames:
                        finish(max_error, cycle)
                        resume_at = began + 2.0
                if index == frames and began >= resume_at:
                    reset()
                    cycle, index, max_error = cycle + 1, 0, 0.0
                    resume_at = began + 1.0
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"{episode_label}\n{variation['variant_preset']} ({variation['appearance_preset']})\nSpace / R",
                    f"Recorded actions\n30 Hz\n{-data.qpos[drawer_qpos]*1000:.1f} mm\nPause / Restart",
                )
            )
            viewer.sync()
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - began)))
    if quit_review.is_set():
        raise SystemExit(20)


if __name__ == "__main__":
    main()
