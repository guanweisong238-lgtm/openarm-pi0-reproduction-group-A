# 本文件负责：可视化回放A2导出的单条示教，记录跟随误差和实际限幅。
# 默认加载CELL场景及其顶部相机，支持只定位首帧并保存三路图像。
# 先平滑移动到示教首帧，再按数据频率逐条下发位置目标。
# 比较的是执行动作前的实际状态与示教状态，避免错开一个时间步。
# 本脚本不修改控制增益或限位；结果只作探索性回放诊断，不代表抓取成功。

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco.viewer
import numpy as np
from PIL import Image

from simulation.env import ROOT
from simulation.policy_env import (
    OpenArmPolicyEnv,
    PolicyInterfaceConfig,
    POLICY_ACTUATORS,
)


# 一、读取并校验回放包
def load_episode(path):
    """读取纯数组NPZ，检查关节顺序、数值和时间索引，返回回放数据。"""
    with np.load(path, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}

    expected = [
        name.removesuffix("_ctrl").replace("_finger1", "_gripper") + ".pos"
        for name in POLICY_ACTUATORS
    ]
    assert bundle["joint_names"].tolist() == expected

    states, actions = bundle["states"], bundle["actions"]
    assert states.ndim == 2 and states.shape[1] == 16
    assert states.shape == actions.shape
    assert len(states) > 0
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    np.testing.assert_allclose(
        bundle["timestamps"],
        np.arange(len(states)) / float(bundle["fps"]),
        atol=1e-5,
        rtol=0,
    )
    return bundle


# 二、执行一个时间步并刷新窗口
def advance(env, viewer, action):
    """执行已反归一化的位置目标，按名义周期限制显示速度。

    窗口提前关闭时抛出InterruptedError，由主流程保存已完成部分。
    """
    if not viewer.is_running():
        raise InterruptedError("窗口已关闭")

    began = time.monotonic()
    with viewer.lock():
        observation = env.step(action, render=False)
    viewer.sync()
    time.sleep(max(
        0, 1 / env.config.control_hz - (time.monotonic() - began)
    ))
    return observation


# 三、平滑定位到示教起点
def prepare_pose(env, viewer, target):
    """两秒移动、五秒保持，返回首帧定位后的最大实际位置误差。"""
    native = env.to_native(target)
    if np.any(native < env.lower) or np.any(native > env.upper):
        raise ValueError("首帧姿态超出仿真限位")

    start = env.get_observation(render=False)["control.applied"]
    count = round(2 * env.config.control_hz)
    for index in range(1, count + 1):
        alpha = 0.5 - 0.5 * np.cos(np.pi * index / count)
        advance(env, viewer, start + alpha * (target - start))
    for _ in range(round(5 * env.config.control_hz)):
        observation = advance(env, viewer, target)

    return float(np.abs(observation["observation.state"] - target).max())


# 四、保存实际完成的回放片段和诊断指标
def save_result(output, bundle, records, initial_error, completed, *, scene_path=None):
    """保存逐帧记录和汇总；提前中断时明确标记，不报告完整回放成功。"""
    output.mkdir(parents=True, exist_ok=True)
    count = len(records)

    actual = np.asarray(
        [row["actual_before"] for row in records], dtype=np.float64
    ).reshape(-1, 16)
    applied = np.asarray(
        [row["applied_action"] for row in records], dtype=np.float64
    ).reshape(-1, 16)
    requested = bundle["actions"][:count]
    reference = bundle["states"][:count]
    errors = actual - reference
    clipped = np.abs(applied - requested) > 1e-5

    np.savez_compressed(
        output / "replay.npz",
        actual_before=actual,
        reference_state=reference,
        requested_action=requested,
        applied_action=applied,
        timestamps=np.asarray([row["time"] for row in records]),
        reference_timestamps=bundle["timestamps"][:count],
        joint_names=bundle["joint_names"],
    )
    report = {
        "scene_path": str(scene_path) if scene_path is not None else None,
        "completed": completed,
        "executed_frames": count,
        "expected_frames": len(bundle["actions"]),
        "initial_pose_max_abs_error": initial_error,
        "initial_pose_within_0_02_rad": initial_error < 0.02,
        "mean_abs_state_error": float(np.abs(errors).mean()) if count else None,
        "max_abs_state_error": float(np.abs(errors).max()) if count else None,
        "clipped_frames": int(np.any(clipped, axis=1).sum()),
        "clipped_frames_per_joint": clipped.sum(axis=0).tolist(),
        "joint_names": bundle["joint_names"].tolist(),
        "calibration_verified": False,
        "task_success_verified": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print("结果目录:", output, flush=True)


# 五、保存首帧定位后的相机样张和保持目标
def save_initial_pose(env, output, target, initial_error):
    """保存同一物理时刻的三路RGB和首帧误差，不推进物理时间。"""
    output.mkdir(parents=True, exist_ok=True)
    observation = env.get_observation(render=True)
    # 使用纯数组快照在A1和A2独立Python环境之间交换观测，禁止pickle依赖。
    # 同时导出模型顺序的限位，以便推理端诊断预测目标，不在此处修改预测。
    np.savez_compressed(
        output / "policy_observation.npz",
        **{
            key: np.asarray(observation[key])
            for key in (
                "observation.state", "prompt", "timestamp",
                "observation.images.ceiling",
                "observation.images.wrist_left",
                "observation.images.wrist_right",
            )
        },
        action_lower=env.to_policy(env.lower),
        action_upper=env.to_policy(env.upper),
    )
    for key, label in (
        ("observation.images.ceiling", "ceiling"),
        ("observation.images.wrist_left", "wrist_left"),
        ("observation.images.wrist_right", "wrist_right"),
    ):
        path = output / f"{label}.png"
        Image.fromarray(observation[key]).save(path)
        print("首帧样张:", path, flush=True)
    summary = {
        "timestamp": observation["timestamp"],
        "target_state": np.asarray(target).tolist(),
        "actual_state": observation["observation.state"].tolist(),
        "max_abs_state_error": initial_error,
        "calibration_verified": False,
        "task_success_verified": False,
    }
    (output / "initial_pose.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


def hold_pose(env, viewer, target):
    """持续下发固定目标并刷新窗口，关闭窗口或Ctrl+C后返回。"""
    while viewer.is_running():
        try:
            advance(env, viewer, target)
        except (InterruptedError, KeyboardInterrupt):
            break


# 六、可视化回放主流程
def main():
    """加载指定场景，定位并导出首帧，可选择保持观察或继续回放。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument(
        "--scene", type=Path,
        default=ROOT / "openarm_mujoco/v2/cell/demo.xml",
        help="场景MJCF路径，默认CELL demo",
    )
    parser.add_argument(
        "--pose-only", action="store_true",
        help="只定位示教首帧并保持，关闭窗口结束，不播放动作序列",
    )
    parser.add_argument("--output", type=Path, help="结果目录，必须尚不存在")
    args = parser.parse_args()
    args.scene = args.scene.resolve()
    bundle = load_episode(args.episode)

    config = PolicyInterfaceConfig.from_file()
    if not np.isclose(float(bundle["fps"]), config.control_hz):
        raise ValueError("回放频率与环境配置不一致")

    # 每次运行单独保存，避免覆盖先前pedestal或CELL的诊断记录。
    output = args.output or (
        ROOT / "outputs/a2_interface" /
        f"{args.scene.parent.name}_{args.episode.stem}_"
        f"{datetime.now():%Y%m%d-%H%M%S-%f}"
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps({
        "scene_path": str(args.scene),
        "episode_path": str(args.episode.resolve()),
        "pose_only": args.pose_only,
        "control_hz": config.control_hz,
    }, indent=2, ensure_ascii=False) + "\n")
    print("场景:", args.scene, flush=True)
    print("结果目录:", output, flush=True)
    records = []
    initial_error = None
    completed = False

    with OpenArmPolicyEnv(config, scene_path=args.scene) as env:
        with mujoco.viewer.launch_passive(
            env.env.model, env.env.data
        ) as viewer:
            with viewer.lock():
                # 使用原生固定顶部相机，与离屏导出的ceiling图像采用相同外参。
                # 窗口宽高比可能不同；比较数据时使用导出的960×600图像。
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(
                    env.env.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_ceiling"
                )
            viewer.sync()

            try:
                print("先移动到示教首帧并稳定……", flush=True)
                initial_error = prepare_pose(
                    env, viewer, bundle["states"][0]
                )
                print(
                    f"首帧定位最大误差：{initial_error:.6f} rad",
                    flush=True,
                )
                with viewer.lock():
                    save_initial_pose(
                        env, output, bundle["states"][0], initial_error
                    )
                if args.pose_only:
                    print("首帧定位结束，保持姿态供观察。关闭窗口结束。", flush=True)
                    hold_pose(env, viewer, bundle["states"][0])
                    return

                print(f"开始回放{len(bundle['actions'])}帧示教……", flush=True)
                time_zero = env.env.data.time

                for index, action in enumerate(bundle["actions"]):
                    # 数据中的state_t对应执行action_t之前的状态。
                    before = env.get_observation(render=False)
                    after = advance(env, viewer, action)
                    records.append({
                        "actual_before": before["observation.state"],
                        "applied_action": after["control.applied"],
                        "time": before["timestamp"] - time_zero,
                    })
                    if index % 30 == 0:
                        print(
                            f"回放进度：{index + 1}/{len(bundle['actions'])}",
                            flush=True,
                        )

                completed = True
            except (InterruptedError, KeyboardInterrupt):
                print("回放提前中止，将保存已完成部分。", flush=True)
            finally:
                # 初始化未完成时不生成具有误导性的回放报告。
                if initial_error is not None and not args.pose_only:
                    save_result(
                        output, bundle, records, initial_error, completed,
                        scene_path=args.scene,
                    )

            if completed:
                print("示教播放结束，保持最终目标。关闭窗口结束。", flush=True)
                hold_pose(env, viewer, bundle["actions"][-1])


# 七、命令行入口
if __name__ == "__main__":
    main()
