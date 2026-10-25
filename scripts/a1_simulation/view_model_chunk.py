# 本文件负责：在CELL窗口中可视化执行模型预测动作块的前几个时间步。
# 示教数据仅用于恢复推理快照对应的初始姿态，不用于回放示教动作。
# 核对当前状态与推理输入一致后，执行已保存的模型预测并记录实际状态。
# 执行后停止推进物理时间，保持窗口供观察；不进行循环推理或任务成功判定。

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from simulation.env import ROOT
from simulation.policy_env import OpenArmPolicyEnv, PolicyInterfaceConfig
from scripts.a1_simulation.replay_episode import (
    advance,
    load_episode,
    prepare_pose,
)


# 一、读取模型推理产物及对应的原始观测。
def load_prediction(directory):
    """读取原始模型动作和输入状态，检查形状、数值及来源记录。"""
    summary = json.loads((directory / "summary.json").read_text())
    actions = np.load(directory / "actions.npy", allow_pickle=False)

    if actions.shape != (16, 16) or not np.isfinite(actions).all():
        raise ValueError("模型预测必须是有限的(16,16)动作块")

    with np.load(summary["observation_path"], allow_pickle=False) as data:
        state = data["observation.state"].copy()

    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("推理输入状态必须是有限的16维向量")

    return summary, actions, state


# 二、打开窗口、恢复首帧并短执行模型预测。
def main():
    """先核对输入状态，再逐步执行模型动作，保存记录后冻结画面。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    summary, actions, expected_state = load_prediction(args.inference)
    bundle = load_episode(args.episode)
    if not 1 <= args.steps <= len(actions):
        raise ValueError("steps必须在1到16之间")

    config = PolicyInterfaceConfig.from_file()
    if not np.isclose(float(bundle["fps"]), config.control_hz):
        raise ValueError("初始化数据频率与环境配置不一致")

    scene = ROOT / "openarm_mujoco/v2/cell/demo.xml"
    output = (
        ROOT / "outputs/a2_interface"
        / f"model_visible_{datetime.now():%Y%m%d-%H%M%S-%f}"
    )
    output.mkdir(parents=True, exist_ok=False)

    with OpenArmPolicyEnv(config, scene_path=scene) as env:
        with mujoco.viewer.launch_passive(
            env.env.model, env.env.data
        ) as viewer:
            # 使用CELL原有顶部相机，方便与示教图像比较。
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(
                    env.env.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    "camera_ceiling",
                )
            viewer.sync()

            print("先恢复模型推理时的初始姿态……", flush=True)
            try:
                prepare_pose(env, viewer, bundle["states"][0])
            except (InterruptedError, KeyboardInterrupt):
                print("初始化中止，未执行模型动作。", flush=True)
                return

            current = env.get_observation(render=False)
            np.testing.assert_allclose(
                current["observation.state"],
                expected_state,
                atol=1e-6,
                rtol=0,
                err_msg="当前状态与模型推理输入不同，停止执行",
            )

            print("输入状态核对通过。3秒后执行模型动作……", flush=True)
            # 等待期间只刷新画面，不推进物理，保持输入状态不变。
            deadline = time.monotonic() + 3
            while viewer.is_running() and time.monotonic() < deadline:
                viewer.sync()
                time.sleep(0.02)

            states = [current["observation.state"].copy()]
            applied, timestamps = [], [current["timestamp"]]

            try:
                for index, action in enumerate(actions[:args.steps]):
                    observation = advance(env, viewer, action)
                    states.append(observation["observation.state"].copy())
                    applied.append(observation["control.applied"].copy())
                    timestamps.append(observation["timestamp"])
                    print(
                        f"已执行模型动作：{index + 1}/{args.steps}",
                        flush=True,
                    )
            except (InterruptedError, KeyboardInterrupt):
                print("执行中止，保存已完成部分。", flush=True)

            # 三、分别记录模型请求、实际下发目标和实际测得状态。
            count = len(applied)
            applied_array = np.asarray(applied).reshape(-1, 16)
            requested = actions[:count]
            np.savez_compressed(
                output / "execution.npz",
                states=np.asarray(states),
                requested_actions=requested,
                applied_actions=applied_array,
                timestamps=np.asarray(timestamps),
            )
            report = {
                "execution_type": "saved_model_prediction_short_chunk",
                "checkpoint": summary["checkpoint"],
                "inference_directory": str(args.inference.resolve()),
                "scene_path": str(scene),
                "initial_state_matches_snapshot": True,
                "executed_steps": count,
                "completed": count == args.steps,
                "elapsed_simulation_seconds": timestamps[-1] - timestamps[0],
                "clipped_steps": int(np.any(
                    np.abs(applied_array - requested) > 1e-6, axis=1
                ).sum()),
                "closed_loop_verified": False,
                "task_success_verified": False,
            }
            (output / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            )
            print("结果目录:", output, flush=True)
            print("已停止推进物理时间。观察后关闭窗口结束。", flush=True)

            # 四、冻结物理状态，仅刷新窗口，避免保持目标时继续运动。
            try:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
            except KeyboardInterrupt:
                pass


# 五、命令行入口。
if __name__ == "__main__":
    main()