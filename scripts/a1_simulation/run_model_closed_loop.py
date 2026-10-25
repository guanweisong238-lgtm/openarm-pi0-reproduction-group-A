# 本文件负责：在CELL场景中运行有限轮次的模型闭环联调。
# 每轮重新采集实际状态及三路RGB，调用A2独立环境推理，再执行动作块前几步。
# 推理期间暂停物理时间，保持窗口刷新；记录各轮观测、模型目标和实际状态。
# 整次运行复用一个常驻A2模型进程，启动时预热一次，每轮只推理一次。
# 当前推理期间仍暂停物理时间，不自动判定抓取成功或实时性。
# 可指定检查点及任务仿真时长；任务模式只用示教首帧初始化，后续全部由模型控制。

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from simulation.env import ROOT
from scripts.a1_simulation.resident_policy_client import ResidentPolicyClient
from simulation.policy_env import OpenArmPolicyEnv, PolicyInterfaceConfig
from scripts.a1_simulation.replay_episode import (
    advance,
    load_episode,
    prepare_pose,
)


# 一、独立项目路径，不依赖同级openpi源码目录。
A2_ROOT = ROOT.parent / "openarm-pi0-reproduction-group-A"
A2_PYTHON = A2_ROOT / ".venv/bin/python"
INFER_SCRIPT = A2_ROOT / "scripts/a2_pi0/resident_inference.py"


# 二、采集当前观测，保存给A2独立进程读取。
def export_observation(env, path):
    """保存同一物理时刻的实际状态、RGB、指令及模型顺序限位。"""
    observation = env.get_observation(render=True)
    keys = (
        "observation.state",
        "prompt",
        "timestamp",
        "observation.images.ceiling",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    )
    np.savez_compressed(
        path,
        **{key: np.asarray(observation[key]) for key in keys},
        action_lower=env.to_policy(env.lower),
        action_upper=env.to_policy(env.upper),
    )
    return observation


# 三、执行每轮闭环，并将结果保存到独立目录。
def main():
    """恢复示教首帧后，有限次重复新观测、推理和短动作执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path,
        default=A2_ROOT / "checkpoints/pi0_openarm_lora/smoke_001/10")
    parser.add_argument("--task-seconds", type=float,
        help="任务阶段仿真时长，指定后按时长计算总步数并替代cycles")
    parser.add_argument("--no-hold", action="store_true",
        help="结束后关闭窗口，默认冻结并保持窗口供观察")
    parser.add_argument("--interface-config", type=Path, default=ROOT / "configs/a2_interface.json", help="接口及外观配置；默认保留原设置")
    args = parser.parse_args()

    if args.cycles < 1 or not 1 <= args.steps <= 16:
        raise ValueError("cycles必须为正数，steps必须在1到16之间")
    if not A2_PYTHON.is_file() or not INFER_SCRIPT.is_file():
        raise FileNotFoundError("A2虚拟环境或推理脚本不存在")

    bundle = load_episode(args.episode)
    config = PolicyInterfaceConfig.from_file(args.interface_config)
    if args.task_seconds is not None:
        if not np.isfinite(args.task_seconds) or args.task_seconds <= 0:
            raise ValueError("任务时长必须为正有限数")
        total_steps = max(1, round(args.task_seconds * config.control_hz))
        args.cycles = math.ceil(total_steps / args.steps)
    else:
        total_steps = args.cycles * args.steps
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(checkpoint / "params")
    if not np.isclose(float(bundle["fps"]), config.control_hz):
        raise ValueError("初始化数据频率与环境配置不一致")

    output = (
        ROOT / "outputs/a2_interface"
        / f"closed_loop_{datetime.now():%Y%m%d-%H%M%S-%f}"
    )
    output.mkdir(parents=True, exist_ok=False)
    scene = ROOT / "openarm_mujoco/v2/cell/demo.xml"
    rounds = []
    completed = False
    # 延迟到首个观测就绪后启动；所有轮次复用同一个模型进程。
    client = ResidentPolicyClient(checkpoint=checkpoint)
    executed_steps = 0
    task_start_time = None

    print("结果目录:", output, flush=True)
    print("模型检查点:", checkpoint, flush=True)
    print(f"计划执行{total_steps}步，约{total_steps / config.control_hz:.3f}秒仿真时间", flush=True)

    with OpenArmPolicyEnv(config, scene_path=scene) as env:
        with mujoco.viewer.launch_passive(
            env.env.model, env.env.data
        ) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(
                    env.env.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    "camera_ceiling",
                )
            viewer.sync()

            try:
                print("先移动到示教首帧……", flush=True)
                initial_error = prepare_pose(
                    env, viewer, bundle["states"][0]
                )
                print(
                    f"首帧最大误差：{initial_error:.6f} rad",
                    flush=True,
                )
                task_start_time = env.env.data.time

                for cycle in range(args.cycles):
                    if not viewer.is_running():
                        raise InterruptedError("窗口已关闭")

                    directory = output / f"cycle_{cycle:03d}"
                    directory.mkdir()
                    snapshot = directory / "observation.npz"

                    # 每轮重新采集，不能复用第一轮图像或示教状态。
                    with viewer.lock():
                        observation = export_observation(env, snapshot)

                    print(
                        f"[{cycle + 1}/{args.cycles}] "
                        f"新观测 t={observation['timestamp']:.3f}s，开始推理……",
                        flush=True,
                    )
                    actions, inference_directory = client.infer(
                        snapshot, directory / "inference.log", viewer
                    )

                    # 等待推理期间物理时间应保持不变。
                    if env.env.data.time != observation["timestamp"]:
                        raise RuntimeError("推理等待期间物理时间发生变化")

                    states = [observation["observation.state"].copy()]
                    applied = []
                    timestamps = [observation["timestamp"]]

                    # 即使执行中断，也保存本轮已完成的控制步。
                    try:
                        # 最后一轮可少于steps，严格限制任务总控制步数。
                        cycle_steps = min(args.steps, total_steps - executed_steps)
                        for action in actions[:cycle_steps]:
                            after = advance(env, viewer, action)
                            states.append(
                                after["observation.state"].copy()
                            )
                            applied.append(
                                after["control.applied"].copy()
                            )
                            timestamps.append(after["timestamp"])
                            executed_steps += 1
                    finally:
                        count = len(applied)
                        applied_array = np.asarray(applied).reshape(-1, 16)
                        np.savez_compressed(
                            directory / "execution.npz",
                            states=np.asarray(states),
                            requested_actions=actions[:count],
                            applied_actions=applied_array,
                            timestamps=np.asarray(timestamps),
                        )
                        rounds.append({
                            "cycle": cycle,
                            "observation_timestamp": observation["timestamp"],
                            "inference_directory": str(inference_directory),
                            "executed_steps": count,
                            "inference_seconds": client.last_inference_seconds,
                            "request_seconds": client.last_request_seconds,
                            "end_timestamp": timestamps[-1],
                            "clipped_steps": int(np.any(
                                np.abs(
                                    applied_array - actions[:count]
                                ) > 1e-6,
                                axis=1,
                            ).sum()),
                        })

                    print(
                        f"[{cycle + 1}/{args.cycles}] "
                        f"已执行{count}步，接下来重新采集观测。",
                        flush=True,
                    )

                completed = True
                # 保存最终三路图像，任务成功由观察或后续独立判据确认。
                with viewer.lock():
                    final_observation = env.get_observation(render=True)
                for camera in ("ceiling", "wrist_left", "wrist_right"):
                    Image.fromarray(final_observation[
                        f"observation.images.{camera}"
                    ]).save(output / f"final_{camera}.png")

            except (InterruptedError, KeyboardInterrupt):
                print("已中止，保存已有记录。", flush=True)
            finally:
                # 正常结束、窗口关闭和异常退出时都释放模型进程。
                client.close()
                report = {
                    "execution_type": "model_task_attempt" if args.task_seconds is not None else "model_closed_loop_smoke",
                    "checkpoint": str(checkpoint),
                    "interface_config": str(args.interface_config.resolve()),
                    "cell_visual_alpha": config.cell_visual_alpha,
                    "initialization_episode": str(args.episode.resolve()),
                    "demonstration_actions_used_for_control": False,
                    "requested_control_steps": total_steps,
                    "elapsed_task_simulation_seconds": (
                        env.env.data.time - task_start_time
                        if task_start_time is not None else 0
                    ),
                    "scene_path": str(scene),
                    "requested_cycles": args.cycles,
                    "steps_per_cycle": args.steps,
                    "completed": completed,
                    "executed_steps": sum(
                        row["executed_steps"] for row in rounds
                    ),
                    "physics_paused_during_inference": True,
                    "inference_mode": "resident_process",
                    "warmup_calls": 1 if client.process is not None else 0,
                    "rounds": rounds,
                    "task_success_verified": False,
                }
                (output / "report.json").write_text(
                    json.dumps(
                        report, indent=2, ensure_ascii=False
                    ) + "\n"
                )

            # 四、结束后冻结物理状态，保留窗口供观察。
            print("报告已保存:", output / "report.json", flush=True)
            if args.no_hold:
                return
            print("已停止推进物理时间，关闭窗口结束。", flush=True)
            try:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
            except KeyboardInterrupt:
                pass


# 五、命令行入口。
if __name__ == "__main__":
    main()
