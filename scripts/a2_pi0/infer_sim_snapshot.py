# 本文件负责：读取A1导出的仿真观测快照，用已训练的LoRA检查点完成推理。
# A1与A2使用独立虚拟环境，通过无pickle的NPZ传递原始状态、RGB和指令。
# 保存已反归一化的16维动作块、耗时和限位诊断，不连接仿真或执行动作。

# 一、先设置项目缓存及JAX显存选项，再导入推理相关依赖。
from training_config import ROOT, get_train_config

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from openpi.policies import policy_config


# 二、快照读取：只接收既定三路RGB、原始状态和任务指令。
def load_snapshot(path):
    """校验A1原始快照，返回模型观测及用于诊断的16维位置限位。

    快照不包含示教动作标签；不在A2重复交换左右臂或预先归一化状态。
    """
    with np.load(path, allow_pickle=False) as archive:
        observation = {
            "observation.state": archive["observation.state"].copy(),
            "prompt": str(archive["prompt"].item()),
        }
        for name in ("ceiling", "wrist_left", "wrist_right"):
            key = f"observation.images.{name}"
            value = archive[key].copy()
            if value.shape != (600, 960, 3) or value.dtype != np.uint8:
                raise ValueError(f"快照图像格式不符合约定：{key} {value.shape} {value.dtype}")
            observation[key] = value
        lower = archive["action_lower"].copy()
        upper = archive["action_upper"].copy()
    state = observation["observation.state"]
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("快照状态必须为有限16维向量")
    if not observation["prompt"].strip():
        raise ValueError("任务指令为空")
    if lower.shape != (16,) or upper.shape != (16,):
        raise ValueError("位置限位必须为16维")
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError("位置限位不合法")
    return observation, lower, upper


# 三、模型推理：采用检查点中的归一化统计，输出恢复到原始位置目标。
def main():
    """执行两次固定噪声推理，检查动作有限值与重复性并保存诊断证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints/pi0_openarm_lora/smoke_001/10",
    )
    args = parser.parse_args()
    observation, lower, upper = load_snapshot(args.observation)
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(args.checkpoint / "params")
    output = ROOT / "logs/a2_pi0" / f"sim_infer_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True, exist_ok=False)
    cfg = get_train_config()
    print("观测快照:", args.observation.resolve(), flush=True)
    print("检查点:", args.checkpoint.resolve(), flush=True)
    print("结果目录:", output, flush=True)
    policy = policy_config.create_trained_policy(
        cfg, args.checkpoint, sample_kwargs={"num_steps": 10}
    )
    noise = np.random.default_rng(42).standard_normal(
        (cfg.model.action_horizon, cfg.model.action_dim)
    ).astype(np.float32)
    predictions, timings = [], []
    for index in range(2):
        began = time.perf_counter()
        actions = np.asarray(policy.infer(observation, noise=noise)["actions"])
        elapsed = time.perf_counter() - began
        if actions.shape != (16, 16) or not np.isfinite(actions).all():
            raise ValueError(f"模型动作格式或数值异常：{actions.shape}")
        predictions.append(actions.copy())
        timings.append(elapsed)
        print(f"推理{index + 1}: shape={actions.shape}, 耗时={elapsed:.3f}s", flush=True)
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-5, atol=1e-6)

    # 四、诊断原始预测的超限情况，不把限幅后的动作当作原始模型输出。
    actions = predictions[-1]
    clipped = np.clip(actions, lower, upper)
    np.save(output / "actions.npy", actions)
    np.save(output / "clipped_actions.npy", clipped)
    summary = {
        "observation_path": str(args.observation.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "action_shape": list(actions.shape),
        "finite": True,
        "repeatability_passed": True,
        "inference_seconds": timings,
        "out_of_range_steps": int(np.any(actions != clipped, axis=1).sum()),
        "out_of_range_per_joint": np.sum(actions != clipped, axis=0).tolist(),
        "max_clipping_amount": float(np.abs(actions - clipped).max()),
        "first_action_max_abs_delta_from_state": float(np.abs(
            actions[0] - observation["observation.state"]
        ).max()),
        "simulation_actions_executed": 0,
        "task_success_verified": False,
        "limitation": "10步试训练模型，仅验证仿真观测到动作输出链路",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("PASS：仿真快照推理通过，尚未执行模型动作", flush=True)


# 五、命令行入口。
if __name__ == "__main__":
    main()
