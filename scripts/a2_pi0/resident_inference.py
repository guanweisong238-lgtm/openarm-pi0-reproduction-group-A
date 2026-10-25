# 本文件负责：常驻加载A2模型，通过文件请求为A1持续提供动作预测。
# 启动时加载检查点并预热一次，之后每个新观测只执行一次推理。
# 观测与动作使用无pickle的NumPy文件，JSON负责请求和状态反馈。
# 不推进仿真时间，不执行动作，不判断抓取成功。

# 一、先设置项目环境，再导入模型依赖。
from training_config import ROOT, get_train_config

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
from openpi.policies import policy_config

from infer_sim_snapshot import load_snapshot


# 二、原子发布JSON，避免读取方看到尚未写完的内容。
def publish(path, value):
    """先写临时文件，再原子替换目标JSON文件。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)


# 三、加载、预热并持续处理请求。
def main():
    """模型加载一次，逐个处理观测快照，直到收到停止文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints/pi0_openarm_lora/smoke_001/10",
        help="本次常驻进程使用的检查点，默认保留10步基线",
    )
    args = parser.parse_args()

    workspace = args.workspace
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        cfg = get_train_config()
        checkpoint = args.checkpoint.resolve()
        if not (checkpoint / "params").is_dir():
            raise FileNotFoundError(checkpoint / "params")
        policy = policy_config.create_trained_policy(
            cfg, checkpoint, sample_kwargs={"num_steps": 10}
        )

        # 固定噪声便于与此前检查比较；后续可另外评估采样策略。
        noise = np.random.default_rng(42).standard_normal(
            (cfg.model.action_horizon, cfg.model.action_dim)
        ).astype(np.float32)

        observation, _, _ = load_snapshot(args.warmup)
        warmup_actions = np.asarray(
            policy.infer(observation, noise=noise)["actions"]
        )
        if (
            warmup_actions.shape != (16, 16)
            or not np.isfinite(warmup_actions).all()
        ):
            raise ValueError("预热输出异常")

        publish(workspace / "ready.json", {
            "ready": True,
            "checkpoint": str(checkpoint),
        })

        while not (workspace / "stop").exists():
            # 请求和响应文件一一对应；已有响应的请求不重复执行。
            for request_path in sorted(workspace.glob("request_*.json")):
                response_path = workspace / request_path.name.replace(
                    "request_", "response_", 1
                )
                if response_path.exists():
                    continue

                try:
                    request = json.loads(request_path.read_text())
                    observation, lower, upper = load_snapshot(
                        Path(request["observation"])
                    )

                    began = time.perf_counter()
                    actions = np.asarray(
                        policy.infer(observation, noise=noise)["actions"]
                    )
                    elapsed = time.perf_counter() - began

                    if (
                        actions.shape != (16, 16)
                        or not np.isfinite(actions).all()
                    ):
                        raise ValueError("预测动作形状或数值异常")

                    directory = Path(request["output"])
                    np.save(directory / "actions.npy", actions)

                    clipped = np.clip(actions, lower, upper)
                    result = {
                        "ok": True,
                        "observation_path": request["observation"],
                        "checkpoint": str(checkpoint),
                        "inference_seconds": elapsed,
                        "out_of_range_steps": int(np.any(
                            actions != clipped, axis=1
                        ).sum()),
                    }
                    publish(directory / "summary.json", result)

                except Exception:
                    result = {
                        "ok": False,
                        "error": traceback.format_exc(),
                    }

                publish(response_path, result)

            time.sleep(0.01)

    except Exception:
        publish(workspace / "fatal.json", {
            "error": traceback.format_exc(),
        })
        raise


# 四、命令行入口。
if __name__ == "__main__":
    main()
