# 本文件负责：在固定验证帧上比较模型预测与示教位置目标，报告原始16维动作误差。
# 每条验证轨迹选取首帧与中间帧；不把动作标签输入策略，不用验证集更新参数。
# 剔除超出本轨迹末端的填充动作，以固定噪声保证不同检查点比较条件一致。
# 此为12帧抽样离线指标，不代表完整验证集结果或仿真抓取成功率。

# 一、先加载环境配置与项目内的LeRobot子集兼容类。
from training_config import ROOT, DATA_ROOT, get_train_config
from local_data_loader import EpisodeSubsetDataset

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from openpi.policies import policy_config


# 二、固定抽样并累计原始位置目标的误差。
def main():
    """加载一个检查点，抽样评估验证轨迹24至29的动作预测。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames-per-episode", type=int, default=2, help="每条轨迹均匀采样帧数；0表示全部")
    parser.add_argument("--split", choices=("train","val"), default="val")
    parser.add_argument("--episode", type=int, help="限定当前划分内的一条轨迹，用于过拟合诊断")
    args = parser.parse_args()
    if args.frames_per_episode < 0:
        raise ValueError("采样数不能为负")
    cfg = get_train_config()
    info = json.loads((DATA_ROOT / "meta/info.json").read_text())
    start, stop = map(int, info["splits"][args.split].split(":"))
    episodes = list(range(start, stop))
    if args.episode is not None:
        if args.episode not in episodes:
            raise ValueError("指定轨迹不属于当前划分")
        episodes = [args.episode]
    dataset = EpisodeSubsetDataset(
        repo_id="local/openarm_pickup", root=DATA_ROOT,
        episodes=episodes, video_backend="pyav",
        delta_timestamps={"action": [i / info["fps"] for i in range(16)]},
    )
    episode_ids = np.asarray([int(value) for value in dataset.hf_dataset["episode_index"]])
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, sample_kwargs={"num_steps": 10})
    noise = np.random.default_rng(42).standard_normal((16, 32)).astype(np.float32)
    absolute_sum, squared_sum = np.zeros(16), np.zeros(16)
    valid_count = 0
    records = []
    for episode in episodes:
        indices = np.flatnonzero(episode_ids == episode)
        # 均匀覆盖开始、接近、抓取、搬运及释放，包含末帧并保留填充掩码。
        selected = indices if args.frames_per_episode == 0 else indices[np.unique(
            np.linspace(0, len(indices)-1, min(len(indices), args.frames_per_episode), dtype=int)
        )]
        for index in selected:
            sample = dataset[int(index)]
            observation = {
                "observation.state": np.asarray(sample["observation.state"]),
                "prompt": sample["task"],
                **{key: np.asarray(sample[key]) for key in (
                    "observation.images.ceiling", "observation.images.wrist_left",
                    "observation.images.wrist_right",
                )},
            }
            began = time.perf_counter()
            prediction = np.asarray(policy.infer(observation, noise=noise)["actions"])
            elapsed = time.perf_counter() - began
            if prediction.shape != (16, 16) or not np.isfinite(prediction).all():
                raise ValueError("预测形状或数值异常")
            target = np.asarray(sample["action"])
            valid = ~np.asarray(sample["action_is_pad"], dtype=bool)
            error = (prediction - target)[valid].astype(np.float64)
            absolute_sum += np.abs(error).sum(axis=0)
            squared_sum += np.square(error).sum(axis=0)
            valid_count += len(error)
            record = {"episode": episode, "subset_index": int(index),
                "valid_action_steps": len(error), "mae": float(np.abs(error).mean()),
                "seconds": elapsed}
            records.append(record)
            print(json.dumps(record), flush=True)

    # 三、按有效时间步加权统计；手臂和夹爪分开报告。
    mae = absolute_sum / valid_count
    rmse = np.sqrt(squared_sum / valid_count)
    arms = [i for i in range(16) if i not in (7, 15)]
    summary = {"checkpoint": str(args.checkpoint.resolve()),
        "split": args.split, "episodes": episodes, "frames_per_episode": args.frames_per_episode, "sample_count": len(records), "valid_action_steps": valid_count,
        "mae_per_joint": mae.tolist(), "rmse_per_joint": rmse.tolist(),
        "mae_all": float(mae.mean()), "mae_arm": float(mae[arms].mean()),
        "mae_gripper": float(mae[[7, 15]].mean()), "samples": records,
        "task_success_verified": False,
        "limitation": "仅按配置均匀抽样的原始位置目标误差，不代表闭环成功率"}
    output = ROOT / "logs/a2_pi0" / f"action_eval_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"PASS：动作离线抽样评估完成，MAE={mae.mean():.6f}，记录={output}", flush=True)


# 四、命令行入口。
if __name__ == "__main__":
    main()
