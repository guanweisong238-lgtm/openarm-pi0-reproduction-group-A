# 本文件负责：将指定轨迹的状态、动作、时间和关节名称导出为独立回放包。
# 输入为已转换的 LeRobot v2.1 数据，输出为不依赖 LeRobot 的 NPZ 文件。
# 保留右臂在前的16维顺序，不修改单位、不限幅、不执行仿真。
# 回放包供A1独立环境读取；同名输出会被替换，原始数据保持不变。

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# 一、项目路径
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/processed/openarm_pickup_v21"


def load_episode(episode_index):
    """读取一条完整轨迹，并验证状态、动作及固定频率时间戳。

    参数：episode_index 为数据集中的轨迹编号。
    返回：包含数值数组和字符串元数据的字典，可直接保存为NPZ。
    异常：轨迹不存在、字段不一致或数值/时间检查失败时停止。
    """
    # 二、读取数据定义和任务文本
    info = json.loads((DATA_ROOT / "meta/info.json").read_text())
    tasks = {}
    for line in (DATA_ROOT / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            task = json.loads(line)
            tasks[task["task_index"]] = task["task"]

    action_names = info["features"]["action"]["names"]
    state_names = info["features"]["observation.state"]["names"]
    if action_names != state_names or len(action_names) != 16:
        raise ValueError("状态和动作的16维名称或顺序不一致")

    # 三、只收集指定轨迹，按轨迹内帧号排序
    rows = []
    columns = [
        "episode_index", "frame_index", "timestamp",
        "observation.state", "action", "task_index",
    ]
    for path in sorted((DATA_ROOT / "data").rglob("*.parquet")):
        rows.extend(
            row
            for row in pq.read_table(path, columns=columns).to_pylist()
            if row["episode_index"] == episode_index
        )
    if not rows:
        raise ValueError(f"未找到轨迹：{episode_index}")
    rows.sort(key=lambda row: row["frame_index"])

    states = np.asarray(
        [row["observation.state"] for row in rows],
        dtype=np.float32,
    )
    actions = np.asarray(
        [row["action"] for row in rows],
        dtype=np.float32,
    )
    timestamps = np.asarray(
        [row["timestamp"] for row in rows],
        dtype=np.float64,
    )
    fps = float(info["fps"])

    # 四、验证形状和时间，确保回放包不混入其他轨迹
    count = len(rows)
    assert states.shape == actions.shape == (count, 16)
    assert np.isfinite(states).all()
    assert np.isfinite(actions).all()
    assert np.isfinite(timestamps).all()
    np.testing.assert_array_equal(
        [row["frame_index"] for row in rows], np.arange(count)
    )
    np.testing.assert_allclose(
        timestamps, np.arange(count) / fps,
        atol=1e-5, rtol=0,
    )

    task_ids = {row["task_index"] for row in rows}
    if len(task_ids) != 1:
        raise ValueError("当前回放包要求一条轨迹只有一个任务指令")
    prompt = tasks[next(iter(task_ids))]

    return {
        "states": states,
        "actions": actions,
        "timestamps": timestamps,
        "joint_names": np.asarray(action_names, dtype=str),
        "episode_index": np.asarray(episode_index),
        "fps": np.asarray(fps),
        "prompt": np.asarray(prompt),
        "source_directory": np.asarray(str(DATA_ROOT)),
        "units": np.asarray("待采集端确认，导出时未做单位换算"),
    }


def main():
    """解析轨迹编号，导出回放包并打印首帧信息供联合核对。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    bundle = load_episode(args.episode)
    output = (
        ROOT / "data/processed/replay"
        / f"episode_{args.episode:06d}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **bundle)

    print("轨迹编号:", args.episode)
    print("指令:", bundle["prompt"].item())
    print("状态形状:", bundle["states"].shape)
    print("动作形状:", bundle["actions"].shape)
    print("频率:", bundle["fps"].item())
    print("首帧状态:", bundle["states"][0])
    print("关节顺序:", bundle["joint_names"].tolist())
    print("回放包:", output)
    print("PASS：示教回放包导出通过")


# 五、命令行入口
if __name__ == "__main__":
    main()