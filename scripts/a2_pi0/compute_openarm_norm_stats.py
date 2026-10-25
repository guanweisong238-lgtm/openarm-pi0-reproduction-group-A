# 本文件负责：仅使用训练集计算 OpenArm 状态和动作的归一化统计。
# 读取转换后的 LeRobot v2.1 数值数据，计算均值、标准差及分位数，处理恒定动作维度。
# 将统计和数据来源记录保存到 assets/pi0_openarm_lora/openarm_pickup/，并验证 OpenPI 可回读。
# 本脚本不修改原始数据、不解码视频，也不训练模型。

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize


# 一、数据和统计产物路径
# 按脚本所在位置定位项目，避免统计文件落到其他 OpenPI 工作副本中。
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/openarm_pickup_v21"
OUTPUT = ROOT / "assets/pi0_openarm_lora/openarm_pickup"


def main():
    """计算训练集逐维统计，保存 OpenPI 格式文件和来源清单。

    输入由 DATA 指定，使用 meta/info.json 中左闭右开的 train 轨迹区间。
    输出为 OUTPUT 下的 norm_stats.json 与 provenance.json，已有同名统计会重写。
    发现轨迹缺失、帧数不符或非有限数值时停止；不纳入验证集或动作块尾部填充帧。
    """
    # 二、解析训练划分并收集原始帧
    # 必须先按轨迹筛选，再计算统计，避免验证信息进入训练预处理。
    info = json.loads((DATA / "meta/info.json").read_text())
    start, stop = map(int, info["splits"]["train"].split(":"))
    train_ids = set(range(start, stop))

    states = []
    actions = []
    observed_ids = set()

    # 直接读取数值列，不解码视频。
    for path in sorted((DATA / "data").rglob("*.parquet")):
        table = pq.read_table(
            path,
            columns=["episode_index", "observation.state", "action"],
        )
        for row in table.to_pylist():
            eid = row["episode_index"]
            if eid in train_ids:
                observed_ids.add(eid)
                states.append(row["observation.state"])
                actions.append(row["action"])

    # 三、针对当前已验收数据快照做完整性检查
    # 4548 帧和 16 维是本项目数据的约定；更换数据版本时需要重新核对。
    assert observed_ids == train_ids, "训练轨迹不完整"
    # 用 float64 累加统计，减少均值和近似恒定维度标准差的数值误差。
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    assert states.shape == actions.shape == (4548, 16)
    assert np.isfinite(states).all()
    assert np.isfinite(actions).all()

    stats = {}
    near_constant = {}

    # 四、每一维独立计算统计；每个原始训练帧只计一次权重
    # 键名使用适配后的 state/actions，而不是数据集中的 observation.state/action。
    for key, values in (("state", states), ("actions", actions)):
        mean = values.mean(axis=0)
        raw_std = values.std(axis=0)

        # 恒定夹爪维度不能除以零。
        std = np.maximum(raw_std, 1e-6)
        near_constant[key] = np.flatnonzero(raw_std < 1e-6).tolist()

        # 当前训练使用均值/标准差；分位数同步保存，不能据此认为分位数归一化已验收。
        stats[key] = normalize.NormStats(
            mean=mean,
            std=std,
            q01=np.quantile(values, 0.01, axis=0),
            q99=np.quantile(values, 0.99, axis=0),
        )

        # 数值往返校验只证明统计运算可逆，不证明动作单位或机器人标定正确。
        normalized = (values - mean) / std
        assert np.isfinite(normalized).all()
        reconstructed = normalized * std + mean
        np.testing.assert_allclose(
            reconstructed, values, rtol=1e-6, atol=1e-7
        )

        print(f"{key}: {values.shape}")
        print("近似恒定的维度:", near_constant[key])
        print("归一化及反归一化检查通过")

    # 五、保存、回读与来源追踪
    # 训练和推理必须使用同一份统计；新数据重算后也需更新对应实验记录。
    OUTPUT.mkdir(parents=True, exist_ok=True)
    normalize.save(OUTPUT, stats)

    # 使用 OpenPI 自己的接口回读，确认文件格式兼容。
    loaded = normalize.load(OUTPUT)
    for key in stats:
        np.testing.assert_allclose(loaded[key].mean, stats[key].mean)
        np.testing.assert_allclose(loaded[key].std, stats[key].std)

    manifest = {
        "dataset": str(DATA.relative_to(ROOT)),
        "train_episode_ids": sorted(train_ids),
        "frames": len(states),
        "normalization": "mean_std",
        "action_representation": "original values, no delta conversion",
        "dimension_order": info["features"]["action"]["names"],
        "std_floor": 1e-6,
        "near_constant_dimensions": near_constant,
        "weighting": "each original training frame counted once",
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    print("统计文件:", OUTPUT / "norm_stats.json")
    print("PASS：训练集统计保存及OpenPI回读通过")


# 六、命令行入口
if __name__ == "__main__":
    main()
