import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/openarm_pickup_v21"
OUTPUT = ROOT / "assets/pi0_openarm_lora/openarm_pickup"


def main():
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

    assert observed_ids == train_ids, "训练轨迹不完整"
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    assert states.shape == actions.shape == (4548, 16)
    assert np.isfinite(states).all()
    assert np.isfinite(actions).all()

    stats = {}
    near_constant = {}

    for key, values in (("state", states), ("actions", actions)):
        mean = values.mean(axis=0)
        raw_std = values.std(axis=0)

        # 恒定夹爪维度不能除以零。
        std = np.maximum(raw_std, 1e-6)
        near_constant[key] = np.flatnonzero(raw_std < 1e-6).tolist()

        stats[key] = normalize.NormStats(
            mean=mean,
            std=std,
            q01=np.quantile(values, 0.01, axis=0),
            q99=np.quantile(values, 0.99, axis=0),
        )

        normalized = (values - mean) / std
        assert np.isfinite(normalized).all()
        reconstructed = normalized * std + mean
        np.testing.assert_allclose(
            reconstructed, values, rtol=1e-6, atol=1e-7
        )

        print(f"{key}: {values.shape}")
        print("近似恒定的维度:", near_constant[key])
        print("归一化及反归一化检查通过")

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


if __name__ == "__main__":
    main()