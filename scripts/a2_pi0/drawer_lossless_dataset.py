"""Read original uint8 RGB observations directly, without video encoding/decoding."""
import json
from pathlib import Path
import numpy as np

EPISODE_ROOT = Path(__file__).resolve().parents[2] / "data/drawer_trial/contact_roundtrip_rgb_001"


class LosslessDrawerDataset:
    def __init__(self, episode_root=EPISODE_ROOT, horizon=16):
        self.root, self.horizon = Path(episode_root), horizon
        self.prompt = json.loads((self.root / "report.json").read_text())["prompt"]
        with np.load(self.root / "episode.npz", allow_pickle=False) as episode:
            self.state = episode["state"].astype(np.float32)
            self.action = episode["action"].astype(np.float32)
            self.images = {name: episode[f"images_{name}"].copy()
                           for name in ("ceiling", "wrist_left", "wrist_right")}
        if self.state.shape != self.action.shape or self.state.shape[1:] != (16,):
            raise ValueError("Invalid original state/action arrays")
        for image in self.images.values():
            if image.dtype != np.uint8 or image.shape != (len(self), 240, 320, 3):
                raise ValueError("Original RGB must be aligned uint8 HWC frames")

    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        indices = index + np.arange(self.horizon)
        return {"observation.state": self.state[index].copy(),
                "action": self.action[np.minimum(indices, len(self)-1)].copy(),
                "action_is_pad": indices >= len(self), "task": self.prompt,
                **{f"observation.images.{name}": frames[index].copy()
                   for name, frames in self.images.items()}}


def transformed_dataset(cfg):
    from local_data_loader import TransformedDataset
    from openpi import transforms
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    return TransformedDataset(LosslessDrawerDataset(horizon=cfg.model.action_horizon), [
        *data.repack_transforms.inputs, *data.data_transforms.inputs,
        transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
        *data.model_transforms.inputs]), data


if __name__ == "__main__":
    from drawer_simulation import DrawerSimulation
    import pyarrow.parquet as pq
    dataset = LosslessDrawerDataset()
    rows = pq.read_table(dataset.root / "lerobot_v21/data/chunk-000/episode_000000.parquet").to_pylist()
    np.testing.assert_array_equal(dataset.state, np.array([r["observation.state"] for r in rows], dtype=np.float32))
    np.testing.assert_array_equal(dataset.action, np.array([r["action"] for r in rows], dtype=np.float32))
    for index in range(len(dataset)):
        sample = dataset[index]
        valid = min(dataset.horizon, len(dataset)-index)
        np.testing.assert_array_equal(sample["action"][:valid], dataset.action[index:index+valid])
        assert int(sample["action_is_pad"].sum()) == dataset.horizon-valid
    environment = DrawerSimulation()
    with np.load(dataset.root / "episode.npz", allow_pickle=False) as episode:
        original_controls = episode["action"].copy()
    selected = {0, 30, 149, 191, 281, 401, 461}
    checks = []
    for index in range(len(dataset)):
        if index in selected:
            raw, live = dataset[index], environment.observation()
            for camera in dataset.images:
                key = f"observation.images.{camera}"
                np.testing.assert_array_equal(raw[key], live[key])
            np.testing.assert_array_equal(raw["observation.state"], live["observation.state"])
            checks.append(index)
        environment.step(original_controls[index])
    environment.close()
    print(json.dumps({"source": "original_npz_uint8_rgb", "frames": len(dataset),
                      "pixel_exact_frames": checks, "all_labels_aligned": True}))
