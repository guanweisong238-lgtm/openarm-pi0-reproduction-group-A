"""按显式清单读取多轨迹无损RGB；未来动作窗口始终留在所属episode内。"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from collect_drawer_trial import ROOT
from drawer_lossless_dataset import LosslessDrawerDataset


class MultiEpisodeDrawerDataset:
    def __init__(self, manifest, split="train", horizon=16, cache_episodes=2):
        self.manifest = json.loads(Path(manifest).read_text())
        if split not in ("train", "validation") or horizon < 1 or cache_episodes < 1:
            raise ValueError("split/horizon/cache_episodes不合法")
        self.episodes = [row for row in self.manifest["episodes"] if row["split"] == split]
        if not self.episodes:
            raise ValueError(f"清单中没有{split}轨迹")
        self.horizon, self.cache_size = horizon, cache_episodes
        self.ends = np.cumsum([row["frames"] for row in self.episodes])
        self.cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if episode == 0 else int(self.ends[episode - 1])
        return episode, int(index) - start

    def __getitem__(self, index):
        episode, frame = self.locate(index)
        if episode not in self.cache:
            dataset = LosslessDrawerDataset(ROOT / self.episodes[episode]["path"], self.horizon)
            if len(dataset) != self.episodes[episode]["frames"]:
                raise ValueError("清单帧数与数据不一致")
            self.cache[episode] = dataset
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        return self.cache[episode][frame]
