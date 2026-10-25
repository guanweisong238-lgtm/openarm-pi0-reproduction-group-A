"""配对采样：两个实验的轨迹选择一致，只替换部分轨迹内的帧。"""
import numpy as np


def make_schedule(rows, steps, seed, early_probability=0., early_frames=75):
    if not rows or steps < 1 or not 0 <= early_probability <= 1 or early_frames < 1:
        raise ValueError('非法采样设置')
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('训练采样器拒绝validation轨迹')
    # 基础流保持原uniform训练的抽样顺序；早期替换使用独立随机流。
    base = np.random.default_rng(seed)
    extra = np.random.default_rng(seed + 100003)
    gates, fractions = extra.random((2, steps))
    schedule = []
    for step in range(steps):
        episode = int(base.integers(len(rows)))
        local = int(base.integers(rows[episode]['frames']))
        forced = bool(gates[step] < early_probability)
        if forced:
            local = int(fractions[step] * min(early_frames, rows[episode]['frames']))
        schedule.append((episode, local, forced))
    return schedule
