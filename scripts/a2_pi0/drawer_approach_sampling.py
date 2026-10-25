"""轨迹均匀、阶段均衡的首接近/夹持采样；不改变原始标签。"""
import numpy as np


def make_schedule(rows, phases, steps, seed):
    if any(r['split']!='train' for r in rows):raise ValueError('拒绝validation梯度采样')
    rng=np.random.default_rng(seed);schedule=[]
    for _ in range(steps):
        episode=int(rng.integers(len(rows)));stage=str(rng.choice(['start','approach','grasp'],p=[.3,.35,.35]))
        low,high=phases[rows[episode]['episode_id']]['phase_ranges'][stage]
        if not 0<=low<high<=rows[episode]['frames']:raise ValueError('阶段越界')
        schedule.append((episode,int(rng.integers(low,high)),stage))
    return schedule
