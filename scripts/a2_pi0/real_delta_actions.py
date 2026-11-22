"""Delta-action conversion for task C; absolute state remains the observation."""
import numpy as np
from real_training_support import ROOT

def to_deltas(absolute, state):
    absolute = np.asarray(absolute, np.float32)
    state = np.asarray(state, np.float32)
    return np.concatenate([(absolute[..., :1, :] - state[..., None, :]),
                           np.diff(absolute, axis=-2)], axis=-2)

def encode_deltas(absolute, state, mean, std):
    return ((to_deltas(absolute, state) - mean) / (std + 1e-6)).astype(np.float32)

def decode_deltas(normalized, state, mean, std):
    return np.cumsum(np.asarray(normalized)[..., :16] * (std + 1e-6) + mean, axis=-2) + np.asarray(state)[..., None, :]

def delta_stats(dataset):
    total = np.zeros(16, np.float64)
    square = np.zeros(16, np.float64)
    count = 0
    for row in dataset.entries:
        with np.load(ROOT / row['samples_path'], allow_pickle=False) as z:
            n = row['full_window_samples']
            actions = z['actions'][:n-1]
            states = z['state'][1:n]
            delta = to_deltas(actions, states).astype(np.float64)
            total += delta.sum(axis=(0, 1)); square += (delta * delta).sum(axis=(0, 1))
            count += delta.shape[0] * delta.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 0))
    if not np.isfinite(std).all() or (std < 1e-6).any():
        raise ValueError('invalid or degenerate train-only delta statistics')
    return mean.astype(np.float32), std.astype(np.float32)
