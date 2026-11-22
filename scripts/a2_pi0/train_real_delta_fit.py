#!/usr/bin/env python3
"""Continue delta LoRA on real demonstrations and measure in-sample angle fit."""
import dataclasses
import json
import shutil
from pathlib import Path

import flax.nnx as nnx
from flax import traverse_util
import jax
import numpy as np

from real_four_camera_loader import RealFourCameraDataset
from real_prefix_pi0 import PrefixPi0Config
from real_training_support import ROOT, get_train_config, load_parameters, make_train_step
from train_real_four_camera import save_adapter, sha
from train_real_recovery_ablation import RUN, real_batch
from openpi.training import optimizer

NAME = 'real_delta_fit_20261122'
STEPS = 288
ARM = np.r_[0:7, 8:15]
GRIP = [7, 15]


def main():
    output = ROOT / 'results' / NAME
    checkpoint_dir = ROOT / 'checkpoints' / NAME
    output.mkdir()
    checkpoint_dir.mkdir()
    source = ROOT / 'checkpoints' / RUN / 'step_1000'
    stats_path = ROOT / 'results' / RUN / 'delta_stats.npz'
    with np.load(stats_path, allow_pickle=False) as stats:
        mean, std = stats['mean'], stats['std']
    ds = RealFourCameraDataset('train', cache_episodes=18)
    rng = np.random.default_rng(20261122)
    # Balance episodes and cover the timeline of each recording.
    per_episode = STEPS // len(ds.entries)
    if per_episode * len(ds.entries) != STEPS:
        raise ValueError('Step budget must be divisible by the number of training episodes')
    probes, updates = [], []
    for i, entry in enumerate(ds.entries):
        offset = int(ds.ends[i - 1]) if i else 0
        groups = np.array_split(np.arange(1, entry['full_window_samples']), per_episode)
        indices = [int(rng.choice(group)) + offset for group in groups]
        probes.extend(indices[2::4])
        updates.extend(indices)
    updates = rng.permutation(updates).tolist()
    (output / 'protocol.json').write_text(json.dumps({
        'source': str(source.relative_to(ROOT)), 'updates': updates, 'probes': probes,
        'probe_split': 'train; all probes are included in updates', 'sampling_steps': 10,
        'known_demo_prefix_steps': 7, 'supervised_output': '16 angle deltas, radians',
    }, indent=2) + '\n')
    print(f'{len(ds.entries)} training episodes, {STEPS} updates, {len(probes)} fixed in-sample probes', flush=True)
    schedule = optimizer.CosineDecaySchedule(warmup_steps=12, peak_lr=1e-5,
                                              decay_steps=STEPS, decay_lr=2e-6)
    cfg = dataclasses.replace(get_train_config(), model=PrefixPi0Config(
        paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora',
        action_horizon=50), num_train_steps=STEPS, lr_schedule=schedule, seed=42)
    graph = nnx.graphdef(nnx.eval_shape(cfg.model.create, jax.random.key(cfg.seed)))
    model = nnx.merge(graph, load_parameters(cfg))
    adapter = source / 'lora_params.npz'
    if sha(adapter) != json.loads((source / 'checkpoint.json').read_text())['lora_sha256']:
        raise ValueError('Source adapter hash mismatch')
    state = nnx.state(model).filter(cfg.trainable_filter)
    with np.load(adapter, allow_pickle=False) as data:
        state.replace_by_pure_dict(traverse_util.unflatten_dict(dict(data), sep='/'))
    nnx.update(model, state)
    params = nnx.state(model)
    del model, state
    tx = optimizer.create_optimizer(cfg.optimizer, schedule, weight_decay_mask=None)
    opt_state = tx.init(params.filter(cfg.trainable_filter))
    train_step = make_train_step(cfg, graph, tx)

    @jax.jit
    def sample(params, obs, actions, key):
        return nnx.merge(graph, params).sample_actions(
            key, obs, known_prefix=actions[:, :7], num_steps=10)

    def measure(step):
        rows = []
        for i, index in enumerate(probes):
            obs, actions = real_batch(ds, index, mean, std)
            predicted = np.asarray(sample(params, obs, actions, jax.random.key(500 + i)))
            error = np.abs(np.cumsum((predicted - np.asarray(actions))[0, :, :16] * (std + 1e-6), axis=0))
            rows.append([error[7:20, ARM].mean(), error[7:20, GRIP].mean(),
                         error[7:, ARM].mean(), error[7:, GRIP].mean()])
        metrics = np.asarray(rows)
        if not np.isfinite(metrics).all():
            raise ValueError('Nonfinite fit measurement')
        keys = ('arm_mae_8_20_rad', 'gripper_mae_8_20_rad',
                'arm_mae_8_50_rad', 'gripper_mae_8_50_rad')
        record = {'step': step, 'mean': dict(zip(keys, metrics.mean(axis=0).tolist())),
                  'p90_window_mae': dict(zip(keys, np.percentile(metrics, 90, axis=0).tolist()))}
        with (output / 'fit.jsonl').open('a') as file:
            file.write(json.dumps(record) + '\n')
        print(record, flush=True)
        return record

    before = measure(0)
    history = []
    for step, index in enumerate(updates, 1):
        obs, actions = real_batch(ds, index, mean, std)
        params, opt_state, metrics = train_step(params, opt_state, jax.random.key(700 + step), obs, actions)
        loss = float(metrics['loss'])
        if not np.isfinite(loss):
            raise ValueError(f'Nonfinite loss at step {step}')
        history.append(loss)
        if step % 48 == 0:
            print(f'{step}/{STEPS} mean training loss {np.mean(history[-48:]):.5f}', flush=True)
        if step == STEPS // 2 or step == STEPS:
            checkpoint = save_adapter(checkpoint_dir, step, params, opt_state, cfg)
            shutil.copy2(stats_path, checkpoint / 'delta_stats.npz')
            result = measure(step)
            (output / 'summary.json').write_text(json.dumps({
                'baseline': before, 'current': result, 'checkpoint': str(checkpoint.relative_to(ROOT)),
                'scope': 'in-sample demo fit on 72 training windows with recorded prefixes',
            }, indent=2) + '\n')


if __name__ == '__main__':
    main()
