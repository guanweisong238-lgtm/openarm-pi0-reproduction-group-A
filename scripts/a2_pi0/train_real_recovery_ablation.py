#!/usr/bin/env python3
"""Compare short real-only and mixed updates from the same delta checkpoint."""
import dataclasses
import json
from pathlib import Path

import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from real_training_support import ROOT, get_train_config, load_parameters, make_train_step
from real_four_camera_loader import CAMERA_MAPPING, PROMPT, RealFourCameraDataset, batch_samples
from real_prefix_pi0 import PrefixPi0Config
from real_delta_actions import to_deltas
from train_real_four_camera import save_adapter, sha
from openpi.training import optimizer

RUN = 'real_delta_prefix_train_1000_20261122'
STEPS = 40


def prepared(dataset, raw, delta, mean, std):
    sample = dataset.adapter(raw)
    sample['state'] = ((sample['state'] - dataset.stats['state'].mean)
                       / (dataset.stats['state'].std + 1e-6)).astype(np.float32)
    sample['actions'] = ((delta - mean) / (std + 1e-6)).astype(np.float32)
    sample = dataset.pad(dataset.resize(sample))
    sample['tokenized_prompt'] = dataset.prompt_tokens
    sample['tokenized_prompt_mask'] = dataset.prompt_mask
    sample.pop('prompt')
    return jax.device_put(batch_samples([sample]))


def real_batch(dataset, index, mean, std):
    raw = dataset.get_raw(index)
    episode, frame = dataset.locate(index)
    delta = to_deltas(dataset.cache[episode]['actions'][frame - 1], raw['state'])
    return prepared(dataset, raw, delta, mean, std)


def recovery_samples():
    folder = ROOT / 'data/processed/real_recovery_synthetic_candidates_v001'
    manifest = json.loads((folder / 'manifest.json').read_text())
    rows = []
    for entry in manifest['entries']:
        path = folder / entry['window_file']
        if sha(path) != entry['window_sha256'] or entry['split'] != 'train_candidate_only':
            raise ValueError('Invalid recovery provenance')
        with np.load(ROOT / entry['trajectory'], allow_pickle=False) as trajectory:
            states = trajectory['state']
            actions = trajectory['action_qpos']
        with np.load(path, allow_pickle=False) as data:
            for frame, delta in zip(data['observation_indices'], data['delta_actions'], strict=True):
                expected = to_deltas(actions[frame-1:frame+49], states[frame])
                np.testing.assert_allclose(delta, expected, atol=1e-7)
                images = {}
                for camera in CAMERA_MAPPING.values():
                    path = ROOT / entry['image_directory'] / f'{frame:04d}_{camera}.jpg'
                    with Image.open(path) as image:
                        rgb = np.asarray(image.convert('RGB'))
                    if rgb.shape != (224, 224, 3):
                        raise ValueError(f'Unexpected image size: {path}')
                    images[camera] = rgb
                rows.append(({'images': images, 'state': states[frame], 'prompt': PROMPT}, delta))
    if len(rows) != manifest['full_window_count']:
        raise ValueError('Recovery window count mismatch')
    return rows


def main():
    out = ROOT / 'results/real_recovery_ablation_20261122'
    out.mkdir()
    with np.load(ROOT / 'results' / RUN / 'delta_stats.npz') as stats:
        mean, std = stats['mean'], stats['std']
    train = RealFourCameraDataset('train', cache_episodes=18)
    val = RealFourCameraDataset('validation', cache_episodes=4)
    recovery = recovery_samples()
    validation_indices = []
    for episode, entry in enumerate(val.entries):
        start = int(val.ends[episode - 1]) if episode else 0
        validation_indices.extend([start + 1, start + entry['full_window_samples'] // 2])
    validation = [real_batch(val, index, mean, std) for index in validation_indices]
    eligible = np.setdiff1d(np.arange(len(train)), np.r_[0, train.ends[:-1]])
    rng = np.random.default_rng(123)
    indices = rng.choice(eligible, STEPS, replace=False)
    synthetic_indices = iter(rng.choice(len(recovery), STEPS // 5, replace=False))
    print(f'Validated {len(recovery)} recovery windows and {len(validation)} real probes', flush=True)
    schedule = optimizer.CosineDecaySchedule(warmup_steps=4, peak_lr=3e-6, decay_steps=STEPS, decay_lr=1e-6)
    cfg = dataclasses.replace(get_train_config(), model=PrefixPi0Config(
        paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora',
        action_horizon=50), num_train_steps=STEPS, lr_schedule=schedule, seed=42)
    graph = nnx.graphdef(nnx.eval_shape(cfg.model.create, jax.random.key(cfg.seed)))
    initial = ROOT / 'checkpoints' / RUN / 'step_1000'
    adapter_path = initial / 'lora_params.npz'
    adapter_hash = sha(adapter_path)
    if adapter_hash != json.loads((initial / 'checkpoint.json').read_text())['lora_sha256']:
        raise ValueError('Starting adapter hash mismatch')
    with np.load(adapter_path) as data:
        adapter = traverse_util.unflatten_dict(dict(data), sep='/')

    @jax.jit
    def evaluate(params, observation, actions, key):
        model = nnx.merge(graph, params)
        loss = jnp.mean(model.compute_loss(key, observation, actions, train=False))
        output = model.sample_actions(key, observation, known_prefix=actions[:, :7], num_steps=10)
        # Prefix positions are identical. Differences after integration are absolute-angle errors.
        residual = jnp.cumsum((output[..., :16] - actions[..., :16]) * (std + 1e-6), axis=1)
        residual = jnp.abs(residual[:, 7:20])
        return jnp.array([loss, residual[..., np.r_[0:7, 8:15]].mean(), residual[..., [7, 15]].mean()])

    tx = optimizer.create_optimizer(cfg.optimizer, schedule, weight_decay_mask=None)
    train_step = make_train_step(cfg, graph, tx)
    report = {'checkpoint': str(initial.relative_to(ROOT)), 'checkpoint_sha256': adapter_hash,
              'probe_metrics': ['flow_loss', 'arm_mae_rad_steps_8_to_20', 'gripper_mae_rad_steps_8_to_20'],
              'real_probe_indices': validation_indices, 'recovery_windows_checked': len(recovery)}
    for mode in ('real_only', 'mixed'):
        model = nnx.merge(graph, load_parameters(cfg))
        state = nnx.state(model).filter(cfg.trainable_filter)
        state.replace_by_pure_dict(adapter)
        nnx.update(model, state)
        params = nnx.state(model)
        del model, state
        opt_state = tx.init(params.filter(cfg.trainable_filter))

        def score():
            results = np.stack([np.asarray(evaluate(params, *sample, jax.random.key(100 + i)))
                                for i, sample in enumerate(validation)])
            if not np.isfinite(results).all():
                raise ValueError('Nonfinite validation metrics')
            return results.mean(axis=0).tolist()

        before = score()
        used = []
        for step, index in enumerate(indices, 1):
            if mode == 'mixed' and step % 5 == 0:
                index = int(next(synthetic_indices))
                batch = prepared(train, *recovery[index], mean, std)
                used.append(index)
            else:
                batch = real_batch(train, int(index), mean, std)
            params, opt_state, metrics = train_step(params, opt_state, jax.random.key(200 + step), *batch)
            if not np.isfinite(float(metrics['loss'])):
                raise ValueError('Nonfinite training loss')
            if step % 10 == 0:
                print(mode, step, float(metrics['loss']), flush=True)
        after = score()
        checkpoint_dir = out / mode
        checkpoint_dir.mkdir()
        checkpoint = save_adapter(checkpoint_dir, STEPS, params, opt_state, cfg)
        report[mode] = {'before': before, 'after': after, 'synthetic_indices': used,
                        'checkpoint': str(checkpoint.relative_to(ROOT))}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(mode, before, after, flush=True)
        del params, opt_state, batch, metrics


if __name__ == '__main__':
    main()
