#!/usr/bin/env python3
"""Train and compare matched offline temporal-input pilots, then plot joint angles."""
import argparse
import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
from PIL import Image
from openpi.training import optimizer as optimizer_lib

from eval_real_handoff_pilot import ARM, CAMS, CONFIG, ROOT, forecast_metrics, replay
from policy_server import PolicyService, RealFourCameraDataset
from compare_pi0_contact_stages import GraspPoint
from prepare_demo0_replay import driver_limits
from real_four_camera_loader import batch_samples
from real_training_support import make_train_step


MODES = ('current', 'vision', 'joints')
TRAIN_LOG = ROOT / 'results/real_20261128_handoff_control200_v001/metrics.jsonl'


def add_history(data, mode, *, past_images=None, delta=None):
    if mode == 'vision':
        for name, image in past_images.items():
            key = 'past_' + name
            data['image'][key] = image
            data['image_mask'][key] = np.bool_(True)
    elif mode == 'joints':
        # The 32-wide pi0 state has 16 padded channels; use 14 for arm motion.
        data['state'][16:30] = delta / np.float32(.1)
    return data


def run(mode, output):
    checkpoint = ROOT / CONFIG['checkpoint']
    service = PolicyService(checkpoint, ROOT / CONFIG['run'])
    train = RealFourCameraDataset('train', root=service.candidate, cache_episodes=18)
    val = RealFourCameraDataset('validation', root=service.candidate)
    rows = [json.loads(line) for line in TRAIN_LOG.read_text().splitlines()]
    indices = [row['paired_sample_indices'][0] for row in rows]
    assert len(indices) == 200 and all(train.locate(i)[1] >= 3 for i in indices)
    assert not {r['episode_id'] for r in train.entries} & {r['episode_id'] for r in val.entries}

    cfg = dataclasses.replace(service.cfg, lr_schedule=optimizer_lib.CosineDecaySchedule(
        warmup_steps=20, peak_lr=2e-5, decay_steps=200, decay_lr=2e-6))
    tx = optimizer_lib.create_optimizer(cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None)
    params = service.params
    del service.params  # The training step donates these buffers.
    opt = tx.init(params.filter(cfg.trainable_filter))
    step = make_train_step(cfg, service.graph, tx)
    key = jax.random.key(43)
    for n, idx in enumerate(indices, 1):
        sample = train[idx]
        if mode == 'vision':
            add_history(sample, mode, past_images=train[idx-3]['image'])
        elif mode == 'joints':
            current, previous = train.get_raw(idx)['state'], train.get_raw(idx-3)['state']
            add_history(sample, mode, delta=current[ARM]-previous[ARM])
        key, subkey = jax.random.split(key)
        params, opt, metrics = step(params, opt, subkey, *batch_samples([sample]))
        values = {name: float(value) for name, value in metrics.items()}
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f'Nonfinite training metrics for {mode}, step {n}')
        if n == 1 or n % 25 == 0:
            print(mode, n, values, flush=True)
    del opt
    service.params = params
    seed = CONFIG['seed']
    report = dict(mode=mode, seed=seed, initial_checkpoint=str(checkpoint.relative_to(ROOT)),
                  training_indices=indices, training_steps=len(indices), episodes={})
    for row in val.entries:
        ep = row['episode_id']
        with np.load(ROOT / row['samples_path'], allow_pickle=False) as z:
            state, demo = z['state'], z['actions'][:, 0]
            timestamp = z['timestamp_ns']
            paths = {cam: z['image_path_'+cam] for cam in CAMS}

        def images(frame):
            result = {}
            for cam, names in paths.items():
                with Image.open(ROOT / str(names[frame])) as image:
                    result[cam] = np.asarray(image.convert('RGB'))
            return result

        predictions, anchors = [], np.arange(12, len(demo)-17, 12)
        for frame in anchors:
            data = service.pre(images(frame), state[frame])
            if mode == 'vision':
                past = service.pre(images(frame-3), state[frame-3])['image']
                add_history(data, mode, past_images=past)
            elif mode == 'joints':
                add_history(data, mode, delta=state[frame, ARM]-state[frame-3, ARM])
            obs = service._make_obs(data)
            raw = np.asarray(service._sample(params, obs, np.zeros((1, 7, 32), np.float32), jax.random.key(seed)))
            q = service.pre.unnormalize(raw[0, :, :16])
            if q.shape != (50, 16) or not np.isfinite(q).all():
                raise ValueError(f'Invalid prediction for {mode} episode {ep}, frame {frame}')
            predictions.append(q)
        q = np.stack(predictions)
        target_frames = anchors[:, None] + np.arange(12, 18)
        error = q[:, 12:18, :][:, :, ARM] - demo[target_frames][:, :, ARM]
        past_move = state[anchors][:, ARM] - state[anchors-3][:, ARM]
        future_move = state[anchors+15][:, ARM] - state[anchors][:, ARM]
        threshold = np.deg2rad(2)
        moving = abs(past_move) >= threshold
        signed = future_move * np.sign(past_move)
        groups = {'all': np.ones_like(moving), 'continuing': moving & (signed >= threshold),
                  'slowing_or_stopped': moving & (abs(future_move) < threshold),
                  'reversing': moving & (signed <= -threshold), 'little_past_motion': ~moving}
        metrics = {name: forecast_metrics(error.transpose(0, 2, 1)[mask]) for name, mask in groups.items() if mask.any()}
        report['episodes'][ep] = dict(metrics=metrics, anchors=anchors.tolist(),
                                     history_gap_ms=((timestamp[anchors]-timestamp[anchors-3])/1e6).tolist(),
                                     new=forecast_metrics(q[:, :6, :][:, :, ARM]-demo[anchors[:, None]+np.arange(6)][:, :, ARM]))
        np.savez_compressed(output / f'{mode}_episode_{ep}.npz', demo=demo, anchors=anchors, prediction=q)
        print(mode, ep, metrics['all'], flush=True)
    with (output / f'{mode}.json').open('x') as stream:
        json.dump(report, stream, indent=2)


def compare(output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    runs = {mode: json.loads((output / f'{mode}.json').read_text()) for mode in MODES}
    assert runs['current']['training_indices'] == runs['vision']['training_indices'] == runs['joints']['training_indices']
    assert len({r['seed'] for r in runs.values()}) == 1
    episodes = ('13', '19', '20', '24')
    assert all(set(r['episodes']) == set(episodes) for r in runs.values())
    pooled = {}
    for mode, report in runs.items():
        records = list(report['episodes'].values())
        pooled[mode] = {}
        for group in ('all', 'continuing', 'slowing_or_stopped', 'reversing', 'little_past_motion', 'new'):
            metrics = ([r['new'] for r in records] if group == 'new' else
                       [r['metrics'][group] for r in records if group in r['metrics']])
            counts = [m['count'] for m in metrics]
            pooled[mode][group] = {'count': sum(counts), 'mae_deg': float(np.average([m['mae_deg'] for m in metrics], weights=counts))}

    colors = {'current': '#476eb6', 'vision': '#d45926', 'joints': '#0a8a72'}
    candidate = Path(json.loads((ROOT / CONFIG['run'] / 'config.json').read_text())['candidate'])
    validation = RealFourCameraDataset('validation', root=candidate)
    sources = {str(row['episode_id']): ROOT / row['samples_path'] for row in validation.entries}
    limits, point = driver_limits(), GraspPoint()
    fixed_replay = {}
    for ep in episodes:
        data = {}
        for mode in MODES:
            with np.load(output / f'{mode}_episode_{ep}.npz') as z:
                data[mode] = dict(z)
        demo, anchors = (data['current'][key] for key in ('demo', 'anchors'))
        for mode in MODES[1:]:
            np.testing.assert_array_equal(data[mode]['anchors'], anchors)
            np.testing.assert_array_equal(data[mode]['demo'], demo)
        with np.load(sources[ep], allow_pickle=False) as source:
            state = source['state']
        variants = {mode: replay(dict(zip(map(int, anchors), v['prediction'], strict=True)), {}, state, demo,
                                 measured=False, limits=limits, point=point) for mode, v in data.items()}
        common = set.intersection(*({t for t, r in v['rows'].items() if r['adjacent_deg'] is not None} for v in variants.values()))
        fixed_replay[ep] = {}
        for mode, variant in variants.items():
            rows = variant['rows']
            rejects = {t: r['reject'] for t, r in rows.items() if r['reject'] is not None}
            fixed_replay[ep][mode] = dict(first_reject_s=min(rejects)/10 if rejects else None,
                                          first_reject_reason=rejects[min(rejects)] if rejects else None,
                                          rejects=len(rejects), common_over_10=sum(rows[t]['adjacent_deg'] > 10 for t in common),
                                          pull_target_mm=variant['pull_target_mm'])
        target = anchors[:, None] + np.arange(12, 18)
        fig, axes = plt.subplots(8, 2, figsize=(15, 18), sharex=True, layout='constrained')
        for side, offset in enumerate((0, 8)):
            for i, ax in enumerate(axes[:, side]):
                joint = offset+i
                ax.plot(np.arange(len(demo))/30, np.rad2deg(demo[:, joint]), color='#202329', lw=1.35, label='Demonstration target')
                for mode in MODES:
                    q = data[mode]['prediction'][:, 12:18, joint]
                    # Break the line in intervals not covered by the plotted forecast slice.
                    line_x = np.pad(target.astype(float)/30, ((0, 0), (0, 1)), constant_values=np.nan).ravel()
                    line_y = np.pad(np.rad2deg(q), ((0, 0), (0, 1)), constant_values=np.nan).ravel()
                    ax.plot(line_x, line_y, color=colors[mode], lw=1.15, marker='.', ms=1.7, label=mode)
                ax.set_title(f'{("Right", "Left")[side]} {f"J{i+1}" if i < 7 else "gripper"}')
                ax.set_ylabel('deg')
                ax.grid(alpha=.18)
        axes[0, 0].legend(loc='best', ncol=2, fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel('Demonstration time (s)')
        fig.suptitle(f'Episode {ep} | Demonstration and model forecasts 0.40-0.57 s ahead', fontsize=17)
        fig.savefig(output / f'episode_{ep}_joint_angles.png', dpi=125)
        plt.close(fig)
        if ep == '13':
            selected = int(np.argmax(np.max(abs(data['current']['prediction'][:, 12:18, 12] - demo[target, 12]), axis=1)))
            frame = int(anchors[selected])
            fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
            frames = np.arange(max(0, frame-3), min(len(demo), frame+31))
            ax.plot(frames/30, np.rad2deg(demo[frames, 12]), color='#202329', lw=2.3, label='Demonstration target')
            forecast = np.arange(frame, min(len(demo), frame+31))
            for mode in MODES:
                ax.plot(forecast/30, np.rad2deg(data[mode]['prediction'][selected, :len(forecast), 12]),
                        color=colors[mode], lw=1.8, label=mode)
            ax.axvspan((frame+12)/30, (frame+17)/30, color='#e0b03a', alpha=.2, label='Scored forecast slice')
            ax.axvline(frame/30, color='#888', ls=':')
            ax.set(title='Episode 13 | Left J5 forecast at largest current-only error', xlabel='Demonstration time (s)', ylabel='Joint angle (deg)')
            ax.grid(alpha=.2)
            ax.legend()
            fig.savefig(output / 'episode_13_left_j5_detail.png', dpi=150)
            plt.close(fig)
    result = dict(hardware_used=False, training_order_matches=True, pooled=pooled, fixed_replay=fixed_replay,
                  episodes={mode: report['episodes'] for mode, report in runs.items()})
    with (output / 'comparison.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(pooled, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=(*MODES, 'compare'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    if args.mode == 'compare':
        compare(args.output)
    else:
        if (args.output / f'{args.mode}.json').exists():
            parser.error('output for this mode already exists')
        run(args.mode, args.output)
