#!/usr/bin/env python3
"""Offline, timestamp-aligned Task C action chunk scheduling and fusion."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_real_replay_handoffs import ARM
from eval_gripper_replay import episode_data, load_data, policy

FPS = 30
HORIZON = 50
BASE_STRIDE = 5
STRIDES = (5, 10, 20, 30, 40, 50)
FUSION_LENGTHS = (4, 8, 16)
COLORS = {'direct': '#ac4338', 'linear': '#2b77ab', 'quintic': '#4d8b58'}


def schedule(anchors, latency_ms, stride, delayed):
    requests = np.flatnonzero(anchors % stride == 0)
    # Wait for the first result before the playback clock starts, as the
    # existing startup does. Subsequent observations arrive in motion.
    events, busy_until, dropped = [(0, 0)], 0.0, 0
    for index in requests[1:]:
        anchor = int(anchors[index])
        if anchor < busy_until:
            dropped += 1
            continue
        delay = float(latency_ms[index] * FPS / 1000) if delayed else 0
        busy_until = anchor + delay
        events.append((int(np.ceil(busy_until)), index))
    return events, dropped


def fuse(anchors, chunks, events, stride, frames, blend_kind, width):
    actions = np.full((frames, 16), np.nan)
    owners = np.full(frames, -1, dtype=int)
    switches = []
    outgoing = None
    for k, (ready, new) in enumerate(events):
        end = min(events[k + 1][0] if k + 1 < len(events) else frames, anchors[new] + HORIZON, frames)
        if end <= ready:
            continue
        new_at = int(ready - anchors[new])
        incoming = chunks[new].copy()
        if k:
            old = events[k - 1][1]
            old_at = int(ready - anchors[old])
            overlap = max(0, min(HORIZON - old_at, HORIZON - new_at))
            length = min(overlap, width, stride) if blend_kind != 'direct' else 0
            if length:
                u = np.arange(1, length + 1) / length
                weight = u if blend_kind == 'linear' else u**3 * (10 - 15*u + 6*u*u)
                old_targets = outgoing[old_at:old_at + length, :][:, ARM]
                new_targets = incoming[new_at:new_at + length, :][:, ARM]
                incoming[new_at:new_at + length, ARM] = old_targets + weight[:, None] * (new_targets - old_targets)
            switches.append(dict(frame=ready, previous=old, new=new, overlap=overlap,
                                 blended=min(length, end - ready)))
        actions[ready:end] = incoming[new_at:new_at + end - ready]
        owners[ready:end] = new
        outgoing = incoming
    return actions, owners, switches


def finite_metrics(config, actions, owners, switches, state, demo, anchors):
    valid = owners >= 0
    valid_positions = np.flatnonzero(valid)
    if not len(valid_positions):
        raise ValueError('No action ready inside the scored interval')
    start, last = int(valid_positions[0]), int(valid_positions[-1])
    speed_step = np.abs(np.diff(actions[:, ARM], axis=0))
    acceleration = np.abs(np.diff(actions[:, ARM], n=2, axis=0)) * FPS**2
    step_limit = policy.VELOCITY[ARM] / FPS + 1e-6
    boundary_jumps, worst_after_switch = [], []
    boundary_overspeed, restarts = 0, 0
    for i, switch in enumerate(switches):
        frame = switch['frame']
        if frame == 0 or not valid[frame - 1]:
            restarts += 1
            continue
        boundary = speed_step[frame - 1]
        boundary_jumps.append(boundary.max())
        stop = min(frame + max(FUSION_LENGTHS) - 1,
                   switches[i + 1]['frame'] - 1 if i + 1 < len(switches) else len(actions) - 1)
        interval = speed_step[frame - 1:stop]
        worst_after_switch.append(np.nanmax(interval))
        boundary_overspeed += int(np.any(boundary > step_limit))
    outside = (actions < policy.LIMITS[:, 0]) | (actions > policy.LIMITS[:, 1])
    error = np.abs(actions - state)
    step = np.abs(np.diff(actions, axis=0, prepend=actions[:1]))
    scope_results = {}
    for name, axes in (('arms', ARM), ('all_axes', np.arange(16))):
        reasons = [
            ('missing_target', ~valid),
            ('expired_observation', valid & ((np.arange(len(valid)) - anchors[owners.clip(min=0)]) * 1000 / FPS
                                               > policy.CONFIG['max_plan_age_ms'])),
            ('position_limit', outside[:, axes].any(axis=1)),
            ('target_feedback_gap', (error[:, axes] > policy.FEEDBACK_GAP[axes]).any(axis=1)),
            ('command_velocity', (step[:, axes] > policy.VELOCITY[axes] / FPS + 1e-6).any(axis=1)),
        ]
        first_gap = np.abs(actions[start, axes] - state[start, axes])
        reasons[3][1][start] = (first_gap > np.minimum(policy.FEEDBACK_GAP[axes], policy.VELOCITY[axes]/FPS)).any()
        stops = [(int(indices[0]), reason) for reason, mask in reasons
                 if len(indices := np.flatnonzero(mask[start:]) + start)]
        stop_frame, reason = min(stops, key=lambda pair: pair[0]) if stops else (len(valid), 'complete')
        scope_results[name] = dict(result=reason, seconds_until_stop=(stop_frame - start) / FPS)
    valid_accel = np.isfinite(acceleration).all(axis=1)
    gap = np.zeros(len(valid), dtype=bool)
    gap[start:] = ~valid[start:]
    return dict(
        **config,
        switches=len(switches), switches_after_gap=restarts,
        boundary_overspeed=boundary_overspeed,
        coverage=float(valid.mean()), gaps=int(gap.sum()),
        first_available_s=start / FPS, last_available_s=last / FPS,
        scheduled_duration_s=len(valid) / FPS, scheduled_target_s=float(valid.sum() / FPS),
        gap_s=float(gap.sum() / FPS),
        median_effective_switch_frames=float(np.median(np.diff([start] + [p['frame'] for p in switches]))) if switches else None,
        median_available_overlap_frames=float(np.median([p['overlap'] for p in switches])) if switches else None,
        median_blended_frames=float(np.median([p['blended'] for p in switches])) if switches else None,
        arm_mae_deg=float(np.rad2deg(np.abs(actions[valid][:, ARM] - demo[valid][:, ARM]).mean())),
        switch_jump_deg=np.rad2deg(boundary_jumps).tolist(),
        switch_peak16_deg=np.rad2deg(worst_after_switch).tolist(),
        arm_acceleration_deg_s2=np.rad2deg(acceleration[valid_accel].max(axis=1)).tolist(),
        guard=scope_results,
    )


def aggregate(rows):
    def values(key):
        return np.concatenate([row[key] for row in rows])

    def quantile(data, percentile):
        return float(np.percentile(data, percentile)) if len(data) else None

    def mean(key):
        available = [x[key] for x in rows if x[key] is not None]
        return float(np.mean(available)) if available else None

    result = {
        key: rows[0][key] for key in ('timing', 'stride', 'fusion', 'requested_blend_frames')
    }
    jumps, peaks, acceleration = (values(key) for key in ('switch_jump_deg', 'switch_peak16_deg', 'arm_acceleration_deg_s2'))
    result.update(
        episodes=len(rows), switches=sum(row['switches'] for row in rows),
        scored_contiguous_switches=len(jumps), switches_after_gap=sum(row['switches_after_gap'] for row in rows),
        dropped_requests=sum(row['dropped_requests'] for row in rows),
        boundary_overspeed=sum(row['boundary_overspeed'] for row in rows),
        jump_median_deg=quantile(jumps, 50), jump_p95_deg=quantile(jumps, 95),
        post_switch_peak16_p95_deg=quantile(peaks, 95), acceleration_p95_deg_s2=quantile(acceleration, 95),
        median_available_overlap_frames=mean('median_available_overlap_frames'),
        median_blended_frames=mean('median_blended_frames'),
        median_effective_switch_frames=mean('median_effective_switch_frames'),
        arm_mae_deg=mean('arm_mae_deg'), mean_startup_inference_s=mean('startup_inference_s'),
        mean_first_available_s=mean('first_available_s'),
        mean_last_available_s=mean('last_available_s'), mean_scheduled_duration_s=mean('scheduled_duration_s'),
        mean_scheduled_target_s=mean('scheduled_target_s'), mean_gap_s=mean('gap_s'), mean_coverage=mean('coverage'),
        task_success_rate=None, task_execution_time_s=None,
    )
    for scope in ('arms', 'all_axes'):
        summary = Counter(row['guard'][scope]['result'] for row in rows)
        result[f'{scope}_guard_completions'] = summary['complete']
        result[f'{scope}_guard_mean_seconds'] = float(np.mean([row['guard'][scope]['seconds_until_stop'] for row in rows]))
        result[f'{scope}_guard_outcomes'] = dict(summary)
    return result


def plot_summary(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, timing in enumerate(('zero_delay', 'measured_latency')):
        for method, color in COLORS.items():
            selected = [r for r in rows if r['timing'] == timing and r['fusion'] == method
                        and r['requested_blend_frames'] in (0, 8)]
            for j, key in enumerate(('jump_median_deg', 'jump_p95_deg')):
                y = [np.nan if row[key] is None else row[key] for row in selected]
                axes[i, j].plot([row['stride'] for row in selected], y, 'o-', color=color,
                                label=method if method == 'direct' else f'{method}, up to 8 frames')
                axes[i, j].set_title(f'{timing.replace("_", " ")} · {("median", "P95")[j]}')
                axes[i, j].set_ylabel('Largest joint step (deg)')
                axes[i, j].grid(alpha=.25)
                axes[i, j].set_xticks(STRIDES)
    axes[0, 0].legend()
    fig.supxlabel('Requested observation interval (frames at 30 Hz)')
    fig.suptitle('Task C · 10 recorded validation episodes')
    fig.tight_layout()
    fig.savefig(out / 'stride_vs_jump.png', dpi=170)
    plt.close(fig)


def plot_example(anchors, windows, events, demo, out):
    stride = 20
    variants = {method: fuse(anchors, windows, events, stride, len(demo), method, 8)
                for method in COLORS}
    switches = [x for x in variants['direct'][2] if x['overlap'] >= 8 and 30 < x['frame'] < len(demo) - 20]
    first = max(range(len(switches)-2), key=lambda n: np.max([
        np.abs(windows[t['previous'], t['frame'] - anchors[t['previous']], ARM]
               - windows[t['new'], t['frame'] - anchors[t['new']], ARM]).max()
        for t in switches[n:n+3]]))
    switches = switches[first:first+3]
    worst = max(switches, key=lambda t: np.abs(windows[t['previous'], t['frame'] - anchors[t['previous']], ARM]
                                               - windows[t['new'], t['frame'] - anchors[t['new']], ARM]).max())
    t = worst['frame']
    delta = windows[worst['previous'], t - anchors[worst['previous']]] - windows[worst['new'], t - anchors[worst['new']]]
    joint = ARM[np.abs(delta[ARM]).argmax()]
    label = f'{"R" if joint < 8 else "L"} J{joint % 8 + 1}'
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, info in zip(axes, switches, strict=True):
        old, new = info['previous'], info['new']
        start, end = anchors[new], anchors[old] + HORIZON
        timestamps = np.arange(start, end) / FPS
        ax.plot(timestamps, np.rad2deg(windows[old, start-anchors[old]:, joint]), label='previous chunk')
        ax.plot(timestamps, np.rad2deg(windows[new, :end-start, joint]), label='new chunk')
        ax.axvline(info['frame'] / FPS, ls='--', color='#333', label='new chunk ready')
        ax.set_ylabel('Angle (deg)')
        ax.legend(loc='best')
    fig.suptitle(f'Episode 5 · {label} · three consecutive overlaps at identical timestamps')
    axes[-1].set_xlabel('Recorded timeline (s)')
    fig.savefig(out / 'consecutive_chunk_overlaps.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    region = np.arange(t - 10, t + 15)
    for method, (trajectory, _, _) in variants.items():
        levels = [np.rad2deg(trajectory[:, joint])]
        levels.extend([np.r_[np.nan, np.diff(levels[0]) * FPS]])
        levels.extend([np.r_[np.nan, np.diff(levels[1]) * FPS]])
        for ax, level in zip(axes, levels, strict=True):
            ax.plot(region / FPS, level[region], label=method, color=COLORS[method])
    for ax, unit in zip(axes, ('Position (deg)', 'Velocity (deg/s)', 'Acceleration (deg/s²)'), strict=True):
        ax.axvline(t / FPS, color='#555', ls='--', alpha=.5)
        ax.set_ylabel(unit)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel('Recorded timeline (s)')
    fig.suptitle(f'Episode 5 · {label} · direct switch, linear and quintic fusion')
    fig.savefig(out / 'fusion_joint_curves.png', dpi=170)
    plt.close(fig)
    np.savez_compressed(out / 'episode_5_stride20_fusion_trajectories.npz',
                        timestamp_s=np.arange(len(demo)) / FPS,
                        switch_frames=[item['frame'] for item in variants['direct'][2]],
                        **{name: values[0] for name, values in variants.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1002)
    args = parser.parse_args()
    grouped = defaultdict(list)
    latency_samples = []
    for entry in load_data():
        if entry['split'] != 'validation':
            continue
        state, demo, _ = episode_data(entry)
        episode = entry['episode_id']
        path = args.out / 'predictions' / f'seed{args.seed}' / f'episode_{episode}.npz'
        with np.load(path, allow_pickle=False) as cache:
            anchors, windows, latency = (cache[key] for key in ('anchors', 'windows', 'latency_ms'))
        np.testing.assert_array_equal(anchors, np.arange(0, len(state), BASE_STRIDE))
        if windows.shape != (len(anchors), HORIZON, 16) or not np.isfinite(windows).all():
            raise ValueError(f'Unexpected predictions for episode {episode}')
        if latency.shape != anchors.shape or not np.isfinite(latency).all() or np.any(latency <= 0):
            raise ValueError(f'Unexpected inference timing for episode {episode}')
        latency_samples.extend(latency.tolist())
        for timing in ('zero_delay', 'measured_latency'):
            for stride in STRIDES:
                events, dropped = schedule(anchors, latency, stride, timing == 'measured_latency')
                if episode == 5 and timing == 'measured_latency' and stride == 20:
                    plot_example(anchors, windows, events, demo, args.out)
                for method, width in [('direct', 0), *((method, width) for method in ('linear', 'quintic')
                                                       for width in FUSION_LENGTHS)]:
                    actions, owners, switches = fuse(anchors, windows, events, stride, len(state), method, width)
                    row = finite_metrics(dict(episode=episode, timing=timing, stride=stride, fusion=method,
                                              requested_blend_frames=width, dropped_requests=dropped,
                                              startup_inference_s=float(latency[0] / 1000) if timing == 'measured_latency' else 0.0),
                                         actions, owners, switches, state, demo, anchors)
                    grouped[timing, stride, method, width].append(row)
    summary = [aggregate(rows) for rows in grouped.values()]
    detailed = [{k: v for k, v in row.items() if k not in ('switch_jump_deg', 'switch_peak16_deg', 'arm_acceleration_deg_s2')}
                for rows in grouped.values() for row in rows]
    report = dict(
        seed=args.seed, checkpoint=policy.CONFIG['checkpoint'], one_serial_worker=True, fps=FPS, horizon=HORIZON,
        latency_samples=len(latency_samples), prediction_latency_median_ms=float(np.median(latency_samples)),
        prediction_latency_p95_ms=float(np.percentile(latency_samples, 95)),
        scope='Recorded observations and fixed wall-clock schedules. No simulator or hardware task success or execution time.',
        summary=summary, episodes=detailed,
    )
    (args.out / 'sliding_window_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (args.out / 'sliding_window_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot_summary(summary, args.out)


if __name__ == '__main__':
    main()
