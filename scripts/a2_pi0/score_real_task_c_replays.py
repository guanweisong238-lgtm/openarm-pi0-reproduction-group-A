#!/usr/bin/env python3
"""Score Task C overlap schedules and demonstration fit from saved replays."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from real_task_c_phases import PHASES, grip_intervals, phase_frames
from compare_real_replay_handoffs import ARM, blend_handoffs

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / 'assets/real_replay_scene/empty_recovery_scene.xml'
FPS = 30


def overlap_targets(data, frames):
    result, lengths = blend_handoffs(data, frames)
    anchors = data['anchor_indices'][1:]
    transition = np.zeros(len(result), dtype=bool)
    if not frames:
        lengths = np.ones_like(anchors)
    for start, length in zip(anchors, lengths):
        transition[start:start + length] = True
    return result, transition


def ee_positions(model, actions):
    data = mujoco.MjData(model)
    joints = [model.joint(f'openarm_{side}_joint{i}').id for side in ('right', 'left') for i in range(1, 8)]
    addresses = model.jnt_qposadr[joints]
    sites = [model.site(f'{side}_ee_control_point').id for side in ('right', 'left')]
    points = np.empty((len(actions), 2, 3))
    for frame, action in enumerate(actions):
        data.qpos[addresses] = action[ARM]
        mujoco.mj_kinematics(model, data)
        points[frame] = data.site_xpos[sites]
    return points


def score_replay(demo, pred, demo_ee, pred_ee, phases, events, transition, anchors, windows):
    error = np.abs(pred - demo)
    ee_error_mm = np.linalg.norm(pred_ee - demo_ee, axis=-1) * 1000
    delta = np.diff(pred[:, ARM], axis=0, prepend=pred[:1, ARM])
    acceleration = np.diff(delta, axis=0, prepend=delta[:1])
    speed = np.linalg.norm(delta * FPS, axis=1)
    demo_speed = np.linalg.norm(np.diff(demo[:, ARM], axis=0, prepend=demo[:1, ARM]) * FPS, axis=1)
    max_second_diff = np.max(np.abs(acceleration), axis=1)
    boundary_accel = transition | np.r_[False, transition[:-1]]
    around_replan = np.zeros(len(pred), dtype=bool)
    for frame in anchors[1:]:
        around_replan[frame:frame + 8] = True
    valid = np.arange(len(pred)) >= 2
    phase_scores = {}
    for i, phase in enumerate(PHASES):
        mask = phases == i
        phase_scores[phase] = {
            'frames': int(mask.sum()),
            **({
                'arm_mae_rad': float(error[mask][:, ARM].mean()),
                'right_ee_mm': float(ee_error_mm[mask, 0].mean()),
                'left_ee_mm': float(ee_error_mm[mask, 1].mean()),
                'right_grip_mae_rad': float(error[mask, 7].mean()),
                'left_grip_mae_rad': float(error[mask, 15].mean()),
            } if mask.any() else {}),
        }
    grip_offsets = {}
    used_events = set()
    for name, frame in events.items():
        right = name.startswith('right')
        closing = not name.endswith(('release', 'release_handle'))
        intervals = grip_intervals(pred[:, 7 if right else 15], right)
        candidates = [pair[0 if closing else 1] for pair in intervals]
        candidates = [i for i in candidates if 0 < i < len(pred) and (right, closing, i) not in used_events and abs(i - frame) <= FPS]
        match = min(candidates, key=lambda i: abs(i - frame), default=None)
        if match is not None:
            used_events.add((right, closing, match))
        grip_offsets[name] = None if match is None else match - frame
    extra_grip_events = []
    for right, channel in [(True, 7), (False, 15)]:
        for start, end in grip_intervals(pred[:, channel], right):
            extra_grip_events.extend(
                (('right' if right else 'left'), ('close' if closing else 'open'), frame)
                for closing, frame in [(True, start), (False, end)]
                if 0 < frame < len(pred) and (right, closing, frame) not in used_events
            )

    # Count dips only when both the adjacent predicted and demonstration speeds support a through motion.
    dips = []
    for i in range(5, len(speed) - 5):
        before, after = slice(i - 5, i - 1), slice(i + 2, i + 6)
        reference = min(np.median(speed[before]), np.median(speed[after]))
        demo_reference = min(np.median(demo_speed[before]), np.median(demo_speed[after]))
        if reference > 0.35 and speed[i] < 0.45 * reference and demo_reference > 0.35 and demo_speed[i] >= 0.7 * demo_reference:
            dips.append(i)
    in_window_delta = np.diff(windows[:, :, ARM], axis=1, n=2)
    horizon_error = np.abs(windows - demo[anchors[:, None] + np.arange(windows.shape[1])])[:, :, ARM]
    return {
        'arm_mae_rad': float(error[:, ARM].mean()),
        'right_ee_mm': float(ee_error_mm[:, 0].mean()),
        'left_ee_mm': float(ee_error_mm[:, 1].mean()),
        'peak_step_rad': float(np.abs(delta).max()),
        'joint_accel_p95_rad_s2': float(np.quantile(max_second_diff[valid], .95) * FPS**2),
        'joint_accel_p99_rad_s2': float(np.quantile(max_second_diff[valid], .99) * FPS**2),
        'joint_accel_near_replan_p95_rad_s2': float(np.quantile(max_second_diff[around_replan & valid], .95) * FPS**2),
        'demo_joint_accel_p95_rad_s2': float(np.quantile(np.max(np.abs(np.diff(demo[:, ARM], axis=0, n=2)), axis=1), .95) * FPS**2),
        'joint_accel_boundary_p95_rad_s2': float(np.quantile(max_second_diff[boundary_accel & valid], .95) * FPS**2),
        'joint_accel_interior_p95_rad_s2': float(np.quantile(max_second_diff[~boundary_accel & valid], .95) * FPS**2),
        'raw_windows_joint_accel_p95_rad_s2': float(np.quantile(np.max(np.abs(in_window_delta), axis=-1), .95) * FPS**2),
        'raw_windows_early_arm_mae_rad': float(horizon_error[:, :10].mean()),
        'raw_windows_late_arm_mae_rad': float(horizon_error[:, 30:].mean()),
        'unsupported_speed_dips_boundary': sum(bool(transition[i]) for i in dips),
        'unsupported_speed_dips_interior': sum(not transition[i] for i in dips),
        'transition_frames': int(transition.sum()),
        'grip_offsets_frames': grip_offsets,
        'extra_sustained_grip_events': extra_grip_events,
        'phases': phase_scores,
    }, speed, demo_speed


def average(rows):
    fields = ('arm_mae_rad', 'right_ee_mm', 'left_ee_mm', 'peak_step_rad',
              'joint_accel_p95_rad_s2', 'joint_accel_p99_rad_s2', 'joint_accel_near_replan_p95_rad_s2',
              'demo_joint_accel_p95_rad_s2',
              'joint_accel_boundary_p95_rad_s2', 'joint_accel_interior_p95_rad_s2',
              'raw_windows_joint_accel_p95_rad_s2', 'raw_windows_early_arm_mae_rad',
              'raw_windows_late_arm_mae_rad')
    result = {key: float(np.mean([row[key] for row in rows])) for key in fields}
    for label, prefix in [('boundary', 'transition'), ('interior', 'interior')]:
        count = sum(row[f'unsupported_speed_dips_{label}'] for row in rows)
        frames = sum(row['transition_frames'] if prefix == 'transition' else row['samples'] - row['transition_frames'] for row in rows)
        result[f'speed_dips_{label}_per_1k_frames'] = count * 1000 / frames
    offsets = [offset for row in rows for offset in row['grip_offsets_frames'].values()]
    matched = [abs(offset) for offset in offsets if offset is not None]
    result.update(clip_count=len(rows), grip_events_matched=len(matched), grip_events_total=len(offsets),
                  mean_matched_abs_grip_offset_frames=float(np.mean(matched)) if matched else None,
                  extra_grip_event_count=sum(len(row['extra_sustained_grip_events']) for row in rows))
    for event in sorted({key for row in rows for key in row['grip_offsets_frames']}):
        event_offsets = [row['grip_offsets_frames'][event] for row in rows if event in row['grip_offsets_frames']]
        matched = [abs(v) for v in event_offsets if v is not None]
        result[f'{event}_timing'] = dict(matched=len(matched), total=len(event_offsets),
                                        mean_abs_offset_frames=float(np.mean(matched)) if matched else None)
    result['phases'] = {}
    for phase in PHASES:
        scores = [row['phases'][phase] for row in rows if row['phases'][phase]['frames']]
        if scores:
            result['phases'][phase] = {
                key: float(np.mean([score[key] for score in scores])) for key in scores[0] if key != 'frames'
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--strides', type=int, nargs='+', default=[10, 15])
    parser.add_argument('--episodes', type=int, nargs='+')
    parser.add_argument('--skip-plots', action='store_true')
    args = parser.parse_args()

    if not args.skip_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    manifest = json.loads(args.manifest.read_text())
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in manifest['entries']:
        if entry['split'] != 'validation' or args.episodes and entry['episode_id'] not in args.episodes:
            continue
        episode, style = entry['episode_id'], entry['operation_style']
        if not args.skip_plots:
            figure, axes = plt.subplots(len(args.strides) * 2, 1, figsize=(13, 2.2 * len(args.strides) * 2), sharex=True, squeeze=False)
        for col, stride in enumerate(args.strides):
            source = args.replay / f'episode_{episode}' / f'stride_{stride}'
            report = json.loads((source / 'report.json').read_text())
            with np.load(source / 'full_episode.npz') as z:
                data = dict(z)
            demo = data['recorded_ik_target']
            demo_ee = ee_positions(model, demo)
            phases, events = phase_frames(demo, style)
            t = (data['timestamp_ns'] - data['timestamp_ns'][0]) / 1e9
            for frames in (0, 4, 8):
                pred, transition = overlap_targets(data, frames)
                metrics, speed, demo_speed = score_replay(
                    demo, pred, demo_ee, ee_positions(model, pred), phases, events, transition,
                    data['anchor_indices'], data['predicted_windows'])
                records.append(dict(episode=episode, operation_style=style, samples=len(demo), stride=stride,
                                    blend=frames, event_frames=events, checkpoint=report['checkpoint'], **metrics))
                if frames and not args.skip_plots:
                    ax = axes[2 * col + (frames == 8), 0]
                    ax.plot(t, demo_speed, lw=0.8, color='0.6', label='Recorded target')
                    ax.plot(t, speed, lw=0.8, color='#125e85', label='Policy target')
                    for a in data['anchor_indices'][1:]:
                        ax.axvline(t[a], lw=0.5, color='#a45e27', alpha=0.40)
                    ax.set(ylabel='Joint speed norm (rad/s)', title=f'Replan every {stride} frames, blend {frames} frames')
                    ax.legend(loc='upper right')
        if not args.skip_plots:
            axes[-1, 0].set_xlabel('Time (s)')
            figure.suptitle(f'Episode {episode} | {style} | Vertical lines mark replans')
            figure.tight_layout()
            figure.savefig(args.output / f'episode_{episode}_speed.png', dpi=135)
            plt.close(figure)

    if not records or args.episodes and {row['episode'] for row in records} != set(args.episodes):
        raise ValueError('No replays or an incomplete requested episode set')
    summary = {}
    for stride in args.strides:
        for frames in (0, 4, 8):
            selected = [row for row in records if (row['stride'], row['blend']) == (stride, frames)]
            summary[f'{stride}+{frames}'] = {
                'all': average(selected),
                **{style: average([row for row in selected if row['operation_style'] == style])
                   for style in sorted({row['operation_style'] for row in selected})},
            }
    (args.output / 'metrics.json').write_text(json.dumps(dict(
        checkpoint=records[0]['checkpoint'],
        scope='Recorded observation replay, no physical contact or rollout metric.',
        phase_note='Approximate phase intervals derived from sustained gripper commands and left arm motion.',
        ee_note='Distances between predicted and demonstrated openarm control points under the same local kinematic model.',
        aggregation='Equal weight per episode, phase score averaged only over episodes with that phase.',
        methods=summary, episodes=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
