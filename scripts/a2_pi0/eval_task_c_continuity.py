#!/usr/bin/env python3
"""Compare arm replanning continuity with matched Task C validation observations."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from eval_gripper_replay import (
    STRIDE, episode_data, generate_predictions, load_data, load_predictions, metrics, policy,
)

ARMS = np.r_[0:7, 8:15]


def evaluate(cache, entries, seed):
    joint_differences, dispatch_jumps, episode_rows = [], [], []
    violations = 0
    for entry in entries:
        state, demo, _ = episode_data(entry)
        episode = entry['episode_id']
        anchors = np.arange(0, len(state), STRIDE)
        windows = load_predictions(cache, seed, episode, anchors)
        arms = windows[:, :, ARMS]
        overlap = np.abs(arms[1:, 0] - arms[:-1, STRIDE])
        jumps = np.abs(arms[1:, 0] - arms[:-1, STRIDE - 1])
        violates = (jumps > policy.VELOCITY[ARMS] / 30 + 1e-6).any(axis=1)
        nominal = arms[:, :STRIDE].reshape(-1, len(ARMS))[:len(demo)]
        joint_differences.extend(overlap.max(axis=1))
        dispatch_jumps.extend(jumps.max(axis=1))
        violations += int(violates.sum())
        episode_rows.append(dict(
            episode=episode,
            arm_mae_deg=float(np.rad2deg(np.abs(nominal - demo[:, ARMS]).mean())),
            **metrics(windows, state, demo, delay_ms=316),
        ))

    def percentiles(values):
        return dict(zip(('median', 'p95', 'max'), map(float, np.rad2deg(np.percentile(values, [50, 95, 100])))))

    def stop_counts(field):
        return dict(Counter(row[field]['reason'] or 'finished' for row in episode_rows))

    return dict(
        transitions=len(joint_differences),
        same_time_max_joint_difference_deg=percentiles(joint_differences),
        dispatch_max_joint_jump_deg=percentiles(dispatch_jumps),
        transitions_over_arm_step_limit=violations,
        mean_episode_arm_mae_deg=float(np.mean([row['arm_mae_deg'] for row in episode_rows])),
        early_right_close_over_one_second=sum(
            row['close_delta_frames'] is not None and row['close_delta_frames'] < -30 for row in episode_rows
        ),
        missing_right_close=sum(row['model_close_frame'] is None for row in episode_rows),
        recorded_feedback_stops=stop_counts('first_stop'),
        target_echo_stops=stop_counts('echo_first_stop'),
        episodes=episode_rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-cache', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1002)
    args = parser.parse_args()
    entries = [entry for entry in load_data() if entry['split'] == 'validation']
    baseline = evaluate(args.baseline_cache, entries, args.seed)
    cache = args.out / 'predictions' / args.checkpoint.parent.name / args.checkpoint.name
    generate_predictions(cache, [args.seed], entries, checkpoint=args.checkpoint, run=args.run)
    report = dict(
        seed=args.seed, stride=STRIDE, assumed_guard_delay_ms=316,
        baseline_checkpoint=policy.CONFIG['checkpoint'], pilot_checkpoint=str(args.checkpoint),
        baseline=baseline, pilot=evaluate(cache, entries, args.seed),
    )
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
