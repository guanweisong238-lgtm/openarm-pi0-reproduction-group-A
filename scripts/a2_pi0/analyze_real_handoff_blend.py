#!/usr/bin/env python3
"""Replay a causal handoff correction on cached task C IK targets."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/real_prefix_schedule_compare_20261122'
EPISODE = ROOT / 'data/processed/real_alignment_prototype_v001/episode_17/samples.npz'
ARM = np.r_[0:7, 8:15]
JOINTS = [f'{side} J{i}' for side in ('right', 'left') for i in range(1, 8)]


def blend_handoffs(target, switches, steps):
    result = target.copy()
    for start, end in zip(switches, [*switches[1:], len(target)]):
        length = min(steps, end - start)
        phase = np.arange(1, length + 1) / steps
        weight = (1 + np.cos(np.pi * phase)) / 2
        offset = result[start - 1, ARM] - target[start, ARM]
        result[start:start + length, ARM] += weight[:, None] * offset
    return result


def score(target, switches, reference, timestamps, horizon):
    delta = np.abs(np.diff(target[:, ARM], axis=0))
    peak = delta.max(axis=1)
    boundary = np.asarray(switches) - 1
    other = np.ones(len(peak), dtype=bool)
    other[boundary] = False
    row = boundary[np.argmax(peak[boundary])]
    joint = int(delta[row].argmax())
    region = np.concatenate([np.arange(i, i + horizon) for i in boundary])
    return {
        'handoff_median_rad': float(np.median(peak[boundary])),
        'handoff_max_rad': float(peak[row]),
        'handoff_over_0_1_rad': int((peak[boundary] > 0.1).sum()),
        'worst_handoff': {
            'frame_one_based': int(row + 2),
            'elapsed_seconds': float((timestamps[row + 1] - timestamps[0]) / 1e9),
            'joint': JOINTS[joint],
            'from_rad': float(target[row, ARM[joint]]),
            'to_rad': float(target[row + 1, ARM[joint]]),
        },
        'outside_handoff_median_rad': float(np.median(peak[other])),
        'first_8_frames_after_handoff_max_rad': float(peak[region].max()),
        'whole_trace_max_rad': float(peak.max()),
        'whole_trace_over_0_1_rad': int((peak > 0.1).sum()),
        'arm_mae_to_recorded_ik_rad': float(np.abs(target[:, ARM] - reference[:, ARM]).mean()),
    }


def main():
    original = json.loads((SOURCE / 'report.json').read_text())
    frames = original['coverage_frames']
    mode = 'prefix_pilot_9'
    switches = original['modes'][mode]['switch_frames']
    with np.load(SOURCE / 'traces.npz', allow_pickle=False) as traces:
        target = traces[mode][:frames]
        reference = traces['recorded_ik_target'][:frames]
        times = traces['timestamp_ns'][:frames]
    with np.load(EPISODE, allow_pickle=False) as samples:
        np.testing.assert_array_equal(times, samples['timestamp_ns'][:frames])
        observed = samples['state'][:frames]
    if not np.isfinite(target).all():
        raise ValueError('nonfinite predictions')
    report = {'episode': 17, 'frames': frames, 'switches': len(switches),
              'simulated_delay_cycles': 9, 'inherited_flow_steps': original['sampling_steps']}
    report['recorded_observation'] = score(observed, switches, reference, times, 8)
    report['raw_prediction'] = score(target, switches, reference, times, 8)
    for steps in (4, 8, 12):
        result = blend_handoffs(target, switches, steps)
        np.testing.assert_array_equal(result[:, [7, 15]], target[:, [7, 15]])
        np.testing.assert_array_equal(result[np.array(switches) + steps - 1], target[np.array(switches) + steps - 1])
        report[f'blend_{steps}'] = score(result, switches, reference, times, 8)
        report[f'blend_{steps}']['transition_max_rad'] = float(max(
            np.abs(np.diff(result[start - 1:start + steps, ARM], axis=0)).max()
            for start in switches
        ))
    path = SOURCE / 'handoff_blend_analysis.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
