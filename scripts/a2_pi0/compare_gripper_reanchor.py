#!/usr/bin/env python3
"""Measure gripper drift and handoffs from cached episode 17 predictions."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'results/real_delta_schedule_1000_ep17_20261122'
ANCHORED = ORIGINAL.with_name(ORIGINAL.name + '_gripper_anchor')
ARM = np.r_[0:7, 8:15]
GRIP = [7, 15]


def load(folder):
    report = json.loads((folder / 'report.json').read_text())
    count = report['coverage_frames']
    with np.load(folder / 'traces.npz') as traces:
        return (traces['delta_pilot_9'][:count], traces['recorded_ik_target'][:count],
                np.array(report['modes']['delta_pilot_9']['switch_frames']))


def score(target, reference, switches):
    gripper = target[:, GRIP]
    excess = np.maximum(np.array([-.7854, 0]) - gripper, gripper - np.array([0, .7854]))
    change = np.abs(np.diff(gripper, axis=0)).max(axis=1)
    return {
        'arm_mae_rad': float(np.abs(target[:, ARM] - reference[:, ARM]).mean()),
        'gripper_mae_rad': float(np.abs(gripper - reference[:, GRIP]).mean()),
        'gripper_range_excess_gt_0_05_frames': int((excess.max(axis=1) > .05).sum()),
        'gripper_switch_median_rad': float(np.median(change[switches - 1])),
        'gripper_switch_max_rad': float(change[switches - 1].max()),
        'gripper_whole_max_step_rad': float(change.max()),
    }


def main():
    initial, reference, switches = load(ORIGINAL)
    anchored, anchored_reference, anchored_switches = load(ANCHORED)
    np.testing.assert_array_equal(reference, anchored_reference)
    np.testing.assert_array_equal(switches, anchored_switches)
    results = {'original': score(initial, reference, switches),
               'anchored': score(anchored, reference, switches)}
    for steps in (4, 8, 12):
        target = anchored.copy()
        weights = (1 + np.cos(np.pi * np.arange(1, steps + 1) / steps)) / 2
        for switch in switches:
            count = min(steps, len(target) - switch)
            offset = target[switch - 1, GRIP] - anchored[switch, GRIP]
            target[switch:switch + count, GRIP] += weights[:count, None] * offset
        results[f'anchored_blend_{steps}'] = score(target, reference, switches)
    report = {'frames': len(reference), 'assumed_delay_ms': 300, 'flow_steps': 10,
              'recorded_observations': True, 'blend_changes_feed_back_into_policy': False,
              'metrics': results}
    (ANCHORED / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
