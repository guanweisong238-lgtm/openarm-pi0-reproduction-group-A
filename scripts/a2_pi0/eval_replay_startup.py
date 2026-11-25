#!/usr/bin/env python3
"""Separate model startup error from scheduling delay using the Dora guard replay."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_gripper_replay import STRIDE, episode_data, load_data, load_predictions, replay_guards
from eval_overhead_grasp_geometry import gated_windows

ARMS = np.r_[0:7, 8:15]
LEFT_J5 = 12


def reanchor_first_plan(windows, initial_state, transition_steps):
    result = windows.copy()
    first = result[0]
    fraction = np.minimum(np.arange(len(first)) / transition_steps, 1)
    weight = 1 - fraction**2 * (3 - 2 * fraction)
    offset = initial_state[ARMS] - first[0, ARMS]
    first[:, ARMS] += weight[:, None] * offset
    return result


def joint_comparison(windows, state, demo):
    step = 12
    target = float(windows[0, step, LEFT_J5])
    # With the observed delay, this target lands about eight frames later.
    observations = {'at_plan_start': state[0, LEFT_J5], 'same_step': state[step, LEFT_J5],
                    'eight_frames_later': state[step + 8, LEFT_J5],
                    'demo_target_same_step': demo[step, LEFT_J5]}
    return dict(joint='left_J5', step=step, model_target_rad=target,
                references={label: dict(qpos_rad=float(position), absolute_gap_rad=abs(target - float(position)))
                            for label, position in observations.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    geometry = json.loads(args.geometry.read_text())['evidence']
    rows, timing, joint = [], {}, None
    for entry in load_data():
        if entry['split'] != 'validation':
            continue
        episode = entry['episode_id']
        state, demo, _ = episode_data(entry)
        anchors = np.arange(0, len(demo), STRIDE)
        ready = np.isin(anchors, geometry[str(episode)]['ready_after_drawer'])
        for seed in (1001, 1002, 1003):
            windows = load_predictions(args.cache, seed, episode, anchors)
            gated = gated_windows(windows, ready)
            aligned12 = reanchor_first_plan(gated, state[0], 12)
            aligned24 = reanchor_first_plan(gated, state[0], 24)
            for mode, plans in [('geometry', gated), ('start12', aligned12), ('start24', aligned24)]:
                rows.append(dict(episode=episode, seed=seed, mode=mode,
                                 recorded=replay_guards(plans, state, 260),
                                 echo=replay_guards(plans, state, 260, echo_targets=True)))
            if episode == 34 and seed == 1002:
                joint = joint_comparison(windows, state, demo)
                for delay in (0, 130, 260, 390):
                    timing[delay] = {mode: replay_guards(plans, state, delay)
                                     for mode, plans in [('geometry', gated), ('start24', aligned24)]}
    args.out.write_text(json.dumps(dict(joint=joint, timing=timing, results=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
