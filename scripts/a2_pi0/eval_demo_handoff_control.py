"""Offline Task C positive control: recorded action windows vs cached model plans."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from compare_real_replay_handoffs import ARM, blend_handoffs
from eval_gripper_replay import STRIDE, load_data, load_predictions, policy, replay_guards, variant


def evaluate(windows, state, demo, anchors, delay):
    nominal = windows[:, :STRIDE].reshape(-1, 16)[:len(state)]
    times = anchors[1:]
    step = np.abs(np.diff(nominal[:, ARM], axis=0))
    diffs = {
        'boundary': step[times - 1].max(axis=1),
        'same_time': np.abs(windows[1:, 0] - windows[:-1, STRIDE])[:, ARM].max(axis=1),
        'boundary_feedback': np.abs(nominal[times] - state[times])[:, ARM].max(axis=1),
        'all_steps': step.max(axis=1),
    }
    guards = {key: replay_guards(windows, state, delay, echo_targets=echo)
              for key, echo in [('recorded', False), ('echo', True)]}
    return diffs, dict(**guards, arm_mae_deg=float(np.rad2deg(np.abs(nominal - demo)[:, ARM].mean())),
                       boundary_over_velocity=int((step[times - 1] > policy.VELOCITY[ARM] / 30 + 1e-6).any(axis=1).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--guided', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--delay-ms', type=int, default=316)
    args = p.parse_args()
    rows, values = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for entry in load_data():
        if entry['split'] != 'validation':
            continue
        ep = entry['episode_id']
        with np.load(policy.ROOT / entry['samples_path']) as z:
            state, actions = z['state'], z['actions']
            demo = actions[:, 0]
            anchors = np.arange(0, len(state), STRIDE)
            index = anchors[:, None] + np.arange(actions.shape[1])
            valid = index < len(state)
            np.testing.assert_array_equal(valid, z['action_valid'][anchors])
            for key in ('action_timestamp_ns_right', 'action_timestamp_ns_left', 'actions'):
                x = z[key]
                np.testing.assert_array_equal(x[anchors][valid], x[index[valid], 0])
        def record(label, w):
            diff, row = evaluate(w, state, demo, anchors, args.delay_ms)
            rows[label].append(dict(episode=ep, **row))
            for key, data in diff.items():
                values[label][key].append(data)

        record('demo', actions[anchors])
        rows['demo'][-1]['recorded_zero_delay'] = replay_guards(actions[anchors], state, 0)
        for name, cache in [('baseline', args.baseline), ('pilot', args.pilot)]:
            w = load_predictions(cache, 1002, ep, anchors)
            record(name, w)
            clipped = variant(w, None)
            record(name + '_clip', clipped)
            raw = clipped[:, :STRIDE].reshape(-1, 16)[:len(state)]
            stitched, _ = blend_handoffs(dict(anchor_indices=anchors, predicted_windows=clipped,
                                               predicted_ik_target=raw), 8)
            merged = clipped.copy()
            for i, a in enumerate(anchors):
                n = min(STRIDE, len(state) - a)
                merged[i, :n] = stitched[a:a+n]
            record(name + '_clip_blend8', merged)
        if ep in (5, 38):
            with np.load(args.guided / f'episode_{ep}/predictions.npz') as z:
                np.testing.assert_array_equal(z['anchors'], anchors)
                np.testing.assert_array_equal(z['demo'], demo)
                for mode in ('independent', 'guide_arms_8'):
                    record(mode + '_paired_clip', variant(z[mode + '_windows'], None))

    summary = {}
    for name, episodes in rows.items():
        agg = {k: dict(zip(('median', 'p95', 'max'), np.rad2deg(np.percentile(np.concatenate(v), [50, 95, 100]))))
               for k, v in values[name].items()}
        summary[name] = dict(degrees=agg, boundaries=sum(map(len, values[name]['boundary'])),
                             over_velocity=sum(x['boundary_over_velocity'] for x in episodes),
                             arm_mae_deg=np.mean([x['arm_mae_deg'] for x in episodes]),
                             stops={k: Counter(x[k]['reason'] or 'no_rejection' for x in episodes)
                                    for k in ('recorded', 'echo')})
    report = dict(seed=1002, delay_ms=args.delay_ms, stride=STRIDE, summary=summary, episodes=rows,
                  inputs={k: str(getattr(args,k)) for k in ('baseline','pilot','guided')})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
