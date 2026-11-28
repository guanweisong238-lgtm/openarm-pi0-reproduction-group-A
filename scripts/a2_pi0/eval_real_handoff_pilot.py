#!/usr/bin/env python3
"""Compare offline 50/12 handoffs on identical and measured dispatch clocks."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from compare_pi0_contact_stages import GraspPoint
from motion_arm import accept_target
from pi0_observations import CONFIG, ROOT, TICK_NS, Plan
from policy_server import PolicyService, RealFourCameraDataset
from prepare_demo0_replay import driver_limits
from real_task_c_phases import grip_intervals


ARM = np.r_[0:7, 8:15]
GRIP = [7, 15]
CAMS = ('global', 'chest', 'wrist_left', 'wrist_right')


def forecast_metrics(error):
    deg = np.rad2deg(abs(error))
    return dict(count=int(deg.size), mae_deg=float(deg.mean()), p95_deg=float(np.quantile(deg, .95)),
                worst_5pct_mae_deg=float(np.sort(deg.ravel())[-max(1, (deg.size+19)//20):].mean()))


def replay(predictions, latencies, state, demo, *, measured, limits, point):
    last = pending = active = None
    rows, handoffs = {}, {}
    next_inference = 0
    for tick, frame in enumerate(range(0, len(state), 3)):
        stamp = tick * TICK_NS
        if pending is not None and pending[0] <= tick:
            _, new = pending
            if active is not None:
                try:
                    old = active.at(stamp)
                except TimeoutError:
                    pass
                else:
                    new_frame = new.observation_ns // TICK_NS * 3
                    old_frame = active.observation_ns // TICK_NS * 3
                    handoffs[tick] = {
                        'new_anchor_frame': new_frame,
                        'max_same_tick_deg': float(np.rad2deg(np.max(abs(new.targets[0, ARM] - old[ARM])))),
                        'left_j5_aligned_old_new_demo_deg': np.rad2deg([
                            active.targets[new_frame-old_frame, 12], new.targets[0, 12], demo[new_frame, 12],
                        ]).tolist(),
                    }
            active, pending = new, None
        if active is not None:
            row = {'reject': None, 'adjacent_deg': None}
            try:
                q = active.at(stamp)
                row['gripper_oob'] = ((q[GRIP] < limits[GRIP, 0]) | (q[GRIP] > limits[GRIP, 1])).tolist()
                row['left_closed_while_demo_open'] = bool(q[15] < .305 <= demo[frame, 15])
                for side, sl in [('right', slice(0, 8)), ('left', slice(8, 16))]:
                    accept_target(q[sl], state[frame, sl], limits[sl], stamp, stamp, side=side)
            except (ValueError, TimeoutError) as exc:
                last = None
                row['reject'] = str(exc)
            else:
                if last is not None:
                    row['adjacent_deg'] = float(np.rad2deg(np.max(abs(q[ARM] - last[ARM]))))
                row['left_j5_deg'] = float(np.rad2deg(q[12]))
                row['left_x_mm'] = float(point(q, 'left')[0] * 1000)
                last = q
            rows[tick] = row
        if tick >= next_inference and frame in predictions and pending is None:
            delay = (latencies[frame] + TICK_NS - 1) // TICK_NS if measured else 3
            pending = (tick + delay, Plan(stamp, predictions[frame]))
            next_inference = tick + 4
    pull = (max(grip_intervals(demo[:, 15], False), key=lambda pair: pair[1] - pair[0])[0] + 2) // 3
    start_x, end_x = (rows.get(t, {}).get('left_x_mm') for t in (pull, pull + 10))
    # Keep post-rejection ticks for diagnosis only; a live run stops on the first rejection.
    j5_goal = float(np.rad2deg(demo[20, 12]))
    j5_near = next((t/10 for t, row in rows.items() if 20 <= 3*t < 60
                    and 'left_j5_deg' in row and abs(row['left_j5_deg'] - j5_goal) <= 3), None)
    return dict(rows=rows, handoffs=handoffs, pull_tick=pull, first_j5_near_demo_frame20_s=j5_near,
                pull_target_mm=None if start_x is None or end_x is None else start_x - end_x,
                demo_pull_mm=float((point(demo[pull*3], 'left')[0] - point(demo[(pull+10)*3], 'left')[0]) * 1000))


def evaluate(checkpoint, history_training_log=None):
    service = PolicyService(checkpoint, ROOT / 'results' / checkpoint.parent.name)
    data = RealFourCameraDataset('validation', root=service.candidate)
    limits, point = driver_limits(), GraspPoint()
    seed = CONFIG['seed']
    service.warmup()
    history_model = None
    if history_training_log:
        train = RealFourCameraDataset('train', root=service.candidate, cache_episodes=18)
        pairs = [json.loads(line)['paired_sample_indices'] for line in history_training_log.read_text().splitlines()]
        assert len(pairs) == 200 and not {r['episode_id'] for r in train.entries} & {r['episode_id'] for r in data.entries}
        features, errors = [], []
        for old, new in pairs:
            ep, frame = train.locate(old)
            assert train.locate(new) == (ep, frame+12)
            if frame < 3:
                continue
            raw = train.get_raw(old)
            earlier = train.get_raw(old-3)['state']
            q, _ = service.infer(raw['images'], raw['state'], seed=seed)
            features.append(raw['state'][ARM] - earlier[ARM])
            errors.append(raw['actions'][12:18, ARM] - q[12:18, ARM])
        x, y = np.stack(features), np.stack(errors)
        mean, scale = x.mean(axis=0), x.std(axis=0).clip(min=1e-6)
        design = (x - mean) / scale
        bias = y.mean(axis=0)
        # Ridge on standardized features, with an unpenalized bias. No validation fitting.
        reg = np.eye(len(ARM)) * (.1 * len(x))
        weights = np.linalg.solve(design.T @ design + reg, design.T @ (y-bias).reshape(len(x), -1))
        history_model = (mean, scale, bias, weights)
        print(f'Fitted history probe on {len(x)} training anchors', flush=True)
    episodes = {}
    for entry in data.entries:
        episode = entry['episode_id']
        with np.load(ROOT / entry['samples_path'], allow_pickle=False) as source:
            state, demo = source['state'], source['actions'][:, 0]
            paths = {cam: source[f'image_path_{cam}'] for cam in CAMS}
        predictions, latencies = {}, {}
        anchors = set(range(0, len(state)-3, 12))
        regression_anchor = 520
        regression = None

        def predict(frame):
            started = time.perf_counter_ns()
            images = {}
            for cam, names in paths.items():
                with Image.open(ROOT / str(names[frame])) as image:
                    images[cam] = np.asarray(image.convert('RGB'))
            q, _ = service.infer(images, state[frame], seed=seed)
            elapsed = time.perf_counter_ns() - started
            if not np.isfinite(q).all() or q.shape != (50, 16):
                raise ValueError(f'Invalid plan for episode {episode}, frame {frame}')
            return q, elapsed

        for frame in sorted(anchors):
            predictions[frame], latencies[frame] = predict(frame)
        # If a replan takes more than four ticks, the next observation is taken on
        # the first available dispatch tick, not snapped to the original grid.
        frame = 0
        measured_predictions, measured_latencies = {}, {}
        while frame < len(state)-3:
            if frame not in predictions:
                q, elapsed = predict(frame)
                predictions[frame], latencies[frame] = q, elapsed
            measured_predictions[frame], measured_latencies[frame] = predictions[frame], latencies[frame]
            frame += 3 * max(4, (latencies[frame] + TICK_NS - 1) // TICK_NS)
        if episode == 24:
            q, _ = predict(regression_anchor)
            regression = {'anchor': regression_anchor, 'offset': 18, 'target_rad': float(q[18, 15]),
                          'demo_rad': float(demo[regression_anchor+18, 15]), 'passes_open_threshold': bool(q[18, 15] >= .305)}
        timings = np.array(list(measured_latencies.values())) / 1e6
        paired = sorted(f for f in anchors if f+12 in anchors and f+18 <= len(demo))
        goals = np.stack([demo[f+12:f+18, ARM] for f in paired])
        old = np.stack([predictions[f][12:18, ARM] for f in paired])
        new = np.stack([predictions[f+12][:6, ARM] for f in paired])
        episodes[episode] = dict(
            p95_ms=float(np.quantile(timings, .95)), max_ms=float(timings.max()),
            over_inference_timeout=int(np.sum(timings > CONFIG['inference_timeout_ms'])),
            fixed=replay({f: predictions[f] for f in anchors}, latencies, state, demo, measured=False, limits=limits, point=point),
            measured=replay(measured_predictions, measured_latencies, state, demo, measured=True, limits=limits, point=point),
            forecast={'old': forecast_metrics(old-goals), 'new': forecast_metrics(new-goals)},
        )
        if history_model:
            mean, scale, bias, weights = history_model
            frames = np.asarray(paired[1:])
            assert paired[0] == 0 and np.all(frames >= 3)
            past = state[frames][:, ARM] - state[frames-3][:, ARM]
            future = state[frames+15][:, ARM] - state[frames][:, ARM]
            raw_error = old[1:] - goals[1:]
            correction = (((past-mean)/scale) @ weights).reshape(raw_error.shape)
            moving = abs(past) >= np.deg2rad(2)
            same_direction_motion = future * np.sign(past)
            groups = {'all': np.ones_like(moving), 'continuing': moving & (same_direction_motion >= np.deg2rad(2)),
                      'slowing_or_stopped': moving & (abs(future) < np.deg2rad(2)),
                      'reversing': moving & (same_direction_motion <= -np.deg2rad(2)), 'little_past_motion': ~moving}
            episodes[episode]['history_probe'] = {group: {name: forecast_metrics(err.transpose(0, 2, 1)[mask])
                                                   for name, err in [('raw', raw_error), ('bias_only', raw_error+bias),
                                                                     ('history', raw_error+bias+correction)]}
                                                   for group, mask in groups.items() if mask.any()}
        if episode == 24:
            episodes[episode]['left_gripper_regression'] = regression
        print(checkpoint.parent.name, episode, f'P95 {np.quantile(timings, .95):.1f} ms', flush=True)
    return dict(checkpoint=str(checkpoint), seed=seed, hardware_used=False, episodes=episodes,
                history_fit_anchors=len(x) if history_model else None,
                protocol={'prediction_frames': 50, 'replan_frames': 12, 'dispatch_hz': 10,
                          'fixed_delay_ms': 300, 'gripper_open_rad': .305,
                          'scope': 'Recorded feedback and target FK only; no hardware or transport. Post-rejection ticks are diagnostic.'})


def summarize(paths):
    runs = [json.loads(path.read_text()) for path in paths]
    weighted = 'weighted' in Path(runs[2]['checkpoint']).parent.name
    names = ('original', 'control', 'weighted' if weighted else 'directed')
    candidate = names[2]
    assert all(set(r['episodes']) == {'13', '19', '20', '24'} and not r['hardware_used'] for r in runs)
    assert len({r['seed'] for r in runs}) == 1
    training = []
    for run in runs[1:]:
        name = Path(run['checkpoint']).parent.name
        directory = ROOT / 'results' / name
        config = json.loads((directory / 'config.json').read_text())
        complete = json.loads((directory / 'summary.json').read_text())
        metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        assert len(metrics) == complete['steps_completed'] == 200
        assert config['directed_handoff'] and config['pair_stride'] == 12 and config['pair_overlap'] == 6
        assert all(m['paired_sample_indices'][1] - m['paired_sample_indices'][0] == 12 for m in metrics)
        training.append((config, complete, metrics))
    a, b = training
    assert all(a[0][key] == b[0][key] for key in ('seed', 'initial_adapter', 'batch_size', 'schedule', 'candidate_manifest_sha256'))
    assert [a[0]['continuity_weight'], b[0]['continuity_weight']] == [0.0, 0.0 if weighted else 0.05]
    assert [r[0].get('old_forecast_weight', 1.0) for r in training] == [1.0, 2.0 if weighted else 1.0]
    assert [m['paired_sample_indices'] for m in a[2]] == [m['paired_sample_indices'] for m in b[2]]
    summary = {}
    for clock in ('fixed', 'measured'):
        report = summary[clock] = {}
        for episode in ('13', '19', '20', '24'):
            variants = [r['episodes'][episode][clock] for r in runs]
            rows = [r['rows'] for r in variants]
            common = set.intersection(*({t for t, row in series.items() if row['adjacent_deg'] is not None} for series in rows))
            outcome = report[episode] = {}
            for name, series, r in zip(names, rows, variants, strict=True):
                rejects = {t for t, row in series.items() if row['reject'] is not None}
                first = min(rejects, key=int, default=None)
                added = rejects & {t for t, row in rows[0].items() if row['reject'] is None}
                changes = [series[t]['adjacent_deg'] for t in common]
                outcome[name] = dict(first_reject_s=None if first is None else int(first)/10,
                                     first_reject_reason=None if first is None else series[first]['reject'],
                                     rejected_ticks=len(rejects), added_rejected_ticks=sorted(map(int, added)),
                                     left_right_gripper_oob=np.sum([v.get('gripper_oob', [False, False])[::-1] for v in series.values()], axis=0).tolist(),
                                     left_closed_while_demo_open=sum(v.get('left_closed_while_demo_open', False) for v in series.values()),
                                     common_over_10=sum(x > 10 for x in changes), common_max_deg=max(changes, default=0),
                                     own_over_10=sum((v['adjacent_deg'] or 0) > 10 for v in series.values()),
                                     pull_target_mm=r['pull_target_mm'],
                                     first_j5_near_demo_frame20_s=r['first_j5_near_demo_frame20_s'])
            outcome.update(common_adjacent_ticks=len(common), demo_pull_mm=variants[0]['demo_pull_mm'])
            if episode == '13' and clock == 'fixed':
                def correction(handoff):
                    old, new, demo = handoff['left_j5_aligned_old_new_demo_deg']
                    return abs(old-demo) - abs(new-demo)

                tick = max(variants[0]['handoffs'], key=lambda t: correction(variants[0]['handoffs'][t]))
                probe = outcome['left_j5_largest_original_correction'] = {'dispatch_s': int(tick)/10}
                for name, r in zip(names, variants, strict=True):
                    probe[name] = r['handoffs'].get(tick)
    summary['timing'] = {name: {ep: {k: row[k] for k in ('p95_ms', 'max_ms', 'over_inference_timeout')}
                               for ep, row in r['episodes'].items()} for name, r in zip(names, runs, strict=True)}
    summary['episode24_left_gripper'] = {name: run['episodes']['24']['left_gripper_regression']
                                         for name, run in zip(names, runs, strict=True)}
    summary['common_over_10_totals'] = {mode: {name: sum(ep[name]['common_over_10'] for ep in summary[mode].values())
                                              for name in names} for mode in ('fixed', 'measured')}
    summary['objective_checks'] = {'candidate_had_no_inference_timeouts': all(
        row['over_inference_timeout'] == 0 for row in summary['timing'][candidate].values())}
    for mode in ('fixed', 'measured'):
        totals = summary['common_over_10_totals'][mode]
        episodes = summary[mode].values()
        summary['objective_checks'][mode] = {
            'no_earlier_first_rejections': all(
                (float('inf') if ep[candidate]['first_reject_s'] is None else ep[candidate]['first_reject_s']) >=
                (float('inf') if ep['original']['first_reject_s'] is None else ep['original']['first_reject_s'])
                for ep in episodes),
            'no_added_rejected_ticks': all(not ep[candidate]['added_rejected_ticks'] for ep in episodes),
            'at_least_one_third_fewer_over_10': totals['original'] > 0 and 3*totals[candidate] <= 2*totals['original'],
            'fewer_over_10_than_control': totals[candidate] < totals['control'],
        }
    summary['training'] = {'paired_orders_match': True}
    for name, (config, complete, metrics) in zip(names[1:], training, strict=True):
        possible = len(metrics) * config['pair_overlap'] * len(ARM)
        hits = int(sum(m['eligible_arm_targets'] for m in metrics)) if 'eligible_arm_targets' in metrics[0] else None
        summary['training'][name] = dict(steps=complete['steps_completed'], eligible_arm_targets=hits,
                                         possible_arm_targets=possible, eligible_fraction=None if hits is None else hits/possible,
                                         final_validation_flow_loss=complete['final_validation_flow_loss'])
    if weighted:
        summary['forecast'] = {name: {ep: row['forecast'] for ep, row in run['episodes'].items()}
                               for name, run in zip(names, runs, strict=True)}
        pooled = summary['forecast_pooled_mae_deg'] = {}
        for name, episodes in summary['forecast'].items():
            pooled[name] = {stage: float(np.average([r[stage]['mae_deg'] for r in episodes.values()],
                                                    weights=[r[stage]['count'] for r in episodes.values()]))
                            for stage in ('old', 'new')}
        count = sum(row['old']['mae_deg'] < summary['forecast']['control'][ep]['old']['mae_deg']
                    for ep, row in summary['forecast'][candidate].items())
        summary['objective_checks']['forecast'] = dict(old_at_least_10pct_better=pooled[candidate]['old'] <= .9*pooled['control']['old'],
                                                       episodes_old_better=count, new_not_worse=pooled[candidate]['new'] <= pooled['control']['new'])
        summary['history'] = {ep: row['history_probe'] for ep, row in runs[0]['episodes'].items()}
        summary['history_fit_anchors'] = runs[0]['history_fit_anchors']
        groups = ('all', 'continuing', 'slowing_or_stopped', 'reversing', 'little_past_motion')
        summary['history_pooled_mae_deg'] = {}
        for group in groups:
            selected = [ep[group] for ep in summary['history'].values() if group in ep]
            if selected:
                summary['history_pooled_mae_deg'][group] = {
                    key: float(np.average([m[key]['mae_deg'] for m in selected], weights=[m[key]['count'] for m in selected]))
                    for key in ('raw', 'bias_only', 'history')
                } | {'count': sum(m['raw']['count'] for m in selected)}
        for mode in ('fixed', 'measured'):
            summary['objective_checks'][mode]['at_most_15_over_10'] = summary['common_over_10_totals'][mode][candidate] <= 15
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--checkpoint', type=Path)
    modes.add_argument('--compare', type=Path, nargs=3, metavar=('ORIGINAL', 'CONTROL', 'CANDIDATE'))
    parser.add_argument('--history-training-log', type=Path, help='Fit a past-displacement residual probe on these training anchors only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'{args.output} already exists')
    result = evaluate(args.checkpoint, args.history_training_log) if args.checkpoint else summarize(args.compare)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
