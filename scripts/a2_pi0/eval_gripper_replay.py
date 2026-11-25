#!/usr/bin/env python3
"""Compare gripper changes on recorded observations and replay the Dora guards offline."""
import argparse
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

from real_task_c_phases import grip_intervals

sys.path.insert(0, str(Path(__file__).resolve().parent / 'deploy'))
import dora_policy_node as policy

GRIP = [7, 15]
STRIDE = policy.CONFIG['replan_steps']
FPS = 30
WRIST_IMAGE_SIZE = (12, 8)
GLOBAL_IMAGE_SIZE = (16, 12)
GRASP_IMAGE_SIZE = (20, 18)
GRASP_REGION = (0.56, 0.61, 0.83, 0.90)
MODES = ('baseline', 'clip', 'pose_gate', 'wrist_pose_gate', 'drawer_progress_gate',
         'global_gate', 'global_grasp_region_gate', 'global_grasp_region_drawer_gate')


def load_data():
    from hashlib import sha256

    root = policy.ROOT
    run = json.loads((root / policy.CONFIG['run'] / 'config.json').read_text())
    manifest = root / run['candidate'] / 'manifest.json'
    raw = manifest.read_bytes()
    if sha256(raw).hexdigest() != run['candidate_manifest_sha256']:
        raise ValueError('Replay manifest hash mismatch')
    return json.loads(raw)['entries']


def episode_data(entry):
    with np.load(policy.ROOT / entry['samples_path'], allow_pickle=False) as data:
        cameras = {name: data[f'image_path_{name}'] for name in ('global', 'wrist_right')}
        return data['state'], data['actions'][:, 0], cameras


def image_features(path, size):
    with Image.open(policy.ROOT / path) as image:
        return image_features_from_rgb(image.convert('RGB'), size)


def global_features(paths):
    full, region = [], []
    for path in paths:
        with Image.open(policy.ROOT / path) as image:
            image = image.convert('RGB')
            full.append(image_features_from_rgb(image, GLOBAL_IMAGE_SIZE))
            box = tuple(round(f * bound) for f, bound in zip(GRASP_REGION, image.size * 2))
            region.append(image_features_from_rgb(image.crop(box), GRASP_IMAGE_SIZE))
    return np.stack(full), np.stack(region)


def image_features_from_rgb(image, size):
    rgb = np.asarray(image.resize(size), dtype=np.float32) / 255
    red = rgb[:, :, 0] - rgb[:, :, 1:].max(axis=2)
    return np.stack([rgb.mean(axis=2), red]).ravel()


def close_frame(target):
    intervals = grip_intervals(target[:, 7], True)
    return intervals[0][0] if intervals else None


def observation_features(state, paths, indices):
    wrist = np.stack([image_features(path, WRIST_IMAGE_SIZE) for path in paths['wrist_right'][indices]])
    global_full, global_region = global_features(paths['global'][indices])
    # Relative joint changes distinguish reaching the same arm pose at startup
    # from reaching it after pulling the drawer. All history ends at the anchor.
    moved = state - state[0]
    left_travel = np.maximum.accumulate(np.linalg.norm(moved[:, 8:15], axis=1))
    pose = state[indices, :7]
    wrist_pose = np.column_stack([pose, wrist])
    progress = np.column_stack([wrist_pose, state[indices, 8:], moved[indices, :7],
                                moved[indices, 8:], left_travel[indices]])
    return (pose, wrist_pose, progress, global_full, global_region), left_travel[indices]


def train_gates(entries):
    features, labels, grasp_travel = [], [], []
    for entry in entries:
        if entry['split'] != 'train':
            continue
        state, target, paths = episode_data(entry)
        close = close_frame(target)
        if close is None:
            raise ValueError(f"Missing training grasp in episode {entry['episode_id']}")
        # Sample the approach and the 0.5 seconds before the demonstrated close.
        # The grasp-time label comes only from the training split.
        indices = np.unique(np.r_[np.arange(0, close, 6), np.arange(max(0, close - 15), close + 1, 3)])
        episode, travel = observation_features(state, paths, indices)
        features.append(episode)
        labels.append(indices >= close - 15)
        grasp_travel.append(travel[-1])
    positive = np.concatenate(labels)
    weights = np.where(positive, 0.5 / positive.sum(), 0.5 / (~positive).sum())
    gates = []
    for rows in zip(*features, strict=True):
        x = np.concatenate(rows)
        width = x.shape[1]
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.04)
        standardized = (x - mean) / scale
        weighted = standardized.T * weights
        coefficients = np.linalg.solve(weighted @ standardized + 0.1 * np.eye(width),
                                       weighted @ (positive.astype(float) - 0.5))
        gates.append((mean, scale, coefficients))
    return gates, float(min(grasp_travel))


def gate_scores(state, paths, anchors, gates, drawer_floor):
    inputs, travel = observation_features(state, paths, anchors)
    scores = [((matrix - mean) / scale) @ coefficients + 0.5
              for matrix, (mean, scale, coefficients) in zip(inputs, gates, strict=True)]
    scores[2][travel < drawer_floor] = -np.inf
    scores.append(np.where(travel >= drawer_floor, scores[-1], -np.inf))
    return scores


def variant(windows, scores):
    result = windows.copy()
    limits = policy.LIMITS[GRIP]
    bounds = limits.astype(result.dtype)
    lo = np.where(bounds[:, 0] < limits[:, 0], np.nextafter(bounds[:, 0], np.inf), bounds[:, 0])
    hi = np.where(bounds[:, 1] > limits[:, 1], np.nextafter(bounds[:, 1], -np.inf), bounds[:, 1])
    result[:, :, GRIP] = np.clip(result[:, :, GRIP], lo, hi)
    if scores is not None:
        # Arm once the current observation looks ready. Latching preserves release
        # behavior later in the episode. The pilot controls J8 only.
        ready = np.flatnonzero(scores >= 0.5)
        end = int(ready[0]) if ready.size else len(windows)
        result[:end, :, 7] = lo[0]
    return result


def metrics(windows, states, demo, *, delay_ms=260):
    nominal = windows[:, :STRIDE].reshape(-1, 16)[:len(demo)]
    model_close, demo_close = close_frame(nominal), close_frame(demo)
    if demo_close is None:
        raise ValueError('Missing grasp in validation demonstration')
    grippers = windows[:, :, GRIP].astype(float)
    bounds = policy.LIMITS[GRIP]
    excess = np.maximum(0, np.maximum(bounds[:, 0] - grippers, grippers - bounds[:, 1]))
    # NumPy compares with float64 bounds in the actual guard. Clip to float32
    # values on the inside of the bounds, so compare with the same precision here.
    return dict(
        demo_close_frame=demo_close, model_close_frame=model_close,
        close_delta_frames=None if model_close is None else model_close - demo_close,
        right_oob_blocks=int((excess[:, :, 0] > 0).any(axis=1).sum()),
        left_oob_blocks=int((excess[:, :, 1] > 0).any(axis=1).sum()),
        right_max_excess_rad=float(excess[:, :, 0].max()),
        left_max_excess_rad=float(excess[:, :, 1].max()),
        first_stop=replay_guards(windows, states, delay_ms),
        echo_first_stop=replay_guards(windows, states, delay_ms, echo_targets=True),
    )


def replay_guards(windows, states, delay_ms, *, echo_targets=False):
    """Feed recorded feedback and cached plans to the actual policy state machine."""
    class Replay:
        now = 10_000_000_000
        start = now
        stop = None
        sent = 0

        def submit(self, _operation, *_args):
            return self

        def done(self):
            return self.now >= self.due

        def result(self):
            return windows[controller.seq - 1], delay_ms, None

        def cancel(self):
            return False

        def send_output(self, _name, _value, _metadata=None):
            pass

        def log(self, event, **fields):
            if event == 'observation':
                self.due = self.now + delay_ms * 1_000_000
            elif event == 'dispatch':
                self.sent += 1
            elif event == 'reject':
                details = {k: fields[k] for k in ('step', 'axis', 'value', 'gap_rad') if k in fields}
                self.stop = dict(reason=str(fields['reason']), block=controller.seq,
                                 at_s=round((self.now - self.start) / 1e9, 3),
                                 dispatched=self.sent, **details)

    replay = Replay()
    controller = policy.Execution(replay, replay, 'transport')
    controller.loading = None
    controller.status = dict.fromkeys(policy.SIDES, ('aligned', 1))
    wall_offset = 100_000_000_000
    with (patch.object(policy.time, 'monotonic_ns', lambda: replay.now),
          patch.object(policy.time, 'time_ns', lambda: replay.now + wall_offset),
          patch.object(policy, 'log', replay.log)):
        for ms in range(0, (len(states) * 1000 + FPS - 1) // FPS):
            replay.now = replay.start + ms * 1_000_000
            state = states[ms * FPS // 1000]
            if echo_targets:
                # Match the mock's instant target echo after the recorded initial pose.
                state = states[0] if controller.last_target is None else controller.last_target
            stamp = replay.now + wall_offset
            controller.frames = dict.fromkeys(policy.CAMERAS, (stamp, b''))
            controller.feedback = {'right': (stamp, state[:8]), 'left': (stamp, state[8:])}
            controller.tick()
            if replay.stop:
                break
    return replay.stop or dict(reason=None, dispatched=replay.sent, block=controller.seq)


def load_predictions(directory, seed, episode, anchors):
    with np.load(directory / f'seed{seed}' / f'episode_{episode}.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['anchors'], anchors)
        windows = data['windows']
        if windows.shape != (len(anchors), 50, 16) or not np.isfinite(windows).all():
            raise ValueError(f'Invalid predictions for seed {seed}, episode {episode}')
        return windows


def generate_predictions(directory, seeds, entries, *, checkpoint=None, run=None,
                         stride=STRIDE, record_latency=False):
    from hashlib import sha256
    import time

    from policy_server import CAM_ORDER, PolicyService, decode_jpeg

    root = policy.ROOT
    service = PolicyService(checkpoint or root / policy.CONFIG['checkpoint'],
                            run or root / policy.CONFIG['run'])
    if record_latency:
        service.warmup()
    for seed in seeds:
        folder = directory / f'seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            episode = entry['episode_id']
            output = folder / f'episode_{episode}.npz'
            if output.exists():
                continue
            source = root / entry['samples_path']
            if sha256(source.read_bytes()).hexdigest() != entry['samples_sha256']:
                raise ValueError(f'Replay sample hash mismatch for episode {episode}')
            with np.load(source, allow_pickle=False) as samples:
                state = samples['state']
                paths = {camera: samples[f'image_path_{camera}'] for camera in CAM_ORDER}
            anchors = np.arange(0, len(state), stride)
            windows, latency = [], []
            for frame in anchors:
                start = time.perf_counter()
                images = {camera: decode_jpeg((root / paths[camera][frame]).read_bytes()) for camera in CAM_ORDER}
                targets, _ = service.infer(images, state[frame], seed=seed)
                windows.append(targets)
                if record_latency:
                    latency.append((time.perf_counter() - start) * 1000)
            with output.with_suffix('.tmp').open('wb') as handle:
                timing = {'latency_ms': np.asarray(latency)} if record_latency else {}
                np.savez_compressed(handle, anchors=anchors, windows=windows, **timing)
            output.with_suffix('.tmp').replace(output)
            print(f'Seed {seed}, episode {episode}: {len(anchors)} plans', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1001, 1002, 1003])
    parser.add_argument('--generate', action='store_true', help='Run the model for missing cached predictions')
    args = parser.parse_args()
    entries = load_data()
    validation = [entry for entry in entries if entry['split'] == 'validation']
    if args.generate:
        generate_predictions(args.cache, args.seeds, validation)
    gates, drawer_floor = train_gates(entries)
    rows = []
    for entry in validation:
        episode = entry['episode_id']
        state, demo, paths = episode_data(entry)
        anchors = np.arange(0, len(state), STRIDE)
        scores = gate_scores(state, paths, anchors, gates, drawer_floor)
        for seed in args.seeds:
            windows = load_predictions(args.cache, seed, episode, anchors)
            candidates = [windows, variant(windows, None), *(variant(windows, score) for score in scores)]
            for mode, candidate in zip(MODES, candidates, strict=True):
                rows.append(dict(episode=episode, seed=seed, mode=mode, plans=len(windows),
                                 **metrics(candidate, state, demo)))
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(plan_stride=STRIDE, assumed_inference_delay_ms=260,
                  grasp_label='First sustained right closure, 10 frames above -0.305 rad',
                  gate='Training-only ridge classifier; 0.5 threshold; latched at a 20-frame observation',
                  drawer_floor_rad=drawer_floor,
                  global_grasp_region=GRASP_REGION,
                  guards='Dora policy state machine, virtual clock, synthetic fresh timestamps and cached plans',
                  feedback='Recorded states, and a separate instant target echo after the recorded initial state',
                  scope='Full recorded observations including the tail; each plan scored over 50 predicted steps; '
                        'closing time over the nominal splice without the replay delay; no dynamics')
    (args.out / 'summary.json').write_text(json.dumps(dict(config=config, results=rows), indent=2) + '\n')
    with (args.out / 'comparison.csv').open('w', newline='') as handle:
        fields = [key for key in rows[0] if not key.endswith('first_stop')]
        writer = csv.DictWriter(handle, fields + ['stop_reason', 'stop_at_s', 'dispatched',
                                                  'echo_stop_reason', 'echo_stop_at_s', 'echo_dispatched'])
        writer.writeheader()
        for row in rows:
            stop = row['first_stop']
            echo = row['echo_first_stop']
            writer.writerow(dict({key: row[key] for key in fields}, stop_reason=stop['reason'],
                                 stop_at_s=stop.get('at_s'), dispatched=stop['dispatched'],
                                 echo_stop_reason=echo['reason'], echo_stop_at_s=echo.get('at_s'),
                                 echo_dispatched=echo['dispatched']))


if __name__ == '__main__':
    main()
