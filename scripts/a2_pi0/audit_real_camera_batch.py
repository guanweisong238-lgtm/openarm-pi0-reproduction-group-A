#!/usr/bin/env python3
"""Read-only raw capture audit with timestamp metrics and sampled JPEG checks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml

CAMERAS = ('global', 'chest', 'wrist_left', 'wrist_right')


def audit_episode(root, episode, sheets):
    directory = root / 'episodes' / str(episode)
    frames = defaultdict(list)
    endings = []
    with (root / 'camera_timing' / f'episode_{episode}.jsonl').open() as log:
        for line in log:
            row = json.loads(line)
            if row['event'] == 'frame':
                frames[row['camera']].append(row)
            elif row['event'] in ('end', 'capture_error'):
                endings.append(row)
    if [row.get('command') for row in endings] != ['success'] or set(frames) != set(CAMERAS):
        raise ValueError('Missing four-camera log or clean success ending')
    streams, cameras, joints = {}, {}, {}
    for camera in CAMERAS:
        rows = frames[camera]
        times = np.array([row['timestamp_ns'] for row in rows], dtype=np.int64)
        seq = np.array([row['source_sequence'] for row in rows])
        if (np.diff(times) <= 0).any() or (np.diff(seq) != 1).any():
            raise ValueError(f'{camera}: timestamp or source sequence issue')
        folder = directory / 'cameras' / camera
        logged = {f'{t}.jpeg': row for t, row in zip(times, rows)}
        actual = {file.name: file.stat().st_size for file in folder.glob('*.jpeg')}
        if actual != {name: row['image_bytes'] for name, row in logged.items()}:
            raise ValueError(f'{camera}: timestamp filenames or sizes disagree with log')
        transform = 'identity-v1' if camera == 'chest' else 'rot180-v1'
        if any(row['image_transform'] != transform for row in rows):
            raise ValueError(f'{camera}: inconsistent transform')
        gray = []
        for index in np.unique(np.linspace(0, len(rows) - 1, 7, dtype=int)):
            row = rows[index]
            path = folder / f'{times[index]}.jpeg'
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != row['jpeg_sha256']:
                raise ValueError(f'{path}: JPEG hash mismatch')
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (720, 1280):
                raise ValueError(f'{path}: JPEG decode or dimensions')
            gray.append(float(image.mean()))
        streams[camera] = times
        cameras[camera] = {'frames': len(times), 'fps': (len(times)-1)*1e9/int(times[-1]-times[0]),
                           'gap_max_ms': float(np.diff(times).max()/1e6), 'sampled_gray': gray}
    for kind in ('obs', 'action'):
        for arm in ('right', 'left'):
            name = f'{kind}_{arm}'
            table = pq.read_table(directory / kind / 'arms' / arm / 'state.parquet')
            times = table['timestamp'].to_numpy().astype('datetime64[ns]').astype(np.int64)
            if len(times) < 2 or (np.diff(times) <= 0).any() or (times < 0).any():
                raise ValueError(f'{name}: invalid timestamps')
            for col in ('qpos', 'qvel', 'qtorque') if kind == 'obs' else ('qpos',):
                values = np.asarray(table[col].to_pylist(), dtype=float)
                if values.shape != (len(times), 8) or not np.isfinite(values).all():
                    raise ValueError(f'{name}: invalid {col}')
                if col == 'qpos':
                    step_max = float(np.abs(np.diff(values[:, :7], axis=0)).max())
                    gripper_span = float(np.ptp(values[:, 7]))
            streams[name] = times
            joints[name] = {'rows': len(times), 'hz': (len(times)-1)*1e9/int(times[-1]-times[0]),
                            'gap_max_ms': float(np.diff(times).max()/1e6),
                            'arm_max_raw_step_rad': step_max, 'gripper_span_rad': gripper_span}
    first = max(times[0] for times in streams.values())
    last = min(times[-1] for times in streams.values())
    global_times = streams['global']
    earliest = int(global_times[np.searchsorted(global_times, first)])
    count = (int(last) - earliest) * 30 // 1_000_000_000 + 1
    if count < 50:
        raise ValueError('Less than one complete 50-step common window')
    grid = earliest + np.arange(count, dtype=np.int64) * 1_000_000_000 // 30
    matched = {name: times[np.searchsorted(times, grid, side='right') - 1] for name, times in streams.items()}
    ages = {name: (grid - times)/1e6 for name, times in matched.items()}
    if episode in sheets:
        tiles = []
        for fraction in (.02, .22, .42, .62, .82, .98):
            i = round((count - 1)*fraction)
            row = []
            for camera in CAMERAS:
                image = cv2.imread(str(directory/'cameras'/camera/f'{matched[camera][i]}.jpeg'))
                image = cv2.resize(image, (320, 180))
                cv2.rectangle(image, (0, 0), (320, 24), (0, 0, 0), -1)
                cv2.putText(image, f'ep {episode} | {camera} | {i/30:.1f}s', (5, 17),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
                row.append(image)
            tiles.append(np.hstack(row))
        cv2.imwrite(str(sheets[episode]), np.vstack(tiles))
    return {'episode':episode, 'cameras':cameras, 'joints':joints, 'common_seconds':float((last-first)/1e9),
            'aligned_frames_estimate':count, 'full_50_step_windows_estimate':count-49,
            'causal_age_ms':{name:{'max':float(age.max()),'p95':float(np.percentile(age,95)),
                                   'over_50ms':int((age>50).sum())} for name,age in ages.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = yaml.safe_load((args.data/'metadata.yaml').read_text())
    episodes = sorted(int(p.name) for p in (args.data/'episodes').iterdir() if p.is_dir() and p.name.isdigit())
    if len(metadata['episodes']) != len(episodes):
        raise ValueError('Metadata episode count disagrees with directories')
    sheets = {i: args.output/f'episode_{i}_four_views.jpg' for i in (0, 26, 33, 48) if i in episodes}
    results, failures = [], []
    for episode in episodes:
        try:
            results.append(audit_episode(args.data, episode, sheets))
        except Exception as exc:
            failures.append({'episode':episode, 'error':str(exc)})
    report = {'dataset':str(args.data.resolve()), 'episodes_found':len(episodes),
              'metadata_success_count':sum(x.get('success') is True for x in metadata['episodes']),
              'episodes_passed':len(results), 'failures':failures, 'episodes':results,
              'task_metadata':metadata['tasks']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print({key: report[key] for key in ('episodes_found','metadata_success_count','episodes_passed','failures')})
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
