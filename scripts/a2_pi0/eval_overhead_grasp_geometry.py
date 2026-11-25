#!/usr/bin/env python3
"""Find the visible cube and right fingers in Task C recordings and score grasp timing."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from eval_gripper_replay import (
    FPS, STRIDE, close_frame, episode_data, load_data, load_predictions, policy, variant,
)


def masks(image):
    blue, green, red = np.moveaxis(image.astype(np.int16), 2, 0)
    pink = (red > 155) & (green >= 75) & (blue >= 65) & (red - green > 45) & (red - blue > 36)
    finger = (red > 105) & (green < 100) & (blue < 110) & (red - green > 65) & (red - blue > 55)
    return pink, finger


def components(mask):
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    return labels, stats


def find_cube(image):
    # Fixed tabletop region of the 1280x720 overhead camera in this recording batch.
    pink, _ = masks(image[330:, 750:1100])
    labels, stats = components(pink)
    candidates = []
    for index, (x, y, width, height, area) in enumerate(stats[1:], 1):
        fill = area / (width * height)
        if not (45 <= width <= 110 and 32 <= height <= 110 and 1950 <= area <= 6800 and fill > .54):
            continue
        aspect = max(width, height) / min(width, height)
        score = area * fill**2 / aspect**.65 + 11 * x
        reference = labels[y:y + height, x:x + width] == index
        candidates.append((score, int(x + 750), int(y + 330), int(width), int(height), reference))
    return max(candidates, key=lambda item: item[0])[1:] if candidates else None


def locate(image, cube):
    if cube is None:
        return dict(visibility=None, upper_tip=None, lower_tip=None, reach=None, balance=None)
    x, y, width, height, reference = cube
    left, top = x - 115, y - 110
    right, bottom = min(x + width + 60, image.shape[1]), min(y + height + 110, image.shape[0])
    pink, red = masks(image[top:bottom, left:right])
    box = np.s_[y - top:y + height - top, x - left:x + width - left]
    visibility = float((pink[box] & reference).sum() / reference.sum())
    result = dict(visibility=visibility, upper_tip=None, lower_tip=None, reach=None, balance=None)
    if visibility < .57:
        return result
    # Keep the cube out of the finger search. The two fingers should be on
    # different sides of the cube, approaching along the image's rightward axis.
    red[box] = False
    labels, stats = components(red)
    center_y = y + height / 2
    candidates = [[], []]
    for index, (dx, dy, span_x, span_y, area) in enumerate(stats[1:], 1):
        if not (span_x >= 36 and span_y >= 12 and 400 <= area <= 4300):
            continue
        py, px = np.nonzero(labels[dy:dy + span_y, dx:dx + span_x] == index)
        px, py = px + dx + left, py + dy + top
        tip_x = float(np.quantile(px, .92))
        tip_y = float(np.median(py[px >= tip_x]))
        if not (x - 55 <= tip_x <= x + width + 40 and .22 * height <= abs(tip_y - center_y) <= 1.65 * height):
            continue
        # A finger whose distal pixels hit the image edge cannot be localized.
        if (py[px >= tip_x] >= image.shape[0] - 3).any():
            continue
        score = tip_x - .20 * abs(tip_y - center_y) + min(area, 1700) / 200
        candidates[int(tip_y >= center_y)].append((score, tip_x, tip_y))
    upper, lower = [max(tips)[1:] if tips else None for tips in candidates]
    result.update(upper_tip=upper, lower_tip=lower)
    if upper is not None and lower is not None:
        result['reach'] = (min(upper[0], lower[0]) - x) / width
        result['balance'] = abs((upper[1] + lower[1]) / 2 - center_y) / height
    return result


def read_image(path):
    image = cv2.imread(str(policy.ROOT / path))
    if image is None or image.shape != (720, 1280, 3):
        raise ValueError(f'Unexpected overhead image dimensions: {path}')
    return image


def left_travel(state):
    return np.maximum.accumulate(np.linalg.norm(state[:, 8:15] - state[0, 8:15], axis=1))


def ready(detection):
    return detection['reach'] is not None and detection['reach'] >= -.12 and detection['balance'] <= .35


def gated_windows(windows, readiness):
    targets = variant(windows, None)
    committed = False
    for target, allow in zip(targets, readiness, strict=True):
        if not committed and not allow:
            target[:, 7] = np.float32(policy.LIMITS[7, 0])
        if close_frame(target[:STRIDE]) is not None:
            committed = True
    return targets


def nominal_trace(windows, sample_count):
    return windows[:, :STRIDE].reshape(-1, 16)[:sample_count]


def annotate(path, cube, detection, frame, label):
    image = read_image(path)
    if cube is not None and detection['visibility'] >= .57:
        x, y, width, height = cube[:4]
        cv2.rectangle(image, (x, y), (x + width, y + height), (20, 225, 250), 2)
    for key in ('upper_tip', 'lower_tip'):
        point = detection[key]
        if point is not None:
            cv2.circle(image, tuple(round(p) for p in point), 6, (250, 235, 20), 2)
    crop = image[470:, 760:1080].copy()
    cv2.rectangle(crop, (0, 0), (320, 34), (28, 27, 31), -1)
    cv2.putText(crop, f'{label}  {frame / FPS:.2f}s', (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .52, (244, 244, 244), 1, cv2.LINE_AA)
    return crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    entries = load_data()
    threshold = min(left_travel(state)[close_frame(target)]
                    for state, target, _ in (episode_data(e) for e in entries if e['split'] == 'train'))
    rows, evidence = [], {}
    args.out.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if entry['split'] != 'validation':
            continue
        episode = entry['episode_id']
        state, demo, cameras = episode_data(entry)
        paths = cameras['global']
        cube = find_cube(read_image(paths[0]))
        demonstrated = close_frame(demo)
        anchors = np.arange(0, len(demo), STRIDE)
        detections = [locate(read_image(paths[i]), cube) for i in anchors]
        geometry = np.array([ready(d) for d in detections])
        progress = geometry & (left_travel(state)[anchors] >= threshold)
        evidence[episode] = dict(cube=None if cube is None else cube[:4],
                                 ready_anchors=anchors[geometry].tolist(),
                                 ready_after_drawer=anchors[progress].tolist(),
                                 demonstration=locate(read_image(paths[demonstrated]), cube))
        for seed in (1001, 1002, 1003):
            windows = load_predictions(args.cache, seed, episode, anchors)
            for method, candidate in [('baseline', windows),
                                      ('geometry', gated_windows(windows, geometry)),
                                      ('geometry_and_drawer', gated_windows(windows, progress))]:
                found = close_frame(nominal_trace(candidate, len(demo)))
                rows.append(dict(episode=episode, seed=seed, method=method, demo_frame=demonstrated,
                                 close_frame=found, delta_frames=None if found is None else found - demonstrated))
        if episode == 38:
            baseline = load_predictions(args.cache, 1002, episode, anchors)
            frames = (close_frame(nominal_trace(baseline, len(demo))), demonstrated, 300)
            examples = {label: locate(read_image(paths[frame]), cube) for frame, label in zip(
                frames, ('early', 'demo', 'after_pickup'), strict=True)}
            evidence[episode]['examples'] = examples
            panels = [annotate(paths[frame], cube, detection, frame, label)
                      for (label, detection), frame in zip(examples.items(), frames, strict=True)]
            cv2.imwrite(str(args.out / 'episode38-marked.jpg'), np.hstack(panels))
    with (args.out / 'grasp_timing.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / 'geometry.json').write_text(json.dumps(
        dict(progress_rad=float(threshold), evidence=evidence, results=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
