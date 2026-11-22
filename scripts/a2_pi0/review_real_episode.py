#!/usr/bin/env python3
"""Review synchronized camera images using host receive timestamps (read-only)."""
import argparse
from bisect import bisect_left
from pathlib import Path
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/real_data_20261121_193807/episodes'
CAMERAS = ('global', 'chest', 'wrist_left', 'wrist_right')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=int)
    parser.add_argument('--dataset', type=Path, default=DATA.parent,
                        help='Dataset directory containing episodes/.')
    parser.add_argument('--fps', type=float, default=15, help='Playback speed in frames per second (default 15).')
    parser.add_argument('--check', action='store_true', help='Decode the first frame without opening a window.')
    parser.add_argument('--auto-next', action='store_true', help='Exit automatically at the end of the episode.')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    streams = {}
    for camera in CAMERAS:
        files = sorted((args.dataset / 'episodes' / str(args.episode) / 'cameras' / camera).glob('*.jpeg'))
        if not files:
            parser.error(f'No frames found for episode {args.episode}, camera {camera}')
        streams[camera] = ([int(f.stem) for f in files], files)
    reference = streams['global'][0]
    def render(index):
        t = reference[index]
        tiles = []
        for camera in CAMERAS:
            timestamps, files = streams[camera]
            pos = bisect_left(timestamps, t)
            choices = [j for j in (pos - 1, pos) if 0 <= j < len(timestamps)]
            best = min(choices, key=lambda j: abs(timestamps[j] - t))
            image = cv2.imread(str(files[best]))
            if image is None:
                raise RuntimeError(f'Cannot decode {files[best]}')
            image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
            text = f'{camera}  offset {(timestamps[best] - t) / 1e6:+.1f} ms'
            cv2.rectangle(image, (0, 0), (640, 32), (0, 0, 0), -1)
            cv2.putText(image, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(image)
        result = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
        caption = f'Episode {args.episode} | {index + 1}/{len(reference)} | {(t-reference[0])/1e9:.2f}s | Space pause, A/D frame, J/L 1s, R restart, Q quit'
        cv2.rectangle(result, (0, 688), (1280, 720), (0, 0, 0), -1)
        cv2.putText(result, caption, (8, 710), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        return result
    if args.check:
        frame = render(0)
        print(f'Episode {args.episode}: {len(reference)} global frames; first 4 images decoded; display {frame.shape}')
        return
    print('Space: pause/play; A/D: previous/next frame; J/L: back/forward ~1 second; R: restart; Q/Esc: quit', flush=True)
    print('Offsets are host receive-time differences, not exposure synchronization.', flush=True)
    index, paused = 0, False
    cv2.namedWindow('Real episode review', cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow('Real episode review', render(index))
        key = cv2.waitKey(0 if paused else max(1, round(1000 / args.fps))) & 0xff
        if key in (ord('q'), 27): break
        if key == ord(' '): paused = not paused
        elif key == ord('a'): index = max(0, index - 1); paused = True
        elif key == ord('d'): index = min(len(reference)-1, index + 1); paused = True
        elif key == ord('j'): index = max(0, index - 30); paused = True
        elif key == ord('l'): index = min(len(reference)-1, index + 30); paused = True
        elif key == ord('r'): index = 0
        elif not paused:
            if index == len(reference)-1:
                if args.auto_next: break
                paused = True
            else: index += 1
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
