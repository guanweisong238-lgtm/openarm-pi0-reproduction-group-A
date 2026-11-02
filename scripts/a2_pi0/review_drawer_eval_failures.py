"""Interactively replay failed evaluation trials using their recorded actions."""
import argparse
import json
import os
from pathlib import Path
import threading
import time
from collections import deque

os.environ.setdefault('MUJOCO_GL', 'glfw')
import mujoco
import numpy as np
from drawer_simulation import DrawerSimulation

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'logs/a2_pi0/grasp_improvement_20261102/baseline020_final100')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/processed/openarm_drawer_eval_v004_final/manifest.json')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    rows = {row['episode_id']: row for row in json.loads(args.manifest.read_text())['episodes']}
    cases = []
    for path in sorted(args.run.glob('trial_*/summary.json')):
        summary = json.loads(path.read_text())
        if not summary['judge']['success']:
            cases.append((path.parent, summary))
    if not cases:
        raise ValueError('No failed trials found')
    if not args.headless:
        import mujoco.viewer
    for number, (directory, summary) in enumerate(cases, 1):
        with np.load(directory / 'closed_loop.npz', allow_pickle=False) as source:
            data = {key: source[key].copy() for key in ('applied', 'qpos_before', 'qpos', 'qvel_before', 'qvel')}
        env = DrawerSimulation(ROOT / rows[summary['episode_id']]['path'])
        maximum = 0.0

        def step(index):
            nonlocal maximum
            maximum = max(maximum, float(np.max(np.abs(env.data.qpos - data['qpos_before'][index]))),
                          float(np.max(np.abs(env.data.qvel - data['qvel_before'][index]))))
            env.step(data['applied'][index])
            maximum = max(maximum, float(np.max(np.abs(env.data.qpos - data['qpos'][index]))),
                          float(np.max(np.abs(env.data.qvel - data['qvel'][index]))))
            if maximum > 1e-9:
                raise RuntimeError(f'Replay diverged: {maximum}')

        def finish():
            print(f'REPLAY_PASS episode={summary["episode_id"]} steps={len(data["applied"])} max_state_error={maximum}', flush=True)

        try:
            if args.headless:
                for index in range(len(data['applied'])):
                    step(index)
                finish()
                continue
            events = deque()
            next_case = False
            threads_before = set(threading.enumerate())
            with mujoco.viewer.launch_passive(env.model, env.data, key_callback=events.append,
                                             show_left_ui=False, show_right_ui=False) as viewer:
                viewer.cam.lookat[:] = [.40, -.24, .53]
                viewer.cam.distance = .9
                viewer.cam.azimuth = -140
                viewer.cam.elevation = -25
                index, paused = 0, False
                resume_at = time.perf_counter() + 4
                print(f'GUI_READY failure={number}/{len(cases)} episode={summary["episode_id"]} mode={summary.get("failure_mode")} Space=pause R=restart N=next Q=quit', flush=True)
                while viewer.is_running():
                    began = time.perf_counter()
                    while events:
                        key = events.popleft()
                        if key in (ord('Q'), ord('q')):
                            return
                        if key in (ord('N'), ord('n')):
                            next_case = True
                        if key == 32:
                            paused = not paused
                        if key in (ord('R'), ord('r')):
                            with viewer.lock():
                                env.reset()
                            index, maximum = 0, 0.0
                            resume_at = began + 1
                    if next_case:
                        break
                    with viewer.lock():
                        viewer.perturb.active = 0
                        viewer.perturb.active2 = 0
                        env.data.xfrc_applied[:] = 0
                        if not paused and began >= resume_at and index < len(data['applied']):
                            step(index)
                            index += 1
                            if index == len(data['applied']):
                                finish()
                    status = 'Finished: R replay / N next' if index == len(data['applied']) else 'Paused' if paused else 'Playing'
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      'Recorded failure\nFailure mode\nTime\nOpening\nStatus\nSpace / R / N / Q',
                                      f'{number}/{len(cases)}  Episode {summary["episode_id"]}\n{summary.get("failure_mode")}\n'
                                      f'{index/30:.1f} / {len(data["applied"])/30:.1f} s\n'
                                      f'{-env.data.qpos[env.drawer_qpos]*1000:.1f} mm\n{status}\nPause / Replay / Next / Quit'))
                    viewer.sync()
                    time.sleep(max(0, 1/30 - (time.perf_counter() - began)))
            for thread in set(threading.enumerate()) - threads_before:
                if '_launch_internal' in thread.name:
                    thread.join(timeout=10)
            if not next_case:
                return
        finally:
            env.close()


if __name__ == '__main__':
    main()
