#!/usr/bin/env python3
"""Render recorded, baseline and prefix IK targets; teacher-forced, no physics."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import mujoco
import numpy as np
from analyze_real_handoff_blend import blend_handoffs

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'results/real_prefix_schedule_compare_20261122'
SCENE = ROOT / 'assets/real_replay_scene/empty_recovery_scene.xml'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycles', type=int, choices=(6, 9), default=9)
    parser.add_argument('--output', type=Path, default=RESULT / 'comparison_300ms.mp4')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--blend', action='store_true', help='compare prefix targets before and after 8-step handoff correction')
    args = parser.parse_args()
    report = json.loads((RESULT / 'report.json').read_text())
    with np.load(RESULT / 'traces.npz', allow_pickle=False) as data:
        n = report['coverage_frames']
        traces = [data['recorded_ik_target'][:n].copy(), data[f'unconditioned_baseline_{args.cycles}'][:n].copy(), data[f'prefix_pilot_{args.cycles}'][:n].copy()]
    if args.blend:
        switches_list = report['modes'][f'prefix_pilot_{args.cycles}']['switch_frames']
        traces = [traces[0], traces[2], blend_handoffs(traces[2], switches_list, 8)]
    if any(x.shape != (n, 16) or not np.isfinite(x).all() for x in traces):
        raise ValueError('invalid or incomplete trace')
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model)
    for i in range(model.ngeom):
        body = int(model.geom_bodyid[i]); name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
        if body and name != 'table' and not name.startswith('openarm_'): model.geom_group[i] = 5
    option = mujoco.MjvOption(); option.geomgroup[5] = 0
    ids = [model.joint(f'openarm_{side}_joint{i}').id for side in ('right', 'left') for i in range(1, 8)]
    adr = model.jnt_qposadr[ids]
    fingers = np.array([[model.joint(f'openarm_{side}_finger_joint{i}').id for i in (1, 2)] for side in ('right', 'left')])
    fadr = model.jnt_qposadr[fingers]; lo = model.jnt_range[fingers[:, 0], 0]; hi = model.jnt_range[fingers[:, 0], 1]
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [.25, 0, .85]; camera.distance = 1.25; camera.azimuth = 130; camera.elevation = -20
    renderer = mujoco.Renderer(model, height=270, width=480)
    titles = ['Recorded IK target', 'Original model, no prefix', '7-step prefix pilot']
    if args.blend:
        titles = ['Recorded IK target', 'Prefix before correction', 'Prefix with 8-step correction']
    switches = set(report['modes'][f'prefix_pilot_{args.cycles}']['switch_frames'])
    def render(i):
        panels = []
        for q, title in zip(traces, titles):
            data.qpos[adr] = q[i, np.r_[0:7, 8:15]]
            data.qpos[fadr] = np.clip(q[i, [7, 15]], lo, hi)[:, None]
            mujoco.mj_forward(model, data); renderer.update_scene(data, camera=camera, scene_option=option)
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(img, (0, 0), (480, 26), (0, 0, 0), -1)
            cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
            panels.append(img)
        footer = np.zeros((76, 1440, 3), dtype=np.uint8)
        lines = [f'Episode 17 | frame {i+1}/{n} | {i/30:.2f}s | assumed inference {args.cycles}/30 s | '+ ('PLAN SWITCH' if i in switches else ''),
                 'Recorded observations drive predictions; poses show IK targets only. No physics or closed-loop task evaluation.',
                 'First 583 frames compared; final 7 frames omitted. Gripper display clipped to simulated limits.']
        for j, line in enumerate(lines): cv2.putText(footer, line, (7, 19 + 24*j), cv2.FONT_HERSHEY_SIMPLEX, .47, (255,255,255), 1, cv2.LINE_AA)
        return np.vstack((np.hstack(panels), footer))
    try:
        if args.check:
            for i in (0, n//2, n-1): render(i)
            print(f'PASS: rendered first, middle, final of {n} compared frames'); return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1440, 346))
        if not writer.isOpened(): raise RuntimeError('cannot open video writer')
        try:
            for i in range(n):
                writer.write(render(i))
                if (i+1) % 100 == 0: print(f'rendered {i+1}/{n}', flush=True)
        finally: writer.release()
        print(f'Wrote {args.output}, {n} frames at 30 fps')
    finally: renderer.close()

if __name__ == '__main__': main()
