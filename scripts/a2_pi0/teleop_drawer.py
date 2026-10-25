"""鼠标XY/滑条Z控制右夹持点，键盘开合；数值录制后异步重放生成无损RGB。"""
import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np

from collect_drawer_trial import ROOT, write_json
from drawer_simulation import DrawerSimulation

sys.path[:0] = [str(ROOT / '.cache/drawer-deps'), str(ROOT / '.cache/openarm-control')]
from contact_drawer_controller import ContactDrawerController, finger_handle_contacts


class Teleop:
    def __init__(self, warmup=75):
        self.env = DrawerSimulation()
        self.controller = ContactDrawerController(self.env.model, self.env.data)
        with np.load(self.env.episode_root / 'episode.npz') as e:
            self.prefix = e['action'][:warmup].copy()
        self.recording = False
        self.rows = []
        self.paused = False
        self.message = ''
        self.reset()

    def reset(self):
        self.env.reset()
        for action in self.prefix:
            self.env.step(action)
        self.action = self.env.data.ctrl[self.env.actuator_ids].copy()
        self.target, self.rotation = self.controller.right.king.forward_kinematics()
        self.accepted = self.target.copy()
        self.origin = self.target.copy()
        self.grip = float(self.action[7])
        self.recording = False
        self.rows = []
        self.paused = False
        self.message = 'Ready: drag XY, adjust Z, O=open C=close'

    def start(self):
        if self.recording or self.rows:
            self.message = 'Save current recording or reset before starting another'
            return
        self.env.steps = 0  # 让新episode的16/17子步调度从相位0开始。
        self.initial = np.zeros(mujoco.mj_stateSize(self.env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.env.model, self.env.data, self.initial, mujoco.mjtState.mjSTATE_INTEGRATION)
        self.initial_eq = self.env.model.eq_data.copy()
        self.start_time = self.env.data.time
        self.recording = True
        self.paused = False
        self.message = 'RECORDING: S saves, Space pauses physics'

    def step(self):
        if self.paused:
            return
        delta = self.target - self.accepted
        goal = self.accepted + delta * min(1., .002 / max(np.linalg.norm(delta), 1e-12))
        try:
            q = self.controller._solve_contact_pose(goal, self.action[:7], self.rotation, branch_limit=.08)
            self.action[:7] = q
            self.accepted = goal
        except RuntimeError:
            self.message = 'IK rejected target; move target back toward actual gripper'
        self.action[7] += np.clip(self.grip - self.action[7], -.005, .005)
        self.action = np.clip(self.action, self.env.low, self.env.high)
        if self.recording:
            d, m = self.env.data, self.env.model
            counts, forces = finger_handle_contacts(m, d, include_stems=True)
            native = d.ctrl.copy()
            native[self.env.actuator_ids] = self.action
            self.rows.append(dict(state=d.qpos[self.env.qpos_ids].copy(), qpos=d.qpos.copy(),
                action=self.action.copy(), native_action=native, eq_active=d.eq_active.copy(),
                eq_data=m.eq_data.copy(), drawer=float(d.qpos[self.env.drawer_qpos]),
                timestamp=float(d.time-self.start_time), finger_contact_counts=counts,
                finger_normal_forces=forces))
        self.env.step(self.action)

    def save(self, output):
        if not self.rows:
            self.message = 'No recorded frames; press R first'
            return None
        self.recording, self.paused = False, True
        output.mkdir(parents=True, exist_ok=False)
        arrays = {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
        arrays.update(initial_integration_state=self.initial, initial_eq_data=self.initial_eq,
            terminal_qpos=self.env.data.qpos.copy(), terminal_state=self.env.data.qpos[self.env.qpos_ids].copy(),
            terminal_timestamp=np.asarray(self.env.data.time-self.start_time),
            actuator_names=np.asarray([self.env.model.actuator(int(i)).name for i in self.env.actuator_ids]))
        arrays['substeps'] = np.diff(np.rint(np.arange(len(self.rows)+1)/30/self.env.model.opt.timestep)).astype(int)
        np.savez_compressed(output/'episode.npz', **arrays)
        opening = arrays['drawer'][0] - arrays['drawer']
        open_hold = len(opening) >= 30 and bool(np.any(np.convolve((opening >= .08).astype(int), np.ones(30, int), 'valid') == 30))
        final_open = float(arrays['drawer'][0]-self.env.data.qpos[self.env.drawer_qpos])
        closed = len(opening) >= 30 and bool(np.all(np.abs(opening[-30:]) <= .005)) and abs(final_open) <= .005
        bilateral = bool(np.any(np.all(arrays['finger_contact_counts'] > 0, axis=1)))
        report = dict(self.env.provenance)
        # 只继承场景/相机等来源信息，不沿用专家轨迹的测量值。
        for key in ('lerobot_frames_verified','lerobot_path','contact_metrics','clipped_frames'):
            report.pop(key,None)
        report.update(expert='human_mouse', frames=len(self.rows), seconds=len(self.rows)/30,
            mode='contact', image_shape=None, exported_lerobot=False, human_recording=True,
            contact_scope='handle_assembly: drawer_handle + drawer_handle_stem1 + drawer_handle_stem2',
            warmup_frames=len(self.prefix), initialization='Actual replay of warmup prefix before human control; full integration state captured at R',
            expert_returned_success=None, upstream_returned_success=None,
            success_open_then_close=bool(open_hold and closed and bilateral),
            opened_8cm_held_1s_during_episode=bool(open_hold), final_closed_5mm_held_1s=bool(closed),
            success_8cm_held_1s=False, peak_open_m=float(opening.max()), final_open_m=final_open,
            weld_frames=int(np.any(arrays['eq_active'][:,self.env.weld_mask], axis=1).sum()),
            grasp_diagnostics=[],
            expert_parameters={'pitch_degrees':None,'orientation_matrix':self.rotation.tolist(),
                               'grasp_point_local_m':[0.,0.,-.16]},
            contact_metrics={'bilateral_contact_frames':int(np.all(arrays['finger_contact_counts']>0,axis=1).sum()),
                             'any_finger_handle_contact_frames':int(np.any(arrays['finger_contact_counts']>0,axis=1).sum())},
            replay_verified=False, replay_max_qpos_error=None, rgb_export_status='pending',
            stages=[], limitations=['Human teleoperation; warm-start conditions differ from ready-start demonstrations',
                                    'Success is measured, not manually verified; failures must not enter success dataset'],
            teleop=dict(position_step_limit_m=.002, joint_step_limit_rad=.08,
                        orientation='fixed at initial actual gripper orientation', gravity='same as DrawerSimulation',
                        gripper_limits=[float(self.env.low[7]), float(self.env.high[7])]))
        write_json(output/'report.json', report)
        replay = DrawerSimulation(output)
        try:
            error = 0.
            for i, action in enumerate(arrays['action']):
                error = max(error, float(np.max(np.abs(replay.data.qpos-arrays['qpos'][i]))))
                replay.step(action)
            error = max(error, float(np.max(np.abs(replay.data.qpos-arrays['terminal_qpos']))))
        finally:
            replay.close()
        report.update(replay_max_qpos_error=error, replay_verified=error < 1e-6)
        write_json(output/'report.json', report)
        if error >= 1e-6:
            raise RuntimeError(f'Replay mismatch {error}; numeric data retained for diagnosis')
        self.rows = []
        self.message = f'Saved {output.name}; paused. Reset before next episode.'
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    teleop = Teleop()
    if args.self_check:
        teleop.start()
        teleop.target[1] += .005
        for _ in range(30): teleop.step()
        teleop.grip = -.20
        for _ in range(15): teleop.step()
        out = ROOT/'data/drawer_human'/('selfcheck_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        teleop.save(out)
        print('PASS teleop movement, gripper control, numeric capture and full replay:', out, flush=True)
        teleop.env.close()
        return
    import tkinter as tk
    import mujoco.viewer
    window = tk.Tk()
    window.title('OpenArm 人工示教：鼠标XY，滑条Z，O/C开合')
    tk.Label(window, text='拖动平面移动夹持点；Z滑条调高度\nO 张开 / C 闭合（每次0.01 rad）\nR 新录制 / S 保存 / 空格 暂停 / Home 重置\nQ退出；未保存录制会丢弃').pack()
    canvas = tk.Canvas(window, width=400, height=400, bg='#eeeeee')
    canvas.pack()
    z = tk.DoubleVar(value=float(teleop.target[2]))
    tk.Scale(window, from_=.70, to=.30, resolution=.001, variable=z,
             orient='vertical', length=200, label='世界Z高度(m)').pack(side='right')
    status = tk.StringVar()
    tk.Label(window, textvariable=status, wraplength=480).pack()
    events = deque()
    def drag(event):
        teleop.target[0] = teleop.origin[0] + (200-event.y)/200*.15
        teleop.target[1] = teleop.origin[1] + (event.x-200)/200*.15
    canvas.bind('<Button-1>', drag)
    canvas.bind('<B1-Motion>', drag)
    window.bind('<KeyPress>', lambda event: events.append(event.keysym.upper()))
    for text, key in [('张开 O','O'),('闭合 C','C'),('开始录制 R','R'),('保存 S','S'),('重置 Home','HOME')]:
        tk.Button(window, text=text, command=lambda k=key: events.append(k)).pack(side='left')
    session = ROOT/'data/drawer_human'/datetime.now().strftime('session_%Y%m%d_%H%M%S')
    running, episode = True, 0
    def close():
        nonlocal running
        running = False
    window.protocol('WM_DELETE_WINDOW', close)
    with mujoco.viewer.launch_passive(teleop.env.model, teleop.env.data,
            key_callback=lambda key: events.append('SPACE' if key==32 else chr(key).upper() if 0<=key<128 else '')) as viewer:
        viewer.cam.lookat[:] = [.40,-.24,.53]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .85,-140,-25
        print('TELEOP_READY: XY mouse panel; O/C grip; R record; S save; Space pause; Home reset; Q quit', flush=True)
        try:
            while running and viewer.is_running():
                began = time.perf_counter()
                window.update()
                while events:
                    key = events.popleft()
                    if key == 'Q': running=False
                    elif key == 'O': teleop.grip=max(float(teleop.env.low[7]),teleop.grip-.01)
                    elif key == 'C': teleop.grip=min(float(teleop.env.high[7]),teleop.grip+.01)
                    elif key == 'SPACE': teleop.paused=not teleop.paused
                    elif key == 'R': teleop.start()
                    elif key == 'HOME': teleop.reset();z.set(float(teleop.target[2]))
                    elif key == 'S' and teleop.rows:
                        episode+=1
                        out=teleop.save(session/f'episode_{episode:03d}')
                        if out:
                            with (out/'rgb_export.log').open('w') as stream:
                                subprocess.Popen([sys.executable,str(ROOT/'scripts/a2_pi0/export_human_drawer_rgb.py'),
                                    '--episode-root',str(out)],stdout=stream,stderr=subprocess.STDOUT)
                            print('SAVED_NUMERIC:',out,'RGB export running',flush=True)
                with viewer.lock():
                    # 关闭viewer直接施力，确保运动只能由可记录的16维命令产生。
                    viewer.perturb.active=0
                    viewer.perturb.active2=0
                    teleop.env.data.xfrc_applied[:]=0
                    teleop.target[2]=z.get()
                    teleop.step()
                canvas.delete('all')
                canvas.create_line(200,0,200,400,fill='#aaa')
                canvas.create_line(0,200,400,200,fill='#aaa')
                x=200+(teleop.target[1]-teleop.origin[1])/.15*200
                y=200-(teleop.target[0]-teleop.origin[0])/.15*200
                canvas.create_oval(x-6,y-6,x+6,y+6,fill='red')
                counts,forces=finger_handle_contacts(teleop.env.model,teleop.env.data,include_stems=True)
                opening=-teleop.env.data.qpos[teleop.env.drawer_qpos]*1000
                status.set(f'{teleop.message}\nframes={len(teleop.rows)} grip target={teleop.grip:.3f} rad\nXYZ={np.round(teleop.target,3)}\nOpening={opening:.1f} mm contacts={counts.tolist()} forces={np.round(forces,1)} N')
                viewer.sync()
                time.sleep(max(0.,1/30-(time.perf_counter()-began)))
        finally:
            teleop.env.close()
            window.destroy()


if __name__=='__main__':
    main()
