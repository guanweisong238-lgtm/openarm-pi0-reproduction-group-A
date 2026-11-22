#!/usr/bin/env python3
"""Paired MuJoCo recovery evaluation using live simulated observations."""
import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import flax.nnx as nnx
from flax import traverse_util
import jax
import mujoco as mj
import numpy as np

from real_delta_actions import decode_deltas, encode_deltas
from real_four_camera_loader import PROMPT, RealFourCameraDataset
from real_prefix_pi0 import PrefixPi0Config
from real_training_support import ROOT, get_train_config, load_parameters
from train_real_four_camera import sha
from train_real_recovery_ablation import RUN, prepared

ARM = np.r_[0:7, 8:15]
GRIP = [7, 15]
SIDES = ('right', 'left')
RATE = 30
STEPS = 150
DELAY = 9
RECOVERY_ERROR = .08
RECOVERY_FRAMES = 10
CAMERAS = ('global', 'chest', 'wrist_left', 'wrist_right')


class Simulation:
    def __init__(self):
        self.model = mj.MjModel.from_xml_path(str(ROOT / 'assets/real_replay_scene/empty_recovery_scene.xml'))
        self.data = mj.MjData(self.model)
        m = self.model
        names = [f'{side}_{kind}' for side in SIDES
                 for kind in [*(f'joint{i}' for i in range(1, 8)), 'finger_joint1']]
        joints = [m.joint('openarm_' + name).id for name in names]
        self.qpos = m.jnt_qposadr[joints]
        self.extra_fingers = [m.jnt_qposadr[m.joint(f'openarm_{side}_finger_joint2').id] for side in SIDES]
        self.controls = np.array([m.actuator(name.replace('finger_joint1', 'finger1') + '_ctrl').id for name in names])
        self.low = np.maximum(m.jnt_range[joints, 0], m.actuator_ctrlrange[self.controls, 0])
        self.high = np.minimum(m.jnt_range[joints, 1], m.actuator_ctrlrange[self.controls, 1])
        self.renderer = mj.Renderer(m, height=224, width=224)
        self.body_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or '' for i in range(m.nbody)]

    def reset(self, state):
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos] = state
        self.data.qpos[self.extra_fingers] = state[GRIP]
        self.data.ctrl[self.controls] = state
        mj.mj_forward(self.model, self.data)

    def observe(self):
        return self.data.qpos[self.qpos].copy()

    def contact(self):
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist >= -.0005:
                continue
            bodies = [self.body_names[self.model.geom_bodyid[g]] for g in (contact.geom1, contact.geom2)]
            same_hand = all('finger' in b for b in bodies) and any(
                all(side in b for b in bodies) for side in SIDES)
            if not same_hand and any(b.startswith('openarm_') for b in bodies):
                return bodies
        return None

    def image(self, view):
        camera = mj.MjvCamera()
        mj.mjv_defaultFreeCamera(self.model, camera)
        if view == 'global':
            camera.lookat[:] = [.25, 0, .75]
            camera.distance, camera.azimuth, camera.elevation = 1.3, 130, -20
        elif view == 'chest':
            camera.lookat[:] = [.30, 0, .48]
            camera.distance, camera.azimuth, camera.elevation = .85, 180, -30
        else:
            side = view.split('_')[1]
            camera.lookat[:] = self.data.body(f'openarm_{side}_ee_base_link').xpos
            camera.distance, camera.azimuth, camera.elevation = .32, 110 if side == 'left' else 250, -35
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def step(self, target, frame):
        self.data.ctrl[self.controls] = target
        while self.data.time < (frame + 1) / RATE - 1e-10:
            mj.mj_step(self.model, self.data)
            if not np.isfinite(self.observe()).all():
                return 'nonfinite_state'
            if self.contact():
                return 'contact_abort'
            state = self.observe()[ARM]
            if np.any(state < self.low[ARM] - .025) or np.any(state > self.high[ARM] + .025):
                return 'joint_limit_abort'
        return None


def make_cases(sim, dataset):
    cases = []
    for entry in dataset.entries:
        with np.load(ROOT / entry['samples_path'], allow_pickle=False) as data:
            states = data['state'][:30]
        for frame in (0, 25):
            goal = states[frame].astype(float)
            if (np.maximum(sim.low - goal, goal - sim.high)[ARM] > .005).any():
                continue
            goal = np.clip(goal, sim.low, sim.high)
            sim.reset(goal)
            if sim.contact():
                continue
            rng = np.random.default_rng(9000 + 100 * entry['episode_id'] + frame)
            for _ in range(30):
                start = goal.copy()
                start[ARM] += rng.uniform(-.21, .21, len(ARM))
                if np.any(start < sim.low) or np.any(start > sim.high):
                    continue
                if np.max(abs(start[ARM] - goal[ARM])) < .14:
                    continue
                sim.reset(start)
                if not sim.contact():
                    cases.append((entry['episode_id'], frame, goal, start))
                    break
    if len(cases) < 3:
        raise RuntimeError(f'Only {len(cases)} paired starts passed initial checks')
    return cases


def rollout(sim, start, goal, predict, keep_video=False):
    sim.reset(start)
    plan = predict(sim, np.repeat(start[None, :], 7, axis=0), 0)
    origin = 0
    pending = None
    command = start.copy()
    errors = [float(np.max(abs(start[ARM] - goal[ARM])))]
    states = [start.copy()]
    targets, frames = [], []
    first_recovery = None
    stable = 0
    relapsed = False
    status = 'completed'
    clipped = 0
    for frame in range(STEPS):
        if frame % 20 == 14:
            previous = plan[frame - 1 - origin:frame + 6 - origin].copy()
            previous[:, GRIP] += sim.observe()[GRIP] - previous[0, GRIP]
            pending = predict(sim, previous, frame)
        if frame >= 14 + DELAY and (frame - 14 - DELAY) % 20 == 0:
            plan, pending = pending, None
            origin = frame - DELAY - 1
        target = plan[frame - origin]
        if not np.isfinite(target).all():
            status = 'nonfinite_target'
            break
        clipped += int(np.any(target < sim.low) or np.any(target > sim.high))
        bounded = np.clip(target, sim.low, sim.high)
        command += np.clip(bounded - command, -1.2 / RATE, 1.2 / RATE)
        stop = sim.step(command, frame)
        targets.append(command.copy())
        state = sim.observe()
        states.append(state)
        errors.append(float(np.max(abs(state[ARM] - goal[ARM]))))
        if keep_video:
            frames.append(sim.image('global'))
        if stop:
            status = stop
            break
        stable = stable + 1 if errors[-1] <= RECOVERY_ERROR else 0
        if first_recovery is None and stable >= RECOVERY_FRAMES:
            first_recovery = float(sim.data.time)
        if first_recovery is not None and errors[-1] > RECOVERY_ERROR:
            relapsed = True
    result = {'status': status, 'initial_error_rad': errors[0], 'best_error_rad': min(errors),
              'final_error_rad': errors[-1], 'first_recovery_seconds': first_recovery,
              'relapsed': relapsed, 'sim_seconds': float(sim.data.time),
              'recovered_and_held': first_recovery is not None and not relapsed and status == 'completed',
              'frames_with_projected_target': clipped}
    trace = {'state': np.array(states), 'targets': np.array(targets), 'arm_max_error_rad': np.array(errors)}
    return result, trace, frames


def save_video(path, left, right):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), RATE, (448, 250))
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    try:
        for i in range(max(len(left), len(right))):
            panel = np.zeros((250, 448, 3), np.uint8)
            for column, (frames, title) in enumerate(((left, 'Real only'), (right, 'Mixed recovery'))):
                x = 224 * column
                if i < len(frames):
                    panel[26:, x:x+224] = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
                elif frames:
                    title += ' - STOPPED'
                cv2.putText(panel, title, (x+4, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            writer.write(panel)
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/real_recovery_closed_loop_20261122')
    out = parser.parse_args().out
    out.mkdir(parents=True)
    validation = RealFourCameraDataset('validation')
    sim = Simulation()
    cases = make_cases(sim, validation)
    print('Paired starts:', [(episode, frame) for episode, frame, _, _ in cases], flush=True)
    with np.load(ROOT / 'results' / RUN / 'delta_stats.npz', allow_pickle=False) as stats:
        mean, std = stats['mean'], stats['std']
    cfg = dataclasses.replace(get_train_config(), model=PrefixPi0Config(
        paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora', action_horizon=50), seed=42)
    graph = nnx.graphdef(nnx.eval_shape(cfg.model.create, jax.random.key(cfg.seed)))
    base = load_parameters(cfg)
    ablation = json.loads((ROOT / 'results/real_recovery_ablation_20261122/report.json').read_text())

    @jax.jit
    def infer(params, obs, prefix, key):
        return nnx.merge(graph, params).sample_actions(key, obs, known_prefix=prefix, num_steps=10)

    results, videos = {}, {}
    try:
        for mode in ('real_only', 'mixed'):
            checkpoint = ROOT / ablation[mode]['checkpoint']
            archive = checkpoint / 'lora_params.npz'
            if sha(archive) != json.loads((checkpoint / 'checkpoint.json').read_text())['lora_sha256']:
                raise ValueError(f'Checkpoint hash mismatch: {archive}')
            model = nnx.merge(graph, base)
            trainable = nnx.state(model).filter(cfg.trainable_filter)
            with np.load(archive, allow_pickle=False) as params:
                trainable.replace_by_pure_dict(traverse_util.unflatten_dict(dict(params), sep='/'))
            nnx.update(model, trainable)
            params = nnx.state(model)

            def predict(environment, previous, frame):
                state = environment.observe()
                images = {view: environment.image(view) for view in CAMERAS}
                raw = {'state': state, 'images': images, 'prompt': PROMPT}
                obs, _ = prepared(validation, raw, np.zeros((50, 16)), mean, std)
                prefix = np.pad(encode_deltas(previous, state, mean, std), ((0, 0), (0, 16)))[None]
                output = np.asarray(infer(params, obs, prefix, jax.random.key(3100 + frame)))[0]
                return decode_deltas(output, state, mean, std)

            rows = []
            for index, (episode, frame, goal, start) in enumerate(cases):
                result, trace, frames = rollout(sim, start, goal, predict, keep_video=index == 0)
                np.savez_compressed(out / f'{mode}_ep{episode}_frame{frame}.npz', goal=goal, **trace)
                rows.append({'episode': episode, 'reference_frame': frame, **result})
                if index == 0:
                    videos[mode] = frames
                print(mode, episode, frame, result, flush=True)
            results[mode] = {'trials': rows, 'recovered': sum(row['first_recovery_seconds'] is not None for row in rows),
                             'recovered_and_held': sum(row['recovered_and_held'] for row in rows),
                             'mean_final_error_rad': float(np.mean([row['final_error_rad'] for row in rows])),
                             'contact_aborts': sum(row['status'] == 'contact_abort' for row in rows)}
        save_video(out / 'paired_first_case.mp4', videos['real_only'], videos['mixed'])
        report = {'scene': 'empty MuJoCo with four approximate rendered views', 'sim_rate_hz': RATE,
                  'flow_steps': 10, 'horizon': 50, 'replan_every_frames': 20, 'inference_starts_after_steps': 14,
                  'assumed_inference_delay_frames': DELAY, 'command_rate_limit_rad_s': 1.2,
                  'recovery_max_arm_error_rad': RECOVERY_ERROR, 'recovery_consecutive_frames': RECOVERY_FRAMES,
                  'maximum_test_seconds': STEPS / RATE, 'reference_given_to_policy': False,
                  'results': results}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        sim.renderer.close()


if __name__ == '__main__':
    main()
