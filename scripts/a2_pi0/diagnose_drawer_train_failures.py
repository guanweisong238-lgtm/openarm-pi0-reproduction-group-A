"""Diagnose recorded training failures and reconstruct exact recovery snapshots (no policy inference)."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from drawer_simulation import DrawerSimulation
from drawer_success_training_config import ROOT, MANIFEST


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episode-ids', type=int, nargs='+', default=[9, 10, 11])
    args = p.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    rows = {r['episode_id']: r for r in manifest['episodes']}
    if any(rows[i]['split'] != 'train' for i in args.episode_ids):
        raise ValueError('Only training parents may produce recovery snapshots')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for eid in args.episode_ids:
        trial, = args.run.glob(f'trial_*_episode_{eid:03d}')
        env = DrawerSimulation(ROOT / rows[eid]['path'])
        try:
            with np.load(trial / 'closed_loop.npz') as src:
                closed = {k: src[k] for k in src.files}
            with np.load(env.episode_root / 'episode.npz') as src:
                demo_q = src['qpos'].copy()
                demo_contact = ((src['finger_contact_counts'] > 0) &
                                (src['finger_normal_forces'] > .01)).all(1)
            scratch = mujoco.MjData(env.model)
            site = env.model.site('right_ee_control_point').id
            handle = env.model.geom('drawer_handle').id

            def geometry(q):
                scratch.qpos[:] = q
                mujoco.mj_kinematics(env.model, scratch)
                tip = scratch.site_xpos[site] + scratch.site_xmat[site].reshape(3, 3) @ np.array([0, 0, -.16])
                return tip - scratch.geom_xpos[handle]

            xyz = np.array([geometry(q) for q in closed['qpos']])
            demo_xyz = np.array([geometry(q) for q in demo_q])
            near = int(np.linalg.norm(xyz, axis=1).argmin())
            touched = np.flatnonzero((closed['contacts'] > 0).any(1))
            # Early deviation, approaching stage, and closest approach. Snapshot indices
            # count already executed actions, unlike 1-based post-action report steps.
            selected = sorted(set([0, 12, 30, 60, near]))
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            dest = args.output / f'episode_{eid:03d}'
            dest.mkdir()
            errors = {'qpos': 0., 'qvel': 0.}
            snapshots = []
            for i, action in enumerate(closed['applied']):
                if i in selected:
                    state = np.zeros(mujoco.mj_stateSize(env.model, spec))
                    mujoco.mj_getState(env.model, env.data, state, spec)
                    path = dest / f'snapshot_{i:04d}.npz'
                    np.savez_compressed(path, integration_state=state, eq_data=env.model.eq_data,
                                        qpos=env.data.qpos, qvel=env.data.qvel,
                                        applied_prefix=closed['applied'][:i],
                                        origin=f'{trial.resolve()} before action {i}; train parent {eid}')
                    snapshots.append(str(path.resolve().relative_to(ROOT)))
                env.step(action)
                for key in errors:
                    errors[key] = max(errors[key], float(np.abs(getattr(env.data, key)-closed[key][i]).max()))
            if max(errors.values()) > 1e-9:
                raise RuntimeError(f'Replay mismatch {errors}')
            bilateral = ((closed['contacts'] > 0) & (closed['forces'] > .01)).all(1)
            streak = maximum = 0
            for hit in bilateral:
                streak = streak + 1 if hit else 0
                maximum = max(maximum, streak)
            out = dict(episode_id=eid, split='train', source_trial=str(trial),
                       source_sha256=hashlib.sha256((trial/'closed_loop.npz').read_bytes()).hexdigest(),
                       first_contact_step=int(touched[0]+1) if len(touched) else None,
                       bilateral_force_steps=int(bilateral.sum()), max_bilateral_streak=maximum,
                       closest_step=near+1, closest_tip_minus_handle_mm=(xyz[near]*1000).tolist(),
                       closest_distance_mm=float(np.linalg.norm(xyz[near])*1000),
                       demo_first_bilateral_frame=int(np.flatnonzero(demo_contact)[0]) if demo_contact.any() else None,
                       demo_bilateral_tip_minus_handle_median_mm=(np.median(demo_xyz[demo_contact],axis=0)*1000).tolist(),
                       right_arm_clipped_steps=int((np.abs(closed['requested'][:,:7]-closed['applied'][:,:7])>1e-9).any(1).sum()),
                       replay_max_errors=errors, snapshots=snapshots,
                       geometry_samples=[dict(post_action_step=i+1, tip_minus_handle_mm=(xyz[i]*1000).tolist(),
                                              gripper_target=float(closed['applied'][i,7]))
                                         for i in [0,11,29,59,near,719] if i < len(xyz)])
            results.append(out)
            np.savez_compressed(dest/'geometry.npz', policy_tip_minus_handle=xyz, demo_tip_minus_handle=demo_xyz)
            print(json.dumps(out, ensure_ascii=False), flush=True)
        finally:
            env.close()
    (args.output/'diagnosis.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
