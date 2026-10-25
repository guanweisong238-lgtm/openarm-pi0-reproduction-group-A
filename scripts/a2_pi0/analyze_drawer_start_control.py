"""将闭环最初几步与同轨迹示教对齐，分别报告目标误差和实际状态偏离。"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from drawer_simulation import DrawerSimulation
from drawer_success_training_config import ROOT, MANIFEST


def analyze(directory, rows):
    summary = json.loads((directory/'summary.json').read_text())
    row = rows[summary['episode_id']]
    env = DrawerSimulation(ROOT/row['path'])
    scratch = mujoco.MjData(env.model)
    site = env.model.site('right_ee_control_point').id

    def tip(qpos):
        scratch.qpos[:] = qpos
        mujoco.mj_kinematics(env.model, scratch)
        return (scratch.site_xpos[site] +
                scratch.site_xmat[site].reshape(3, 3) @ np.array([0., 0., -.16]))

    try:
        with np.load(directory/'closed_loop.npz') as live, np.load(ROOT/row['path']/'episode.npz') as teacher:
            pred = live['action_blocks'][0]
            target = teacher['action'][:len(pred)]
            target_qpos, pred_qpos = teacher['qpos'][0].copy(), teacher['qpos'][0].copy()
            target_qpos[env.qpos_ids] = target[0]
            pred_qpos[env.qpos_ids] = pred[0]
            result = dict(summary=summary,
                initial_qpos_exact=bool(np.array_equal(live['qpos_before'][0], teacher['qpos'][0])),
                first_action_position_mm=float(np.linalg.norm(tip(pred_qpos)-tip(target_qpos))*1000),
                first_action_right7_abs_error_rad=abs(pred[0,:7]-target[0,:7]).tolist(),
                first_chunk_right7_mae_rad=float(abs(pred[:,:7]-target[:,:7]).mean()),
                peak_contact_counts=live['contacts'].max(axis=0).tolist(),
                state_deviation=[])
            replay_error = 0.
            for index in range(4):
                env.step(teacher['action'][index])
                replay_error = max(replay_error, float(abs(env.data.qpos-teacher['qpos'][index+1]).max()))
            result['separate_teacher_replay_4step_max_qpos_error'] = replay_error
            for steps in (1, 4, 30, 120, 360):
                if steps > len(live['qpos']) or steps >= len(teacher['qpos']):
                    continue
                actual, reference = live['qpos'][steps-1], teacher['qpos'][steps]
                result['state_deviation'].append(dict(steps=steps,
                    max_qpos_error_rad=float(abs(actual-reference).max()),
                    right7_rms_rad=float(np.sqrt(np.mean((actual[env.qpos_ids[:7]]-reference[env.qpos_ids[:7]])**2))),
                    tip_position_mm=float(np.linalg.norm(tip(actual)-tip(reference))*1000)))
            return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = {r['episode_id']: r for r in json.loads(MANIFEST.read_text())['episodes']}
    trials = sorted(p.parent for p in args.run_dir.glob('trial_*/summary.json'))
    if not trials:
        raise ValueError('没有已保存的试验')
    results = [analyze(p, rows) for p in trials]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
