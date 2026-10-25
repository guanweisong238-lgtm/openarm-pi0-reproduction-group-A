"""为已保存的真实闭环起步片段评分；不推理、不注入状态或专家动作。"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from drawer_success_training_config import ROOT,MANIFEST
from drawer_simulation import DrawerSimulation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-frames',type=int,default=9)
    parser.add_argument('--prefix-only',action='store_true',help='从较长真实闭环中只评分最初若干步')
    args=parser.parse_args()
    if not 1<=args.expected_frames<=9:parser.error('本诊断仅覆盖首9帧')
    summary=json.loads((args.trial_directory/'summary.json').read_text())
    row=next(r for r in json.loads(MANIFEST.read_text())['episodes'] if r['episode_id']==summary['episode_id'])
    with np.load(ROOT/row['path']/'episode.npz') as data:
        reference=np.concatenate([data['qpos'],data['terminal_qpos'][None]],axis=0)
    with np.load(args.trial_directory/'closed_loop.npz') as data:
        before=data['qpos_before'].copy();actual=data['qpos'].copy()
    total_frames=len(actual)
    if args.prefix_only:
        before=before[:args.expected_frames];actual=actual[:args.expected_frames]
    if len(actual)!=args.expected_frames:raise ValueError('闭环帧数与预设检查范围不一致')
    np.testing.assert_allclose(before[0],reference[0],atol=1e-10,rtol=0)
    env=DrawerSimulation(ROOT/row['path']);scratch=mujoco.MjData(env.model)
    site=env.model.site('right_ee_control_point').id
    def pose(q):
        scratch.qpos[:]=q;mujoco.mj_kinematics(env.model,scratch)
        rot=scratch.site_xmat[site].reshape(3,3).copy()
        return scratch.site_xpos[site]+rot@np.array([0.,0.,-.16]),rot
    values=[]
    try:
        for i,q in enumerate(actual):
            position,rotation=pose(q);target,orientation=pose(reference[i+1])
            values.append(dict(after_control_step=i+1,
                position_mm=float(np.linalg.norm(position-target)*1000),
                attitude_deg=float(np.degrees(np.arccos(np.clip((np.trace(rotation.T@orientation)-1)/2,-1,1)))),
                max_right_joint_error_rad=float(np.max(abs(q[env.qpos_ids[:7]]-reference[i+1,env.qpos_ids[:7]])))))
        result=dict(trial_directory=str(args.trial_directory),episode_id=summary['episode_id'],seed=summary['noise_seed'],
            scope='first nine physical control steps only; not approach/grasp/task success',
            teacher_actions_executed=summary['teacher_actions_executed'],intermediate_states_injected=summary['intermediate_states_injected'],
            frames=len(actual),source_total_frames=total_frames,scored_prefix=args.prefix_only,
            position_gate_mm=5.,attitude_gate_deg=2.,values=values,
            worst_position_mm=max(v['position_mm'] for v in values),worst_attitude_deg=max(v['attitude_deg'] for v in values))
        result['passed']=bool(result['worst_position_mm']<5 and result['worst_attitude_deg']<2
            and result['teacher_actions_executed']==0 and result['intermediate_states_injected']==0)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='values'}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
