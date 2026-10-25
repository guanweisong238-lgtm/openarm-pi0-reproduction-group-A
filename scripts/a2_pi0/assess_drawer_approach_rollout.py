"""接近/首次夹持真实闭环评分：5帧持续双侧组件接触，不等同完整任务成功。"""
from drawer_success_training_config import ROOT,MANIFEST
from drawer_simulation import DrawerSimulation
import argparse,json
from pathlib import Path
import mujoco
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=json.loads((args.trial_directory/'summary.json').read_text())
    row=next(r for r in json.loads(MANIFEST.read_text())['episodes'] if r['episode_id']==summary['episode_id'])
    phase_path=ROOT/'results/drawer_approach_phases_v001/phases.json'
    phase=next(r for r in json.loads(phase_path.read_text())['episodes'] if r['episode_id']==row['episode_id'])
    with np.load(ROOT/row['path']/'episode.npz') as e:reference=e['qpos'].copy()
    with np.load(args.trial_directory/'closed_loop.npz') as e:
        qpos=e['qpos'].copy();before=e['qpos_before'].copy();contacts=e['contacts'].copy();forces=e['forces'].copy();applied=e['applied'].copy()
    assert len(qpos)>0
    np.testing.assert_allclose(before[0],reference[0],atol=1e-10,rtol=0)
    both=np.all(contacts>0,axis=1)&np.all(forces>.01,axis=1)
    stable=np.convolve(both.astype(int),np.ones(5,int),mode='valid')==5 if len(both)>=5 else np.array([],bool)
    streak=longest=0
    for value in both:streak=streak+1 if value else 0;longest=max(longest,streak)
    env=DrawerSimulation(ROOT/row['path']);scratch=mujoco.MjData(env.model);sid=env.model.site('right_ee_control_point').id
    def pose(q):
        scratch.qpos[:]=q;mujoco.mj_kinematics(env.model,scratch);r=scratch.site_xmat[sid].reshape(3,3).copy()
        return scratch.site_xpos[sid]+r@np.array([0,0,-.16]),r
    deviations=[];target,orientation=pose(reference[phase['first_stable_contact_frame']]);replay_error=0.
    try:
        for i,q in enumerate(qpos):
            p,r=pose(q)
            deviations.append(dict(step=i+1,distance_to_reference_contact_mm=float(np.linalg.norm(p-target)*1000),
                attitude_to_reference_contact_deg=float(np.degrees(np.arccos(np.clip((np.trace(r.T@orientation)-1)/2,-1,1)))),
                bilateral=bool(both[i])))
            replay_error=max(replay_error,float(np.max(abs(env.data.qpos-before[i]))))
            env.step(applied[i]);replay_error=max(replay_error,float(np.max(abs(env.data.qpos-q))))
        if replay_error>=1e-10:raise ValueError('动作重放与实际闭环状态不一致')
    finally:env.close()
    completed=len(qpos)==phase['phase_end_exclusive'] and summary['fully_executed']
    result=dict(episode_id=row['episode_id'],split=row['split'],initialization=row['initialization_group'],
        seed=summary['noise_seed'],frames=len(qpos),expected_frames=phase['phase_end_exclusive'],
        action_chunk=summary['action_chunk'],noise_mode=summary.get('noise_mode','fixed'),
        longest_bilateral_frames=longest,first_stable_bilateral_step=int(np.flatnonzero(stable)[0]+1) if stable.any() else None,
        minimum_distance_to_reference_contact_mm=min(v['distance_to_reference_contact_mm'] for v in deviations),
        final_distance_to_reference_contact_mm=deviations[-1]['distance_to_reference_contact_mm'],
        final_attitude_to_reference_contact_deg=deviations[-1]['attitude_to_reference_contact_deg'],
        replay_max_qpos_error=replay_error,completed=completed,
        passed=bool(completed and stable.any() and summary['teacher_actions_executed']==0 and summary['intermediate_states_injected']==0),
        task_success_verified=False,scope='first stable bilateral handle contact only; not secure grasp/pull/roundtrip success',
        trial_directory=str(args.trial_directory),trajectory=deviations)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trajectory'}),flush=True)


if __name__=='__main__':main()
