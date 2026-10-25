"""独立重放策略动作，按真实双指接触/拉动/80mm保持1秒评分。"""
from pathlib import Path
import argparse,json
import numpy as np
from drawer_success_training_config import ROOT,MANIFEST
from drawer_simulation import DrawerSimulation
from live_drawer_success import update_judge,finger_handle_contacts


def assess(trial):
    summary=json.loads((trial/'summary.json').read_text())
    row=next(r for r in json.loads(MANIFEST.read_text())['episodes'] if r['episode_id']==summary['episode_id'])
    env=DrawerSimulation(ROOT/row['path']);state={};error=0.;contact_error=0.;velocity_error=0.;first_success=None
    with np.load(trial/'closed_loop.npz') as d:raw={k:d[k].copy() for k in d.files}
    velocity_recorded='qvel' in raw and 'qvel_before' in raw
    try:
        for i,action in enumerate(raw['applied']):
            if 'qpos_before' in raw:error=max(error,float(np.max(np.abs(env.data.qpos-raw['qpos_before'][i]))))
            if velocity_recorded:velocity_error=max(velocity_error,float(np.max(np.abs(env.data.qvel-raw['qvel_before'][i]))))
            env.step(action);error=max(error,float(np.max(np.abs(env.data.qpos-raw['qpos'][i]))))
            if velocity_recorded:velocity_error=max(velocity_error,float(np.max(np.abs(env.data.qvel-raw['qvel'][i]))))
            counts,forces=finger_handle_contacts(env.model,env.data,include_stems=True)
            if not np.array_equal(counts,raw['contacts'][i]):raise ValueError('接触计数重放不一致')
            contact_error=max(contact_error,float(np.max(np.abs(forces-raw['forces'][i]))))
            state=update_judge(state,-float(env.data.qpos[env.drawer_qpos]),counts,forces,task='open')
            if state['success'] and first_success is None:first_success=i+1
        permitted=summary['teacher_actions_executed']==0 and summary['intermediate_states_injected']==0
        velocity_required=summary.get('state_representation')=='position_velocity16'
        valid=permitted and error<1e-9 and contact_error<1e-7 and (not velocity_recorded or velocity_error<1e-9) and (not velocity_required or velocity_recorded)
        return dict(trial=str(trial),episode_id=row['episode_id'],split=row['split'],
            noise_seed=summary['noise_seed'],action_chunk=summary['action_chunk'],noise_mode=summary['noise_mode'],
            criterion='bilateral >.01N for5 steps, contact pulling, opening>=.08m held30 steps',
            replay_max_qpos_error=error,replay_max_contact_force_error=contact_error,
            replay_max_qvel_error=velocity_error if velocity_recorded else None,velocity_sensor_evidence_required=velocity_required,
            teacher_actions_executed=summary['teacher_actions_executed'],intermediate_states_injected=summary['intermediate_states_injected'],
            peak_open_mm=float(np.max(raw['opening'])*1000),first_success_step=first_success,
            judge=state,evidence_valid=valid,model_pull_success=bool(valid and state.get('success')))
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();results=[assess(t) for t in args.trials]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    for result in results:print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
