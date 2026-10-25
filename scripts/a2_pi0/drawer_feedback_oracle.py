"""仅供生成监督标签的路径投影反馈专家；禁止把其运行记为模型成功。"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from drawer_success_training_config import ROOT
from drawer_simulation import DrawerSimulation
sys.path[:0]=[str(ROOT/'.cache/drawer-deps'),str(ROOT/'.cache/openarm-control')]
from contact_drawer_controller import finger_handle_contacts


class FeedbackOracle:
    def __init__(self,reference=ROOT/'data/drawer_recovery/continuous_pilot_ready',lookahead=6):
        with np.load(Path(reference)/'episode.npz') as d:
            self.q=d['state'][:,:8].copy();self.actions=d['action'][:,:8].copy()
        self.scale=np.maximum(self.q.std(0),.1);self.lookahead=lookahead

    def action(self,env):
        q=env.data.qpos[env.qpos_ids].copy();v=env.data.qvel[env.velocity_ids]
        distance=np.mean(((self.q-q[:8])/self.scale)**2,axis=1)
        progress=int(np.flatnonzero(distance<=distance.min()+.0004)[-1])
        goal=self.actions[min(progress+self.lookahead,len(self.actions)-1)]
        right=q[:8].copy();right[:7]+=np.clip(.5*(goal[:7]-q[:7])-.03*v[:7],-.08,.08)
        right[7]=goal[7]
        return np.r_[right,env.data.ctrl[env.actuator_ids[8:]]],progress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=40)
    parser.add_argument('--snapshot',type=Path)
    args=parser.parse_args()
    from live_drawer_success import update_judge
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);env=DrawerSimulation();oracle=FeedbackOracle();records=[];judge={}
    import mujoco
    if args.snapshot:
        with np.load(args.snapshot) as s:
            env.model.eq_data[:]=s['eq_data'];mujoco.mj_setState(env.model,env.data,s['integration_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(env.model,env.data);env.steps=len(s['applied_prefix'])
    initial=np.zeros(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model,env.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION)
    eq=env.model.eq_data.copy();offset=env.steps
    for i in range(round(args.seconds*30)):
        action,progress=oracle.action(env);applied=env.step(action)
        c,f=finger_handle_contacts(env.model,env.data,include_stems=True);opening=-float(env.data.qpos[env.drawer_qpos])
        judge=update_judge(judge,opening,c,f,task='open')
        records.append(dict(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),action=applied,opening=opening,contacts=c,forces=f,reference_progress=progress))
        if i%100==0:print('ORACLE',i+1,progress,opening*1000,c,flush=True)
        if judge['success']:break
    np.savez_compressed(args.output/'expert_rollout.npz',**{k:np.asarray([r[k] for r in records]) for k in records[0]},initial_integration_state=initial,initial_eq_data=eq,initial_control_step_offset=offset)
    env.model.eq_data[:]=eq;mujoco.mj_setState(env.model,env.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(env.model,env.data);env.steps=offset;error=0.
    for r in records:env.step(r['action']);error=max(error,float(np.max(np.abs(env.data.qpos-r['qpos']))))
    report=dict(scripted_expert=True,model_success_evidence=False,judge=judge,steps=len(records),peak_open_mm=max(r['opening'] for r in records)*1000,replay_max_qpos_error=error)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True);env.close()


if __name__=='__main__':main()
