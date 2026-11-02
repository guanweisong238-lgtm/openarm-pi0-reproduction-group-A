"""学习关节目标的小网络基线，通用PD负责稳定；明确非Pi0模型。"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from torch import nn
from drawer_success_training_config import ROOT,MANIFEST
from drawer_simulation import DrawerSimulation
from drawer_feedback_oracle import FeedbackOracle,finger_handle_contacts
from live_drawer_success import update_judge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--load',type=Path)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--episode-ids',type=int,nargs='+',default=[12])
    p.add_argument('--closed-target',type=float,default=-.0625)
    p.add_argument('--queries',type=Path,help='Train-only multi-parent feedback query data')
    p.add_argument('--seconds',type=float,default=40)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(42);rng=np.random.default_rng(42)
    (args.output/'training_source.py').write_bytes(Path(__file__).read_bytes())
    model=nn.Sequential(nn.Linear(8,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,8))
    if args.load:
        checkpoint=torch.load(args.load,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['weights'])
        xm,xs,ym,ys=[checkpoint[k] for k in ('xm','xs','ym','ys')]
    else:
        oracle=FeedbackOracle()
        with np.load((args.queries or ROOT/'data/drawer_feedback_queries/v001')/'physical_queries.npz') as data:
            q=data['state'][:,:8].copy()
            parents=data['parent_episode_id'].copy() if 'parent_episode_id' in data else np.full(len(q),12)
        allowed={r['episode_id'] for r in json.loads(MANIFEST.read_text())['episodes'] if r['split']=='train'}
        if not set(parents.tolist())<=allowed:raise ValueError('Reject held-out query parents')
        oracles={12:oracle}
        for parent in sorted(set(parents.tolist())-{12}):
            oracles[parent]=FeedbackOracle(ROOT/f'data/drawer_recovery/targeted_20261101/episode{parent:03d}_step0030')
        goal=[]
        for state,parent in zip(q,parents):
            oracle=oracles[int(parent)]
            d=np.mean(((oracle.q-state)/oracle.scale)**2,1);k=int(np.flatnonzero(d<=d.min()+.0004)[-1]);goal.append(oracle.actions[min(k+oracle.lookahead,len(oracle.actions)-1)])
        goal=np.asarray(goal);xm=q.mean(0);xs=np.maximum(q.std(0),.03);ym=goal[:,:7].mean(0);ys=np.maximum(goal[:,:7].std(0),.03)
        x=torch.tensor((q-xm)/xs,dtype=torch.float32);y=torch.tensor((goal[:,:7]-ym)/ys,dtype=torch.float32);closed=torch.tensor(goal[:,7]>-.24,dtype=torch.float32)
        order=rng.permutation(len(q));train=order[120:];validation=order[:120]
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        for step in range(1,args.steps+1):
            index=rng.choice(train,128);prediction=model(x[index]);loss=(prediction[:,:7]-y[index]).square().mean()+nn.functional.binary_cross_entropy_with_logits(prediction[:,7],closed[index])
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if step%2000==0:
                with torch.no_grad():
                    pred=model(x[validation]);mae=float(np.abs(pred[:,:7].numpy()*ys+ym-goal[validation,:7]).mean());accuracy=float(((pred[:,7]>0)==closed[validation].bool()).float().mean())
                print('GOAL_TRAIN',step,float(loss),'heldout_goal_mae_rad',mae,'gripper_accuracy',accuracy,flush=True)
            for group in optimizer.param_groups:group['lr']=1e-4+9e-4*(1-step/args.steps)
        torch.save(dict(weights=model.state_dict(),xm=xm,xs=xs,ym=ym,ys=ys,model_type='MLP_joint_goal_feedback',is_pi0=False,gripper_label='close intent: desired target above open by0.01rad',gripper_slew=.007),args.output/'model.pt')
    rows={r['episode_id']:r for r in json.loads(MANIFEST.read_text())['episodes']};results=[];model.eval()
    for eid in args.episode_ids:
        env=DrawerSimulation(ROOT/rows[eid]['path']);left=env.data.ctrl[env.actuator_ids[8:]].copy();record=[];judge={};grip_command=float(env.data.ctrl[env.actuator_ids[7]])
        for call in range(int(np.ceil(args.seconds*30/4))):
            q=env.data.qpos[env.qpos_ids].copy()
            with torch.no_grad():pred=model(torch.tensor((q[:8]-xm)/xs,dtype=torch.float32)).numpy()
            goal=pred[:7]*ys+ym;close_intent=bool(pred[7]>0)
            for _ in range(4):
                grip_command=min(args.closed_target,grip_command+.007) if close_intent else -.25
                q=env.data.qpos[env.qpos_ids].copy();v=env.data.qvel[env.velocity_ids].copy();command=q[:8].copy();command[:7]+=np.clip(.5*(goal-q[:7])-.03*v[:7],-.08,.08);command[7]=grip_command
                before=env.data.qpos.copy();vb=env.data.qvel.copy();applied=env.step(np.r_[command,left]);counts,forces=finger_handle_contacts(env.model,env.data,include_stems=True);opening=-float(env.data.qpos[env.drawer_qpos]);judge=update_judge(judge,opening,counts,forces,task='open')
                record.append(dict(qpos_before=before,qpos=env.data.qpos.copy(),qvel_before=vb,qvel=env.data.qvel.copy(),applied=applied,requested=np.r_[command,left],contacts=counts,forces=forces,opening=opening))
                if judge['success']:break
            if judge['success']:break
        folder=args.output/f'trial_episode_{eid}';folder.mkdir();np.savez_compressed(folder/'closed_loop.npz',**{k:np.asarray([r[k] for r in record]) for k in record[0]})
        summary=dict(model_type='MLP_joint_goal_feedback',is_pi0=False,checkpoint=str(args.load or args.output/'model.pt'),episode_id=eid,split=rows[eid]['split'],noise_seed=42,noise_mode='deterministic',action_chunk=4,state_representation='right_position_velocity8',teacher_actions_executed=0,intermediate_states_injected=0,judge=judge,steps=len(record),peak_open_mm=max(r['opening'] for r in record)*1000,
            gripper_closed_target=args.closed_target,control='learned goal, generic joint feedback kp0.5 kd0.03 maxdelta0.08; gripper learned binary mode, slew0.007rad/step')
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');results.append(summary);print('GOAL_MODEL_RESULT',summary,flush=True);env.close()
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
