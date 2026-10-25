"""固定柜体的轻量位置/速度反馈基线；单独标注，不能冒充Pi0结果。"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from drawer_success_training_config import ROOT,MANIFEST
from drawer_simulation import DrawerSimulation
from live_drawer_success import update_judge,finger_handle_contacts


def kernel(a,b,linear=False):
    distance=np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0)
    return np.exp(-distance/a.shape[1])+(a@b.T/a.shape[1]+1 if linear else 0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/drawer_recovery/continuous_v003/manifest.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode-ids',type=int,nargs='+',default=[12,2,6])
    p.add_argument('--seconds',type=float,default=24)
    p.add_argument('--ridge',type=float,default=1e-4)
    p.add_argument('--queries',type=Path,help='使用物理分支专家查询替代连续示教')
    p.add_argument('--linear-kernel',action='store_true')
    p.add_argument('--binary-gripper',action='store_true',help='模型夹爪预测按开/闭两种执行器目标执行')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    source=json.loads(args.manifest.read_text());features=[];targets=[]
    if args.queries:
        with np.load(args.queries/'physical_queries.npz') as d:
            q=d['state'][:,:8];v=d['velocity'][:,:8];y=d['expert_actions'][:,:4,:8].copy()
            y[:,:,:7]-=q[:,None,:7];features.append(np.concatenate([q,v],1));targets.append(y.reshape(len(q),32))
    for row in ([] if args.queries else source['episodes']):
        if row['split']!='train':raise ValueError('拒绝validation训练')
        env=DrawerSimulation(ROOT/row['path'])
        with np.load(ROOT/row['path']/'episode.npz') as d:
            q=d['state'][:,:8];v=d['qvel'][:,env.velocity_ids[:8]];a=d['action'][:,:8]
            y=a[np.minimum(np.arange(len(a))[:,None]+np.arange(4),len(a)-1)].copy()
            y[:,:,:7]-=q[:,None,:7]
            features.append(np.concatenate([q,v],1));targets.append(y.reshape(len(a),32))
        env.close()
    x=np.concatenate(features);y=np.concatenate(targets);xm=x.mean(0);xs=np.maximum(x.std(0),np.r_[np.repeat(.02,8),np.repeat(.1,8)])
    ym=y.mean(0);ys=np.maximum(y.std(0),.005);x=(x-xm)/xs
    w=cho_solve(cho_factor(kernel(x,x,args.linear_kernel)+np.eye(len(x))*args.ridge),(y-ym)/ys)
    train_mae=float(np.abs((kernel(x,x,args.linear_kernel)@w)*ys+ym-y).mean())
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'model.npz',support=x,weights=w,input_mean=xm,input_std=xs,target_mean=ym,target_std=ys,linear_kernel=args.linear_kernel)
    model_info=dict(model_type='kernel_ridge_state_feedback',is_pi0=False,input='actual right8 positions+right8 velocities',
        target='4 future right8 servo targets, right7 residual and absolute gripper',images_used=False,
        training_manifest=str(args.queries/'READY.json' if args.queries else args.manifest),training_manifest_sha256=hashlib.sha256((args.queries/'READY.json' if args.queries else args.manifest).read_bytes()).hexdigest(),
        training_frames=len(x),ridge=args.ridge,linear_kernel=args.linear_kernel,binary_gripper=args.binary_gripper,train_joint_mae_rad=train_mae,
        scope='fixed cabinet control baseline; no runtime teacher/trajectory index/clock')
    (args.output/'model_info.json').write_text(json.dumps(model_info,indent=2)+'\n');print('TRAIN',model_info,flush=True)
    rows={r['episode_id']:r for r in json.loads(MANIFEST.read_text())['episodes']}
    results=[]
    for eid in args.episode_ids:
        row=rows[eid];env=DrawerSimulation(ROOT/row['path']);record=[];judge={};left=env.data.ctrl[env.actuator_ids[8:]].copy()
        while len(record)<round(args.seconds*30):
            q=env.data.qpos[env.qpos_ids].copy();v=env.data.qvel[env.velocity_ids].copy()
            value=(np.r_[q[:8],v[:8]]-xm)/xs
            prediction=(kernel(value[None],x,args.linear_kernel)@w*ys+ym).reshape(4,8);prediction[:,:7]+=q[:7]
            if args.binary_gripper:prediction[:,7]=np.where(prediction[:,7]>-.15625,0.,-.25)
            for predicted in prediction[:min(4,round(args.seconds*30)-len(record))]:
                requested=np.r_[predicted,left];before=env.data.qpos.copy();vb=env.data.qvel.copy();applied=env.step(requested)
                c,f=finger_handle_contacts(env.model,env.data,include_stems=True);opening=-float(env.data.qpos[env.drawer_qpos])
                judge=update_judge(judge,opening,c,f,task='open')
                record.append(dict(qpos_before=before,qpos=env.data.qpos.copy(),qvel_before=vb,qvel=env.data.qvel.copy(),
                    requested=requested,applied=applied,contacts=c,forces=f,opening=opening))
                if judge['success']:break
            if judge['success']:break
        folder=args.output/f'trial_episode_{eid}';folder.mkdir()
        np.savez_compressed(folder/'closed_loop.npz',**{k:np.asarray([r[k] for r in record]) for k in record[0]})
        summary=dict(model_type='kernel_ridge_state_feedback',is_pi0=False,checkpoint=str(args.output/'model.npz'),
            episode_id=eid,split=row['split'],noise_seed=42,noise_mode='deterministic',action_chunk=4,
            state_representation='right_position_velocity8',binary_gripper=args.binary_gripper,teacher_actions_executed=0,intermediate_states_injected=0,
            judge=judge,steps=len(record),peak_open_mm=max(r['opening'] for r in record)*1000)
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');results.append(summary);print('BASELINE_RESULT',summary,flush=True);env.close()
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
