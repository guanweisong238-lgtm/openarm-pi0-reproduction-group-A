"""物理分支监督查询：当前真实观测+从同一完整状态执行反馈专家的未来动作。"""
import os
os.environ['MUJOCO_GL']='egl'
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES','/usr/share/glvnd/egl_vendor.d/50_mesa.json')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
from pathlib import Path
import argparse,dataclasses,hashlib,json
import numpy as np
import mujoco
from drawer_success_training_config import ROOT,MANIFEST,get_success_config
from drawer_simulation import DrawerSimulation
from drawer_feedback_oracle import FeedbackOracle,finger_handle_contacts
from drawer_velocity_config import VelocityDeltaSuccessDataConfig,VELOCITY_NORM_ROOT
from drawer_success_cache import flatten


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=1200)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output/'collector_source.py').write_bytes(Path(__file__).read_bytes())
    env=DrawerSimulation();oracle=FeedbackOracle();spec=mujoco.mjtState.mjSTATE_INTEGRATION
    def getstate():
        state=np.zeros(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,state,spec);return state
    bases=[]
    trajectories=[('expert',ROOT/'results/drawer_feedback_oracle_v002/expert_rollout.npz','action'),
        ('student15',ROOT/'logs/a2_pi0/success_live_20261025-024400-924028/trial_01_episode_12/closed_loop.npz','applied'),
        ('kernel1',ROOT/'results/drawer_krr_baseline_v001/trial_episode_12/closed_loop.npz','applied')]
    for kind,path,key in trajectories:
        env.reset()
        with np.load(path) as source:
            for i,action in enumerate(source[key]):
                if kind=='expert' or i<120:
                    bases.append((kind,i,getstate(),env.steps))
                env.step(action)
                if kind!='expert' and i>=120:break
    from openpi import transforms
    cfg=get_success_config();cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,action_horizon=4))
    data=VelocityDeltaSuccessDataConfig().create(cfg.assets_dirs,cfg.model)
    transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(data.norm_stats,use_quantiles=False),*data.model_transforms.inputs])
    rng=np.random.default_rng(26025);writers=[];fields=[];physical=[];rejected=0;error=0.
    original_eq=env.initial_eq.copy()
    for sample in range(args.samples):
        attempts=0
        while True:
            attempts+=1
            if attempts>1000:raise RuntimeError('无法找到物理有效查询，请检查接触过滤')
            kind,source_frame,initial,offset=bases[0] if rng.random()<.1 else bases[int(rng.integers(len(bases)))]
            env.model.eq_data[:]=original_eq;mujoco.mj_setState(env.model,env.data,initial,spec);mujoco.mj_forward(env.model,env.data);env.steps=offset
            counts,_=finger_handle_contacts(env.model,env.data,include_stems=True);contact=bool(np.all(counts>0))
            _,progress=oracle.action(env)
            amplitude=0 if sample%4==0 else (.0005 if contact or progress>=110 else float(rng.choice([.003,.01,.02])))
            env.data.qpos[env.qpos_ids[:7]]+=rng.normal(0,amplitude,7)
            env.data.qpos[env.qpos_ids[:7]]=np.clip(env.data.qpos[env.qpos_ids[:7]],env.low[:7],env.high[:7])
            env.data.qvel[env.velocity_ids[:7]]+=rng.normal(0,.01 if contact else .05,7)
            mujoco.mj_forward(env.model,env.data)
            after_counts,_=finger_handle_contacts(env.model,env.data,include_stems=True)
            penetration=min((c.dist for c in env.data.contact),default=0.)
            if penetration<-.004 or (contact and not np.all(after_counts>0)):
                rejected+=1;continue
            break
        state=getstate();obs=env.observation();qpos=env.data.qpos.copy();qvel=env.data.qvel.copy()
        actions=[];future=[]
        for _ in range(16):
            action,_=oracle.action(env);actions.append(env.step(action));future.append(env.data.qpos.copy())
        actions=np.asarray(actions);future=np.asarray(future)
        env.model.eq_data[:]=original_eq;mujoco.mj_setState(env.model,env.data,state,spec);mujoco.mj_forward(env.model,env.data);env.steps=offset
        for a,expected in zip(actions,future):env.step(a);error=max(error,float(np.max(np.abs(env.data.qpos-expected))))
        if error>1e-9:raise RuntimeError('查询分支动作回放不一致')
        obs['action']=actions[:4].astype(np.float32);value=transform(obs)
        leaves=list(flatten(value))
        if not writers:
            for number,(key,array) in enumerate(leaves):
                dtype=np.float32 if array.dtype.kind=='f' else array.dtype
                name=f'field_{number:02d}.npy';writer=np.lib.format.open_memmap(args.output/name,mode='w+',dtype=dtype,shape=(args.samples,*array.shape))
                writers.append((key,writer));fields.append(dict(path=key,file=name,dtype=str(np.dtype(dtype))))
        value=dict(leaves)
        for key,writer in writers:writer[sample]=value[key]
        physical.append(dict(state=obs['observation.state'],velocity=obs['observation.velocity'],qpos=qpos,qvel=qvel,
            initial_state=state,control_offset=offset,expert_actions=actions,after_qpos=future,source_kind=kind,source_frame=source_frame))
        if (sample+1)%100==0:print('FEEDBACK_QUERIES',sample+1,'rejected',rejected,'replay_error',error,flush=True)
    for _,writer in writers:writer.flush()
    np.savez_compressed(args.output/'physical_queries.npz',**{k:np.asarray([r[k] for r in physical]) for k in physical[0]},initial_eq_data=original_eq)
    metadata=dict(schema='model_ready_feedback_queries_v001',frames=args.samples,fields=fields,
        episodes=[dict(episode_id='feedback_queries',frames=args.samples,split='train',parent_episode_id=12)],
        action_horizon=4,state_representation='position_velocity16',action_representation='right7_delta',
        source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        norm_stats_sha256=hashlib.sha256((VELOCITY_NORM_ROOT/'norm_stats.json').read_bytes()).hexdigest(),
        oracle_source_sha256=hashlib.sha256((ROOT/'scripts/a2_pi0/drawer_feedback_oracle.py').read_bytes()).hexdigest(),
        replay_max_qpos_error=error,rejected_queries=rejected,model_success_evidence=False,
        scope='expert-labeled physical branches around successful teacher and real failed student states; parent train12 only')
    (args.output/'READY.json').write_text(json.dumps(metadata,indent=2)+'\n');env.close()


if __name__=='__main__':main()
