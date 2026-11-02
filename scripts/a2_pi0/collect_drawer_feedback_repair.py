"""Train-only feedback branches with state-conditioned labels, matching the original successful oracle protocol."""
import os
os.environ['MUJOCO_GL']='glfw'
from pathlib import Path
import argparse,dataclasses,hashlib,json
import mujoco
import numpy as np
from drawer_simulation import DrawerSimulation
from drawer_feedback_oracle import FeedbackOracle,finger_handle_contacts
from drawer_success_training_config import ROOT,MANIFEST,get_success_config
from drawer_velocity_config import VelocityDeltaSuccessDataConfig,VELOCITY_NORM_ROOT
from drawer_success_cache import CachedInputs,flatten


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples-per-parent',type=int,default=400)
    args=p.parse_args()
    rows={r['episode_id']:r for r in json.loads(MANIFEST.read_text())['episodes']}
    parent_ids=[9,10,11]
    assert all(rows[i]['split']=='train' for i in parent_ids)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'collector_source.py').write_bytes(Path(__file__).read_bytes())
    from openpi import transforms
    cfg=get_success_config();cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,action_horizon=4))
    data=VelocityDeltaSuccessDataConfig().create(cfg.assets_dirs,cfg.model)
    transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(data.norm_stats,use_quantiles=False),*data.model_transforms.inputs])
    base=CachedInputs(ROOT/'data/drawer_feedback_queries/v001');total=len(base)+len(parent_ids)*args.samples_per_parent
    writers=[];fields=[]
    for number,(key,values) in enumerate(base.fields):
        name=f'field_{number:02d}.npy';out=np.lib.format.open_memmap(args.output/name,mode='w+',dtype=values.dtype,shape=(total,*values.shape[1:]))
        out[:len(base)]=values;writers.append((key,out));fields.append(dict(path=key,file=name,dtype=str(values.dtype)))
    keys=['state','velocity','qpos','qvel','expert_actions','initial_state','control_offset']
    with np.load(base.root/'physical_queries.npz') as d:physical={k:[d[k]] for k in keys}
    parents=[np.full(len(base),12)];rng=np.random.default_rng(20261102);episodes=list(base.metadata['episodes'])
    rejected=0;max_error=0.;index=len(base);spec=mujoco.mjtState.mjSTATE_INTEGRATION
    for eid in parent_ids:
        env=DrawerSimulation(ROOT/rows[eid]['path'])
        oracle=FeedbackOracle(ROOT/f'data/drawer_recovery/targeted_20261101/episode{eid:03d}_step0030')
        def state():
            s=np.zeros(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,s,spec);return s
        bases=[]
        # Include expert visited states and both real failure policies, preserving each exact full state.
        for kind,run in [('expert',None),('student16','drawer_eval_isolated_full_20261101'),('student17','drawer_eval_017_full_20261101')]:
            env.reset()
            if run:
                trial,=(ROOT/'logs/a2_pi0'/run).glob(f'trial_*_episode_{eid:03d}')
                with np.load(trial/'closed_loop.npz') as d:actions=d['applied'][:180]
            else:actions=[None]*180
            for frame,action in enumerate(actions):
                bases.append((kind,frame,state(),env.steps))
                if action is None:action,_=oracle.action(env)
                env.step(action)
        local={k:[] for k in keys}
        for sample in range(args.samples_per_parent):
            for attempt in range(1000):
                if rng.random()<.1:kind,frame,s,offset=bases[0]
                else:kind,frame,s,offset=bases[int(rng.integers(len(bases)))]
                env.model.eq_data[:]=env.initial_eq;mujoco.mj_setState(env.model,env.data,s,spec);mujoco.mj_forward(env.model,env.data);env.steps=offset
                c,_=finger_handle_contacts(env.model,env.data,include_stems=True);both=bool((c>0).all())
                amp=0 if sample%4==0 else (.0005 if both else float(rng.choice([.003,.01,.02])))
                env.data.qpos[env.qpos_ids[:7]]=np.clip(env.data.qpos[env.qpos_ids[:7]]+rng.normal(0,amp,7),env.low[:7],env.high[:7])
                env.data.qvel[env.velocity_ids[:7]]+=rng.normal(0,.01 if both else .05,7)
                mujoco.mj_forward(env.model,env.data)
                after,_=finger_handle_contacts(env.model,env.data,include_stems=True)
                if min((c.dist for c in env.data.contact),default=0)<-.004 or (both and not (after>0).all()):
                    rejected+=1;continue
                break
            else:raise RuntimeError('No physical query after 1000 attempts')
            initial=state();obs=env.observation();qpos=env.data.qpos.copy();qvel=env.data.qvel.copy();future=[];actions=[]
            for _ in range(16):
                action,_=oracle.action(env);actions.append(env.step(action));future.append(env.data.qpos.copy())
            actions=np.asarray(actions)
            env.model.eq_data[:]=env.initial_eq;mujoco.mj_setState(env.model,env.data,initial,spec);mujoco.mj_forward(env.model,env.data);env.steps=offset
            for action,expected in zip(actions,future):
                env.step(action);max_error=max(max_error,float(np.abs(env.data.qpos-expected).max()))
            if max_error>1e-9:raise RuntimeError('Branch replay mismatch')
            local['state'].append(obs['observation.state'].copy());local['velocity'].append(obs['observation.velocity'].copy())
            for k,v in [('qpos',qpos),('qvel',qvel),('expert_actions',actions),('initial_state',initial),('control_offset',offset)]:local[k].append(v)
            obs['action']=actions[:4].astype(np.float32);value=dict(flatten(transform(obs)))
            for key,out in writers:out[index]=value[key]
            index+=1
            if (sample+1)%50==0:print('REPAIR_QUERIES',eid,sample+1,'rejected',rejected,'replay',max_error,flush=True)
        for k in keys:physical[k].append(np.asarray(local[k]))
        parents.append(np.full(args.samples_per_parent,eid))
        episodes.append(dict(episode_id=f'feedback_parent_{eid}',parent_episode_id=eid,frames=args.samples_per_parent,split='train'))
        env.close()
    for _,out in writers:out.flush()
    np.savez_compressed(args.output/'physical_queries.npz',**{k:np.concatenate(v) for k,v in physical.items()},parent_episode_id=np.concatenate(parents))
    meta=dict(base.metadata);meta.update(frames=total,fields=fields,episodes=episodes,actual_gradient_parent_episode_ids=[9,10,11,12],
        scope='Train-only state feedback expert branches; no time-indexed rollout label supervision; not model success evidence',
        replay_max_qpos_error=max_error,rejected_queries=rejected,model_success_evidence=False)
    (args.output/'READY.json').write_text(json.dumps(meta,indent=2)+'\n');print('REPAIR_READY',total,flush=True)


if __name__=='__main__':main()
