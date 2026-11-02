"""Feedback supervision on independently randomized training scenes, with scene-aware provenance."""
import os
os.environ['MUJOCO_GL']='glfw'
import argparse,dataclasses,hashlib,json
from pathlib import Path
import numpy as np
import mujoco
from drawer_success_training_config import ROOT,MANIFEST,get_success_config
from drawer_success_cache import CachedInputs,flatten
from drawer_velocity_config import VelocityDeltaSuccessDataConfig
from drawer_simulation import DrawerSimulation
from drawer_feedback_oracle import FeedbackOracle,finger_handle_contacts
from live_drawer_success import update_judge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--base',type=Path,default=ROOT/'data/drawer_feedback_queries/v003_feedback_repair')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples-per-scene',type=int,default=64)
    p.add_argument('--seed',type=int,default=920531)
    p.add_argument('--contact-focus-probability',type=float,default=0.)
    p.add_argument('--scene-sampling-weight',type=float,default=1.)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'collector_source.py').write_bytes(Path(__file__).read_bytes())
    doc=json.loads(args.manifest.read_text())
    assert doc['parent_manifest_sha256']==hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert not doc['heldout_states_used']
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;pilots=[];usable=[]
    allowed={r['episode_id'] for r in json.loads(MANIFEST.read_text())['episodes'] if r['split']=='train'}
    def make_oracle(row,path):
        oracle=FeedbackOracle(path)
        end=row.get('reference_end_frame')
        if end:
            oracle.q=oracle.q[:end];oracle.actions=oracle.actions[:end];oracle.scale=np.maximum(oracle.q.std(0),.1)
        return oracle
    def state(env):
        value=np.zeros(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,value,spec);return value
    for row in doc['episodes']:
        assert row['parent_episode_id'] in allowed and row['split']=='train'
        path=ROOT/row['path'];env=DrawerSimulation(path);oracle=make_oracle(row,path);bases=[];judge={}
        try:
            for _ in range(720):
                bases.append((state(env),env.steps))
                action,_=oracle.action(env);env.step(action);c,f=finger_handle_contacts(env.model,env.data,include_stems=True)
                judge=update_judge(judge,-float(env.data.qpos[env.drawer_qpos]),c,f,task='open')
                if judge['success']:break
            report=dict(path=row['path'],judge=judge,steps=len(bases),model_success_evidence=False)
            pilots.append(report)
            if judge['success']:usable.append((row,bases))
            print('AUGMENT_FEEDBACK_PILOT',row['augmentation_index'],judge['success'],len(bases),flush=True)
        finally:env.close()
    (args.output/'expert_pilots.json').write_text(json.dumps(pilots,indent=2)+'\n')
    if not usable:raise RuntimeError('No randomized feedback experts passed')
    from openpi import transforms
    cfg=get_success_config();cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,action_horizon=4))
    data=VelocityDeltaSuccessDataConfig().create(cfg.assets_dirs,cfg.model)
    transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(data.norm_stats,use_quantiles=False),*data.model_transforms.inputs])
    base=CachedInputs(args.base);total=len(base)+len(usable)*args.samples_per_scene;writers=[];fields=[]
    for number,(key,values) in enumerate(base.fields):
        name=f'field_{number:02d}.npy';out=np.lib.format.open_memmap(args.output/name,mode='w+',dtype=values.dtype,shape=(total,*values.shape[1:]))
        out[:len(base)]=values;writers.append((key,out));fields.append(dict(path=key,file=name,dtype=str(values.dtype)))
    keys=['state','velocity','qpos','qvel','expert_actions','initial_state','control_offset']
    with np.load(base.root/'physical_queries.npz') as d:
        physical={k:[d[k]] for k in keys};parents=[d['parent_episode_id']]
    sources=[np.full(len(base),'v003_feedback_repair',dtype='<U160')]
    rng=np.random.default_rng(args.seed);index=len(base);episodes=list(base.metadata['episodes']);rejected=0;max_error=0.
    for row,bases in usable:
        path=ROOT/row['path'];env=DrawerSimulation(path);oracle=make_oracle(row,path);local={k:[] for k in keys}
        try:
            # Identify approach/closure states on this independent TRAIN reference.
            # Never inspect held-out states to choose training query indices.
            contact_bases=[]
            if args.contact_focus_probability:
                for value,offset in bases:
                    mujoco.mj_setState(env.model,env.data,value,spec);mujoco.mj_forward(env.model,env.data)
                    counts,_=finger_handle_contacts(env.model,env.data,include_stems=True)
                    opening=-float(env.data.qpos[env.drawer_qpos])
                    if (counts>0).any() and opening<.04:contact_bases.append((value,offset))
            for sample in range(args.samples_per_scene):
                for attempt in range(1000):
                    if contact_bases and rng.random()<args.contact_focus_probability:
                        s,offset=contact_bases[int(rng.integers(len(contact_bases)))]
                    else:s,offset=bases[0] if rng.random()<.1 else bases[int(rng.integers(len(bases)))]
                    env.model.eq_data[:]=env.initial_eq;mujoco.mj_setState(env.model,env.data,s,spec);mujoco.mj_forward(env.model,env.data);env.steps=offset
                    c,_=finger_handle_contacts(env.model,env.data,include_stems=True);both=bool((c>0).all())
                    amp=0 if sample%4==0 else (.0005 if both else float(rng.choice([.003,.01,.02])))
                    env.data.qpos[env.qpos_ids[:7]]=np.clip(env.data.qpos[env.qpos_ids[:7]]+rng.normal(0,amp,7),env.low[:7],env.high[:7])
                    env.data.qvel[env.velocity_ids[:7]]+=rng.normal(0,.01 if both else .05,7);mujoco.mj_forward(env.model,env.data)
                    c,_=finger_handle_contacts(env.model,env.data,include_stems=True)
                    if min((c.dist for c in env.data.contact),default=0)<-.004 or (both and not (c>0).all()):rejected+=1;continue
                    break
                else:raise RuntimeError('No valid physical branch')
                initial=state(env);obs=env.observation();qpos=env.data.qpos.copy();qvel=env.data.qvel.copy();actions=[];future=[]
                for _ in range(16):
                    action,_=oracle.action(env);actions.append(env.step(action));future.append(env.data.qpos.copy())
                actions=np.asarray(actions)
                env.model.eq_data[:]=env.initial_eq;mujoco.mj_setState(env.model,env.data,initial,spec);mujoco.mj_forward(env.model,env.data);env.steps=offset
                for action,expected in zip(actions,future):env.step(action);max_error=max(max_error,float(np.abs(env.data.qpos-expected).max()))
                if max_error>1e-9:raise RuntimeError('Branch replay mismatch')
                for k,v in [('state',obs['observation.state'].copy()),('velocity',obs['observation.velocity'].copy()),
                            ('qpos',qpos),('qvel',qvel),('expert_actions',actions),('initial_state',initial),('control_offset',offset)]:local[k].append(v)
                obs['action']=actions[:4].astype(np.float32);value=dict(flatten(transform(obs)))
                for key,out in writers:out[index]=value[key]
                index+=1
            for k in keys:physical[k].append(np.asarray(local[k]))
            parents.append(np.full(args.samples_per_scene,row['parent_episode_id']));sources.append(np.full(args.samples_per_scene,row['path'],dtype='<U160'))
            episodes.append(dict(episode_id=f'augmented_train{row["parent_episode_id"]}_{row["augmentation_index"]}',frames=args.samples_per_scene,split='train',
                                 parent_episode_id=row['parent_episode_id'],scene_source=row['path'],scene_overrides=row['scene_overrides'],
                                 sampling_weight=args.scene_sampling_weight))
            print('AUGMENT_QUERIES',row['augmentation_index'],args.samples_per_scene,'total',index,flush=True)
        finally:env.close()
    for _,out in writers:out.flush()
    np.savez_compressed(args.output/'physical_queries.npz',**{k:np.concatenate(v) for k,v in physical.items()},parent_episode_id=np.concatenate(parents),scene_source=np.concatenate(sources))
    meta=dict(base.metadata);meta.update(frames=total,fields=fields,episodes=episodes,source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        augmentation_manifest=str(args.manifest.resolve()),augmentation_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        independently_randomized_training_scenes=sum(row['augmentation_index']>=0 for row,_ in usable),
        verified_training_reference_scenes=len(usable),scope='Training-parent feedback plus independent continuous train12 scene augmentations; no held-out starts or labels',
        actual_gradient_parent_episode_ids=sorted(set(np.concatenate(parents).tolist())),
        query_seed=args.seed,contact_focus_probability=args.contact_focus_probability,
        replay_max_qpos_error=max_error,rejected_augmented_queries=rejected,model_success_evidence=False)
    (args.output/'READY.json').write_text(json.dumps(meta,indent=2)+'\n');print('AUGMENT_DATA_READY',total,flush=True)


if __name__=='__main__':main()
