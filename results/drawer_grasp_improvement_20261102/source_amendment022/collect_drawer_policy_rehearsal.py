"""Replay successful training rollouts and preserve their recorded policy predictions as rehearsal labels."""
import os
os.environ['MUJOCO_GL']='glfw'
import argparse,dataclasses,hashlib,json
from pathlib import Path
import numpy as np
import mujoco
from drawer_success_training_config import ROOT,MANIFEST,get_success_config
from drawer_success_cache import flatten
from drawer_velocity_config import VelocityDeltaSuccessDataConfig,VELOCITY_NORM_ROOT
from drawer_simulation import DrawerSimulation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy-label',default='recorded_policy016')
    p.add_argument('--sampling-weight',type=float,default=4.)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest={r['episode_id']:r for r in json.loads(MANIFEST.read_text())['episodes']}
    rows=[r for r in json.loads((args.run/'index.json').read_text()) if r['split']=='train' and r['success'] and r['evidence_valid']]
    source_config=json.loads((args.run/'run_config.json').read_text())
    assert rows
    total=sum(r['inference_count'] for r in rows)
    from openpi import transforms
    cfg=get_success_config();cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,action_horizon=4))
    data=VelocityDeltaSuccessDataConfig().create(cfg.assets_dirs,cfg.model)
    transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(data.norm_stats,use_quantiles=False),*data.model_transforms.inputs])
    writers=[];fields=[];physical=[];episodes=[];index=0;error=0.;spec=mujoco.mjtState.mjSTATE_INTEGRATION
    for row in rows:
        eid=row['episode_id'];assert manifest[eid]['split']=='train'
        trial=args.run/row['trial'];env=DrawerSimulation(ROOT/manifest[eid]['path'])
        with np.load(trial/'closed_loop.npz') as d:closed={k:d[k] for k in d.files}
        try:
            for call,block in enumerate(closed['action_blocks']):
                offset=call*4;obs=env.observation();initial=np.zeros(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,initial,spec)
                error=max(error,float(np.abs(env.data.qpos-closed['qpos_before'][offset]).max()),float(np.abs(env.data.qvel-closed['qvel_before'][offset]).max()))
                physical.append(dict(state=obs['observation.state'].copy(),velocity=obs['observation.velocity'].copy(),
                    qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),initial_state=initial,control_offset=env.steps,
                    target_actions=block.copy(),parent_episode_id=eid))
                obs['action']=block.astype(np.float32);values=list(flatten(transform(obs)))
                if not writers:
                    for number,(key,array) in enumerate(values):
                        name=f'field_{number:02d}.npy';out=np.lib.format.open_memmap(args.output/name,mode='w+',dtype=array.dtype,shape=(total,*array.shape))
                        writers.append((key,out));fields.append(dict(path=key,file=name,dtype=str(array.dtype)))
                values=dict(values)
                for key,out in writers:out[index]=values[key]
                index+=1
                for action in closed['applied'][offset:offset+4]:env.step(action)
            if error>1e-9:raise RuntimeError('Recorded policy rollout replay mismatch')
            episodes.append(dict(episode_id=f'rehearsal_{args.policy_label}_{eid}',frames=len(closed['action_blocks']),split='train',parent_episode_id=eid,
                sampling_weight=args.sampling_weight,label_source=args.policy_label+' prediction, not expert'))
            print('REHEARSAL_READY_PARENT',eid,len(closed['action_blocks']),error,flush=True)
        finally:env.close()
    for _,out in writers:out.flush()
    assert index==total
    np.savez_compressed(args.output/'physical_queries.npz',**{k:np.asarray([r[k] for r in physical]) for k in physical[0]})
    meta=dict(schema='model_ready_feedback_queries_v001',frames=total,episodes=episodes,fields=fields,
        action_horizon=4,state_representation='position_velocity16',action_representation='right7_delta',physical_target_field='target_actions',
        source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),norm_stats_sha256=hashlib.sha256((VELOCITY_NORM_ROOT/'norm_stats.json').read_bytes()).hexdigest(),
        source_run=str(args.run.resolve()),source_run_config_sha256=hashlib.sha256((args.run/'run_config.json').read_bytes()).hexdigest(),
        policy_label=args.policy_label,source_checkpoint=source_config['checkpoint'],
        replay_max_qpos_qvel_error=error,scope='Rehearsal of prior model behavior on successful train starts only; no held-out data',model_success_evidence=False)
    (args.output/'READY.json').write_text(json.dumps(meta,indent=2)+'\n');(args.output/'collector_source.py').write_bytes(Path(__file__).read_bytes())
    print('REHEARSAL_DATA_READY',total,flush=True)


if __name__=='__main__':main()
