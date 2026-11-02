"""Append verified train-only recovery queries to original endpoint supervision, without changing statistics."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from drawer_success_training_config import ROOT, MANIFEST, get_success_config
from drawer_success_cache import CachedInputs, flatten
from drawer_velocity_config import VelocityDeltaSuccessDataConfig, VELOCITY_NORM_ROOT


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--base', type=Path, default=ROOT/'data/drawer_feedback_queries/v001')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    doc = json.loads(args.manifest.read_text())
    if doc['parent_manifest_sha256'] != digest(MANIFEST):
        raise ValueError('Parent manifest mismatch')
    train = {r['episode_id'] for r in json.loads(MANIFEST.read_text())['episodes'] if r['split']=='train'}
    base = CachedInputs(args.base)
    if base.metadata['norm_stats_sha256'] != digest(VELOCITY_NORM_ROOT/'norm_stats.json'):
        raise ValueError('Original statistics mismatch')
    plans = []
    for row in doc['episodes']:
        path = ROOT/row['path']
        r = json.loads((path/'report.json').read_text())
        if (row['split']!='train' or row['parent_episode_id'] not in train or not r['train_ready']
            or not r['judge']['success'] or not r['replay_verified'] or r['gl_backend']!='glfw'
            or digest(path/'episode.npz')!=row['episode_sha256']):
            raise ValueError(f'Unverified recovery {path}')
        with np.load(path/'episode.npz') as d:
            frames = d['image_frame_indices'] if 'image_frame_indices' in d else np.arange(len(d['state']))
            # Retain full 16-step physical labels, model learns first four.
            eligible = [(j,int(i)) for j,i in enumerate(frames) if i%4==0 and i+16<=len(d['state'])]
        plans.append((row, r, eligible))
    added = sum(len(x[2]) for x in plans)
    if added == 0:
        raise ValueError('No accepted recovery queries')
    args.output.mkdir(parents=True, exist_ok=False)
    from openpi import transforms
    cfg = get_success_config()
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, action_horizon=4))
    data = VelocityDeltaSuccessDataConfig().create(cfg.assets_dirs,cfg.model)
    sm=np.asarray(data.norm_stats['state'].mean)
    ss=np.asarray(data.norm_stats['state'].std)+1e-6
    am=np.asarray(data.norm_stats['actions'].mean)
    ast=np.asarray(data.norm_stats['actions'].std)+1e-6
    transform = transforms.compose([*data.data_transforms.inputs,
        transforms.Normalize(data.norm_stats,use_quantiles=False),*data.model_transforms.inputs])
    writers, fields = [], []
    total = len(base)+added
    for num,(key,values) in enumerate(base.fields):
        name=f'field_{num:02d}.npy'
        out=np.lib.format.open_memmap(args.output/name,mode='w+',dtype=values.dtype,shape=(total,*values.shape[1:]))
        out[:len(base)]=values
        writers.append((key,out))
        fields.append(dict(path=key,file=name,dtype=str(values.dtype)))
    with np.load(args.base/'physical_queries.npz') as d:
        physical={k:[d[k]] for k in ('state','velocity','qpos','qvel','expert_actions')}
    parents=[np.full(len(base),12,dtype=np.int64)]
    origins=[np.full(len(base),'original_v001',dtype='<U120')]
    source_frames=[np.arange(len(base),dtype=np.int64)]
    index=len(base)
    episodes=list(base.metadata['episodes'])
    for row,r,eligible in plans:
        path=ROOT/row['path']
        local={k:[] for k in physical}
        from drawer_simulation import DrawerSimulation
        env=DrawerSimulation(path)
        with np.load(path/'episode.npz') as archive:
            d={k:archive[k] for k in archive.files}
            for image_index,frame in eligible:
                velocity=d['qvel'][frame]
                # Joint velocity indices are taken from the actual scene, never qpos indices.
                obs={'observation.state':d['state'][frame].astype(np.float32),
                     'observation.velocity':velocity[env.velocity_ids].astype(np.float32),
                     'prompt':r['prompt'],'action':d['action'][frame:frame+4].astype(np.float32)}
                for name in r['camera_mapping']:
                    obs['observation.images.'+name]=d['images_'+name][image_index]
                absolute_actions=obs['action'].copy()
                transformed=dict(flatten(transform(obs)))
                state=transformed[('state',)]
                actions=transformed[('actions',)]
                if state.shape!=(32,) or actions.shape!=(4,32) or state[8:16].any() or state[24:32].any():
                    raise ValueError('Endpoint representation mismatch')
                decoded=actions[:,:16]*ast+am
                decoded[:,:7]+=state[:7]*ss[:7]+sm[:7]
                np.testing.assert_allclose(decoded,absolute_actions,atol=3e-6,rtol=1e-6)
                np.testing.assert_allclose(state[16:24]*ss[16:24]+sm[16:24],obs['observation.velocity'][:8],atol=3e-6,rtol=1e-6)
                if set(transformed)!=set(k for k,_ in writers):
                    raise ValueError('Transform field mismatch')
                for key,out in writers:
                    value=transformed[key]
                    if not np.isfinite(value).all():raise ValueError('Nonfinite transformed data')
                    out[index]=value
                local['state'].append(obs['observation.state'])
                local['velocity'].append(obs['observation.velocity'])
                local['qpos'].append(d['qpos'][frame])
                local['qvel'].append(d['qvel'][frame])
                local['expert_actions'].append(d['action'][frame:frame+16])
                index+=1
        env.close()
        del env
        for k in physical:physical[k].append(np.asarray(local[k]))
        parents.append(np.full(len(eligible),row['parent_episode_id'],dtype=np.int64))
        origins.append(np.full(len(eligible),row['path'],dtype='<U120'))
        source_frames.append(np.array([i for _,i in eligible],dtype=np.int64))
        episodes.append(dict(episode_id=row['episode_id'],frames=len(eligible),split='train',
                             parent_episode_id=row['parent_episode_id']))
        print('PREPARED',row['episode_id'],len(eligible),flush=True)
    for _,out in writers:out.flush()
    if index!=total:raise RuntimeError('Query count mismatch')
    np.savez_compressed(args.output/'physical_queries.npz',
                        **{k:np.concatenate(v) for k,v in physical.items()},
                        parent_episode_id=np.concatenate(parents),source_path=np.concatenate(origins),
                        source_frame=np.concatenate(source_frames))
    meta=dict(base.metadata)
    meta.update(frames=total,fields=fields,episodes=episodes,source_manifest_sha256=digest(MANIFEST),
                base_data=str(args.base.resolve()),base_metadata_sha256=digest(args.base/'READY.json'),
                recovery_manifest=str(args.manifest.resolve()),recovery_manifest_sha256=digest(args.manifest),
                added_recovery_queries=added,actual_gradient_parent_episode_ids=sorted({12,*[r['parent_episode_id'] for r,_,_ in plans]}),
                scope='Original train12 queries plus train9/10/11 expert corrections. Correlated queries, not independent success trajectories.',
                physical_provenance='New query full initial state reconstructible from recovery source episode initial state and source_frame actions; original query provenance remains in base_data.',
                model_success_evidence=False)
    (args.output/'preparer_source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'READY.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('QUERY_DATA_READY',total,'new',added,flush=True)


if __name__=='__main__':
    main()
