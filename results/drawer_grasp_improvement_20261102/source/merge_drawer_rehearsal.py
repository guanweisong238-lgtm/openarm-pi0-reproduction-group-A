"""Combine expert feedback and explicitly labeled prior-policy rehearsal, preserving original artifacts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from drawer_success_cache import CachedInputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--rehearsal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();a=CachedInputs(args.base);b=CachedInputs(args.rehearsal)
    for k in ['schema','action_horizon','norm_stats_sha256','source_manifest_sha256']:
        if a.metadata[k]!=b.metadata[k]:raise ValueError(f'Mismatch {k}')
    args.output.mkdir(parents=True,exist_ok=False);bf=dict(b.fields);fields=[]
    if {k for k,_ in a.fields}!=set(bf):raise ValueError('Field mismatch')
    for number,(key,value) in enumerate(a.fields):
        other=bf[key]
        if value.shape[1:]!=other.shape[1:]:raise ValueError('Field shape mismatch')
        name=f'field_{number:02d}.npy';out=np.lib.format.open_memmap(args.output/name,mode='w+',dtype=value.dtype,shape=(len(a)+len(b),*value.shape[1:]))
        out[:len(a)]=value;out[len(a):]=other;out.flush();fields.append(dict(path=key,file=name,dtype=str(value.dtype)))
    keys=['state','velocity','qpos','qvel','initial_state','control_offset','parent_episode_id']
    with np.load(args.base/'physical_queries.npz') as x,np.load(args.rehearsal/'physical_queries.npz') as y:
        physical={k:np.concatenate([x[k],y[k]]) for k in keys}
        physical['target_actions']=np.concatenate([x[a.metadata.get('physical_target_field','expert_actions')][:,:4],y[b.metadata['physical_target_field']][:,:4]])
        physical['label_source']=np.r_[np.full(len(a),'expert_feedback'),np.full(len(b),'recorded_policy016')]
    np.savez_compressed(args.output/'physical_queries.npz',**physical)
    meta=dict(a.metadata);meta.update(frames=len(a)+len(b),fields=fields,episodes=a.rows+b.rows,physical_target_field='target_actions',
        base_dataset=str(args.base.resolve()),base_metadata_sha256=hashlib.sha256((args.base/'READY.json').read_bytes()).hexdigest(),
        rehearsal_dataset=str(args.rehearsal.resolve()),rehearsal_metadata_sha256=hashlib.sha256((args.rehearsal/'READY.json').read_bytes()).hexdigest(),
        scope='Train-only expert scene feedback plus recorded successful train behavior of policy016; mixed target_actions provenance is explicit',
        model_success_evidence=False)
    (args.output/'READY.json').write_text(json.dumps(meta,indent=2)+'\n');(args.output/'merger_source.py').write_bytes(Path(__file__).read_bytes())
    print('REHEARSAL_MERGE_READY',len(a)+len(b),'expert',len(a),'rehearsal',len(b),flush=True)


if __name__=='__main__':main()
