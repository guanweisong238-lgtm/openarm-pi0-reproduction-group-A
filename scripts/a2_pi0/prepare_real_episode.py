#!/usr/bin/env python3
"""Build an auditable episode-local 30 Hz alignment prototype; raw files stay untouched."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/real_data_20261121_193807'
CAMERAS = ('global', 'chest', 'wrist_left', 'wrist_right')
ARMS = ('right','left')
PROMPT = '用左手拉开抽屉，右手抓起方块并放入抽屉、释放，然后用左手推合抽屉。'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def previous(ts, target):
    idx = np.searchsorted(ts, target, side='right') - 1
    if (idx < 0).any(): raise ValueError('target precedes source; would require look-ahead')
    return idx

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=int,default=0)
    parser.add_argument('--dataset',type=Path,default=RAW)
    parser.add_argument('--output',type=Path,help='New output directory; defaults to episode-specific prototype directory')
    parser.add_argument('--prompt', default=PROMPT)
    args=parser.parse_args(); RAW_PATH=args.dataset.resolve(); ep=RAW_PATH/'episodes'/str(args.episode); out=args.output or ROOT/'data/processed/real_alignment_prototype_v001'/f'episode_{args.episode}'
    if out.exists(): raise SystemExit(f'Output already exists: {out}; choose a new path to avoid overwriting.')
    cams={}
    for cam in CAMERAS:
        paths=sorted((ep/'cameras'/cam).glob('*.jpeg'))
        if not paths: raise ValueError(f'missing camera {cam}')
        ts=np.array([int(p.stem) for p in paths],dtype=np.int64)
        if np.any(np.diff(ts)<=0): raise ValueError(f'non-monotonic camera {cam}')
        cams[cam]=(ts,paths)
    tables={}; source_hashes={}
    for typ in ('obs','action'):
        for arm in ARMS:
            p=ep/typ/'arms'/arm/'state.parquet'; t=pq.read_table(p,columns=['timestamp','qpos']).to_pandas()
            ts=t.timestamp.to_numpy().astype('datetime64[ns]').astype(np.int64); q=np.stack(t.qpos).astype(np.float32)
            if q.shape!=(len(ts),8) or not np.isfinite(q).all() or (np.diff(ts)<=0).any(): raise ValueError(f'bad data {p}')
            tables[(typ,arm)]=(ts,q); source_hashes[str(p.relative_to(ROOT))]=sha(p)
    start=max([int(t[0][0]) for t in cams.values()]+[int(t[0][0]) for t in tables.values()]); end=min([int(t[0][-1]) for t in cams.values()]+[int(t[0][-1]) for t in tables.values()]);
    # A global-camera timestamp anchors a regular 30 Hz integer grid.
    ref=cams['global'][0]; anchor=int(ref[np.searchsorted(ref,start)]); count=((end-anchor)*30)//1_000_000_000+1
    if count<=0: raise ValueError('no common time range')
    times=anchor+(np.arange(count,dtype=np.int64)*1_000_000_000)//30
    result={'timestamp_ns':times}; source_ages={}; patharrays=[]
    for cam,(ts,paths) in cams.items():
        ids=previous(ts,times); matched=ts[ids]; ages=(times-matched)/1e6
        result[f'image_timestamp_ns_{cam}']=matched; result[f'image_index_{cam}']=ids
        result[f'image_path_{cam}']=np.array([str(paths[i].relative_to(ROOT)) for i in ids]);patharrays.append(ids)
        source_ages[cam]={'median_ms':float(np.median(ages)),'p95_ms':float(np.percentile(ages,95)),'max_ms':float(ages.max()),'over_50ms':int((ages>50).sum())}
    states=[]
    for arm in ARMS:
        ts,q=tables[('obs',arm)]; ids=previous(ts,times); states.append(q[ids]);result[f'obs_index_{arm}']=ids; result[f'obs_timestamp_ns_{arm}']=ts[ids]
        age=(times-ts[ids])/1e6;source_ages['obs_'+arm]={'median_ms':float(np.median(age)),'p95_ms':float(np.percentile(age,95)),'max_ms':float(age.max())}
    result['state']=np.concatenate(states,axis=1)
    # Each query uses future points on the same regular grid. Windows never cross episodes.
    index=np.arange(count)[:,None]+np.arange(50)[None,:]; valid=index<count; clipped=np.minimum(index,count-1); target=times[clipped]
    chunks=[]
    for arm in ARMS:
        ts,q=tables[('action',arm)]; ids=previous(ts,target);chunks.append(q[ids]); result[f'action_index_{arm}']=ids;result[f'action_timestamp_ns_{arm}']=ts[ids]
        age=(target-ts[ids])/1e6;source_ages['action_'+arm]={'median_ms':float(np.median(age[valid])),'p95_ms':float(np.percentile(age[valid],95)),'max_ms':float(age[valid].max())}
    result['actions']=np.concatenate(chunks,axis=2);result['action_valid']=valid
    out.mkdir(parents=True,exist_ok=False); np.savez_compressed(out/'samples.npz',**result)
    timing=RAW_PATH/'camera_timing'/f'episode_{args.episode}.jsonl';meta=RAW_PATH/'metadata.yaml';source_hashes[str(timing.relative_to(ROOT))]=sha(timing);source_hashes[str(meta.relative_to(ROOT))]=sha(meta)
    report={'episode_id':str(args.episode),'sample_count':int(count),'fps':30,'horizon':50,'prompt':args.prompt,'anchor':'first global frame at or after all streams start','selection':'latest source timestamp <= grid time (causal zero-order hold); images are references to raw jpeg; no exposure sync claim','camera_order':CAMERAS,'vector_order':'right J1..J7, right gripper, left J1..J7, left gripper','tail':'repeat final in-episode action; action_valid=false for padding; do not train padded steps unless loss applies mask','full_windows':int(valid.all(axis=1).sum()),'padded_windows':int((~valid.all(axis=1)).sum()),'source_age_ms':source_ages,'source_sha256':source_hashes,'note':'prototype, no rejection threshold, not an approved training dataset; check camera freshness and label before training'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
