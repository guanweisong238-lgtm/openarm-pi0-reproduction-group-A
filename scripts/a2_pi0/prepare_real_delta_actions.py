#!/usr/bin/env python3
"""Materialize aligned delta labels and train-only normalization; no robot commands."""
import hashlib,json
import numpy as np
from real_four_camera_loader import RealFourCameraDataset
from real_delta_actions import ROOT,to_deltas

out=ROOT/'data/processed/real_delta_actions_v001'
if out.exists(): raise FileExistsError(out)
out.mkdir(parents=True)
total=np.zeros(16,np.float64);square=total.copy();count=0;rows=[]
for split in ('train','validation'):
 ds=RealFourCameraDataset(split)
 for entry in ds.entries:
  with np.load(ROOT/entry['samples_path'],allow_pickle=False) as z:
   n=entry['full_window_samples']; a=z['actions'][:n-1]; state=z['state'][1:n]; delta=to_deltas(a,state)
   if not np.isfinite(delta).all():raise ValueError('nonfinite action')
   error=float(np.max(np.abs(np.cumsum(delta,axis=1)+state[:,None,:]-a)))
   if error>2e-5:raise ValueError('delta reconstruction failed')
   path=out/f"episode_{entry['episode_id']}.npz";np.savez_compressed(path,delta_actions=delta)
   if split=='train':
    x=delta.astype(np.float64);total+=x.sum(axis=(0,1));square+=(x*x).sum(axis=(0,1));count+=x.shape[0]*x.shape[1]
   rows.append({'episode_id':entry['episode_id'],'split':split,'observation_start_index':1,'windows':len(delta),'reconstruction_max_abs_rad':error,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
mean=total/count;std=np.sqrt(np.maximum(square/count-mean*mean,0))
if not np.isfinite(std).all() or (std<1e-6).any():raise ValueError('bad stats')
np.savez(out/'delta_stats.npz',mean=mean.astype(np.float32),std=std.astype(np.float32))
(out/'manifest.json').write_text(json.dumps({'description':'at observation i, first delta is absolute action i-1 minus observed state i; remaining 49 deltas are successive absolute target differences; indices are episode local','stats':'train split only, all 50 deltas of eligible windows','count':count,'entries':rows},ensure_ascii=False,indent=2)+'\n')
print('prepared',len(rows),'episodes; training delta vectors',count,'maximum reconstruction error',max(x['reconstruction_max_abs_rad'] for x in rows))
