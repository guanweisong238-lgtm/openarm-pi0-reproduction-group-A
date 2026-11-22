#!/usr/bin/env python3
"""Package synthetic recovery candidates, with explicit provenance; no training merge."""
import json,hashlib
from pathlib import Path
import numpy as np
from real_delta_actions import to_deltas
from real_training_support import ROOT
src=ROOT/'results/real_recovery_candidates_projected_v2_1000_ep0_20261122';out=ROOT/'data/processed/real_recovery_synthetic_candidates_v001'
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True);report=json.loads((src/'report.json').read_text());entries=[]
for case in report['cases']:
 if case['exit_code']!=0 or case['result']['status']!='recovered':continue
 frame=case['frame'];folder=src/f'episode_0_frame_{frame}'
 with np.load(folder/'trajectory.npz') as z:states=z['state'];actions=z['action_qpos'];times=z['timestamp_s'];goal=z['goal_arm']
 if len(states)!=len(actions) or len(times)!=len(states) or not np.isfinite(states).all() or not np.isfinite(actions).all() or not (np.diff(times)>0).all():raise ValueError('invalid simulation trajectory')
 for i in range(len(states)):
  for camera in ('global','chest','wrist_left','wrist_right'):
   if not (folder/'images'/f'{i:04d}_{camera}.jpg').is_file():raise ValueError('missing image')
 # Prefix convention: observation i; targets i-1 ... i+48, only full windows.
 indices=np.arange(1,len(states)-49,dtype=np.int32)
 if len(indices):
  absolute=np.stack([actions[i-1:i+49] for i in indices]);delta=to_deltas(absolute,states[indices]);reconstructed=np.cumsum(delta,axis=1)+states[indices,None,:]
  if np.max(abs(reconstructed-absolute))>2e-5:raise ValueError('delta roundtrip')
  target=out/f'ep0_frame{frame}_windows.npz';np.savez_compressed(target,observation_indices=indices,delta_actions=delta)
  entries.append({'source_episode':0,'reference_frame':frame,'trajectory':str((folder/'trajectory.npz').relative_to(ROOT)),'image_directory':str((folder/'images').relative_to(ROOT)),'window_file':target.name,'window_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'frames':len(states),'full_50_step_windows':len(indices),'split':'train_candidate_only'})
manifest={'dataset_status':'synthetic_candidates_not_merged','entries':entries,'full_window_count':sum(x['full_50_step_windows'] for x in entries),'provenance':'offline model predicted arm targets inserted into empty MuJoCo; predicted grippers projected to simulator limits; same-index demo observation used as oracle reference','images':'synthetic camera views, uncalibrated and no drawer/cube','action_label':'delta rad; first action target i-1 minus simulated observed state i; later deltas target-to-target','caveats':['not a model-controlled rollout','gripper initialization is assumed after projection','not demonstrated task recovery or task success','synthetic image and empty environment domain shift','do not treat as real demonstrations; do not merge automatically into real training until reviewed','source validation episode excluded; no independent recovery validation set']}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False,indent=2))
