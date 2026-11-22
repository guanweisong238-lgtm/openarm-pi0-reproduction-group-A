#!/usr/bin/env python3
"""Compare teacher-forced 14/20 schedules on held-out episode; no robot control."""
import dataclasses,hashlib,json,math,time
import numpy as np
import flax.nnx as nnx
from flax import traverse_util
import jax
from real_training_support import ROOT,get_train_config,load_parameters
from real_four_camera_loader import RealFourCameraDataset,batch_samples,DEFAULT
from real_prefix_pi0 import PrefixPi0Config
from real_delta_actions import encode_deltas,decode_deltas

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 out=ROOT/'results/real_delta_schedule_pilot_20261122'
 if out.exists():raise FileExistsError(out)
 baseline=ROOT/'checkpoints/real_pi0_four_camera_lora_001/step_1000';pilot=ROOT/'checkpoints/real_delta_prefix_pilot_100_20261122/step_0100'
 ds=RealFourCameraDataset('validation',cache_episodes=4);e=next(i for i,r in enumerate(ds.entries) if r['episode_id']==17);entry=ds.entries[e];offset=0 if e==0 else int(ds.ends[e-1]);last=entry['full_window_samples']-1
 with np.load(ROOT/entry['samples_path'],allow_pickle=False) as z:reference=z['actions'][:,0,:].copy();times=z['timestamp_ns'].copy()
 n=len(times);mean=np.asarray(ds.stats['actions'].mean,np.float32);std=np.asarray(ds.stats['actions'].std,np.float32)
 cfg=dataclasses.replace(get_train_config(),model=PrefixPi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50),seed=42)
 print('loading model',flush=True);params=load_parameters(cfg);abstract=nnx.eval_shape(cfg.model.create,jax.random.key(42));graph=nnx.graphdef(abstract);model=nnx.merge(graph,params)
 @jax.jit
 def ordinary(p,obs,key):return nnx.merge(graph,p).sample_actions(key,obs,num_steps=10)
 @jax.jit
 def conditioned(p,obs,prefix,key):return nnx.merge(graph,p).sample_actions(key,obs,known_prefix=prefix,num_steps=10)
 def adapter(path):
  meta=json.loads((path/'checkpoint.json').read_text())
  if sha(path/'lora_params.npz')!=meta['lora_sha256']:raise ValueError('adapter hash mismatch')
  state=nnx.state(model).filter(cfg.trainable_filter);ex=traverse_util.flatten_dict(state.to_pure_dict(),sep='/')
  with np.load(path/'lora_params.npz',allow_pickle=False) as f:
   if set(f.files)!=set(ex):raise ValueError('adapter mismatch')
   state.replace_by_pure_dict(traverse_util.unflatten_dict({k:f[k] for k in f.files},sep='/'))
  nnx.update(model,state);return nnx.state(model)
 def unnorm(x):return np.asarray(x)[0,:,:16]*(std+1e-6)+mean
 p0=adapter(baseline);obs0,_=jax.device_put(batch_samples([ds[offset]]));initial=unnorm(ordinary(p0,obs0,jax.random.key(91000)))
 with np.load(ROOT/'results/real_delta_prefix_pilot_100_20261122/delta_stats.npz') as f: dm=f['mean']; dsigma=f['std']
 prepared=[]
 # Both schemes use the same actual recorded observations at the same post-action frame.
 for r in range(14,last+1,20):
  obs=jax.device_put(batch_samples([ds[offset+r]]))[0];state=ds.cache[e]['state'][r].copy();prepared.append((r,obs,state))
 results={};arrays={};begin=time.monotonic()
 for mode,path in [('delta_pilot',pilot)]:
  params=adapter(path)
  for cycles in (6,9):
   output=np.full((n,16),np.nan,dtype=np.float32);current=initial.copy();origin=0;cursor=0;switches=[];prefix_errors=[]
   for r,obs,state in prepared:
    switch=r+cycles
    if switch>=n:break
    # Already executed old actions plus any extra while inference runs.
    if cursor<origin or switch>origin+50 or r-1<origin or r+6>origin+50:raise ValueError('plan has a gap / prefix unavailable')
    output[cursor:switch]=current[cursor-origin:switch-origin]
    previous=current[r-1-origin:r+6-origin]
    if previous.shape!=(7,16):raise ValueError('bad 14..20 prefix')
    if mode=='delta_pilot':
     normal=encode_deltas(previous,state,dm,dsigma);prefix=np.pad(normal,((0,0),(0,16)))[None,:,:]
     generated=np.asarray(conditioned(params,obs,jax.device_put(prefix),jax.random.key(90000+r)))[0]
     predicted=decode_deltas(generated,state,dm,dsigma)
     prefix_errors.append(float(np.max(np.abs(predicted[:7]-previous))))
     next_origin=r-1
    else:
     predicted=unnorm(ordinary(params,obs,jax.random.key(90000+r)));next_origin=r
    current=predicted;origin=next_origin;cursor=switch;switches.append(switch)
   end=min(n,origin+50);output[cursor:end]=current[cursor-origin:end-origin]
   arrays[f'{mode}_{cycles}']=output;results[f'{mode}_{cycles}']={'coverage_frames':end,'switch_frames':switches,'max_prefix_error_rad':max(prefix_errors,default=None)}
   print(mode,'simulated inference delay cycles',cycles,'coverage',end,'switches',len(switches),'elapsed',round(time.monotonic()-begin,1),flush=True)
 common=min(x['coverage_frames'] for x in results.values());arm=np.r_[0:7,8:15]
 for name,info in results.items():
  q=arrays[name][:common];delta=np.abs(np.diff(q[:,arm],axis=0)).max(axis=1);boundary=np.array([x-1 for x in info['switch_frames'] if 0<x<common],int);other=np.ones(len(delta),bool);other[boundary]=False
  info.update({'evaluated_frames':common,'arm_boundary_median_rad':float(np.median(delta[boundary])),'arm_boundary_max_rad':float(np.max(delta[boundary])),
   'arm_boundary_over_0_1_count':int((delta[boundary]>.1).sum()),'boundary_count':len(boundary),'arm_nonboundary_median_rad':float(np.median(delta[other])),
   'ik_target_mae_rad':float(np.abs(q-reference[:common]).mean())})
 out.mkdir();np.savez_compressed(out/'traces.npz',recorded_ik_target=reference,timestamp_ns=times,**arrays)
 report={'episode':17,'coverage_frames':common,'modes':results,'delay_cycles':{'6':'ideal arrival at step 20: 200ms','9':'simulated using ~275ms measured baseline, switch at next 30Hz boundary: 300ms'},'seed':'91000 at bootstrap, 90000 + observation index for replans',
  'sampling_steps':10,'prefix_train':'100 updates on preprocessed normalized angle deltas, recorded prefix during training, previous predicted prefix at inference; first absolute plan from existing baseline',
  'limitations':['Recorded validation observations, not model-controlled camera or robot feedback; no physical rollout.','Baseline supplies initial absolute plan; delta pilot handles replans. Different checkpoints and limited training prevent attributing differences solely to delta representation.','Delay simulated from a small baseline benchmark, not actual asynchronous timing or prefix-model benchmark.','Step boundaries are offline grid conventions, must be validated against real control timestamps.','No extra clipping, interpolation or safety behavior was applied.']}
 (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
