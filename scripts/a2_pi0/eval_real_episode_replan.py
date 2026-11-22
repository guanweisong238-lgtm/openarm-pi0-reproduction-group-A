#!/usr/bin/env python3
"""Full-episode offline teacher-forced replanning from held-out recorded observations."""
import argparse,dataclasses,hashlib,json,time
from pathlib import Path
import numpy as np
import flax.nnx as nnx
from flax import traverse_util
import jax
from real_training_support import ROOT,get_train_config,load_parameters
from real_four_camera_config import FourCameraPi0Config
from real_four_camera_loader import RealFourCameraDataset,batch_samples,DEFAULT

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_episode(outdir, episode, stride, row, timestamps, targets, states, anchors, allwindows, a, meta, timings):
 n=len(timestamps)
 result=np.empty((n,16),dtype=np.float32);source=np.empty(n,dtype=np.int32)
 for j,i in enumerate(anchors):
  end=anchors[j+1] if j+1<len(anchors) else n;length=end-i
  if not 1<=length<=50:raise ValueError('gap or unsupported last segment')
  result[i:end]=allwindows[j][:length];source[i:end]=i
 err=np.abs(result-targets);hold=np.abs(states-targets)
 outdir.mkdir(parents=True)
 np.savez_compressed(outdir/'full_episode.npz',timestamp_ns=timestamps,observed_state=states,recorded_ik_target=targets,predicted_ik_target=result,replan_anchor=source,anchor_indices=np.asarray(anchors,dtype=np.int32),predicted_windows=np.stack(allwindows))
 out={'episode_id':episode,'split':'validation','samples':n,'fps':30,'checkpoint':f'{a.name}/step_{a.step:04d}','source_samples_path':row['samples_path'],'checkpoint_sha256':meta['lora_sha256'],'replan_stride_samples':stride,'replans':len(anchors),'replan_anchors':anchors,'num_flow_steps':10,'seed_rule':'70000 + anchor sample index' if a.noise_mode=='vary' else '70000 for every anchor',
 'rtc_prefix_frames':a.rtc_prefix,
 'replay_prepare_and_inference_p95_ms':float(np.quantile(timings,.95)*1000) if timings else None,
 'conditioning':'Every replan uses recorded real images and measured state from this validation episode, not model-generated simulation observations; last forecast fills final 50 steps.','mean_absolute_error_rad':float(err.mean()),'hold_observed_mae_rad':float(hold.mean()),'per_joint_mae_rad':err.mean(0).tolist(),
 'recorded_episode_used_for_prior_validation':True,'unseen_episode_in_training_split':True,'object_initial_position_independence_verified':False,'calibrated_physical_rollout':False,'task_success_verified':False,
 'limitations':['Same collection session and validation episode previously used for monitoring; exact unseen physical start and object pose not established.','Model does not control simulation: kinematic targets are plotted.','Recorded real observations are supplied at each replan; this is teacher-forced, not closed loop.','IK targets precede driver safety processing; no commands sent to hardware.']}
 (outdir/'report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(f'Wrote episode {episode}, stride {stride}: {len(anchors)} plans',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--episode',type=int,default=17)
 p.add_argument('--episodes',type=int,nargs='+')
 p.add_argument('--all-validation',action='store_true')
 p.add_argument('--stride',type=int,default=15)
 p.add_argument('--strides',type=int,nargs='+')
 p.add_argument('--noise-mode',choices=('vary','fixed'),default='vary')
 p.add_argument('--out',type=Path,default=ROOT/'results/real_full_episode_17_offline_20261121')
 p.add_argument('--name',default='real_pi0_four_camera_lora_001')
 p.add_argument('--step',type=int,default=1000)
 p.add_argument('--rtc-prefix',type=int,default=0,help='Experimental RTC guidance on overlapping arm targets, measured in frames')
 a=p.parse_args()
 strides=sorted(set(a.strides or [a.stride]))
 if any(not 1<=stride<=50 for stride in strides):p.error('stride must be 1..50')
 if not 0<=a.rtc_prefix<=50-max(strides) or a.rtc_prefix and len(strides)!=1:p.error('RTC requires one stride and sufficient overlap')
 multi=a.all_validation or a.episodes is not None or a.strides is not None
 if a.out.exists() and not multi:raise FileExistsError(a.out)
 run=ROOT/'results'/a.name;config=json.loads((run/'config.json').read_text());candidate=Path(config.get('candidate',str(DEFAULT)))
 if sha(candidate/'manifest.json')!=config['candidate_manifest_sha256'] or sha(candidate/'norm_stats.json')!=config['norm_stats_sha256']:raise ValueError('data or normalization changed')
 ds=RealFourCameraDataset('validation',root=candidate,cache_episodes=1)
 requested=set(a.episodes or [a.episode])
 entries=[(e,row) for e,row in enumerate(ds.entries) if a.all_validation or row['episode_id'] in requested]
 if not a.all_validation and len(entries)!=len(requested):raise ValueError('requested episode is not in the held-out validation split')
 cfg=dataclasses.replace(get_train_config(),model=FourCameraPi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50),seed=config['seed'])
 params=load_parameters(cfg);abstract=nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed));graph=nnx.graphdef(abstract);model=nnx.merge(graph,params)
 ck=ROOT/'checkpoints'/a.name/f'step_{a.step:04d}';meta=json.loads((ck/'checkpoint.json').read_text())
 if sha(ck/'lora_params.npz')!=meta['lora_sha256'] or sha(ck/'norm_stats.json')!=config['norm_stats_sha256']:raise ValueError('checkpoint or statistics hash mismatch')
 state=nnx.state(model).filter(cfg.trainable_filter);expected=traverse_util.flatten_dict(state.to_pure_dict(),sep='/')
 with np.load(ck/'lora_params.npz',allow_pickle=False) as f:
  if set(f.files)!=set(expected):raise ValueError('adapter keys mismatch')
  state.replace_by_pure_dict(traverse_util.unflatten_dict({k:f[k] for k in f.files},sep='/'))
 nnx.update(model,state);params=nnx.state(model)
 @jax.jit
 def sample(params,obs,key):return nnx.merge(graph,params).sample_actions(key,obs,num_steps=10)
 @jax.jit
 def sample_with_rtc(params,obs,key,previous,mask):
  return nnx.merge(graph,params).sample_actions(key,obs,num_steps=10,previous_actions=previous,guidance_mask=mask)
 mean=np.asarray(ds.stats['actions'].mean,np.float32);std=np.asarray(ds.stats['actions'].std,np.float32)
 for e,row in entries:
  episode=row['episode_id']
  start=0 if e==0 else int(ds.ends[e-1]);last=row['full_window_samples']-1
  destinations={stride:(a.out/f'episode_{episode}'/f'stride_{stride}' if multi else a.out) for stride in strides}
  by_stride={}
  for stride,outdir in destinations.items():
   if outdir.exists():
    report_file=outdir/'report.json'
    if not report_file.is_file() or not (outdir/'full_episode.npz').is_file():
     raise ValueError(f'incomplete replay: {outdir}')
    prior=json.loads(report_file.read_text())
    seed_rule='70000 + anchor sample index' if a.noise_mode=='vary' else '70000 for every anchor'
    if (prior['checkpoint_sha256'],prior['episode_id'],prior['replan_stride_samples'],prior['seed_rule'],prior['source_samples_path'],prior.get('rtc_prefix_frames',0))!=(meta['lora_sha256'],episode,stride,seed_rule,row['samples_path'],a.rtc_prefix):
     raise ValueError(f'replay configuration changed: {outdir}')
    continue
   anchors=list(range(0,last+1,stride))
   if anchors[-1]!=last:anchors.append(last)
   by_stride[stride]=anchors
  if not by_stride:continue
  with np.load(ROOT/row['samples_path'],allow_pickle=False) as z:
   timestamps=z['timestamp_ns'].copy();targets=z['actions'][:,0,:].copy();states=z['state'].copy()
  if last+50!=len(timestamps):raise ValueError('unexpected eligible-window boundary')
  required=sorted(set().union(*by_stride.values()))
  predictions={};begin=time.monotonic()
  elapsed_samples=[]
  previous_output=None
  print(f'validation episode {episode}: {len(timestamps)} frames, {len(required)} real-observation replans',flush=True)
  for j,i in enumerate(required):
   before=time.monotonic()
   seed=70000+i if a.noise_mode=='vary' else 70000
   obs,_=jax.device_put(batch_samples([ds[start+i]]));key=jax.random.key(seed)
   if a.rtc_prefix and previous_output is not None:
    offset=i-required[j-1]
    mask=np.zeros_like(previous_output);ref=np.zeros_like(previous_output)
    ref[:,:a.rtc_prefix]=previous_output[:,offset:offset+a.rtc_prefix]
    decay=np.arange(a.rtc_prefix,0,-1,dtype=np.float32)/a.rtc_prefix
    mask[:,:a.rtc_prefix,:7]=decay[None,:,None]
    mask[:,:a.rtc_prefix,8:15]=decay[None,:,None]
    previous_output=np.asarray(sample_with_rtc(params,obs,key,ref,mask))
   else:
    previous_output=np.asarray(sample(params,obs,key))
   output=previous_output[0,:,:16]*(std+1e-6)+mean
   if j>int(bool(a.rtc_prefix)):elapsed_samples.append(time.monotonic()-before)
   if output.shape!=(50,16) or not np.isfinite(output).all():raise ValueError('invalid prediction')
   predictions[i]=output
   if j%25==0 or j==len(required)-1:print(f'predictions {j+1}/{len(required)} elapsed {time.monotonic()-begin:.1f}s',flush=True)
  for stride,anchors in by_stride.items():
   write_episode(destinations[stride],episode,stride,row,timestamps,targets,states,anchors,[predictions[i] for i in anchors],a,meta,elapsed_samples)

if __name__=='__main__':main()
