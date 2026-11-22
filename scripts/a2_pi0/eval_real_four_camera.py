#!/usr/bin/env python3
"""Offline fixed-probe evaluation of saved four-camera LoRA adapters."""
import dataclasses,hashlib,json
import numpy as np
import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
from real_training_support import ROOT,get_train_config
from real_training_support import load_parameters
from real_four_camera_config import FourCameraPi0Config
from real_four_camera_loader import RealFourCameraDataset,batch_samples,DEFAULT

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 name='real_pi0_four_camera_lora_001';run=ROOT/'results'/name;config=json.loads((run/'config.json').read_text())
 if sha(DEFAULT/'manifest.json')!=config['candidate_manifest_sha256'] or sha(DEFAULT/'norm_stats.json')!=config['norm_stats_sha256']:raise ValueError('dataset differs from training')
 cfg=dataclasses.replace(get_train_config(),model=FourCameraPi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50),seed=config['seed'])
 params=load_parameters(cfg);abstract=nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed));graph=nnx.graphdef(abstract);model=nnx.merge(graph,params)
 ds=RealFourCameraDataset('validation',cache_episodes=4);probes=[]
 for e,row in enumerate(ds.entries):
  start=0 if e==0 else int(ds.ends[e-1]);n=row['full_window_samples']
  for i in np.linspace(0,n-1,20,dtype=int):probes.append((row['episode_id'],int(i),start+int(i)))
 @jax.jit
 def evaluate(params,obs,actions,key):return jnp.mean(nnx.merge(graph,params).compute_loss(key,obs,actions,train=False))
 results={}
 for step in (500,1000):
  check=ROOT/'checkpoints'/name/f'step_{step:04d}';meta=json.loads((check/'checkpoint.json').read_text())
  if sha(check/'lora_params.npz')!=meta['lora_sha256']:raise ValueError('adapter hash mismatch')
  state=nnx.state(model).filter(cfg.trainable_filter);expected=traverse_util.flatten_dict(state.to_pure_dict(),sep='/')
  with np.load(check/'lora_params.npz',allow_pickle=False) as f:
   if set(f.files)!=set(expected):raise ValueError('unexpected adapter keys')
   flat={k:f[k] for k in f.files};state.replace_by_pure_dict(traverse_util.unflatten_dict(flat,sep='/'))
  nnx.update(model,state);p=nnx.state(model);losses=[]
  for i,(eid,frame,index) in enumerate(probes):
   obs,actions=jax.device_put(batch_samples([ds[index]]));loss=float(evaluate(p,obs,actions,jax.random.key(30000+i)))
   if not np.isfinite(loss):raise ValueError(f'nonfinite loss at {eid}/{frame}')
   losses.append({'episode_id':eid,'sample_index':frame,'loss':loss})
  per={str(e):float(np.mean([x['loss'] for x in losses if x['episode_id']==e])) for e in sorted({x['episode_id'] for x in losses})}
  results[str(step)]={'mean':float(np.mean([x['loss'] for x in losses])),'episode_means':per,'losses':losses,'adapter_sha256':meta['lora_sha256']}
  print(f'step {step}: mean {results[str(step)]["mean"]:.6f}; per-episode {per}',flush=True)
 report={'protocol':'20 evenly spaced full-window starts per validation episode; fixed flow rng per paired probe; independent of gradients but same validation episodes used for monitoring',
 'validation_episode_ids':sorted({x[0] for x in probes}),'probe_count':len(probes),'results':results,
 'note':'offline flow-matching loss; not robot success, generalization guarantee, or independent final test'}
 (run/'expanded_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
