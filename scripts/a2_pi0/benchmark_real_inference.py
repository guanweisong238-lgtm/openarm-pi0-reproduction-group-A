#!/usr/bin/env python3
"""Measure warmed sequential data preparation and 10-step 4-camera policy inference."""
import argparse,dataclasses,hashlib,json,time
from pathlib import Path
import flax.nnx as nnx
from flax import traverse_util
import jax
import numpy as np
from real_training_support import ROOT,get_train_config,load_parameters
from real_four_camera_config import FourCameraPi0Config
from real_prefix_pi0 import PrefixPi0Config
from real_four_camera_loader import RealFourCameraDataset,batch_samples,DEFAULT

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--num-steps',type=int,default=10);parser.add_argument('--prefix',action='store_true');parser.add_argument('--out',type=Path,default=ROOT/'results/real_inference_latency_20261122');args=parser.parse_args()
 if args.num_steps<1:parser.error('num-steps must be positive')
 dest=args.out;dest.mkdir(parents=True,exist_ok=False)
 run=json.loads((ROOT/'results/real_pi0_four_camera_lora_001/config.json').read_text()); model_class=PrefixPi0Config if args.prefix else FourCameraPi0Config;cfg=dataclasses.replace(get_train_config(),model=model_class(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50),seed=run['seed'])
 dataset=RealFourCameraDataset('validation',cache_episodes=4);started=time.perf_counter();params=load_parameters(cfg);abstract=nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed));graph=nnx.graphdef(abstract);model=nnx.merge(graph,params)
 ck=ROOT/('checkpoints/real_prefix_prevframe_pilot_100_20261122/step_0100' if args.prefix else 'checkpoints/real_pi0_four_camera_lora_001/step_1000');meta=json.loads((ck/'checkpoint.json').read_text())
 if sha(ck/'lora_params.npz')!=meta['lora_sha256'] or sha(DEFAULT/'norm_stats.json')!=run['norm_stats_sha256']:raise ValueError('checkpoint or stats mismatch')
 state=nnx.state(model).filter(cfg.trainable_filter);expected=traverse_util.flatten_dict(state.to_pure_dict(),sep='/')
 with np.load(ck/'lora_params.npz',allow_pickle=False) as f:
  if set(f.files)!=set(expected):raise ValueError('adapter keys mismatch')
  state.replace_by_pure_dict(traverse_util.unflatten_dict({k:f[k] for k in f.files},sep='/'))
 nnx.update(model,state);params=nnx.state(model)
 @jax.jit
 def sample(params,obs,prefix,key):
  model=nnx.merge(graph,params)
  return model.sample_actions(key,obs,known_prefix=prefix,num_steps=args.num_steps) if args.prefix else model.sample_actions(key,obs,num_steps=args.num_steps)
 print('model loaded, warm-up compile',flush=True); warmobs,warmactions=jax.device_put(batch_samples([dataset[0]]));warmstart=time.perf_counter();jax.block_until_ready(sample(params,warmobs,warmactions[:,:7,:],jax.random.key(0)));warmup=time.perf_counter()-warmstart
 records=[]
 for i in range(20):
  index=int(i*(len(dataset)-1)/19);t0=time.perf_counter();obs,actions=batch_samples([dataset[index]]);t1=time.perf_counter();obs,actions=jax.device_put((obs,actions));jax.block_until_ready((obs,actions));t2=time.perf_counter();out=sample(params,obs,actions[:,:7,:],jax.random.key(81000+i));jax.block_until_ready(out);t3=time.perf_counter()
  records.append({'index':index,'prepare_ms':(t1-t0)*1000,'device_transfer_ms':(t2-t1)*1000,'model_ms':(t3-t2)*1000,'total_ms':(t3-t0)*1000});print(f'{i+1}/20 total_ms={records[-1]["total_ms"]:.1f} model_ms={records[-1]["model_ms"]:.1f}',flush=True)
 keys=['prepare_ms','device_transfer_ms','model_ms','total_ms'];summary={k:{'median':float(np.percentile([r[k] for r in records],50)),'p95':float(np.percentile([r[k] for r in records],95)),'max':float(max(r[k] for r in records))} for k in keys}
 result={'hardware':str(jax.devices()[0]),'sampling_steps':args.num_steps,'prefix_conditioned':args.prefix,'warmup_compile_and_first_ms':warmup*1000,'total_process_seconds':time.perf_counter()-started,'count':len(records),'summary_ms':summary,'fraction_over_200ms':sum(r['total_ms']>200 for r in records)/len(records),'records':records,'note':'offline JPEG decode/resize + observation assembly + model; cached process, no real camera capture, no IPC/driver/network, not deployment latency or deadline guarantee; benchmark prefix comes from recorded targets'}
 (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
