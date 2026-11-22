#!/usr/bin/env python3
"""Independently reload a saved four-camera LoRA adapter and compare held-out probes."""
import argparse,dataclasses,hashlib,json
from pathlib import Path
import numpy as np
import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
from real_training_support import ROOT,get_train_config
from real_training_support import load_parameters
from real_four_camera_config import FourCameraPi0Config
from real_four_camera_loader import RealFourCameraDataset,batch_samples

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',default='real_pi0_four_camera_lora_001');parser.add_argument('--step',type=int,default=1000);args=parser.parse_args()
 root=ROOT/'results'/args.name;check=ROOT/'checkpoints'/args.name/f'step_{args.step:04d}';meta=json.loads((check/'checkpoint.json').read_text());run=json.loads((root/'config.json').read_text())
 if sha(check/'lora_params.npz')!=meta['lora_sha256'] or sha(check/'norm_stats.json')!=meta['normalization_sha256']:raise ValueError('checkpoint hash mismatch')
 if sha(root/'manifest.json')!=run['candidate_manifest_sha256'] or sha(root/'norm_stats.json')!=run['norm_stats_sha256']:raise ValueError('run dataset hash mismatch')
 from real_four_camera_loader import DEFAULT
 candidate=Path(run.get('candidate',str(DEFAULT)))
 if sha(candidate/'manifest.json')!=run['candidate_manifest_sha256'] or sha(candidate/'norm_stats.json')!=run['norm_stats_sha256']:raise ValueError('current loader dataset differs from run')
 cfg0=get_train_config();cfg=dataclasses.replace(cfg0,model=FourCameraPi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50),seed=run['seed'])
 params=load_parameters(cfg);abstract=nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed));graph=nnx.graphdef(abstract);model=nnx.merge(graph,params)
 state=nnx.state(model).filter(cfg.trainable_filter); expected=traverse_util.flatten_dict(state.to_pure_dict(),sep='/')
 with np.load(check/'lora_params.npz',allow_pickle=False) as f:
  if set(f.files)!=set(expected):raise ValueError('adapter parameter keys do not match')
  flat={k:f[k] for k in f.files}
  for k,v in expected.items():
   if flat[k].shape!=v.shape or flat[k].dtype!=v.dtype:raise ValueError(f'adapter shape/dtype mismatch: {k}')
  state.replace_by_pure_dict(traverse_util.unflatten_dict(flat,sep='/'))
 nnx.update(model,state);params=nnx.state(model)
 @jax.jit
 def evaluate(params,obs,actions,key):return jnp.mean(nnx.merge(graph,params).compute_loss(key,obs,actions,train=False))
 ds=RealFourCameraDataset('validation',root=candidate,cache_episodes=4);lookup={row['episode_id']:i for i,row in enumerate(ds.entries)};loss=[]
 for i,item in enumerate(run['validation_probe_samples']):
  e=lookup[item['episode_id']];offset=0 if e==0 else int(ds.ends[e-1]);sample=jax.device_put(batch_samples([ds[offset+item['sample_index']]]));loss.append(float(evaluate(params,*sample,jax.random.key(10000+i))))
 records=[json.loads(l) for l in (root/'validation.jsonl').read_text().splitlines()];expected_loss=next(x['sample_losses'] for x in records if x['step']==args.step)
 error=float(np.max(np.abs(np.array(loss)-np.array(expected_loss))))
 result={'step':args.step,'samples':len(loss),'recorded_mean':float(np.mean(expected_loss)),'restored_mean':float(np.mean(loss)),'max_absolute_difference':error,'recorded_losses':expected_loss,'restored_losses':loss,'checkpoint_sha256':meta['lora_sha256'],
         'exact_match':error==0,'diagnostic_tolerance':1e-3,'within_diagnostic_tolerance':error<=1e-3,
         'note':'The tolerance is a practical diagnostic, not a verified numerical reproducibility guarantee; cause of nonzero difference is not established.'}
 (root/f'independent_restore_{args.step}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if error>1e-3:raise ValueError(f'validation difference exceeds diagnostic tolerance: {error}')
if __name__=='__main__':main()
