#!/usr/bin/env python3
"""Small offline prefix-conditioned LoRA pilot; no controller or robot commands."""
import dataclasses,json,logging,shutil,time
import numpy as np
import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
from real_training_support import ROOT,get_train_config,load_parameters,make_train_step
from real_four_camera_loader import RealFourCameraDataset,batch_samples,DEFAULT
from real_prefix_pi0 import PrefixPi0Config
from train_real_four_camera import save_adapter,verify_adapter,sha
from openpi.training import optimizer as optimizer_lib

NAME='real_delta_prefix_train_1000_20261122'
from real_delta_actions import delta_stats, encode_deltas

def shifted_batch(dataset,index):
 """Observation at frame i; 50 IK targets start at i-1 (last applied action)."""
 episode,frame=dataset.locate(index)
 if frame<1:raise ValueError('prefix requires a previous frame within this episode')
 item=dataset[index]  # Loads and checks the episode before accessing its cached numeric arrays.
 episode_id=dataset.entries[episode]['episode_id']
 raw=DELTA_LABELS[episode_id][frame-1]
 normalized=((raw-DELTA_MEAN)/(DELTA_STD+1e-6)).astype(np.float32)
 item['actions']=np.pad(normalized,((0,0),(0,16)))
 return jax.device_put(batch_samples([item]))

def main():
 output=ROOT/'results'/NAME;checkpoints=ROOT/'checkpoints'/NAME
 if output.exists() or checkpoints.exists():raise FileExistsError(NAME)
 output.mkdir();checkpoints.mkdir();logging.basicConfig(level=logging.INFO,force=True,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(output/'run.log')])
 initial=ROOT/'checkpoints/real_pi0_four_camera_lora_001/step_1000';meta=json.loads((initial/'checkpoint.json').read_text());run=json.loads((ROOT/'results/real_pi0_four_camera_lora_001/config.json').read_text())
 if sha(initial/'lora_params.npz')!=meta['lora_sha256'] or sha(DEFAULT/'manifest.json')!=run['candidate_manifest_sha256'] or sha(DEFAULT/'norm_stats.json')!=run['norm_stats_sha256']:raise ValueError('input hash mismatch')
 c0=get_train_config();schedule=optimizer_lib.CosineDecaySchedule(warmup_steps=10,peak_lr=1e-5,decay_steps=1000,decay_lr=1e-6)
 cfg=dataclasses.replace(c0,model=PrefixPi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50),num_train_steps=1000,lr_schedule=schedule,seed=42)
 ds=RealFourCameraDataset('train',cache_episodes=18);val=RealFourCameraDataset('validation',cache_episodes=4)
 global DELTA_MEAN,DELTA_STD,DELTA_LABELS
 delta_root=ROOT/'data/processed/real_delta_actions_v001'
 delta_manifest=json.loads((delta_root/'manifest.json').read_text())
 DELTA_LABELS={}
 for row in delta_manifest['entries']:
  path=delta_root/row['file']
  if sha(path)!=row['sha256']:raise ValueError('delta label hash mismatch')
  with np.load(path,allow_pickle=False) as f:DELTA_LABELS[row['episode_id']]=f['delta_actions']
 with np.load(delta_root/'delta_stats.npz',allow_pickle=False) as f:DELTA_MEAN=f['mean'];DELTA_STD=f['std']
 shutil.copy2(delta_root/'delta_stats.npz',output/'delta_stats.npz')
 shutil.copy2(delta_root/'manifest.json',output/'delta_manifest.json')
 samples=[];picked=[]
 for e,row in enumerate(val.entries):
  start=0 if e==0 else int(val.ends[e-1]);n=row['full_window_samples']
  for i in (1,n//2):samples.append(shifted_batch(val,start+i));picked.append({'episode_id':row['episode_id'],'observation_index':i,'action_start_index':i-1})
 config={'experiment':NAME,'initial_adapter_sha256':meta['lora_sha256'],'steps':1000,'sampling_steps':10,'prefix':7,'training_objective':'normalized delta angles: first target minus observed state at frame i; later targets minus preceding targets; targets start at i-1; first 7 are known prefix, loss on remaining 43',
  'training_prefix_source':'preprocessed delta labels from recorded targets (teacher forcing), not previous predicted targets','state_and_images':'actual recorded observation at start of prefix','caveats':['not an online controller; IK targets not final CAN commands','no asynchronous rollout; latency model separate','validation uses recorded prefixes, not predicted prefixes'],
  'train_episode_ids':[x['episode_id'] for x in ds.entries],'validation_probe_samples':picked,'manifest_sha256':run['candidate_manifest_sha256'],'norm_sha256':run['norm_stats_sha256'],'delta_stats_source':'training windows only, includes bootstrap delta and 49 successive differences','schedule':repr(schedule)}
 (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
 snap=output/'source_snapshot';snap.mkdir()
 for name in ('train_real_delta_1000.py','real_delta_actions.py','real_prefix_pi0.py','real_action_window.py','real_training_support.py','real_four_camera_pi0.py','real_four_camera_config.py','real_four_camera_loader.py','train_real_four_camera.py'):shutil.copy2(ROOT/'scripts/a2_pi0'/name,snap/name)
 for name in ('manifest.json','norm_stats.json','review_decision.json'):shutil.copy2(DEFAULT/name,output/name)
 logging.info('loading base model and existing 1000-step adapter')
 params=load_parameters(cfg);abstract=nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed));graph=nnx.graphdef(abstract);model=nnx.merge(graph,params);state=nnx.state(model).filter(cfg.trainable_filter);expected=traverse_util.flatten_dict(state.to_pure_dict(),sep='/')
 with np.load(initial/'lora_params.npz',allow_pickle=False) as f:
  if set(f.files)!=set(expected):raise ValueError('incompatible starting adapter')
  state.replace_by_pure_dict(traverse_util.unflatten_dict({k:f[k] for k in f.files},sep='/'))
 nnx.update(model,state);params=nnx.state(model)
 tx=optimizer_lib.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None);opt_state=tx.init(params.filter(cfg.trainable_filter));train_step=make_train_step(cfg,graph,tx)
 @jax.jit
 def evaluate(params,obs,actions,key):return jnp.mean(nnx.merge(graph,params).compute_loss(key,obs,actions,train=False))
 @jax.jit
 def sample(params,obs,prefix,key):return nnx.merge(graph,params).sample_actions(key,obs,known_prefix=prefix,num_steps=10)
 def valpass(step):
  vals=[float(evaluate(params,*item,jax.random.key(11000+i))) for i,item in enumerate(samples)]
  if not np.isfinite(vals).all():raise RuntimeError('nonfinite validation')
  rec={'step':step,'suffix_flow_loss':float(np.mean(vals)),'sample_losses':vals}
  with (output/'validation.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  logging.info('validation step=%d suffix_loss=%.6f',step,rec['suffix_flow_loss']);return rec
 def prefix_check():
  obs,actions=samples[0];p=actions[:,:7,:];result=sample(params,obs,p,jax.random.key(999));jax.block_until_ready(result)
  if not bool(jnp.isfinite(result).all()):raise RuntimeError('nonfinite generated suffix')
  err=float(jnp.max(jnp.abs(result[:,:7,:]-p)))
  if err!=0:raise RuntimeError(f'prefix not preserved: {err}')
  return err
 prefix_before=prefix_check();before=valpass(0);rng=jax.random.key(43)
 eligible=[]
 for e,row in enumerate(ds.entries):
  start_index=0 if e==0 else int(ds.ends[e-1]);eligible.extend(range(start_index+1,start_index+row['full_window_samples']))
 order=np.random.default_rng(42).permutation(eligible);start=time.monotonic()
 for step in range(1,1001):
  index=int(order[step-1]);batch=shifted_batch(ds,index);rng,key=jax.random.split(rng);params,opt_state,metrics=train_step(params,opt_state,key,*batch);metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
  if not np.isfinite(list(metrics.values())).all():raise ValueError('nonfinite training metrics')
  with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,'sample_index':index,**metrics})+'\n')
  if step==1 or step%100==0:logging.info('step=%d/1000 loss=%.5f elapsed=%.1fs',step,metrics['loss'],time.monotonic()-start)
  if step in (250,500,750):valpass(step)
 after=valpass(1000);prefix_after=prefix_check();path=save_adapter(checkpoints,1000,params,opt_state,cfg);restore=verify_adapter(path,params,graph,cfg,evaluate,samples[0]);stats=jax.devices()[0].memory_stats() or {}
 result={'steps':1000,'training_windows':1000,'initial_suffix_flow_loss':before['suffix_flow_loss'],'final_suffix_flow_loss':after['suffix_flow_loss'],'prefix_max_abs_error_before':prefix_before,'prefix_max_abs_error_after':prefix_after,'checkpoint':str(path.relative_to(ROOT)),'restore_check':restore,'peak_gpu_bytes':stats.get('peak_bytes_in_use'),'training_seconds':time.monotonic()-start,'trained_prefix_source':'recorded teacher actions only; online uses previous predicted actions','robot_executed':False}
 (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');logging.info('PILOT COMPLETE %s',json.dumps(result))
if __name__=='__main__':main()
