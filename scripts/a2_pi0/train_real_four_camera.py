#!/usr/bin/env python3
"""Train real four-camera pi0 LoRA with episode-isolated fixed validation probes."""
import argparse,dataclasses,hashlib,json,logging,os,shutil,time
from pathlib import Path
from datetime import datetime
from real_training_support import ROOT,get_train_config,BASE_PARAMS
from real_four_camera_config import FourCameraPi0Config,FOUR_CAMERA_KEYS,CAMERA_MAPPING
from real_four_camera_loader import RealFourCameraDataset,batch_samples,DEFAULT,PROMPT
from real_training_support import load_parameters,make_train_step
from real_task_c_phases import PHASES,phase_frames
import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi.training import optimizer as optimizer_lib


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def phase_balanced_order(train, previous_indices, steps, rng, priority):
 labels=np.empty(len(train),dtype=np.int8)
 buckets={}
 start=0
 for row in train.entries:
  source=ROOT/row['samples_path']
  if sha(source)!=row['samples_sha256']:raise ValueError(f'source hash mismatch: {source}')
  with np.load(source,allow_pickle=False) as z:
   phase,_=phase_frames(z['actions'][:,0],row['operation_style'])
  count=row['full_window_samples']
  # Favor observations leading into the phase, while the event is still ahead.
  labels[start:start+count]=phase[8:count+8]
  fresh=np.arange(start,start+count)
  fresh=fresh[~np.isin(fresh,list(previous_indices))]
  for stage in np.unique(labels[fresh]):
   buckets[(row['episode_id'],PHASES[stage])]=rng.permutation(fresh[labels[fresh]==stage]).tolist()
  start+=count
 if start!=len(train) or steps>sum(map(len,buckets.values())):raise ValueError('insufficient fresh phase windows')

 # Draw one window per nonempty episode/phase pair in randomly shuffled rounds.
 # Reserve every third draw for nominated phases when those phases still have windows.
 selected=[]
 general_round=[]
 priority_round=[]
 for step in range(steps):
  if not general_round:general_round=list(buckets);rng.shuffle(general_round)
  if not priority_round:
   priority_round=[key for key in buckets if key[1] in priority]
   rng.shuffle(priority_round)
  if step%3==2 and priority_round:
   key=priority_round.pop()
  else:
   key=general_round.pop()
  selected.append(buckets[key].pop())
  if not buckets[key]:
   del buckets[key]
   if key in general_round:general_round.remove(key)
   if key in priority_round:priority_round.remove(key)
 return np.asarray(selected),labels


def save_adapter(root,step,params,opt_state,cfg,candidate):
 path=root/f'step_{step:04d}';path.mkdir(exist_ok=False)
 trainable=params.filter(cfg.trainable_filter)
 flat=traverse_util.flatten_dict(trainable.to_pure_dict(),sep='/')
 arrays={k:np.asarray(v) for k,v in flat.items()}
 np.savez(path/'lora_params.npz',**arrays)
 shutil.copy2(candidate/'norm_stats.json',path/'norm_stats.json')
 meta={'step':step,'base_checkpoint':str(BASE_PARAMS.relative_to(ROOT)),
       'lora_sha256':sha(path/'lora_params.npz'),
       'normalization_sha256':sha(path/'norm_stats.json'),'note':'LoRA adapter only; requires exact base weights, local four-camera model and run protocol; optimizer not saved and resume is not implemented'}
 (path/'checkpoint.json').write_text(json.dumps(meta,indent=2)+'\n')
 return path


def verify_adapter(path,params,graphdef,cfg,evaluate,sample):
 model=nnx.merge(graphdef,params);trainable=nnx.state(model).filter(cfg.trainable_filter)
 expected=traverse_util.flatten_dict(trainable.to_pure_dict(),sep='/')
 with np.load(path/'lora_params.npz',allow_pickle=False) as z:
  assert set(z.files)==set(expected)
  loaded={k:z[k] for k in z.files}
  for k,v in expected.items():np.testing.assert_array_equal(loaded[k],np.asarray(v))
  trainable.replace_by_pure_dict(traverse_util.unflatten_dict(loaded,sep='/'))
 nnx.update(model,trainable);restored=nnx.state(model)
 a=float(evaluate(params,*sample,jax.random.key(9876)))
 b=float(evaluate(restored,*sample,jax.random.key(9876)))
 if not np.isfinite([a,b]).all() or abs(a-b)>1e-6:raise RuntimeError(f'adapter restore mismatch: {a}, {b}')
 return {'baseline_loss':a,'restored_loss':b,'absolute_difference':abs(a-b),'arrays':len(expected)}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--steps',type=int,default=1000);p.add_argument('--eval-every',type=int,default=100);p.add_argument('--save-every',type=int,default=500)
 p.add_argument('--name',default='real_pi0_four_camera_lora_001');p.add_argument('--seed',type=int,default=42)
 p.add_argument('--candidate',type=Path,default=DEFAULT)
 p.add_argument('--init-run',help='Warm start from the final adapter of this run, excluding its sampled windows')
 p.add_argument('--init-step',type=int,help='Select an intermediate checkpoint from --init-run')
 p.add_argument('--continuity-weight',type=float,default=0.0,help='Weight for same-time arm agreement on paired observations')
 p.add_argument('--phase-balanced',action='store_true',help='Spread fresh windows across Task C episodes and operation stages')
 p.add_argument('--priority-phases',nargs='+',choices=PHASES,default=[],help='Allocate every third phase-balanced draw to these stages')
 p.add_argument('--peak-lr',type=float,default=2.5e-5)
 args=p.parse_args()
 candidate=args.candidate.resolve()
 if args.steps<2 or min(args.eval_every,args.save_every)<1 or not args.name.replace('_','').replace('-','').isalnum():p.error('invalid steps, intervals, or experiment name')
 if args.priority_phases and not args.phase_balanced:p.error('--priority-phases requires --phase-balanced')
 if not np.isfinite(args.peak_lr) or args.peak_lr<=0:p.error('--peak-lr must be finite and positive')
 if args.init_step is not None and (not args.init_run or args.init_step<1):p.error('--init-step requires --init-run and a positive step')
 if not np.isfinite(args.continuity_weight) or args.continuity_weight<0:p.error('--continuity-weight must be finite and nonnegative')
 pair_stride,pair_overlap=20,8
 if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
 output=ROOT/'results'/args.name;checkpoints=ROOT/'checkpoints'/args.name
 if output.exists() or checkpoints.exists():raise FileExistsError(f'experiment already exists: {output} / {checkpoints}')
 output.mkdir(parents=True);checkpoints.mkdir(parents=True)
 logging.basicConfig(level=logging.INFO,force=True,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(output/'run.log')])
 cfg0=get_train_config();model_cfg=FourCameraPi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50)
 schedule=optimizer_lib.CosineDecaySchedule(warmup_steps=min(100,max(1,args.steps//10)),peak_lr=args.peak_lr,decay_steps=args.steps,decay_lr=args.peak_lr/10)
 cfg=dataclasses.replace(cfg0,model=model_cfg,num_train_steps=args.steps,lr_schedule=schedule,seed=args.seed)
 train=RealFourCameraDataset('train',root=candidate,cache_episodes=18);val=RealFourCameraDataset('validation',root=candidate,cache_episodes=4)
 previous_indices=set()
 initial_adapter=None
 if args.init_run:
  previous_name=args.init_run
  lineage={args.name}
  while previous_name:
   if previous_name in lineage:raise ValueError('cyclic warm-start lineage')
   lineage.add(previous_name)
   previous=ROOT/'results'/previous_name
   previous_config=json.loads((previous/'config.json').read_text())
   previous_summary=json.loads((previous/'summary.json').read_text())
   if previous_config['seed']!=args.seed:raise ValueError('expected the same model seed in warm-start lineage')
   for filename,key in (('manifest.json','candidate_manifest_sha256'),('norm_stats.json','norm_stats_sha256')):
    if sha(candidate/filename)!=previous_config[key]:raise ValueError(f'initial run uses different {filename}')
   rows=[json.loads(line) for line in (previous/'metrics.jsonl').read_text().splitlines()]
   if len(rows)!=previous_summary['steps_completed']:raise ValueError('incomplete prior training log')
   previous_indices.update(row['sample_index'] for row in rows)
   if initial_adapter is None:
    initial_adapter=(ROOT/'checkpoints'/previous_name/f'step_{args.init_step:04d}'
                     if args.init_step is not None else ROOT/previous_summary['last_checkpoint'])
   previous_name=previous_config.get('init_run')
  if args.steps>len(train)-len(previous_indices):raise ValueError('not enough fresh training windows')
  metadata=json.loads((initial_adapter/'checkpoint.json').read_text())
  if sha(initial_adapter/'lora_params.npz')!=metadata['lora_sha256'] or sha(initial_adapter/'norm_stats.json')!=sha(candidate/'norm_stats.json'):raise ValueError('initial adapter hash mismatch')
 selected=[];samples=[]
 for e,row in enumerate(val.entries):
  start=0 if e==0 else int(val.ends[e-1]);count=row['full_window_samples']
  for offset in (0,count//2):
   index=start+offset; samples.append(jax.device_put(batch_samples([val[index]])));selected.append({'episode_id':row['episode_id'],'sample_index':offset,'operation_style':row.get('operation_style','unspecified')})
 run_config={'seed':args.seed,'steps':args.steps,'batch_size':2 if args.continuity_weight else 1,'eval_every':args.eval_every,'save_every':args.save_every,
  'model':'pi0_base + gemma_2b_lora + gemma_300m_lora','trainable':'LoRA only','frozen_dtype':'bfloat16',
  'action_horizon':50,'action_semantics':'absolute IK target, right arm then left arm, rad','training_frequency_hz':30,
  'optimizer':repr(cfg.optimizer),'schedule':repr(schedule),'image_mapping':CAMERA_MAPPING,'prompt':PROMPT,
  'image_resize':'OpenPI 224x224 resize-with-pad','normalization':'train split only z-score',
  'train_samples':len(train),'validation_samples':len(val),'validation_probe_samples':selected,
  'validation_note':'two fixed probes per held-out episode with fixed flow random keys, not task success or generalization estimate',
  'candidate':str(candidate),'candidate_manifest_sha256':sha(candidate/'manifest.json'),'norm_stats_sha256':sha(candidate/'norm_stats.json'),'review_decision_sha256':sha(candidate/'review_decision.json'),
  'init_run':args.init_run,'initial_adapter':str(initial_adapter) if initial_adapter else None,'excluded_previous_windows':len(previous_indices),
  'continuity_weight':args.continuity_weight,'pair_stride':pair_stride,'pair_overlap':pair_overlap,
  'phase_balanced':args.phase_balanced,'priority_phases':args.priority_phases,
  'base_checkpoint':str(BASE_PARAMS.relative_to(ROOT)), 'note':'exploratory; consult review_decision.json; no resume support'}
 (output/'config.json').write_text(json.dumps(run_config,ensure_ascii=False,indent=2)+'\n')
 snapshot=output/'source_snapshot';snapshot.mkdir()
 for name in ('train_real_four_camera.py','real_four_camera_loader.py','real_four_camera_config.py','real_four_camera_pi0.py','real_training_support.py','real_task_c_phases.py'):
  shutil.copy2(ROOT/'scripts/a2_pi0'/name,snapshot/name)
 for name in ('manifest.json','norm_stats.json','review_decision.json'):shutil.copy2(candidate/name,output/name)
 logging.info('Loading pretrained model; training samples=%d, validation probes=%d',len(train),len(samples))
 params=load_parameters(cfg);abstract=nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed));graphdef=nnx.graphdef(abstract)
 if initial_adapter:
  model=nnx.merge(graphdef,params)
  state=nnx.state(model).filter(cfg.trainable_filter)
  expected=traverse_util.flatten_dict(state.to_pure_dict(),sep='/')
  with np.load(initial_adapter/'lora_params.npz',allow_pickle=False) as adapter:
   if set(adapter.files)!=set(expected):raise ValueError('initial adapter keys mismatch')
   flat={key:adapter[key] for key in adapter.files}
   if any(flat[key].shape!=value.shape or flat[key].dtype!=value.dtype for key,value in expected.items()):
    raise ValueError('initial adapter shape or dtype mismatch')
  state.replace_by_pure_dict(traverse_util.unflatten_dict(flat,sep='/'))
  nnx.update(model,state)
  params=nnx.state(model)
 tx=optimizer_lib.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None);opt_state=tx.init(params.filter(cfg.trainable_filter))
 step_fn=make_train_step(cfg,graphdef,tx,continuity_weight=args.continuity_weight,pair_stride=pair_stride,pair_overlap=pair_overlap)
 @jax.jit
 def evaluate(params,observation,actions,key):
  model=nnx.merge(graphdef,params)
  return jnp.mean(model.compute_loss(key,observation,actions,train=False))
 def validate(step):
  losses=[float(evaluate(params,*sample,jax.random.key(10000+i))) for i,sample in enumerate(samples)]
  if not np.isfinite(losses).all():raise RuntimeError('nonfinite validation loss')
  groups={g:float(np.mean([loss for loss,s in zip(losses,selected) if s['operation_style']==g])) for g in sorted({s['operation_style'] for s in selected})}
  record={'step':step,'mean_flow_loss':float(np.mean(losses)),'mean_flow_loss_by_operation_style':groups,'sample_losses':losses,'samples':selected}
  with (output/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  logging.info('validation step=%d loss=%.6f',step,record['mean_flow_loss']);return record
 initial=validate(0);rng=jax.random.key(args.seed+1);order_rng=np.random.default_rng(args.seed);cursor=0;started=time.monotonic();updates=0
 if args.phase_balanced:
  indices,phase_labels=phase_balanced_order(train,previous_indices,args.steps,order_rng,args.priority_phases)
 else:
  indices=order_rng.permutation(len(train))
  if initial_adapter:indices=indices[~np.isin(indices,list(previous_indices))]
 if initial_adapter:
  rng=jax.random.fold_in(rng,len(previous_indices))
 for step in range(1,args.steps+1):
  if cursor==len(indices):indices=order_rng.permutation(len(train));cursor=0
  index=int(indices[cursor]);cursor+=1
  episode,offset=train.locate(index)
  sample_indices=[index]
  if args.continuity_weight:
   count=train.entries[episode]['full_window_samples']
   if count<=pair_stride:raise ValueError('episode is too short for paired training')
   if offset+pair_stride<count:sample_indices.append(index+pair_stride)
   else:sample_indices.insert(0,index-pair_stride)
  batch=jax.device_put(batch_samples([train[i] for i in sample_indices]));rng,key=jax.random.split(rng);before=time.monotonic()
  params,opt_state,metrics=step_fn(params,opt_state,key,*batch);metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
  if not np.isfinite(list(metrics.values())).all():raise RuntimeError(f'nonfinite at step {step}: {metrics}')
  updates+=metrics['update_norm']>0
  record={'step':step,'sample_index':index,'episode_id':train.entries[episode]['episode_id'],**metrics,'step_seconds':time.monotonic()-before}
  if args.continuity_weight:record['paired_sample_indices']=sample_indices
  if args.phase_balanced:record['phase_at_forecast_frame_8']=PHASES[phase_labels[index]]
  with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  if step==1 or step%25==0:logging.info('step=%d/%d loss=%.6f grad=%.5f elapsed=%.1fs',step,args.steps,metrics['loss'],metrics['grad_norm'],time.monotonic()-started)
  if step%args.eval_every==0 or step==args.steps:final=validate(step)
  if step%args.save_every==0 or step==args.steps:
   path=save_adapter(checkpoints,step,params,opt_state,cfg,candidate);logging.info('saved %s',path)
 if not updates:raise RuntimeError('no nonzero parameter updates')
 restored=verify_adapter(path,params,graphdef,cfg,evaluate,samples[0]);logging.info('adapter restore check: %s',restored)
 stats=jax.devices()[0].memory_stats() or {}
 summary={'steps_completed':args.steps,'initial_validation_flow_loss':initial['mean_flow_loss'],'final_validation_flow_loss':final['mean_flow_loss'],
  'validation_scope':'two fixed probes per held-out episode; not a success-rate estimate','last_checkpoint':str(path.relative_to(ROOT)),
  'adapter_restore_check':restored,'gpu_peak_bytes':stats.get('peak_bytes_in_use'),'training_seconds':time.monotonic()-started,'task_success_verified':False}
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');logging.info('TRAINING COMPLETE %s',json.dumps(summary))
if __name__=='__main__':main()
