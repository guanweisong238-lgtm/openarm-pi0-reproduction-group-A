#!/usr/bin/env python3
"""Compare sampled 50-step actions to held-out recorded IK targets (not robot execution)."""
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
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--samples-per-episode',type=int,default=10)
 parser.add_argument('--num-steps',type=int,default=10)
 parser.add_argument('--out',type=Path,default=ROOT/'results/real_action_eval_20261121')
 parser.add_argument('--repeat-only',action='store_true')
 parser.add_argument('--name',default='real_pi0_four_camera_lora_001')
 args=parser.parse_args()
 if args.samples_per_episode<3 or args.num_steps<1:parser.error('invalid counts')
 if args.out.exists():raise FileExistsError(args.out)
 args.out.mkdir(parents=True)
 run=ROOT/'results'/args.name; config=json.loads((run/'config.json').read_text());candidate=Path(config.get('candidate',str(DEFAULT)))
 if sha(candidate/'manifest.json')!=config['candidate_manifest_sha256'] or sha(candidate/'norm_stats.json')!=config['norm_stats_sha256']:raise ValueError('training data differs')
 cfg=dataclasses.replace(get_train_config(),model=FourCameraPi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50),seed=config['seed'])
 ds=RealFourCameraDataset('validation',root=candidate,cache_episodes=4); probes=[]
 for e,row in enumerate(ds.entries):
  start=0 if e==0 else int(ds.ends[e-1]);n=row['full_window_samples']
  for j in np.linspace(0,n-1,args.samples_per_episode,dtype=int):probes.append({'episode_id':row['episode_id'],'operation_style':row.get('operation_style','unspecified'),'sample_index':int(j),'dataset_index':start+int(j),'relative_position':float(j/(n-1))})
 if args.repeat_only:probes=probes[:1]
 print('probes',len(probes),'loading pretrained model',flush=True);params=load_parameters(cfg);abstract=nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed));graph=nnx.graphdef(abstract);model=nnx.merge(graph,params)
 @jax.jit
 def sample(params,obs,key):return nnx.merge(graph,params).sample_actions(key,obs,num_steps=args.num_steps)
 stats=ds.stats['actions'];mean=np.asarray(stats.mean,dtype=np.float32);std=np.asarray(stats.std,dtype=np.float32)
 all_outputs={};metrics={};ground=[];states=[];check_hashes={};started=time.monotonic()
 for step in ((1000,) if args.repeat_only else (500,1000)):
  checkpoint=ROOT/'checkpoints'/args.name/f'step_{step:04d}';meta=json.loads((checkpoint/'checkpoint.json').read_text())
  if sha(checkpoint/'lora_params.npz')!=meta['lora_sha256'] or sha(checkpoint/'norm_stats.json')!=config['norm_stats_sha256']:raise ValueError('checkpoint hash mismatch')
  check_hashes[str(step)]=meta['lora_sha256'];state=nnx.state(model).filter(cfg.trainable_filter);expected=traverse_util.flatten_dict(state.to_pure_dict(),sep='/')
  with np.load(checkpoint/'lora_params.npz',allow_pickle=False) as f:
   if set(f.files)!=set(expected):raise ValueError('adapter parameter mismatch')
   flat={k:f[k] for k in f.files};state.replace_by_pure_dict(traverse_util.unflatten_dict(flat,sep='/'))
  nnx.update(model,state);params=nnx.state(model);pred=[]
  for i,p in enumerate(probes):
   raw=ds.get_raw(p['dataset_index']);obs,actions=jax.device_put(batch_samples([ds[p['dataset_index']]])); key=jax.random.key(50000+i)
   out=np.asarray(sample(params,obs,key))[0,:,:16];out=out*(std+1e-6)+mean
   if not np.isfinite(out).all():raise ValueError(f'nonfinite prediction at {p}')
   pred.append(out)
   if step==((1000,) if args.repeat_only else (500,1000))[0]:ground.append(raw['actions']);states.append(raw['state'])
   if i==0:
    repeat=np.asarray(sample(params,obs,key))[0,:,:16]*(std+1e-6)+mean
    p['repeat_max_abs_rad_'+str(step)]=float(np.max(np.abs(out-repeat)))
   if (i+1)%10==0 or i==0:print(f'checkpoint {step}: {i+1}/{len(probes)} elapsed {time.monotonic()-started:.1f}s',flush=True)
  p=np.stack(pred); target=np.stack(ground);initial=np.stack(states); err=np.abs(p-target); baseline=np.abs(initial[:,None,:]-target)
  per={str(e):float(err[[j for j,x in enumerate(probes) if x['episode_id']==e]].mean()) for e in sorted({x['episode_id'] for x in probes})}
  phases={}
  for name,condition in [('early',lambda x:x<1/3),('middle',lambda x:1/3<=x<2/3),('late',lambda x:x>=2/3)]:
   idx=[j for j,x in enumerate(probes) if condition(x['relative_position'])]
   if idx: phases[name]={'n':len(idx),'mae_rad':float(err[idx].mean())}
  grouped={}
  for g in sorted({x['operation_style'] for x in probes}):
   idx=[j for j,x in enumerate(probes) if x['operation_style']==g]
   grouped[g]={'windows':len(idx),'mae_rad':float(err[idx].mean()),'hold_baseline_mae_rad':float(baseline[idx].mean())}
  metrics[str(step)]={'mean_absolute_error_rad':float(err.mean()),'by_operation_style':grouped,'root_mean_square_error_rad':float(np.sqrt(np.mean((p-target)**2))),
   'mean_error_joints_rad':float(err[:,:,[*range(7),*range(8,15)]].mean()),'right_gripper_mae_rad':float(err[:,:,7].mean()),'left_gripper_mae_rad':float(err[:,:,15].mean()),
   'first_step_mae_rad':float(err[:,0,:].mean()),'last_step_mae_rad':float(err[:,-1,:].mean()),
   'per_joint_mae_rad':err.mean(axis=(0,1)).tolist(),'per_episode_mae_rad':per,'relative_time_thirds':phases,
   'hold_current_baseline_mae_rad':float(baseline.mean()),'repeat_max_abs_rad':probes[0]['repeat_max_abs_rad_'+str(step)]}
  all_outputs['prediction_'+str(step)]=p
  print(f'checkpoint {step} MAE {err.mean():.6f} rad; grippers right {err[:,:,7].mean():.6f}, left {err[:,:,15].mean():.6f}',flush=True)
 np.savez_compressed(args.out/'predictions.npz',target=np.stack(ground),observed_state=np.stack(states),**all_outputs)
 report={'protocol':'20?','num_samples':len(probes),'num_steps':args.num_steps,'seed_rule':'50000 + probe index, same noise for both checkpoints',
 'samples_per_episode':args.samples_per_episode,'probes':probes,'checkpoint_sha256':check_hashes,'metrics':metrics,
 'notes':['Previously monitored validation episodes; no new independent test set.','Targets are recorded IK targets before driver limits, not executed CAN commands.','Early/middle/late are equal relative time thirds, not annotated task phases.','No robot control, success rate or safety judgment.','Same process same-key repetition; cross-process comparison requires separate repeated run.']}
 report['protocol']='equally spaced full-window starts per validation episode; 10-step pi0 flow sampler; all 50 future targets evaluated'
 (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('done',args.out,flush=True)
if __name__=='__main__':main()
