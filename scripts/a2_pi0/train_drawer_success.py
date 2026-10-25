"""v003多轨迹LoRA训练：train-only更新，整轨迹验证，固定三种子动作和FK诊断。"""
from drawer_success_training_config import ROOT,MANIFEST,NORM_ROOT,get_success_config
import argparse
import dataclasses
from datetime import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--exp-name',default='success13_lora_001')
    parser.add_argument('--lr',type=float,default=1e-5,help='warmup后峰值学习率，末尾为其十分之一')
    parser.add_argument('--eval-every',type=int,default=500)
    parser.add_argument('--save-every',type=int,default=1000)
    parser.add_argument('--checkpoint',type=Path,help='从指定权重继续；优化器和本轮学习率日程重新初始化')
    parser.add_argument('--early-probability',type=float,default=0.,help='先均匀抽轨迹，再以此概率强制抽前early-frames帧；剩余概率抽全轨迹')
    parser.add_argument('--early-frames',type=int,default=75)
    parser.add_argument('--dense-approach-eval',action='store_true',help='增加前75帧固定检查点，单列早期/起点及原四帧指标')
    parser.add_argument('--prepare-only',action='store_true',help='CPU缓存与采样/输入检查，不加载模型')
    args=parser.parse_args()
    if args.steps<2 or min(args.eval_every,args.save_every)<1 or not np.isfinite(args.lr) or args.lr<=0:
        parser.error('训练参数不合法')
    if not 0<=args.early_probability<=1 or args.early_frames<1:
        parser.error('早期采样比例须在[0,1]且窗口为正')
    if not args.exp_name or args.exp_name in ('.','..') or any(c in args.exp_name for c in '/\\'):
        parser.error('实验名必须是单个目录名')
    from openpi.training import optimizer as optimizer_lib
    from drawer_success_cache import prepare
    from local_data_loader import collate_numpy
    from openpi.models import model as model_lib
    import jax
    cfg=dataclasses.replace(get_success_config(args.exp_name,args.steps),
        lr_schedule=optimizer_lib.CosineDecaySchedule(warmup_steps=min(100,max(1,args.steps//10)),
            peak_lr=args.lr,decay_steps=args.steps,decay_lr=args.lr/10))
    if not args.prepare_only and cfg.checkpoint_dir.exists():raise FileExistsError(cfg.checkpoint_dir)
    if not args.prepare_only and jax.default_backend()!='gpu':raise RuntimeError('训练需要用户终端的JAX GPU；CPU请用--prepare-only')
    cached={split:prepare(cfg,split) for split in ('train','validation')}
    def batch(split,index):
        value=collate_numpy([cached[split][int(index)]])
        actions=value.pop('actions')
        return model_lib.Observation.from_dict(value),actions
    from drawer_early_sampling import make_schedule
    schedule=make_schedule(cached['train'].rows,args.steps,cfg.seed,args.early_probability,args.early_frames)
    if args.checkpoint:
        args.checkpoint=args.checkpoint.resolve()
        source_protocol=json.loads((args.checkpoint/'input_protocol.json').read_text())
        for key,path in (('manifest_sha256',MANIFEST),('norm_stats_sha256',NORM_ROOT/'norm_stats.json')):
            if source_protocol[key]!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('续训检查点数据/统计不匹配：'+key)
        if not (args.checkpoint/'params').is_dir():raise FileNotFoundError(args.checkpoint/'params')
    # 保留原四帧，另加密起步0～74帧；两组对照使用完全相同的评估帧。
    anchors={}
    for split,cache in cached.items():
        rows=[];start=0
        for episode,row in enumerate(cache.rows):
            legacy=np.unique(np.linspace(0,row['frames']-1,4,dtype=int))
            dense=np.array([0,1,4,8,15,30,45,60,74]) if args.dense_approach_eval else np.array([],dtype=int)
            for local in np.union1d(legacy,dense[dense<row['frames']]):
                rows.append({'episode':episode,'episode_id':row['episode_id'],'local':int(local),
                             'index':start+int(local),'valid':min(16,row['frames']-int(local))})
                rows[-1].update(initialization=row['initialization_group'],legacy_anchor=bool(local in legacy),
                                early_anchor=bool(local<75),start_anchor=bool(local==0))
            start+=row['frames']
        anchors[split]=rows
    # 实际验证缓存中的首末动作、跨轨迹索引、split隔离和模型输入。
    from openpi.shared import normalize
    stats=normalize.load(NORM_ROOT)['actions'];scale=np.asarray(stats.std)+1e-6;mean=np.asarray(stats.mean)
    for split,cache in cached.items():
        starts=np.r_[0,cache.ends[:-1]]
        for episode,row in enumerate(cache.rows):
            with np.load(ROOT/row['path']/'episode.npz') as source:
                for local in (0,row['frames']-1):
                    index=int(starts[episode])+local
                    obs,actions=batch(split,index)
                    target=source['action'][np.minimum(local+np.arange(16),row['frames']-1)]
                    np.testing.assert_allclose(actions[0,:,:16]*scale+mean,target,atol=2e-6,rtol=1e-6)
                    assert obs.state.shape==(1,32) and actions.shape==(1,16,32)
                    assert cache.locate(index)==(episode,local)
        print(f'INPUT_PASS {split} episodes={len(cache.rows)} frames={len(cache)} anchors={len(anchors[split])}',flush=True)
    train_ids={r['episode_id'] for r in cached['train'].rows}
    val_ids={r['episode_id'] for r in cached['validation'].rows}
    assert train_ids.isdisjoint(val_ids)
    # 只从train抽episode，再在所属episode抽frame。
    rng=np.random.default_rng(cfg.seed)
    counts=np.zeros(len(cached['train'].rows),int)
    for _ in range(10000):counts[int(rng.integers(len(counts)))]+=1
    print('SAMPLING uniform episode -> uniform frame; simulated episode counts',counts.tolist(),flush=True)
    sampling={'early_probability':args.early_probability,'early_frames':args.early_frames,
        'forced_early_updates':sum(forced for _,_,forced in schedule),
        'actual_first15_updates':sum(local<15 for _,local,_ in schedule),
        'actual_early_updates':sum(local<args.early_frames for _,local,_ in schedule),
        'episode_counts':[sum(e==i for e,_,_ in schedule) for i in range(len(cached['train'].rows))]}
    print('ACTUAL_SCHEDULE '+json.dumps(sampling),flush=True)
    if args.prepare_only:
        print('PASS CPU预检；未初始化模型、未训练、未保存检查点',flush=True)
        return
    import flax.nnx as nnx
    from train_smoke import load_parameters,make_train_step,save_checkpoint
    from drawer_training_objective import flow_squared_errors
    from drawer_simulation import DrawerSimulation
    import mujoco
    output=ROOT/'logs/a2_pi0'/datetime.now().strftime('drawer_success_%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'dataset_version':'openarm_drawer_success_v003','manifest':str(MANIFEST),
        'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        'norm_stats_sha256':hashlib.sha256((NORM_ROOT/'norm_stats.json').read_bytes()).hexdigest(),
        'image_source':'original_npz_uint8_rgb',
        'initialization':('checkpoint weights, fresh optimizer and learning rate schedule' if args.checkpoint else 'pi0_base, fresh LoRA and optimizer'),
        'source_checkpoint':str(args.checkpoint) if args.checkpoint else None,
        'sampling':'uniform episode; early-window/full-episode mixture','sampling_details':sampling,
        'dense_approach_eval':args.dense_approach_eval,'objective':'all32 upstream flow loss',
        'train_episode_ids':sorted(train_ids),'validation_episode_ids':sorted(val_ids),
        'anchors':anchors,'closed_loop_tested':False}
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    (output/'config.txt').write_text(repr(cfg)+'\n')
    print('RUN_DIRECTORY='+str(output),flush=True)
    params=(nnx.state(cfg.model.load(model_lib.restore_params(args.checkpoint/'params')))
            if args.checkpoint else load_parameters(cfg))
    graphdef=nnx.graphdef(nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed)))
    tx=optimizer_lib.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None)
    opt_state=tx.init(params.filter(cfg.trainable_filter))
    update=make_train_step(cfg,graphdef,tx)
    @jax.jit
    def flow(current,obs,actions,key):
        return flow_squared_errors(nnx.merge(graphdef,current),key,obs,actions).mean((0,1))
    @jax.jit
    def sample(current,obs,noise):
        return nnx.merge(graphdef,current).sample_actions(jax.random.key(0),obs,noise=noise,num_steps=10)
    noises={seed:np.random.default_rng(seed).standard_normal((1,16,32)).astype(np.float32) for seed in (42,7,99)}
    # 每条轨迹使用自己的场景和原始物理状态，避免把warm-start与ready-start混成同一初态。
    environments={};qposes={}
    records=[]
    try:
        for cache in cached.values():
            for row in cache.rows:
                eid=row['episode_id'];env=DrawerSimulation(ROOT/row['path'])
                environments[eid]=(env,mujoco.MjData(env.model))
                with np.load(ROOT/row['path']/'episode.npz') as e:qposes[eid]=e['qpos'].copy()
        def tip(eid,local,action):
            env,scratch=environments[eid]
            scratch.qpos[:]=qposes[eid][local];scratch.qpos[env.qpos_ids]=action
            mujoco.mj_kinematics(env.model,scratch)
            site=env.model.site('right_ee_control_point').id
            rotation=scratch.site_xmat[site].reshape(3,3).copy()
            return scratch.site_xpos[site]+rotation@np.array([0.,0.,-.16]),rotation
        def evaluate(step):
            record={'step':step,'scope':'fixed held-out/training anchors, not task success','splits':{}}
            for split,rows in anchors.items():
                diagnostics=[]
                for a in rows:
                    obs,target=batch(split,a['index'])
                    values=np.asarray(flow(params,obs,target,jax.random.key(12000+a['episode_id']*10000+a['local'])))
                    if not np.isfinite(values).all():raise RuntimeError('flow诊断出现非有限值')
                    truth=np.asarray(target)[0,:,:16]*scale+mean
                    target_tip,target_rotation=tip(a['episode_id'],a['local'],truth[0])
                    by_seed={}
                    for seed,noise in noises.items():
                        pred=np.asarray(sample(params,obs,noise))[0,:,:16]*scale+mean
                        errors=np.abs(pred[:a['valid']]-truth[:a['valid']])
                        point,rotation=tip(a['episode_id'],a['local'],pred[0])
                        by_seed[str(seed)]={'right_arm_mae_rad':float(errors[:,:7].mean()),
                            'first_action_right_arm_mae_rad':float(errors[0,:7].mean()),
                            'right_gripper_mae_rad':float(errors[:,7].mean()),
                            'left_arm_gripper_mae_rad':float(errors[:,8:16].mean()),
                            'first_action_position_mm':float(np.linalg.norm(point-target_tip)*1000),
                            'first_action_attitude_deg':float(np.degrees(np.arccos(np.clip((np.trace(rotation.T@target_rotation)-1)/2,-1,1))))}
                    diagnostics.append(dict(a,flow_all32=float(values.mean()),flow_right_arm=float(values[:7].mean()),
                        flow_gripper=float(values[7]),flow_padding=float(values[16:].mean()),by_seed=by_seed))
                aggregate={seed:{key:float(np.mean([r['by_seed'][seed][key] for r in diagnostics]))
                                for key in diagnostics[0]['by_seed'][seed]} for seed in ('42','7','99')}
                record['splits'][split]={'flow_all32':float(np.mean([r['flow_all32'] for r in diagnostics])),
                    'by_seed':aggregate,'anchors':diagnostics}
                slices={}
                for name in ('legacy_anchor','early_anchor','start_anchor'):
                    for initialization in ('all','script_ready_start','human_warm_start'):
                        selected=[r for r in diagnostics if r[name] and (initialization=='all' or r['initialization']==initialization)]
                        if selected:
                            slices[name+'/'+initialization]={key:float(np.mean([r['by_seed'][str(seed)][key] for r in selected for seed in (42,7,99)]))
                                for key in diagnostics[0]['by_seed']['42']}
                record['splits'][split]['slices']=slices
                arm=np.mean([v['right_arm_mae_rad'] for v in aggregate.values()])
                position=np.mean([v['first_action_position_mm'] for v in aggregate.values()])
                print(f'EVAL step={step} {split} arm_MAE={arm:.6f} first_position_mm={position:.3f}',flush=True)
                if args.dense_approach_eval:
                    for group in ('script_ready_start','human_warm_start'):
                        early=slices['early_anchor/'+group];start=slices['start_anchor/'+group]
                        print(f"EARLY step={step} {split} {group} early_pos_mm={early['first_action_position_mm']:.3f} start_pos_mm={start['first_action_position_mm']:.3f}",flush=True)
            if not all(np.isfinite(v) for s in record['splits'].values() for r in s['anchors'] for m in r['by_seed'].values() for v in m.values()):
                raise RuntimeError('诊断出现非有限值')
            with (output/'evaluation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            records.append(record)
        evaluate(0)
        key=jax.random.key(cfg.seed+1);started=time.perf_counter()
        train=cached['train'];starts=np.r_[0,train.ends[:-1]]
        for step in range(1,args.steps+1):
            episode,local,forced=schedule[step-1]
            index=int(starts[episode])+local;assert train.locate(index)==(episode,local)
            key,subkey=jax.random.split(key)
            params,opt_state,metrics=update(params,opt_state,subkey,*jax.device_put(batch('train',index)))
            metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
            if not all(np.isfinite(v) for v in metrics.values()) or metrics['update_norm']==0:raise RuntimeError(metrics)
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step,episode_id=train.rows[episode]['episode_id'],local_frame=local,forced_early=forced,**metrics))+'\n')
            if step==1 or step%100==0:print(f"step={step} loss={metrics['loss']:.6f} grad_norm={metrics['grad_norm']:.4f} elapsed={time.perf_counter()-started:.0f}s",flush=True)
            if step%args.eval_every==0 or step==args.steps:evaluate(step)
            if step%args.save_every==0 or step==args.steps:
                checkpoint=save_checkpoint(cfg,params,opt_state,cfg.data.create(cfg.assets_dirs,cfg.model),step=step)
                (checkpoint/'input_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
                (checkpoint/'dataset_manifest.json').write_text(MANIFEST.read_text())
                print('CHECKPOINT='+str(checkpoint),flush=True)
        summary={'steps':args.steps,'checkpoint':str(checkpoint),'initial_evaluation':records[0],
                 'final_evaluation':records[-1],'held_out_episodes':True,'closed_loop_tested':False,'task_success_verified':False}
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print('PASS 训练执行完成；任务成功尚未验证',flush=True)
    finally:
        for env,_ in environments.values():env.close()


if __name__=='__main__':main()
