"""Pi0骨干的固定噪声一步动作回归；实验改造版本，独立记录原flow差异。"""
from drawer_success_training_config import ROOT,MANIFEST,NORM_ROOT,get_success_config
import argparse
import dataclasses
from datetime import datetime
import hashlib
import functools
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--exp-name',default='pull12_endpoint_016')
    parser.add_argument('--lr',type=float,default=3e-6,help='warmup后峰值学习率，末尾为其十分之一')
    parser.add_argument('--eval-every',type=int,default=300)
    parser.add_argument('--save-every',type=int,default=600)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/pi0_openarm_drawer_velocity_v001/pull12_velocity_015/2000',help='从指定权重继续；优化器和本轮学习率日程重新初始化')
    parser.add_argument('--phases',type=Path,default=ROOT/'results/drawer_approach_phases_v001/phases.json')
    parser.add_argument('--prepare-only',action='store_true',help='CPU缓存与采样/输入检查，不加载模型')
    parser.add_argument('--focus-episode',type=int,help='只对指定train轨迹做完整拉取拟合；验证集不参与梯度')
    parser.add_argument('--include-pull',action='store_true',help='扩展到真实拉开8cm保持1秒后的15帧，仍使用原始动作标签')
    parser.add_argument('--skip-initial-eval',action='store_true',help='仅用于已有源模型诊断的续训，仍强制结束时评估')
    parser.add_argument('--recovery-manifest',type=Path,help='独立审核的专家恢复片段清单；不修改v003')
    parser.add_argument('--recovery-probability',type=float,default=.25)
    parser.add_argument('--skip-script-dwells',action='store_true',help='集中拟合时不抽脚本的0.5秒中间等待段；动作标签和末尾拉开保持段不变')
    args=parser.parse_args()
    if args.steps<2 or min(args.eval_every,args.save_every)<1 or not np.isfinite(args.lr) or args.lr<=0:
        parser.error('训练参数不合法')
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
    recovery=None
    if args.recovery_manifest:
        if not 0<args.recovery_probability<=1:raise ValueError('恢复概率应在0和1之间')
        from drawer_recovery_cache import prepare_recovery
        recovery=prepare_recovery(cfg,args.recovery_manifest)
    # 缓存始终由原绝对动作配置构建；残差转换只在运行时进行。
    phase_doc=json.loads(args.phases.read_text())
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if phase_doc['manifest_sha256']!=digest(MANIFEST):raise ValueError('阶段清单版本不匹配')
    phases={r['episode_id']:r for r in phase_doc['episodes']}
    assert set(phases)=={r['episode_id'] for r in json.loads(MANIFEST.read_text())['episodes']}
    source_protocol=json.loads((args.checkpoint/'input_protocol.json').read_text())
    from drawer_start_fit_config import get_stats_location
    norm_root,_=get_stats_location(source_protocol)
    if source_protocol.get('action_representation')!='right7_delta':raise ValueError('需要残差检查点')
    for key,path in [('manifest_sha256',MANIFEST),('norm_stats_sha256',norm_root/'norm_stats.json')]:
        if source_protocol[key]!=digest(path):raise ValueError('数据或统计版本不匹配')
    from prepare_drawer_velocity import prepare_velocity
    from drawer_velocity_config import VELOCITY_NORM_ROOT,VelocityDeltaSuccessDataConfig
    velocity_banks,velocity_metadata=prepare_velocity()
    for split,cache in cached.items():
        assert [r['episode_id'] for r in cache.rows]==velocity_metadata['splits'][split]['episode_ids']
        assert len(cache)==len(velocity_banks[split])
    recovery_velocities={}
    if recovery is not None:
        from drawer_simulation import DrawerSimulation
        for episode,row in enumerate(recovery.rows):
            env=DrawerSimulation(ROOT/row['path'])
            with np.load(ROOT/row['path']/'episode.npz') as data:
                recovery_velocities[episode]=data['qvel'][:,env.velocity_ids].copy()
            env.close()
    norm_root=VELOCITY_NORM_ROOT
    from openpi.shared import normalize,nnx_utils
    import flax.nnx as nnx
    import jax.numpy as jnp
    import optax
    allowed=nnx_utils.PathRegex(r'.*lora.*|(?:state_proj|action_in_proj|action_out_proj)/.*')
    cfg=dataclasses.replace(cfg,name='pi0_openarm_drawer_endpoint_v001',
        data=VelocityDeltaSuccessDataConfig(),freeze_filter=nnx.Not(allowed))
    if not args.prepare_only and cfg.checkpoint_dir.exists():raise FileExistsError(cfg.checkpoint_dir)
    absolute=normalize.load(NORM_ROOT);am=np.asarray(absolute['actions'].mean);asc=np.asarray(absolute['actions'].std)+1e-6
    sm=np.asarray(absolute['state'].mean);ssc=np.asarray(absolute['state'].std)+1e-6
    stats=normalize.load(norm_root)['actions'];mean=np.asarray(stats.mean);scale=np.asarray(stats.std)+1e-6
    velocity_stats=normalize.load(norm_root)['state']
    vm=np.asarray(velocity_stats.mean)[16:];vs=np.asarray(velocity_stats.std)[16:]+1e-6
    def batch(split,index):
        value=collate_numpy([(recovery if split=='recovery' else cached[split])[int(index)]])
        current=value['state'][:,:16]*ssc+sm
        actions=value['actions'][:,:,:16]*asc+am;actions[:,:,:7]-=current[:,None,:7]
        value['actions'][:,:,:16]=((actions-mean)/scale).astype(np.float32)
        value['state'][:,8:16]=0
        if split=='recovery':
            episode,packed=recovery.locate(int(index));row=recovery.rows[episode]
            actual=row.get('source_frame_indices',list(range(row['frames'])))[packed]
            velocity=recovery_velocities[episode][actual]
        else:velocity=velocity_banks[split][int(index)]
        value['state'][:,16:32]=((velocity-vm)/vs).astype(np.float32)
        value['state'][:,24:32]=0
        target=value.pop('actions');return model_lib.Observation.from_dict(value),target
    def decode(target,split,index):
        current=(recovery if split=='recovery' else cached[split])[int(index)]['state'][:16]*ssc+sm
        value=np.asarray(target)[0,:,:16]*scale+mean;value[:,:7]+=current[:7]
        return value
    from drawer_approach_sampling import make_schedule
    schedule=make_schedule(cached['train'].rows,phases,args.steps,cfg.seed)
    if args.focus_episode is not None and args.focus_episode not in {r['episode_id'] for r in cached['train'].rows}:
        raise ValueError('focus必须属于train')
    pull_ends={}
    if args.include_pull:
        for eid,p in phases.items():
            events=np.load(args.phases.parent/p['event_arrays'])
            opening=events['opening']
            holds=[i for i in range(len(opening)-29) if np.all(opening[i:i+30]>=.08)]
            if not holds:raise ValueError(f'ID{eid}没有8cm保持1秒事件')
            pull_ends[eid]=min(p['frames'],holds[0]+30+15)
        rng=np.random.default_rng(cfg.seed);schedule=[]
        eligible=[i for i,r in enumerate(cached['train'].rows) if args.focus_episode is None or r['episode_id']==args.focus_episode]
        for _ in range(args.steps):
            episode=int(rng.choice(eligible));eid=cached['train'].rows[episode]['episode_id'];p=phases[eid]
            stage=str(rng.choice(['start','approach','grasp','pull'],p=[.15,.4,.2,.25]))
            low,high=p['phase_ranges'][stage] if stage!='pull' else (p['phase_end_exclusive'],pull_ends[eid])
            if high<=low:low,high=p['first_stable_contact_frame'],pull_ends[eid]
            schedule.append((episode,int(rng.integers(low,high)),stage))
    elif args.focus_episode is not None:
        raise ValueError('focus实验需要--include-pull')
    excluded_dwells=[]
    if args.skip_script_dwells:
        if args.focus_episode is None:raise ValueError('等待段消融需要明确focus轨迹')
        report=json.loads((ROOT/phases[args.focus_episode]['path']/'report.json').read_text())
        if not report.get('stages'):raise ValueError('该轨迹没有脚本阶段元数据')
        excluded_dwells=[(s['first_frame'],s['first_frame']+s['frames']) for s in report['stages']
            if np.isclose(s['duration'],.5) and s['first_frame']<pull_ends[args.focus_episode]]
        rng=np.random.default_rng(cfg.seed+18000);filtered=[]
        for episode,local,stage in schedule:
            p=phases[args.focus_episode]
            low,high=p['phase_ranges'][stage] if stage!='pull' else (p['phase_end_exclusive'],pull_ends[args.focus_episode])
            eligible=[i for i in range(low,high) if not any(a<=i<b for a,b in excluded_dwells)]
            if not eligible:raise ValueError('移除等待段后阶段没有训练帧')
            if any(a<=local<b for a,b in excluded_dwells):local=int(rng.choice(eligible))
            filtered.append((episode,local,stage))
        schedule=filtered
        assert not any(a<=local<b for _,local,_ in schedule for a,b in excluded_dwells)
        print('EXCLUDED_INTERMEDIATE_DWELLS '+json.dumps(excluded_dwells),flush=True)
    anchors={}
    for split,cache in cached.items():
        rows=[];start=0
        for episode,row in enumerate(cache.rows):
            contact=phases[row['episode_id']]['first_stable_contact_frame'];end=phases[row['episode_id']]['phase_end_exclusive']
            locals_=[0,4,8,contact//2,contact-1,contact+4,end-1]
            if row['episode_id']==12:locals_+=list(range(9))
            if args.include_pull:
                locals_+=(list(range(0,pull_ends[row['episode_id']],12)) if split=='train' else [(end+pull_ends[row['episode_id']])//2])
                locals_+=[pull_ends[row['episode_id']]-1]
            if split=='train' and args.focus_episode is not None and row['episode_id']!=args.focus_episode:
                start+=row['frames'];continue
            for local in sorted(set(locals_)):
                stage='start' if local<9 else 'approach' if local<contact else 'grasp' if local<end else 'pull'
                rows.append(dict(episode=episode,episode_id=row['episode_id'],local=int(local),index=start+int(local),
                    valid=min(16,row['frames']-local),initialization=row['initialization_group'],
                    start_anchor=stage=='start',approach_anchor=stage=='approach',grasp_anchor=stage=='grasp',pull_anchor=stage=='pull'))
            start+=row['frames']
        anchors[split]=rows
    # 实际验证缓存中的首末动作、跨轨迹索引、split隔离和模型输入。
    for split,cache in cached.items():
        starts=np.r_[0,cache.ends[:-1]]
        for episode,row in enumerate(cache.rows):
            with np.load(ROOT/row['path']/'episode.npz') as source:
                for local in sorted(set([0,row['frames']-1,phases[row['episode_id']]['first_stable_contact_frame'],phases[row['episode_id']]['phase_end_exclusive']-1])):
                    index=int(starts[episode])+local
                    obs,actions=batch(split,index)
                    target=source['action'][np.minimum(local+np.arange(16),row['frames']-1)]
                    np.testing.assert_allclose(decode(actions,split,index),target,atol=3e-6,rtol=1e-6)
                    assert obs.state.shape==(1,32) and actions.shape==(1,16,32)
                    assert cache.locate(index)==(episode,local)
        print(f'INPUT_PASS {split} episodes={len(cache.rows)} frames={len(cache)} anchors={len(anchors[split])}',flush=True)
    train_ids={r['episode_id'] for r in cached['train'].rows}
    val_ids={r['episode_id'] for r in cached['validation'].rows}
    assert train_ids.isdisjoint(val_ids)
    recovery_schedule={}
    if recovery is not None:
        rng=np.random.default_rng(cfg.seed+9000);starts=np.r_[0,recovery.ends[:-1]]
        for step in range(args.steps):
            if rng.random()<args.recovery_probability:
                episode=int(rng.integers(len(recovery.rows)));row=recovery.rows[episode]
                local=int(rng.integers(row['frames']))
                recovery_schedule[step]=(int(starts[episode])+local,episode,local)
        for episode,row in enumerate(recovery.rows):
            with np.load(ROOT/row['path']/'episode.npz') as source:
                source_frames=row.get('source_frame_indices',list(range(row['frames'])))
                boundaries={0,row['frames']-1}
                for i in range(1,len(source_frames)):
                    if source_frames[i]!=source_frames[i-1]+1:boundaries.update((i-1,i))
                for local in sorted(boundaries):
                    index=int(starts[episode])+local;obs,target=batch('recovery',index)
                    actual_local=source_frames[local]
                    expected=source['action'][np.minimum(actual_local+np.arange(16),len(source['action'])-1)]
                    np.testing.assert_allclose(decode(target,'recovery',index),expected,atol=3e-6,rtol=1e-6)
        print(f'RECOVERY_INPUT_PASS episodes={len(recovery.rows)} frames={len(recovery)} scheduled_updates={len(recovery_schedule)}',flush=True)
    # 只从train抽episode，再在所属episode抽frame。
    rng=np.random.default_rng(cfg.seed)
    counts=np.zeros(len(cached['train'].rows),int)
    for _ in range(10000):counts[int(rng.integers(len(counts)))]+=1
    print('SAMPLING uniform episode -> stage mixture; simulated episode counts',counts.tolist(),flush=True)
    original_updates=[item for i,item in enumerate(schedule) if i not in recovery_schedule]
    sampling={'phase_probabilities':({'start':.15,'approach':.4,'grasp':.2,'pull':.25} if args.include_pull else {'start':.3,'approach':.35,'grasp':.35}),
        'phase_counts':{p:sum(stage==p for _,_,stage in original_updates) for p in ('start','approach','grasp','pull')},
        'episode_counts':[sum(e==i for e,_,_ in original_updates) for i in range(len(cached['train'].rows))],
        'expert_recovery_updates':len(recovery_schedule)}
    print('ACTUAL_SCHEDULE '+json.dumps(sampling),flush=True)
    if args.prepare_only:
        print('PASS CPU预检；未初始化模型、未训练、未保存检查点',flush=True)
        return
    import flax.nnx as nnx
    from train_smoke import save_checkpoint
    from drawer_training_objective import flow_squared_errors
    from drawer_simulation import DrawerSimulation
    import mujoco
    output=ROOT/'logs/a2_pi0'/datetime.now().strftime('drawer_success_%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True,exist_ok=False)
    source_dir=output/'source';source_dir.mkdir()
    source_hashes={}
    for name in (Path(__file__).name,'drawer_training_objective.py','drawer_delta_config.py',
                 'drawer_start_fit_config.py','drawer_approach_sampling.py','drawer_recovery_cache.py',
                 'openarm_policy.py','drawer_success_training_config.py','local_data_loader.py','train_smoke.py',
                 'drawer_velocity_config.py','prepare_drawer_velocity.py'):
        original=Path(__file__).parent/name
        (source_dir/name).write_bytes(original.read_bytes());source_hashes[name]=digest(original)
    protocol={'dataset_version':'openarm_drawer_success_v003','manifest':str(MANIFEST),
        'source_snapshot_sha256':source_hashes,
        'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        'norm_stats_sha256':digest(norm_root/'norm_stats.json'),'norm_stats_root':str(norm_root),
        'action_representation':'right7_delta','mask_left_normalized_state':True,'train_io_projections':True,
        'state_representation':'position_velocity16','velocity_source':'actual qvel at observation time, actuator order',
        'velocity_bank_identity':velocity_metadata['identity'],
        'velocity_columns_zero_initialized':source_protocol.get('state_representation')!='position_velocity16',
        'recommended_noise_mode':'fixed','recommended_action_chunk':4,
        'time_sampling':'fixed_endpoint','augmentation':False,'first_four_weight':4.,'right8_loss':True,
        'inference_kind':'fixed_noise_endpoint','sampling_steps':1,'fixed_noise_seed':42,
        'phase_manifest':str(args.phases.resolve()),'phase_manifest_sha256':digest(args.phases),
        'image_source':'original_npz_uint8_rgb',
        'initialization':('checkpoint weights, fresh optimizer and learning rate schedule' if args.checkpoint else 'pi0_base, fresh LoRA and optimizer'),
        'source_checkpoint':str(args.checkpoint) if args.checkpoint else None,
        'sampling':'uniform episode; stage mixture','sampling_details':sampling,
        'actual_gradient_episode_ids':sorted({cached['train'].rows[e]['episode_id'] for e,_,_ in schedule}),
        'include_pull':args.include_pull,'pull_end_exclusive':pull_ends,
        'initial_evaluation_skipped':args.skip_initial_eval,
        'recovery_manifest':str(args.recovery_manifest.resolve()) if args.recovery_manifest else None,
        'recovery_manifest_sha256':digest(args.recovery_manifest) if args.recovery_manifest else None,
        'recovery_updates':len(recovery_schedule),
        'excluded_original_dwell_ranges':excluded_dwells,
        'recovery_evaluation_scope':'original v003 anchors only; actual recovery learned behavior requires new closed-loop testing',
        'objective':'right8 deterministic fixed-noise endpoint regression /32, first4 weight4; continuous demonstrations',
        'train_episode_ids':sorted(train_ids),'validation_episode_ids':sorted(val_ids),
        'anchors':anchors,'closed_loop_tested':False}
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    (output/'config.txt').write_text(repr(cfg)+'\n')
    print('RUN_DIRECTORY='+str(output),flush=True)
    params=nnx.state(cfg.model.load(model_lib.restore_params(args.checkpoint/'params')))
    params=nnx_utils.state_map(params,nnx_utils.PathRegex(r'(?:state_proj|action_in_proj|action_out_proj)/.*'),lambda v:v.replace(v.value.astype(jnp.float32)))
    source_params=params
    if source_protocol.get('state_representation')!='position_velocity16':
        assert params.to_pure_dict()['state_proj']['kernel'].shape[0]==32
        params=nnx_utils.state_map(params,nnx_utils.PathRegex(r'state_proj/kernel'),
            lambda v:v.replace(v.value.at[16:32,:].set(0)))
        assert not np.asarray(params.to_pure_dict()['state_proj']['kernel'][16:32]).any()
    from flax import traverse_util
    trainable=traverse_util.flatten_dict(params.filter(cfg.trainable_filter).to_pure_dict(),sep='/')
    assert {k.split('/')[0] for k in trainable if 'lora' not in k}=={'state_proj','action_in_proj','action_out_proj'}
    assert all(v.dtype==jnp.float32 for v in trainable.values())
    print('TRAINABLE_PARAMETERS',sum(v.size for v in trainable.values()),flush=True)
    del trainable
    graphdef=nnx.graphdef(nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed)))
    tx=optimizer_lib.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None)
    opt_state=tx.init(params.filter(cfg.trainable_filter))
    @functools.partial(jax.jit,donate_argnums=(0,1))
    def update(params,opt_state,key,obs,target):
        model=nnx.merge(graphdef,params);model.train()
        def loss(m):
            errors,t=flow_squared_errors(m,key,obs,target,augment=False,time_sampling='fixed_endpoint',return_time=True)
            return ((errors*jnp.array([4.]*4+[1.]*12)[None,:,None])[...,:8].sum(-1)/32).mean(),t.mean()
        (value,t),grads=nnx.value_and_grad(loss,argnums=nnx.DiffState(0,cfg.trainable_filter),has_aux=True)(model)
        old=params.filter(cfg.trainable_filter);updates,next_opt=tx.update(grads,opt_state,old)
        new=optax.apply_updates(old,updates);delta=jax.tree.map(lambda a,b:a-b,new,old);nnx.update(model,new)
        return nnx.state(model),next_opt,dict(loss=value,sampled_time=t,grad_norm=optax.global_norm(grads),update_norm=optax.global_norm(delta))
    @jax.jit
    def flow(current,obs,actions,key):
        return flow_squared_errors(nnx.merge(graphdef,current),key,obs,actions,time_sampling='fixed_endpoint').mean((0,1))
    @jax.jit
    def sample(current,obs,noise):
        return nnx.merge(graphdef,current).sample_actions(jax.random.key(0),obs,noise=noise,num_steps=1)
    noises={seed:np.random.default_rng(seed).standard_normal((1,16,32)).astype(np.float32) for seed in (42,)}
    if source_protocol.get('state_representation')!='position_velocity16':
        probe,_=batch('train',0)
        old_probe=dataclasses.replace(probe,state=np.asarray(probe.state).copy())
        old_probe.state[:,16:32]=0
        before=np.asarray(sample(source_params,old_probe,noises[42]))
        after=np.asarray(sample(params,probe,noises[42]))
        difference=float(np.max(np.abs(before-after)))
        if difference>1e-4:raise RuntimeError(f'速度通道初始化改变了源模型函数: {difference}')
        protocol['velocity_bootstrap_max_normalized_action_difference']=difference
        (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
        print('VELOCITY_BOOTSTRAP_FUNCTION_PARITY',difference,flush=True)
    del source_params

    probe,target=batch('recovery',0)
    endpoint_errors,full_prediction=flow_squared_errors(nnx.merge(graphdef,params),jax.random.key(0),
        probe,target,time_sampling='fixed_endpoint',return_prediction=True)
    sampled=np.asarray(sample(params,probe,noises[42]));difference=float(np.max(np.abs(sampled-np.asarray(full_prediction))))
    protocol['endpoint_full_vs_cached_prediction_max_difference']=difference
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print('ENDPOINT_TRAIN_INFER_NUMERICAL_DIFFERENCE',difference,flush=True)
    if difference>.1:raise RuntimeError('训练与一步推理预测不一致，需要检查实现')
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
                    truth=decode(target,split,a['index'])
                    target_tip,target_rotation=tip(a['episode_id'],a['local'],truth[0])
                    by_seed={}
                    for seed,noise in noises.items():
                        pred=decode(sample(params,obs,noise),split,a['index'])
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
                                for key in diagnostics[0]['by_seed'][seed]} for seed in ('42',)}
                record['splits'][split]={'flow_all32':float(np.mean([r['flow_all32'] for r in diagnostics])),
                    'by_seed':aggregate,'anchors':diagnostics}
                slices={}
                for name in (('start_anchor','approach_anchor','grasp_anchor','pull_anchor') if args.include_pull else ('start_anchor','approach_anchor','grasp_anchor')):
                    for initialization in ('all','script_ready_start','human_warm_start'):
                        selected=[r for r in diagnostics if r[name] and (initialization=='all' or r['initialization']==initialization)]
                        if selected:
                            slices[name+'/'+initialization]={key:float(np.mean([r['by_seed'][str(seed)][key] for r in selected for seed in (42,)]))
                                for key in diagnostics[0]['by_seed']['42']}
                record['splits'][split]['slices']=slices
                record['splits'][split]['fixed_right8_mse_over32']=float(np.mean([(r['flow_right_arm']*7+r['flow_gripper'])/32 for r in diagnostics]))
                arm=np.mean([v['right_arm_mae_rad'] for v in aggregate.values()])
                position=np.mean([v['first_action_position_mm'] for v in aggregate.values()])
                print(f'EVAL step={step} {split} arm_MAE={arm:.6f} first_position_mm={position:.3f}',flush=True)
                for name in (('start_anchor','approach_anchor','grasp_anchor','pull_anchor') if args.include_pull else ('start_anchor','approach_anchor','grasp_anchor')):
                    values=slices[name+'/all']
                    print(f"PHASE_EVAL step={step} {split} {name} position_mm={values['first_action_position_mm']:.3f} attitude_deg={values['first_action_attitude_deg']:.3f}",flush=True)
                if split=='train':
                    startup=[r['by_seed'][str(seed)] for r in diagnostics if r['episode_id']==12 and r['local']<9 for seed in (42,)]
                    record['splits'][split]['retained_id12_start_gate']={
                        'worst_position_mm':max(v['first_action_position_mm'] for v in startup),
                        'worst_attitude_deg':max(v['first_action_attitude_deg'] for v in startup)}
            if not all(np.isfinite(v) for s in record['splits'].values() for r in s['anchors'] for m in r['by_seed'].values() for v in m.values()):
                raise RuntimeError('诊断出现非有限值')
            with (output/'evaluation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            records.append(record)
        if not args.skip_initial_eval:evaluate(0)
        key=jax.random.key(cfg.seed+1);started=time.perf_counter()
        train=cached['train'];starts=np.r_[0,train.ends[:-1]]
        for step in range(1,args.steps+1):
            episode,local,stage=schedule[step-1]
            index=int(starts[episode])+local;assert train.locate(index)==(episode,local)
            source_split='train';episode_id=train.rows[episode]['episode_id']
            if step-1 in recovery_schedule:
                index,episode,local=recovery_schedule[step-1]
                source_split='recovery';stage='expert_recovery';episode_id=recovery.rows[episode]['episode_id']
                local=recovery.rows[episode].get('source_frame_indices',list(range(recovery.rows[episode]['frames'])))[local]
            key,subkey=jax.random.split(key)
            params,opt_state,metrics=update(params,opt_state,subkey,*jax.device_put(batch(source_split,index)))
            metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
            if not all(np.isfinite(v) for v in metrics.values()) or metrics['update_norm']==0:raise RuntimeError(metrics)
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step,episode_id=episode_id,local_frame=local,stage=stage,source_split=source_split,**metrics))+'\n')
            if step==1 or step%100==0:print(f"step={step} loss={metrics['loss']:.6f} grad_norm={metrics['grad_norm']:.4f} elapsed={time.perf_counter()-started:.0f}s",flush=True)
            if step%args.eval_every==0 or step==args.steps:evaluate(step)
            if step%args.save_every==0 or step==args.steps:
                checkpoint=save_checkpoint(cfg,params,opt_state,cfg.data.create(cfg.assets_dirs,cfg.model),step=step)
                (checkpoint/'input_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
                (checkpoint/'dataset_manifest.json').write_text(MANIFEST.read_text())
                print('CHECKPOINT='+str(checkpoint),flush=True)
        summary={'steps':args.steps,'checkpoint':str(checkpoint),'initial_evaluation':None if args.skip_initial_eval else records[0],
                 'final_evaluation':records[-1],'held_out_episodes':True,'closed_loop_tested':False,'task_success_verified':False}
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print('PASS 训练执行完成；任务成功尚未验证',flush=True)
    finally:
        for env,_ in environments.values():env.close()


if __name__=='__main__':main()
