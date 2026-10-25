"""起步学习能力配对诊断：仅train ID12首9帧，比较LoRA和LoRA+I/O。

保持全train统计、图像增强、16步标签和all32目标；小样本拟合不代表任务成功。
"""
from drawer_success_training_config import ROOT,MANIFEST,NORM_ROOT,get_success_config
import argparse
import dataclasses
from datetime import datetime
import hashlib
import functools
import json
from pathlib import Path
import time

import numpy as np

IO_MODULES=('state_proj','action_in_proj','action_out_proj')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/pi0_openarm_drawer_success_v003_lora/success13_early60_500_003/500')
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--eval-every',type=int,default=50)
    parser.add_argument('--variants',nargs='+',choices=('lora','io','io_mask','io_mask_right8'),default=['lora','io'])
    parser.add_argument('--tag',default='start9_pair_001')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--stop-on-fit-pass',action='store_true',help='训练帧连续两次检查通过后停止；仍需独立重载和闭环验收')
    parser.add_argument('--first-four-weight',type=float,default=1.,help='首4步损失权重，后12步保持1')
    parser.add_argument('--time-sampling',choices=('beta','low_mix'),default='beta')
    parser.add_argument('--no-augmentation',action='store_true',help='关闭训练图像增强，其他随机项保留')
    parser.add_argument('--right7-delta',action='store_true',help='独立版本：右7关节相对当前状态的残差，夹爪仍为绝对目标')
    args=parser.parse_args()
    if not np.isfinite(args.first_four_weight) or args.first_four_weight<1:parser.error('首4步权重须>=1且有限')
    if args.steps<2 or args.eval_every<1 or not np.isfinite(args.lr) or args.lr<=0:parser.error('参数不合法')
    if not args.tag or args.tag in ('.','..') or any(c in args.tag for c in '/\\'):parser.error('tag须为单个目录名')
    from drawer_success_cache import prepare
    from local_data_loader import collate_numpy
    from openpi.models import model as model_lib
    from openpi.training import optimizer as optimizer_lib
    from openpi.shared import normalize,nnx_utils
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from flax import traverse_util
    cfg=get_success_config()
    caches={split:prepare(cfg,split) for split in ('train','validation')}
    def index_for(split,eid,local):
        start=0
        for row in caches[split].rows:
            if row['episode_id']==eid:
                if not 0<=local<row['frames']:raise ValueError('帧越界')
                return start+local
            start+=row['frames']
        raise ValueError('轨迹不存在于split')
    train_frames=list(range(9));validation_frames=[0,4,8]
    selected={'train':[(12,f,index_for('train',12,f)) for f in train_frames],
              'validation':[(3,f,index_for('validation',3,f)) for f in validation_frames]}
    rng=np.random.default_rng(cfg.seed);schedule=[]
    while len(schedule)<args.steps:schedule.extend(rng.permutation(9).tolist())
    schedule=schedule[:args.steps]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source=json.loads((args.checkpoint/'input_protocol.json').read_text())
    from drawer_start_fit_config import get_stats_location
    source_norm_root,_=get_stats_location(source)
    if source.get('action_representation')=='right7_delta' and not args.right7_delta:raise ValueError('残差检查点须显式启用right7-delta')
    for key,path in (('manifest_sha256',MANIFEST),('norm_stats_sha256',source_norm_root/'norm_stats.json')):
        if source[key]!=digest(path):raise ValueError('检查点数据或统计不匹配')
    stats=normalize.load(NORM_ROOT)['actions'];mean=np.asarray(stats.mean);scale=np.asarray(stats.std)+1e-6
    absolute_mean,absolute_scale=mean.copy(),scale.copy()
    state_stats=normalize.load(NORM_ROOT)['state'];state_mean=np.array(state_stats.mean);state_scale=np.array(state_stats.std)+1e-6
    for split,frames in selected.items():
        for eid,local,index in frames:
            row=next(r for r in caches[split].rows if r['episode_id']==eid)
            with np.load(ROOT/row['path']/'episode.npz') as e:
                value=caches[split][index]
                np.testing.assert_allclose(value['actions'][:,:16]*scale+mean,e['action'][local:local+16],atol=2e-6)
    target_norm_root=NORM_ROOT
    if args.right7_delta:
        from drawer_delta_config import prepare_delta_stats,DELTA_NORM_ROOT
        prepare_delta_stats();target_norm_root=DELTA_NORM_ROOT
        stats=normalize.load(target_norm_root)['actions'];mean=np.asarray(stats.mean);scale=np.asarray(stats.std)+1e-6
    print('INPUT_PASS train ID12 frames0..8; validation ID3 frames0/4/8; original 16-step labels; full-train statistics',flush=True)
    print('SCHEDULE_COUNTS',np.bincount(schedule,minlength=9).tolist(),flush=True)
    print('ACTION_REPRESENTATION='+('right7_delta, gripper absolute' if args.right7_delta else 'absolute'),flush=True)
    if args.prepare_only:return
    if jax.default_backend()!='gpu':raise RuntimeError('训练需要GPU')
    from train_smoke import save_checkpoint
    from drawer_training_objective import flow_squared_errors
    from drawer_simulation import DrawerSimulation
    import mujoco
    result_root=ROOT/'results'/args.tag
    result_root.mkdir(exist_ok=False)
    pair=[]
    for variant in args.variants:
        train_io=variant!='lora';mask_left=variant in ('io_mask','io_mask_right8');right8=variant=='io_mask_right8'
        allowed=nnx_utils.PathRegex(r'.*lora.*|(?:state_proj|action_in_proj|action_out_proj)/.*')
        current_cfg=dataclasses.replace(cfg,name='pi0_openarm_drawer_start_fit_delta_v001' if args.right7_delta else 'pi0_openarm_drawer_start_fit_v001',exp_name=args.tag+'_'+variant,
            num_train_steps=args.steps,freeze_filter=nnx.Not(allowed) if train_io else cfg.freeze_filter,
            lr_schedule=optimizer_lib.CosineDecaySchedule(warmup_steps=min(20,args.steps//10),peak_lr=args.lr,
                decay_steps=args.steps,decay_lr=args.lr/10))
        if args.right7_delta:
            from drawer_delta_config import DeltaSuccessDataConfig
            current_cfg=dataclasses.replace(current_cfg,data=DeltaSuccessDataConfig(mask_left_state=mask_left))
        if current_cfg.checkpoint_dir.exists():raise FileExistsError(current_cfg.checkpoint_dir)
        output=ROOT/'logs/a2_pi0'/datetime.now().strftime('drawer_start_fit_%Y%m%d-%H%M%S-%f')
        output.mkdir()
        protocol=dict(dataset_version='openarm_drawer_success_v003',manifest_sha256=digest(MANIFEST),norm_stats_sha256=digest(target_norm_root/'norm_stats.json'),
            action_representation='right7_delta' if args.right7_delta else 'absolute',norm_stats_root=str(target_norm_root),
            source_checkpoint=str(args.checkpoint.resolve()),variant=variant,train_io_projections=train_io,
            mask_left_normalized_state=mask_left,right8_loss=right8,right_loss_reduction='sum_right8 / 32' if right8 else 'all32 mean',
            train_episode_ids=[12],train_local_frames=train_frames,validation_episode_ids=[3],validation_local_frames=validation_frames,
            scope='start-frame fitting diagnostic, not task success or broad generalization',steps=args.steps,lr=args.lr,
            schedule=schedule,time_sampling=args.time_sampling,low_time_interval=[.05,.3],low_time_mixture_probability=.5 if args.time_sampling=='low_mix' else 0.,augmentation=not args.no_augmentation,first_four_weight=args.first_four_weight,inference_seeds=[42,7,99],gate_position_mm=2.,gate_attitude_deg=2.,
            stop_on_two_consecutive_fit_passes=args.stop_on_fit_pass,closed_loop_tested=False)
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(output/'config.txt').write_text(repr(current_cfg)+'\n')
        print('RUN_DIRECTORY='+str(output),flush=True);print('VARIANT='+variant,flush=True)
        params=nnx.state(current_cfg.model.load(model_lib.restore_params(args.checkpoint/'params')))
        if train_io:
            params=nnx_utils.state_map(params,nnx_utils.PathRegex(r'(?:state_proj|action_in_proj|action_out_proj)/.*'),
                                       lambda value:value.replace(value.value.astype(jnp.float32)))
        trainable=traverse_util.flatten_dict(params.filter(current_cfg.trainable_filter).to_pure_dict(),sep='/')
        extra={name.split('/')[0] for name in trainable if 'lora' not in name}
        assert extra==(set(IO_MODULES) if train_io else set()),extra
        assert all(value.dtype==jnp.float32 for name,value in trainable.items() if 'lora' not in name)
        print('TRAINABLE_PARAMETERS',sum(value.size for value in trainable.values()),'EXTRA_MODULES',sorted(extra),flush=True)
        del trainable
        graphdef=nnx.graphdef(nnx.eval_shape(current_cfg.model.create,jax.random.key(cfg.seed)))
        tx=optimizer_lib.create_optimizer(current_cfg.optimizer,current_cfg.lr_schedule,weight_decay_mask=None)
        opt_state=tx.init(params.filter(current_cfg.trainable_filter))
        # 各组共用训练实现，实验因素写入protocol.json。
        import optax
        @functools.partial(jax.jit,donate_argnums=(0,1))
        def update(params,opt_state,key,obs,actions):
            model=nnx.merge(graphdef,params);model.train()
            def loss(m):
                errors,sampled_time=flow_squared_errors(m,key,obs,actions,augment=not args.no_augmentation,time_sampling=args.time_sampling,return_time=True)
                if right8:errors=errors*jnp.array([1.]*8+[0.]*24)
                weights=jnp.array([args.first_four_weight]*4+[1.]*12)[None,:,None]
                return (errors*weights).mean(),sampled_time.mean()
            (value,sampled_time),grads=nnx.value_and_grad(loss,argnums=nnx.DiffState(0,current_cfg.trainable_filter),has_aux=True)(model)
            old=params.filter(current_cfg.trainable_filter);updates,next_opt=tx.update(grads,opt_state,old)
            new=optax.apply_updates(old,updates);delta=jax.tree.map(lambda n,o:n-o,new,old);nnx.update(model,new)
            return nnx.state(model),next_opt,dict(loss=value,sampled_time=sampled_time,grad_norm=optax.global_norm(grads),update_norm=optax.global_norm(delta))
        def batch(split,local_index):
            index=selected[split][local_index][2];value=collate_numpy([caches[split][index]])
            if args.right7_delta:
                raw_state=value['state'][:,:16]*state_scale+state_mean
                actions=value['actions'][:,:,:16]*absolute_scale+absolute_mean
                actions[:,:,:7]-=raw_state[:,None,:7]
                value['actions'][:,:,:16]=((actions-mean)/scale).astype(np.float32)
            if mask_left:value['state'][:,8:16]=0
            target=value.pop('actions');return model_lib.Observation.from_dict(value),target
        @jax.jit
        def sample(p,obs,noise):return nnx.merge(graphdef,p).sample_actions(jax.random.key(0),obs,noise=noise,num_steps=10)
        @jax.jit
        def flow(p,obs,target,key):return flow_squared_errors(nnx.merge(graphdef,p),key,obs,target).mean()
        noises={str(seed):np.random.default_rng(seed).standard_normal((1,16,32)).astype(np.float32) for seed in (42,7,99)}
        environments={};qposes={};records=[]
        try:
            for split in selected:
                eid=selected[split][0][0];row=next(r for r in caches[split].rows if r['episode_id']==eid)
                env=DrawerSimulation(ROOT/row['path']);environments[eid]=(env,mujoco.MjData(env.model))
                with np.load(ROOT/row['path']/'episode.npz') as e:qposes[eid]=e['qpos'].copy()
            def tip(eid,local,action):
                env,scratch=environments[eid];scratch.qpos[:]=qposes[eid][local];scratch.qpos[env.qpos_ids]=action
                mujoco.mj_kinematics(env.model,scratch);site=env.model.site('right_ee_control_point').id
                rotation=scratch.site_xmat[site].reshape(3,3).copy()
                return scratch.site_xpos[site]+rotation@np.array([0.,0.,-.16]),rotation
            def evaluate(step):
                record=dict(step=step,splits={})
                for split,frames in selected.items():
                    anchors=[]
                    for i,(eid,local,_) in enumerate(frames):
                        obs,target=batch(split,i);truth=np.asarray(target)[0,:,:16]*scale+mean
                        current_state=qposes[eid][local][environments[eid][0].qpos_ids]
                        if args.right7_delta:truth[:,:7]+=current_state[:7]
                        point,rot=tip(eid,local,truth[0]);by_seed={}
                        for seed,noise in noises.items():
                            pred=np.asarray(sample(params,obs,noise))[0,:,:16]*scale+mean
                            if args.right7_delta:pred[:,:7]+=current_state[:7]
                            pos,rotation=tip(eid,local,pred[0])
                            by_seed[seed]=dict(first_action_position_mm=float(np.linalg.norm(pos-point)*1000),
                                first_action_attitude_deg=float(np.degrees(np.arccos(np.clip((np.trace(rotation.T@rot)-1)/2,-1,1)))),
                                first_action_joint_abs_error_rad=abs(pred[0,:8]-truth[0,:8]).tolist(),
                                right_arm_mae_rad=float(abs(pred[:,:7]-truth[:,:7]).mean()))
                        anchors.append(dict(episode_id=eid,local=local,by_seed=by_seed,
                            flow_all32=float(flow(params,obs,target,jax.random.key(12000+eid*10000+local)))))
                    values=[a['by_seed'][seed] for a in anchors for seed in noises]
                    worst_pos=max(v['first_action_position_mm'] for v in values);worst_rot=max(v['first_action_attitude_deg'] for v in values)
                    if not all(np.isfinite(v[k]) for v in values for k in ('first_action_position_mm','first_action_attitude_deg','right_arm_mae_rad')):raise RuntimeError('非有限评估')
                    record['splits'][split]=dict(anchors=anchors,flow_all32=float(np.mean([a['flow_all32'] for a in anchors])),
                        mean_position_mm=float(np.mean([v['first_action_position_mm'] for v in values])),
                        worst_position_mm=worst_pos,worst_attitude_deg=worst_rot,fit_gate=bool(worst_pos<2 and worst_rot<2))
                    print(f"FIT_EVAL variant={variant} step={step} split={split} mean_pos_mm={record['splits'][split]['mean_position_mm']:.3f} worst_pos_mm={worst_pos:.3f} worst_attitude_deg={worst_rot:.3f} gate={record['splits'][split]['fit_gate']}",flush=True)
                with (output/'evaluation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                records.append(record)
            evaluate(0);key=jax.random.key(cfg.seed+1);started=time.perf_counter();consecutive_passes=0
            for step,local in enumerate(schedule,1):
                key,subkey=jax.random.split(key)
                params,opt_state,metrics=update(params,opt_state,subkey,*jax.device_put(batch('train',local)))
                values={k:float(v) for k,v in jax.device_get(metrics).items()}
                if not all(np.isfinite(v) for v in values.values()) or values['update_norm']==0:raise RuntimeError(values)
                with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,episode_id=12,local_frame=local,**values))+'\n')
                if step==1 or step%50==0:print(f"FIT_TRAIN variant={variant} step={step} loss={values['loss']:.6f} elapsed={time.perf_counter()-started:.0f}s",flush=True)
                if step%args.eval_every==0 or step==args.steps:
                    evaluate(step)
                    consecutive_passes=consecutive_passes+1 if records[-1]['splits']['train']['fit_gate'] else 0
                    if args.stop_on_fit_pass and consecutive_passes>=2:
                        print('STOP two consecutive fit checks passed; independent reload still required',flush=True)
                        break
            checkpoint=save_checkpoint(current_cfg,params,opt_state,current_cfg.data.create(current_cfg.assets_dirs,current_cfg.model),step=step)
            (checkpoint/'input_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
            (checkpoint/'dataset_manifest.json').write_text(MANIFEST.read_text())
            summary=dict(variant=variant,steps=step,planned_steps=args.steps,run_directory=str(output),checkpoint=str(checkpoint),
                initial_evaluation=records[0],final_evaluation=records[-1],scope=protocol['scope'],
                train_fit_passed=bool(records[-1]['splits']['train']['fit_gate'] and
                    (not args.stop_on_fit_pass or consecutive_passes>=2)),
                final_fit_gate=records[-1]['splits']['train']['fit_gate'],
                consecutive_fit_passes=consecutive_passes,closed_loop_tested=False)
            (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');pair.append(summary)
            (result_root/'summary.json').write_text(json.dumps(pair,indent=2)+'\n')
            print('FIT_RESULT '+json.dumps({k:summary[k] for k in ('variant','train_fit_passed','checkpoint')}),flush=True)
        finally:
            for env,_ in environments.values():env.close()
        del params,opt_state,update,sample,flow,graphdef,tx
        jax.clear_caches()
    print('PASS paired fit diagnostic finished; task success not verified',flush=True)


if __name__=='__main__':main()
