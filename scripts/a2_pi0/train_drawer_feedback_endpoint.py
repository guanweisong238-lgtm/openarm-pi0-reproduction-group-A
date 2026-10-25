"""Pi0骨干的4步确定性端点反馈策略；专家只出现在离线监督查询中。"""
from pathlib import Path
import argparse,dataclasses,functools,hashlib,json,time
from datetime import datetime
import numpy as np
from drawer_success_training_config import ROOT,MANIFEST,get_success_config
from drawer_velocity_config import VelocityDeltaSuccessDataConfig,VELOCITY_NORM_ROOT
from drawer_success_cache import CachedInputs
from local_data_loader import collate_numpy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/drawer_feedback_queries/v001')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/pi0_openarm_drawer_velocity_v001/pull12_velocity_015/2000')
    p.add_argument('--exp-name',default='feedback_endpoint_016')
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();digest=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
    cache=CachedInputs(args.data);meta=cache.metadata
    assert meta['schema']=='model_ready_feedback_queries_v001' and meta['action_horizon']==4
    assert meta['source_manifest_sha256']==digest(MANIFEST)
    assert meta['norm_stats_sha256']==digest(VELOCITY_NORM_ROOT/'norm_stats.json')
    source=json.loads((args.checkpoint/'input_protocol.json').read_text())
    assert source['state_representation']=='position_velocity16' and source['norm_stats_sha256']==meta['norm_stats_sha256']
    import jax,jax.numpy as jnp,flax.nnx as nnx,optax
    from openpi.models import model as model_lib
    from openpi.training import optimizer as optimizer_lib
    from openpi.shared import nnx_utils,normalize
    from drawer_training_objective import flow_squared_errors
    from train_smoke import save_checkpoint
    allowed=nnx_utils.PathRegex(r'.*lora.*|(?:state_proj|action_in_proj|action_out_proj)/.*')
    cfg=get_success_config(args.exp_name,args.steps)
    cfg=dataclasses.replace(cfg,name='pi0_openarm_drawer_endpoint_v001',model=dataclasses.replace(cfg.model,action_horizon=4),
        data=VelocityDeltaSuccessDataConfig(),freeze_filter=nnx.Not(allowed),
        lr_schedule=optimizer_lib.CosineDecaySchedule(warmup_steps=100,peak_lr=args.lr,decay_steps=args.steps,decay_lr=args.lr/10))
    if not args.prepare_only and cfg.checkpoint_dir.exists():raise FileExistsError(cfg.checkpoint_dir)
    rng=np.random.default_rng(42);order=rng.permutation(len(cache));nval=max(32,len(cache)//10)
    validation=order[:nval];train=order[nval:]
    assert set(train).isdisjoint(validation)
    anchors={'train':train[:32],'validation':validation[:32]}
    stats=normalize.load(VELOCITY_NORM_ROOT);am=np.asarray(stats['actions'].mean);asc=np.asarray(stats['actions'].std)+1e-6
    sm=np.asarray(stats['state'].mean);ssc=np.asarray(stats['state'].std)+1e-6
    def batch(index):
        value=collate_numpy([cache[int(index)]]);target=value.pop('actions')
        return model_lib.Observation.from_dict(value),target
    def decode(value,index):
        current=cache[int(index)]['state'][:7]*ssc[:7]+sm[:7]
        physical=np.asarray(value)[0,:,:16]*asc+am;physical[:,:7]+=current
        return physical
    with np.load(args.data/'physical_queries.npz') as raw:
        for index in sorted(set([0,len(cache)-1,*order[:30].tolist()])):
            obs,target=batch(index)
            np.testing.assert_allclose(decode(target,index),raw['expert_actions'][index,:4],atol=3e-6,rtol=1e-6)
            assert obs.state.shape==(1,32) and target.shape==(1,4,32)
            assert not obs.state[:,8:16].any() and not obs.state[:,24:32].any()
    print(f'FEEDBACK_INPUT_PASS train={len(train)} query_holdout={len(validation)}; correlated interpolation holdout, not task generalization',flush=True)
    if args.prepare_only:return
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    output=ROOT/'logs/a2_pi0'/datetime.now().strftime('drawer_feedback_%Y%m%d-%H%M%S-%f');output.mkdir(parents=True)
    protocol=dict(dataset_version='drawer_feedback_queries_v001',manifest_sha256=digest(MANIFEST),
        training_data=str(args.data),training_data_metadata_sha256=digest(args.data/'READY.json'),
        norm_stats_root=str(VELOCITY_NORM_ROOT),norm_stats_sha256=digest(VELOCITY_NORM_ROOT/'norm_stats.json'),
        state_representation='position_velocity16',action_representation='right7_delta',action_horizon=4,
        mask_left_normalized_state=True,right8_loss=True,train_io_projections=True,
        inference_kind='fixed_noise_endpoint',sampling_steps=1,fixed_noise_seed=42,
        recommended_noise_mode='fixed',recommended_action_chunk=4,
        source_checkpoint=str(args.checkpoint),source_policy_kind='Pi0 backbone, modified deterministic endpoint policy',
        actual_gradient_parent_episode_ids=[12],physical_validation_episode_ids=[2,3,6],
        query_holdout_scope='same parent training scene, new perturbations; no claim of task generalization',
        train_query_indices=train.tolist(),validation_query_indices=validation.tolist(),
        training_objective='fixed noise42, t1, right8 action MSE, horizon4',teacher_at_runtime=False,
        closed_loop_tested=False,task_success_verified=False)
    (output/'source').mkdir()
    for name in (Path(__file__).name,'drawer_training_objective.py','drawer_velocity_config.py','drawer_start_fit_config.py','live_drawer_success.py'):
        (output/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('RUN_DIRECTORY='+str(output),flush=True)
    params=nnx.state(cfg.model.load(model_lib.restore_params(args.checkpoint/'params')))
    params=nnx_utils.state_map(params,nnx_utils.PathRegex(r'(?:state_proj|action_in_proj|action_out_proj)/.*'),lambda v:v.replace(v.value.astype(jnp.float32)))
    graph=nnx.graphdef(nnx.eval_shape(cfg.model.create,jax.random.key(42)))
    tx=optimizer_lib.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None);opt=tx.init(params.filter(cfg.trainable_filter))
    @functools.partial(jax.jit,donate_argnums=(0,1))
    def update(current,state,key,obs,target):
        model=nnx.merge(graph,current);model.train()
        def loss(m):return flow_squared_errors(m,key,obs,target,time_sampling='fixed_endpoint')[...,:8].mean()
        value,grads=nnx.value_and_grad(loss,argnums=nnx.DiffState(0,cfg.trainable_filter))(model)
        old=current.filter(cfg.trainable_filter);updates,next_state=tx.update(grads,state,old)
        nnx.update(model,optax.apply_updates(old,updates))
        return nnx.state(model),next_state,dict(loss=value,grad_norm=optax.global_norm(grads),update_norm=optax.global_norm(updates))
    noise=np.random.default_rng(42).standard_normal((1,4,32)).astype(np.float32)
    @jax.jit
    def sample(current,obs):return nnx.merge(graph,current).sample_actions(jax.random.key(0),obs,noise=noise,num_steps=1)
    @jax.jit
    def full_prediction(current,obs,target):return flow_squared_errors(nnx.merge(graph,current),jax.random.key(0),obs,target,time_sampling='fixed_endpoint',return_prediction=True)[1]
    probe,target=batch(int(train[0]));difference=float(np.max(np.abs(np.asarray(sample(params,probe))-np.asarray(full_prediction(params,probe,target)))))
    protocol['endpoint_full_vs_cached_prediction_max_difference']=difference
    if difference>.1:raise RuntimeError(f'端点训练/推理数值不一致: {difference}')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('ENDPOINT_NUMERICAL_PARITY',difference,flush=True)
    from drawer_simulation import DrawerSimulation
    import mujoco
    env=DrawerSimulation();scratch=mujoco.MjData(env.model);sid=env.model.site('right_ee_control_point').id
    def pose(action):
        scratch.qpos[:]=env.data.qpos;scratch.qpos[env.qpos_ids[:8]]=action[:8];mujoco.mj_kinematics(env.model,scratch)
        rotation=scratch.site_xmat[sid].reshape(3,3);return scratch.site_xpos[sid]+rotation@np.array([0,0,-.16])
    def evaluate(step):
        record={'step':step,'scope':'held-out feedback queries, not task success','splits':{}}
        for split,indices in anchors.items():
            rows=[]
            for index in indices:
                obs,target=batch(int(index));pred=np.asarray(sample(params,obs));physical=decode(pred,index);truth=decode(target,index)
                rows.append(dict(index=int(index),mse=float(np.mean((pred[...,:8]-target[...,:8])**2)),
                    arm_mae_rad=float(np.abs(physical[:,:7]-truth[:,:7]).mean()),gripper_mae_rad=float(np.abs(physical[:,7]-truth[:,7]).mean()),
                    first_position_mm=float(np.linalg.norm(pose(physical[0])-pose(truth[0]))*1000)))
            record['splits'][split]=dict(fixed_loss=float(np.mean([r['mse'] for r in rows])),
                arm_mae_rad=float(np.mean([r['arm_mae_rad'] for r in rows])),
                first_position_mm=float(np.mean([r['first_position_mm'] for r in rows])),
                worst_position_mm=max(r['first_position_mm'] for r in rows),anchors=rows)
            print('FEEDBACK_EVAL',step,split,{k:v for k,v in record['splits'][split].items() if k!='anchors'},flush=True)
        with (output/'evaluation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return record
    try:
        initial=evaluate(0);started=time.perf_counter();key=jax.random.key(43)
        schedule=rng.choice(train,args.steps,replace=True)
        for step,index in enumerate(schedule,1):
            key,subkey=jax.random.split(key);params,opt,metrics=update(params,opt,subkey,*jax.device_put(batch(int(index))))
            metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
            if not all(np.isfinite(v) for v in metrics.values()) or metrics['update_norm']==0:raise RuntimeError(metrics)
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,query_index=int(index),**metrics))+'\n')
            if step==1 or step%100==0:print(f'step={step} loss={metrics["loss"]:.6f} grad_norm={metrics["grad_norm"]:.4f} elapsed={time.perf_counter()-started:.0f}s',flush=True)
            if step%args.eval_every==0 or step==args.steps:final=evaluate(step)
        checkpoint=save_checkpoint(cfg,params,opt,cfg.data.create(cfg.assets_dirs,cfg.model),step=args.steps)
        (checkpoint/'input_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (checkpoint/'dataset_manifest.json').write_text(MANIFEST.read_text())
        (output/'summary.json').write_text(json.dumps(dict(steps=args.steps,checkpoint=str(checkpoint),initial_evaluation=initial,final_evaluation=final,task_success_verified=False),indent=2)+'\n')
        print('CHECKPOINT='+str(checkpoint),flush=True)
    finally:env.close()


if __name__=='__main__':main()
