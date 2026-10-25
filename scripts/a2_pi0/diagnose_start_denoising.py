"""只读诊断：逐帧关节误差、FK关节替换消融及与原始推理对齐的去噪轨迹。"""
from drawer_success_training_config import ROOT,MANIFEST,NORM_ROOT,get_success_config
import argparse,hashlib,inspect,json,textwrap
from pathlib import Path
import numpy as np


def main():
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import mujoco
    from openpi.models import model as model_lib,pi0
    from openpi.shared import normalize
    from drawer_success_cache import prepare
    from local_data_loader import collate_numpy
    from drawer_start_fit_config import get_stats_location
    from drawer_simulation import DrawerSimulation
    checkpoint=ROOT/'checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_noaug_first4_008_io_mask/300'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/start9_denoising_diagnosis_20261024')
    parser.add_argument('--bridge',action='store_true')
    args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=False)
    protocol=json.loads((checkpoint/'input_protocol.json').read_text());norm_root,asset_id=get_stats_location(protocol)
    for key,path in [('manifest_sha256',MANIFEST),('norm_stats_sha256',norm_root/'norm_stats.json')]:
        assert protocol[key]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads((checkpoint/'assets'/asset_id/'norm_stats.json').read_text())==json.loads((norm_root/'norm_stats.json').read_text())
    assert protocol['action_representation']=='right7_delta'
    cfg=get_success_config();cache=prepare(cfg,'train');offset=0
    for row in cache.rows:
        if row['episode_id']==12:break
        offset+=row['frames']
    else:raise ValueError('ID12缺失')
    env=DrawerSimulation(ROOT/row['path']);scratch=mujoco.MjData(env.model)
    with np.load(ROOT/row['path']/'episode.npz') as e:qposes=e['qpos'].copy();states=e['state'].copy();actions=e['action'].copy()
    stats=normalize.load(norm_root)['actions'];mean=np.asarray(stats.mean);scale=np.asarray(stats.std)+1e-6
    absolute=normalize.load(NORM_ROOT)['actions'];am=np.asarray(absolute.mean);asc=np.asarray(absolute.std)+1e-6
    model=cfg.model.load(model_lib.restore_params(checkpoint/'params'));graph,params=nnx.split(model);del model
    # 保持已安装上游的KV缓存、时间更新和Euler步完全一致，仅附加中间状态返回。
    source=textwrap.dedent(inspect.getsource(pi0.Pi0.sample_actions))
    original='x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))\n    return x_0'
    replacement='''def record(carry, unused):
        next_carry = step(carry)
        return next_carry, (carry[0], carry[1], next_carry[0])
    (x_0, _), history = jax.lax.scan(record, (noise, jnp.asarray(1.0)), None, length=num_steps)
    return x_0, history'''
    assert original in source
    namespace=dict(vars(pi0));exec(source.replace('def sample_actions(', 'def trace_actions(').replace(original,replacement),namespace)
    traced=namespace['trace_actions']
    bridge_source=source.replace('def sample_actions(', 'def bridge_step(').replace('    num_steps:', '    probe_time=1.0,\n    num_steps:').replace(original,'return step((noise, probe_time))')
    exec(bridge_source,namespace)
    bridge_step=namespace['bridge_step']
    @jax.jit
    def bridge(p,obs,x,t):
        next_x,_=bridge_step(nnx.merge(graph,p),jax.random.key(0),obs,num_steps=10,noise=x,probe_time=t)
        return (next_x-x)/(-.1)
    @jax.jit
    def run(p,obs,noise):return traced(nnx.merge(graph,p),jax.random.key(0),obs,num_steps=10,noise=noise)
    @jax.jit
    def standard(p,obs,noise):return nnx.merge(graph,p).sample_actions(jax.random.key(0),obs,num_steps=10,noise=noise)
    def pose(local,action):
        scratch.qpos[:]=qposes[local];scratch.qpos[env.qpos_ids]=action;mujoco.mj_kinematics(env.model,scratch)
        sid=env.model.site('right_ee_control_point').id;rot=scratch.site_xmat[sid].reshape(3,3).copy()
        return scratch.site_xpos[sid]+rot@np.array([0,0,-.16]),rot
    def decode(x,local):
        a=np.asarray(x)[...,:16]*scale+mean;a[...,:7]+=states[local,:7];return a
    def errors(local,pred,truth):
        p,r=pose(local,pred);t,rt=pose(local,truth)
        return dict(position_mm=float(np.linalg.norm(p-t)*1000),attitude_deg=float(np.degrees(np.arccos(np.clip((np.trace(r.T@rt)-1)/2,-1,1)))))
    rows=[];trace_arrays={};max_parity=0.;bridge_rows=[]
    try:
        for local in range(9):
            value=collate_numpy([cache[offset+local]]);target=value.pop('actions')[0,:,:16]*asc+am
            np.testing.assert_allclose(target,actions[local:local+16],atol=2e-6)
            if protocol.get('mask_left_normalized_state'):value['state'][:,8:16]=0
            obs=model_lib.Observation.from_dict(value)
            for seed in (42,7,99):
                noise=np.random.default_rng(seed).standard_normal((1,16,32)).astype(np.float32)
                final,history=run(params,obs,noise);original_pred=np.asarray(standard(params,obs,noise));parity=float(np.max(abs(np.asarray(final)-original_pred)))
                max_parity=max(max_parity,parity);np.testing.assert_allclose(np.asarray(final),original_pred,atol=2e-5,rtol=2e-5)
                if args.bridge and local in (0,8) and seed==42:
                    normalized=(target-mean)/scale
                    normalized[:,:7]=(target[:,:7]-states[local,:7]-mean[:7])/scale[:7]
                    truth32=np.zeros((1,16,32),dtype=np.float32);truth32[0,:,:16]=normalized
                    for t in (.9,.5,.3,.1,.01):
                        x=t*noise+(1-t)*truth32
                        v=np.asarray(bridge(params,obs,x,np.float32(t)))
                        estimate=decode(x-t*v,local)[0]
                        bridge_rows.append(dict(local=local,seed=seed,time=t,clean_estimate=errors(local,estimate[0],target[0]),velocity_error_right7=(v[0,0,:7]-(noise-truth32)[0,0,:7]).tolist()))
                pred=decode(final,local)[0];metric=errors(local,pred[0],target[0]);joint_error=pred[0,:8]-target[0,:8]
                # 将一个关节恢复为真实动作，其他关节预测不变；衡量FK影响，贡献不可简单相加。
                ablations=[]
                for j in range(7):
                    corrected=pred[0].copy();corrected[j]=target[0,j]
                    single=target[0].copy();single[j]=pred[0,j]
                    ablations.append(dict(joint=j+1,corrected=errors(local,corrected,target[0]),single_joint=errors(local,single,target[0])))
                xs,times,next_xs=map(np.asarray,history);steps=[]
                for k,t in enumerate(times):
                    # v=(x_next-x)/dt；x-t*v是基于当前速度推断的干净动作，不是仿真执行动作。
                    velocity=(next_xs[k]-xs[k])/(-.1);estimate=decode(xs[k]-float(t)*velocity,local)[0]
                    steps.append(dict(index=k,time=float(t),clean_estimate=errors(local,estimate[0],target[0]),
                        clean_joint_error_rad=(estimate[0,:8]-target[0,:8]).tolist()))
                rows.append(dict(local=local,seed=seed,**metric,joint_error_rad=joint_error.tolist(),joint_ablations=ablations,
                    chunk_first4_mae_rad=float(abs(pred[:4,:7]-target[:4,:7]).mean()),denoising=steps))
                trace_arrays[f'f{local}_s{seed}_x']=xs;trace_arrays[f'f{local}_s{seed}_next']=next_xs
            print('FRAME_DONE',local,'worst_mm',max(r['position_mm'] for r in rows if r['local']==local),flush=True)
        result=dict(checkpoint=str(checkpoint),protocol=protocol,upstream_sample_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            sample_parity_max_abs=max_parity,rows=rows,bridge_rows=bridge_rows,scope='offline training start frames; no physical rollout, no weight update')
        (out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'trajectories.npz',**trace_arrays)
        print('DIAGNOSIS_SAVED',out,'parity',max_parity,flush=True)
    finally:env.close()


if __name__=='__main__':main()
