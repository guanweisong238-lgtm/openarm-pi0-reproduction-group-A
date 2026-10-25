"""独立重载起步检查点；比较去噪步数/参数精度，不更新模型。"""
from drawer_success_training_config import ROOT,get_success_config,MANIFEST,NORM_ROOT
import argparse
import functools
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--denoise-steps',type=int,nargs='+',default=[10,50])
    parser.add_argument('--precision',choices=('native','bfloat16'),nargs='+',default=['native'])
    parser.add_argument('--episode-id',type=int,help='显式选择训练轨迹，复查阶段训练后的起步保持')
    parser.add_argument('--frames',type=int,nargs='+',help='显式选择局部帧号')
    args=parser.parse_args()
    if min(args.denoise_steps)<1:parser.error('去噪步数须正')
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from openpi.models import model as model_lib
    from openpi.shared import normalize,nnx_utils
    from drawer_success_cache import prepare
    from local_data_loader import collate_numpy
    from drawer_simulation import DrawerSimulation
    import mujoco
    if jax.default_backend()!='gpu':raise RuntimeError('需要GPU')
    protocol=json.loads((args.checkpoint/'input_protocol.json').read_text())
    from drawer_start_fit_config import get_stats_location
    norm_root,asset_id=get_stats_location(protocol)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for key,path in (('manifest_sha256',MANIFEST),('norm_stats_sha256',norm_root/'norm_stats.json')):
        if protocol[key]!=digest(path):raise ValueError(f'{key}与检查点协议不匹配')
    # Orbax资产保存会重新序列化浮点数；比较解析后的统计值，源版本仍按SHA校验。
    if json.loads((args.checkpoint/'assets'/asset_id/'norm_stats.json').read_text())!=json.loads((norm_root/'norm_stats.json').read_text()):
        raise ValueError('保存的统计资产与评估统计不匹配')
    cfg=get_success_config();cache=prepare(cfg,'train');manifest=json.loads(MANIFEST.read_text())
    eid=args.episode_id if args.episode_id is not None else protocol['train_episode_ids'][0]
    if eid not in protocol['train_episode_ids']:parser.error('轨迹不属于该检查点train划分')
    locals_=args.frames if args.frames is not None else protocol.get('train_local_frames',list(range(9)))
    start=0
    for row in cache.rows:
        if row['episode_id']==eid:break
        start+=row['frames']
    else:raise ValueError('训练轨迹不存在')
    if any(not 0<=f<row['frames'] for f in locals_):parser.error('局部帧越界')
    stats=normalize.load(norm_root)['actions'];mean=np.array(stats.mean);scale=np.array(stats.std)+1e-6
    absolute_stats=normalize.load(NORM_ROOT)['actions'];absolute_mean=np.array(absolute_stats.mean);absolute_scale=np.array(absolute_stats.std)+1e-6
    env=DrawerSimulation(ROOT/row['path']);scratch=mujoco.MjData(env.model)
    with np.load(ROOT/row['path']/'episode.npz') as source:qposes=source['qpos'].copy()
    model=cfg.model.load(model_lib.restore_params(args.checkpoint/'params'));graphdef,native=nnx.split(model)
    del model
    @functools.partial(jax.jit,static_argnums=(3,))
    def sample(params,obs,noise,steps):
        return nnx.merge(graphdef,params).sample_actions(jax.random.key(0),obs,noise=noise,num_steps=steps)
    def tip(local,action):
        scratch.qpos[:]=qposes[local];scratch.qpos[env.qpos_ids]=action;mujoco.mj_kinematics(env.model,scratch)
        site=env.model.site('right_ee_control_point').id;rot=scratch.site_xmat[site].reshape(3,3).copy()
        return scratch.site_xpos[site]+rot@np.array([0,0,-.16]),rot
    records=[]
    try:
        for precision in args.precision:
            params=(native if precision=='native' else nnx_utils.state_map(native,nnx.Param,lambda value:value.replace(value.value.astype(jnp.bfloat16))))
            for steps in args.denoise_steps:
                values=[]
                for local in locals_:
                    value=collate_numpy([cache[start+local]])
                    if protocol.get('mask_left_normalized_state'):value['state'][:,8:16]=0
                    target=value.pop('actions')[0,:,:16]*absolute_scale+absolute_mean;obs=model_lib.Observation.from_dict(value)
                    point,rotation=tip(local,target[0])
                    for seed in (42,7,99):
                        noise=np.random.default_rng(seed).standard_normal((1,16,32)).astype(np.float32)
                        pred=np.asarray(sample(params,obs,noise,steps))[0,:,:16]*scale+mean
                        if protocol.get('action_representation')=='right7_delta':pred[:,:7]+=qposes[local][env.qpos_ids[:7]]
                        position,rot=tip(local,pred[0]);values.append(dict(local=local,seed=seed,
                            position_mm=float(np.linalg.norm(position-point)*1000),
                            attitude_deg=float(np.degrees(np.arccos(np.clip((np.trace(rot.T@rotation)-1)/2,-1,1))))))
                row=dict(precision=precision,denoise_steps=steps,mean_position_mm=float(np.mean([v['position_mm'] for v in values])),
                    worst_position_mm=max(v['position_mm'] for v in values),worst_attitude_deg=max(v['attitude_deg'] for v in values),values=values)
                row['fit_gate']=row['worst_position_mm']<2 and row['worst_attitude_deg']<2;records.append(row)
                args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),episode_id=eid,local_frames=locals_,results=records),indent=2)+'\n')
                print('RELOAD_EVAL '+json.dumps({k:v for k,v in row.items() if k!='values'}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
