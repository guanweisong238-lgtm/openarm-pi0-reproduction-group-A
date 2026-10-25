"""同一模型、同一噪声，在示教观测和真实闭环观测上的动作误差对照。"""
from drawer_success_training_config import ROOT,MANIFEST,get_success_config
from drawer_start_fit_config import get_stats_location
from drawer_success_cache import prepare
from local_data_loader import collate_numpy
from drawer_simulation import DrawerSimulation
from pathlib import Path
import json,hashlib
import numpy as np


def main():
    import jax,flax.nnx as nnx,mujoco
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.shared import normalize
    ckpt=ROOT/'checkpoints/pi0_openarm_drawer_approach_v001/approach13_delta_012/1200'
    protocol=json.loads((ckpt/'input_protocol.json').read_text());norm_root,_=get_stats_location(protocol)
    assert protocol['manifest_sha256']==hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert protocol['norm_stats_sha256']==hashlib.sha256((norm_root/'norm_stats.json').read_bytes()).hexdigest()
    cfg=get_success_config();cache=prepare(cfg,'train');stats=normalize.load(norm_root)['actions'];mean=np.asarray(stats.mean);scale=np.asarray(stats.std)+1e-6
    model=cfg.model.load(model_lib.restore_params(ckpt/'params',dtype=jnp.bfloat16));graph,params=nnx.split(model);del model
    @jax.jit
    def sample(current,obs,noise):return nnx.merge(graph,current).sample_actions(jax.random.key(0),obs,noise=noise,num_steps=10)
    results=[]
    for eid,folder in [(12,'success_live_20261024-234423-249381'),(8,'success_live_20261024-234540-696743')]:
        start=0
        for row in cache.rows:
            if row['episode_id']==eid:break
            start+=row['frames']
        trial=ROOT/'logs/a2_pi0'/folder/f'trial_01_episode_{eid}'
        with np.load(trial/'closed_loop.npz') as f:noise=f['inference_noise'].copy();actual=f['action_blocks'].copy()
        with np.load(ROOT/row['path']/'episode.npz') as f:qposes=f['qpos'].copy();state=f['state'].copy();actions=f['action'].copy()
        env=DrawerSimulation(ROOT/row['path']);scratch=mujoco.MjData(env.model);sid=env.model.site('right_ee_control_point').id
        def pose(local,action):
            scratch.qpos[:]=qposes[local];scratch.qpos[env.qpos_ids]=action;mujoco.mj_kinematics(env.model,scratch)
            rot=scratch.site_xmat[sid].reshape(3,3).copy();return scratch.site_xpos[sid]+rot@np.array([0,0,-.16]),rot
        def error(local,action):
            a,r=pose(local,action);b,rt=pose(local,actions[local])
            return dict(position_mm=float(np.linalg.norm(a-b)*1000),attitude_deg=float(np.degrees(np.arccos(np.clip((np.trace(r.T@rt)-1)/2,-1,1)))))
        rows=[]
        try:
            for k,n in enumerate(noise):
                local=4*k;value=collate_numpy([cache[start+local]]);value.pop('actions');value['state'][:,8:16]=0
                normalized=np.asarray(sample(params,model_lib.Observation.from_dict(value),n[None]))[0,:,:16]
                pred=normalized*scale+mean;pred[:,:7]+=state[local,:7]
                rows.append(dict(inference_index=k,local_frame=local,teacher_observation_goal_error=error(local,pred[0]),
                    actual_observation_goal_error=error(local,actual[k,0]),
                    first_action_difference_rad=(pred[0,:8]-actual[k,0,:8]).tolist()))
            results.append(dict(episode_id=eid,precision='bfloat16',noise='exact recorded inference noise, matched by call index',rows=rows))
            print('CASE',eid,'calls',len(rows),'max_teacher_mm',max(r['teacher_observation_goal_error']['position_mm'] for r in rows),
                'max_actual_mm',max(r['actual_observation_goal_error']['position_mm'] for r in rows),flush=True)
        finally:env.close()
    out=ROOT/'results/approach13_delta_012/matched_noise_observations.json';out.write_text(json.dumps(results,indent=2)+'\n');print(out,flush=True)


if __name__=='__main__':main()
