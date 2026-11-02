"""Paired same-observation checkpoint-dtype versus bfloat16 deployment audit, training starts only."""
import os
os.environ['MUJOCO_GL']='glfw'
import argparse,json
from pathlib import Path
import numpy as np
from drawer_success_training_config import ROOT,MANIFEST
from drawer_start_fit_config import get_policy_config
from drawer_policy_loader import create_checkpoint_precision_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import jax,jax.numpy as jnp,flax.nnx as nnx
    import mujoco
    from openpi.policies.policy import Policy
    from drawer_simulation import DrawerSimulation
    from openpi.shared import nnx_utils
    protocol=json.loads((args.checkpoint/'input_protocol.json').read_text())
    cfg=get_policy_config(protocol)
    policy=create_checkpoint_precision_policy(cfg,args.checkpoint,{'num_steps':1})
    graph,state=nnx.split(policy._model)
    dtypes={}
    for value in state.flat_state().values():
        dtype=str(value.value.dtype);dtypes[dtype]=dtypes.get(dtype,0)+value.value.size
    rounded=nnx_utils.state_map(state,nnx.Param,lambda v:v.replace(v.value.astype(jnp.bfloat16)))
    quantized=Policy(nnx.merge(graph,rounded),transforms=[policy._input_transform],output_transforms=[policy._output_transform],sample_kwargs={'num_steps':1})
    rows={r['episode_id']:r for r in json.loads(MANIFEST.read_text())['episodes']}
    code=np.random.default_rng(42).standard_normal((4,32)).astype(np.float32);comparisons=[]
    for eid in [12,9,10,11]:
        assert rows[eid]['split']=='train'
        env=DrawerSimulation(ROOT/rows[eid]['path']);obs=env.observation()
        a=np.asarray(policy.infer(obs,noise=code)['actions']);b=np.asarray(quantized.infer(obs,noise=code)['actions'])
        scratch=mujoco.MjData(env.model);site=env.model.site('right_ee_control_point').id
        def point(action):
            scratch.qpos[:]=env.data.qpos;scratch.qpos[env.qpos_ids[:7]]=action[:7];mujoco.mj_kinematics(env.model,scratch)
            return scratch.site_xpos[site]+scratch.site_xmat[site].reshape(3,3)@np.array([0,0,-.16])
        item=dict(episode_id=eid,max_right7_difference_rad=float(np.abs(a[:,:7]-b[:,:7]).max()),
                  gripper_difference_rad=float(np.abs(a[:,7]-b[:,7]).max()),first_target_position_difference_mm=float(np.linalg.norm(point(a[0])-point(b[0]))*1000))
        comparisons.append(item);print('PRECISION_AUDIT',json.dumps(item),flush=True);env.close()
    report=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_parameter_dtype_element_counts=dtypes,comparisons=comparisons,
                scope='Paired numerical audit only; no task success or causal failure attribution')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('PARAMETER_DTYPES',dtypes,flush=True)


if __name__=='__main__':main()
