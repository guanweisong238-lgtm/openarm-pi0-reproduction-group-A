"""只读输入消融：检查固定模型对状态/图像的依赖，不训练、不执行动作。

黑图、交换状态等是人为干预，不能当作真实闭环表现或证明因果根因。
"""
from drawer_success_training_config import ROOT, MANIFEST, NORM_ROOT, get_success_config
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe-left-components',action='store_true')
    args=parser.parse_args()
    import jax
    if jax.default_backend()!='gpu':raise RuntimeError('需要GPU')
    from openpi.policies import policy_config
    from drawer_start_fit_config import get_policy_config
    from drawer_lossless_dataset import LosslessDrawerDataset
    from drawer_simulation import DrawerSimulation
    import mujoco
    print('LOADING_POLICY',args.checkpoint,flush=True)
    protocol=json.loads((args.checkpoint/'input_protocol.json').read_text())
    policy=policy_config.create_trained_policy(get_policy_config(protocol),args.checkpoint,sample_kwargs={'num_steps':10})
    manifest=json.loads(MANIFEST.read_text());rows={r['episode_id']:r for r in manifest['episodes']}
    stats=json.loads((NORM_ROOT/'norm_stats.json').read_text())['norm_stats']['state']
    mean=np.asarray(stats['mean'],dtype=np.float32)
    datasets={eid:LosslessDrawerDataset(ROOT/rows[eid]['path']) for eid in (12,8)}
    noise=np.random.default_rng(42).standard_normal((16,32)).astype(np.float32)
    results=[]
    for eid in ((12,) if args.probe_left_components else (12,8)):
        env=DrawerSimulation(ROOT/rows[eid]['path']);scratch=mujoco.MjData(env.model)
        with np.load(ROOT/rows[eid]['path']/'episode.npz') as data:qposes=data['qpos'].copy()
        site=env.model.site('right_ee_control_point').id
        def point(action,local):
            scratch.qpos[:]=qposes[local];scratch.qpos[env.qpos_ids]=action
            mujoco.mj_kinematics(env.model,scratch)
            return scratch.site_xpos[site]+scratch.site_xmat[site].reshape(3,3)@np.array([0.,0.,-.16])
        try:
            for local in ((0,) if args.probe_left_components else (0,30)):
                raw=datasets[eid][local]
                obs={key:value for key,value in raw.items() if key.startswith('observation.')}
                obs['prompt']=raw['task']
                original_state=obs['observation.state'].copy()
                variants={}
                names=(['original']+[f'left_mean_dim_{i}' for i in range(8)]+['original_repeat'] if args.probe_left_components else
                       ['original','black_images','left_state_mean','other_start_state','q2_plus','q2_minus','original_repeat'])
                for name in names:
                    current=dict(obs);current['observation.state']=original_state.copy()
                    if name=='black_images':
                        for key in current:
                            if key.startswith('observation.images.'):current[key]=np.zeros_like(current[key])
                    elif name=='left_state_mean':current['observation.state'][8:]=mean[8:]
                    elif name=='other_start_state':current['observation.state']=datasets[8 if eid==12 else 12].state[0].copy()
                    elif name=='q2_plus':current['observation.state'][1]+=.02
                    elif name=='q2_minus':current['observation.state'][1]-=.02
                    elif name.startswith('left_mean_dim_'):
                        dimension=8+int(name.rsplit('_',1)[1]);current['observation.state'][dimension]=mean[dimension]
                    started=time.perf_counter()
                    pred=np.asarray(policy.infer(current,noise=noise)['actions'])
                    variants[name]=dict(first_action=pred[0].tolist(),
                        first_action_position_error_mm=float(np.linalg.norm(point(pred[0],local)-point(raw['action'][0],local))*1000),
                        first_action_right7_mae_rad=float(abs(pred[0,:7]-raw['action'][0,:7]).mean()),
                        inference_seconds=time.perf_counter()-started)
                base=np.array(variants['original']['first_action'])
                for name,value in variants.items():
                    action=np.array(value['first_action'])
                    value['change_from_original_right7_rms_rad']=float(np.sqrt(np.mean((action[:7]-base[:7])**2)))
                    value['change_from_original_tip_mm']=float(np.linalg.norm(point(action,local)-point(base,local))*1000)
                sensitivity=((np.array(variants['q2_plus']['first_action'])[:7]-np.array(variants['q2_minus']['first_action'])[:7])/.04
                             if not args.probe_left_components else np.full(7,np.nan))
                result=dict(episode_id=eid,local_frame=local,q2_finite_difference_gain_right7=sensitivity.tolist() if not args.probe_left_components else None,variants=variants)
                result['repeat_original_max_abs_action_difference']=float(abs(np.array(variants['original_repeat']['first_action'])-base).max())
                if result['repeat_original_max_abs_action_difference']>1e-6:
                    raise RuntimeError('同进程固定输入/噪声重复不一致，不能直接解释消融')
                results.append(result)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),seed=42,
                    scope='artificial input interventions; not task performance',results=results),indent=2)+'\n')
                if args.probe_left_components:
                    print('LEFT_COMPONENT_PROBE '+json.dumps({name:value['change_from_original_tip_mm'] for name,value in variants.items()}),flush=True)
                    continue
                print('PROBE',eid,local,'original_mm',variants['original']['first_action_position_error_mm'],
                    'black_change_mm',variants['black_images']['change_from_original_tip_mm'],
                    'left_mean_change_mm',variants['left_state_mean']['change_from_original_tip_mm'],
                    'swapped_state_change_mm',variants['other_start_state']['change_from_original_tip_mm'],
                    'q2_gain',sensitivity[1],flush=True)
        finally:env.close()
    print('PASS fixed-noise inference probes; no training or physics actions',flush=True)


if __name__=='__main__':main()
