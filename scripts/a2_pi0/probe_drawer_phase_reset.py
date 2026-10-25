"""仅用于定位：在示教中间物理状态重置后运行模型，不能作为完整起点成功证据。"""
import os
os.environ['MUJOCO_GL']='glfw'
from drawer_success_training_config import ROOT,MANIFEST
from drawer_start_fit_config import get_policy_config
from drawer_simulation import DrawerSimulation
from live_drawer_success import update_judge,finger_handle_contacts
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--frame',type=int,default=76)
    p.add_argument('--seconds',type=float,default=12)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.frame<0 or args.frame%4:raise ValueError('噪声配对需要4步边界')
    protocol=json.loads((args.checkpoint/'input_protocol.json').read_text())
    assert protocol['manifest_sha256']==hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    import jax,mujoco,cv2
    from openpi.policies import policy_config
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    policy=policy_config.create_trained_policy(get_policy_config(protocol),args.checkpoint,sample_kwargs={'num_steps':10})
    env=DrawerSimulation();records=[];blocks=[];noises=[];judge={}
    with np.load(env.episode_root/'episode.npz') as d:prefix=d['action'][:args.frame].copy()
    for a in prefix:env.step(a)
    initial=np.zeros(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model,env.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION)
    eq=env.model.eq_data.copy();rng=np.random.default_rng(42)
    for _ in range(args.frame//4+1):noise=rng.standard_normal((16,32)).astype(np.float32)
    args.output.mkdir(parents=True)
    video=cv2.VideoWriter(str(args.output/'diagnostic.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),7.5,(960,240))
    if not video.isOpened():raise RuntimeError('video')
    try:
        for call in range(int(np.ceil(args.seconds*30/4))):
            obs=env.observation();noises.append(noise.copy())
            block=np.asarray(policy.infer(obs,noise=noise)['actions']);blocks.append(block)
            noise=rng.standard_normal((16,32)).astype(np.float32)
            image=np.concatenate([obs['observation.images.'+k] for k in ('ceiling','wrist_left','wrist_right')],axis=1)
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            cv2.putText(image,'DIAGNOSTIC: teacher phase reset, NOT full task success',(8,230),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
            video.write(image)
            for a in block[:min(4,int(args.seconds*30)-len(records))]:
                before=env.data.qpos.copy();applied=env.step(a)
                c,f=finger_handle_contacts(env.model,env.data,include_stems=True)
                opening=-float(env.data.qpos[env.drawer_qpos]);judge=update_judge(judge,opening,c,f,task='open')
                records.append(dict(qpos_before=before,qpos=env.data.qpos.copy(),applied=applied,opening=opening,contacts=c,forces=f))
            if call%10==0:print('PHASE_PROBE',len(records),'opening_mm',opening*1000,'grasped',judge.get('grasped'),flush=True)
            if judge.get('success'):break
        terminal=env.data.qpos.copy();mujoco.mj_setState(env.model,env.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION)
        env.model.eq_data[:]=eq;mujoco.mj_forward(env.model,env.data);env.steps=args.frame;error=0.
        for r in records:env.step(r['applied']);error=max(error,float(np.max(np.abs(env.data.qpos-r['qpos']))))
        result=dict(checkpoint=str(args.checkpoint),reset_teacher_frame=args.frame,
            teacher_prefix_actions_before_policy=len(prefix),teacher_actions_during_policy=0,
            scope='teacher intermediate state counterfactual; not original-start autonomous performance',
            model_pull_success_from_full_initial_state=False,diagnostic_judge=judge,steps=len(records),
            peak_open_mm=max(r['opening'] for r in records)*1000,replay_max_qpos_error=error)
        np.savez_compressed(args.output/'diagnostic.npz',**{k:np.asarray([r[k] for r in records]) for k in records[0]},
            initial_integration_state=initial,initial_eq_data=eq,inference_noise=np.asarray(noises),action_blocks=np.asarray(blocks))
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:video.release();env.close()


if __name__=='__main__':main()
