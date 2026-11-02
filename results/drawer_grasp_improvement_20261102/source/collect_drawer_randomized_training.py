"""Independent continuous train12 scene augmentations, never copying held-out starts or seeds."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import mujoco
import numpy as np
from drawer_success_training_config import ROOT,MANIFEST
from drawer_simulation import DrawerSimulation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenes',type=int,default=24)
    p.add_argument('--seed',type=int,default=830271)
    p.add_argument('--x-range',type=float,nargs=2,default=[.085,.115])
    p.add_argument('--y-range',type=float,nargs=2,default=[-.04,.04])
    p.add_argument('--radius-range',type=float,nargs=2,default=[.004,.014])
    p.add_argument('--joint-jitter',type=float,default=.035)
    p.add_argument('--free-pitch',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'collector_source.py').write_bytes(Path(__file__).read_bytes())
    row=next(r for r in json.loads(MANIFEST.read_text())['episodes'] if r['episode_id']==12)
    assert row['split']=='train'
    base=ROOT/row['path'];provenance=json.loads((base/'report.json').read_text())
    with np.load(base/'episode.npz') as d:initial={k:d[k] for k in ['initial_integration_state','initial_eq_data','actuator_names']}
    rng=np.random.default_rng(args.seed);spec=mujoco.mjtState.mjSTATE_INTEGRATION;attempts=[];accepted=[]
    for i in range(args.scenes):
        scene=args.output/f'scene_{i:03d}';scene.mkdir()
        profile=dict(drawer_offset_x_m=float(rng.uniform(*args.x_range)),drawer_offset_y_m=float(rng.uniform(*args.y_range)),
                     drawer_handle_radius_m=float(rng.uniform(*args.radius_range)))
        r=dict(provenance);r.update(scene_overrides=profile,parent_episode_id=12,parent_split='train',
            augmentation_seed=args.seed,augmentation_index=i,eval_only=False,train_only=True)
        (scene/'report.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(scene/'episode.npz',**initial)
        env=DrawerSimulation(scene)
        try:
            original=env.data.qpos[env.qpos_ids[:7]].copy()
            delta_low=np.maximum(-args.joint_jitter,env.low[:7]-original)
            delta_high=np.minimum(args.joint_jitter,env.high[:7]-original)
            env.data.qpos[env.qpos_ids[:7]]=original+rng.uniform(delta_low,delta_high)
            env.data.ctrl[env.actuator_ids[:7]]=env.data.qpos[env.qpos_ids[:7]]
            mujoco.mj_forward(env.model,env.data)
            s=np.zeros(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,s,spec)
            np.savez_compressed(scene/'episode.npz',initial_integration_state=s,initial_eq_data=env.initial_eq,actuator_names=initial['actuator_names'])
            np.savez_compressed(scene/'snapshot.npz',integration_state=s,eq_data=env.initial_eq,qpos=env.data.qpos,qvel=env.data.qvel,
                                applied_prefix=np.empty((0,16)),origin=f'independent train12 augmentation seed={args.seed}, scene={i}')
            penetration=min((c.dist for c in env.data.contact),default=0.)
        finally:env.close()
        if penetration<-.004:
            attempts.append(dict(scene=i,profile=profile,accepted=False,reason='initial penetration'));continue
        ok=False
        for pitch in [15,12,18]:
            dest=scene/f'expert_pitch{pitch}'
            cmd=[sys.executable,str(ROOT/'scripts/a2_pi0/collect_drawer_recovery.py'),
                 '--snapshot',str(scene/'snapshot.npz'),'--episode-id','12','--episode-root',str(scene),
                 '--output',str(dest),'--no-images','--skip-intermediate-dwells','--pitch-degrees',str(pitch),'--pull-distance','.105']
            if args.free_pitch:cmd.append('--free-pitch')
            with (scene/f'pitch{pitch}.log').open('w') as log:
                status=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
            if status:
                attempts.append(dict(scene=i,profile=profile,pitch=pitch,accepted=False,reason='expert error'));continue
            report=json.loads((dest/'report.json').read_text())
            success=bool(report['judge']['success'] and report['replay_verified'])
            attempts.append(dict(scene=i,profile=profile,pitch=pitch,accepted=success,
                                 peak_open_mm=report['peak_open_m']*1000,path=str(dest.resolve().relative_to(ROOT))))
            if success:
                accepted.append(dict(parent_episode_id=12,split='train',path=str(dest.resolve().relative_to(ROOT)),
                                     episode_sha256=hashlib.sha256((dest/'episode.npz').read_bytes()).hexdigest(),
                                     frames=report['frames'],augmentation_index=i,scene_overrides=profile))
                ok=True;break
        print('AUGMENT_SCENE',i,'accepted',ok,'total',len(accepted),flush=True)
        doc=dict(schema='independent_train_scene_augmentations_v001',seed=args.seed,
                 sampling_config=dict(x_range=args.x_range,y_range=args.y_range,radius_range=args.radius_range,joint_jitter=args.joint_jitter,free_pitch=args.free_pitch),
                 parent_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                 episodes=accepted,attempts=attempts,heldout_states_used=False,model_success_evidence=False)
        (args.output/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
    print('AUGMENT_DONE',len(accepted),'/',args.scenes,flush=True)


if __name__=='__main__':main()
