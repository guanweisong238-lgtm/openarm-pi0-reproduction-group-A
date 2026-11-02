"""从真实策略偏离快照重新规划专家恢复示教；不能用作模型成功证据。"""
from pathlib import Path
import argparse, hashlib, json, sys, os
import numpy as np
from drawer_success_training_config import ROOT, MANIFEST
from drawer_simulation import DrawerSimulation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--episode-id',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-images',action='store_true')
    parser.add_argument('--pitch-degrees',type=float,help='Explicit scripted recovery orientation; required for human parents with null pitch')
    parser.add_argument('--image-stride',type=int,default=1,help='Render only each Nth recorded state; sparse images carry image_frame_indices')
    parser.add_argument('--episode-root',type=Path,help='Explicit independently generated training augmentation of this train parent')
    parser.add_argument('--pull-distance',type=float)
    parser.add_argument('--free-pitch',action='store_true',help='Allow wrist pitch adjustment along the pull path during offline expert generation')
    parser.add_argument('--skip-intermediate-dwells',action='store_true',help='采集连续专家动作版本：省去0.5秒中间等待，仍保留末尾拉开保持')
    args=parser.parse_args()
    collector_source=Path(__file__).read_bytes()
    if args.image_stride<1:parser.error('--image-stride must be positive')
    if args.output.exists():raise FileExistsError(args.output)
    rows=json.loads(MANIFEST.read_text())['episodes']
    row=next(r for r in rows if r['episode_id']==args.episode_id)
    if row['split']!='train':raise ValueError('只采train状态；拒绝把验证状态移入训练')
    sys.path[:0]=[str(ROOT/'.cache/drawer-deps'),str(ROOT/'.cache/openarm-control')]
    import mujoco
    from contact_drawer_controller import ContactDrawerController, finger_handle_contacts
    from openarm_control.trajectory import quintic_polynomial
    render_backend=os.environ.get('MUJOCO_GL','egl')
    from live_drawer_success import update_judge
    os.environ['MUJOCO_GL']=render_backend
    env=DrawerSimulation(args.episode_root or ROOT/row['path']);model,data=env.model,env.data
    snap=np.load(args.snapshot)
    state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
    model.eq_data[:]=snap['eq_data']
    mujoco.mj_setState(model,data,snap['integration_state'],state_spec)
    mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(data.qpos,snap['qpos'])
    np.testing.assert_array_equal(data.qvel,snap['qvel'])
    offset=len(snap['applied_prefix']);env.steps=offset
    initial=snap['integration_state'].copy();initial_eq=model.eq_data.copy();start_time=data.time
    records=[];images={key:[] for key in env.provenance['camera_mapping']};image_indices=[];judge={}
    class Recorder(ContactDrawerController):
        def _park(self,arm):
            # 其他臂由物理位置控制保持；不瞬移qpos/qvel。
            pass

        def _drive(self,active,other,path,dur,grip=None,viewer=None,dt_realtime=False):
            nonlocal judge
            if args.skip_intermediate_dwells and np.isclose(dur,.5):return True
            path=np.atleast_2d(np.asarray(path,float))
            previous=data.qpos[active.king.qpos_indices].copy()
            points=np.vstack([previous,path]) if not np.allclose(path[0],previous) else path
            other_hold=data.qpos[other.king.qpos_indices].copy()
            for index in range(round(dur*30)):
                requested=data.ctrl.copy()
                progress,_,_=quintic_polynomial(index/30,0.,dur,0.,1.)
                requested[active.arm_acts]=active._sample(points,progress)
                requested[other.arm_acts]=other_hold
                if grip is not None:requested[active.grip_act]=grip
                action=np.clip(requested[env.actuator_ids],env.low,env.high)
                counts,forces=finger_handle_contacts(model,data,include_stems=True)
                records.append(dict(state=data.qpos[env.qpos_ids].copy(),qpos=data.qpos.copy(),
                    qvel=data.qvel.copy(),action=action,timestamp=data.time-start_time,
                    drawer=float(data.qpos[env.drawer_qpos]),finger_contact_counts=counts,
                    finger_normal_forces=forces))
                if not args.no_images and (len(records)-1)%args.image_stride==0:
                    obs=env.observation()
                    image_indices.append(len(records)-1)
                    for key in images:images[key].append(obs['observation.images.'+key])
                env.step(action)
                post_counts,post_forces=finger_handle_contacts(model,data,include_stems=True)
                judge=update_judge(judge,-float(data.qpos[env.drawer_qpos]),post_counts,post_forces,task='open')
            return True
    try:
        expert=Recorder(model,data)
        parameters=env.provenance.get('expert_parameters',{})
        pitch=args.pitch_degrees if args.pitch_degrees is not None else parameters.get('pitch_degrees',15.)
        if pitch is None:
            raise ValueError('人工示教无脚本俯仰角；请显式指定 --pitch-degrees，并重新验证专家恢复')
        expert.contact_pitch=pitch
        expert.pull_distance=args.pull_distance if args.pull_distance is not None else parameters.get('nominal_pull_m',.09)
        expert.free_pitch=args.free_pitch or parameters.get('free_pitch',False)
        expert.return_drawer=False
        completed=bool(expert.open_drawer())
        expert._drive(expert.right,expert.left,[data.qpos[expert.right.king.qpos_indices].copy()],1.,
            grip=float(data.ctrl[expert.right.grip_act]))
        terminal=data.qpos.copy()
        arrays={key:np.asarray([r[key] for r in records]) for key in records[0]}
        arrays.update(initial_integration_state=initial,initial_eq_data=initial_eq,
            terminal_qpos=terminal,actuator_names=np.asarray(env.provenance['actuator_order']),
            image_frame_indices=np.asarray(image_indices,dtype=np.int64))
        for key,value in images.items():
            if value:arrays['images_'+key]=np.asarray(value)
        opening=-arrays['drawer'];contact=(arrays['finger_contact_counts']>0).all(1)&(arrays['finger_normal_forces']>.01).all(1)
        opened=any(np.all(opening[i:i+30]>=.08) for i in range(len(opening)-29))
        bilateral=any(np.all(contact[i:i+5]) for i in range(len(contact)-4))
        # 从同一完整初态仅重放动作，逐帧验证；不灌入任何中间状态。
        model.eq_data[:]=initial_eq;mujoco.mj_setState(model,data,initial,state_spec)
        mujoco.mj_forward(model,data);env.steps=offset;error=0.;velocity_error=0.;force_error=0.
        for index,(qpos,action) in enumerate(zip(arrays['qpos'],arrays['action'])):
            error=max(error,float(np.max(np.abs(data.qpos-qpos))))
            velocity_error=max(velocity_error,float(np.max(np.abs(data.qvel-arrays['qvel'][index]))))
            counts,forces=finger_handle_contacts(model,data,include_stems=True)
            np.testing.assert_array_equal(counts,arrays['finger_contact_counts'][index])
            force_error=max(force_error,float(np.max(np.abs(forces-arrays['finger_normal_forces'][index]))))
            env.step(action)
        error=max(error,float(np.max(np.abs(data.qpos-terminal))))
        if error>1e-9 or velocity_error>1e-9 or force_error>1e-7:
            raise RuntimeError(f'动作回放不一致: {error}, {velocity_error}, {force_error}')
        args.output.mkdir(parents=True)
        (args.output/'collector_source.py').write_bytes(collector_source)
        np.savez_compressed(args.output/'episode.npz',**arrays)
        report=dict(env.provenance)
        report.pop('stages',None)
        report.pop('grasp_diagnostics',None)
        report['parent_expert_parameters']=parameters
        report['expert_parameters']=dict(pitch_degrees=pitch,nominal_pull_m=expert.pull_distance,
            free_pitch=expert.free_pitch,return_drawer=False)
        report.update(frames=len(records),seconds=len(records)/30,initial_control_step_offset=offset,
            initialization=str(snap['origin']) if 'origin' in snap.files else 'actual policy deviation snapshot; expert recovery from identical physical state',
            intermediate_dwells_skipped=args.skip_intermediate_dwells,
            parent_episode_id=args.episode_id,parent_split='train',snapshot=str(args.snapshot.resolve()),
            source_episode_root=str(env.episode_root.resolve()),
            snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
            source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            expert_returned_success=completed,opened_8cm_held_1s_during_episode=bool(opened),
            success_8cm_held_1s=bool(np.all(opening[-30:]>=.08)),final_open_m=float(-terminal[env.drawer_qpos]),
            bilateral_contact_5frames=bool(bilateral),peak_open_m=float(opening.max()),
            success_open_then_close=False,final_closed_5mm_held_1s=False,
            initial_state_injections=1,intermediate_state_injections=0,weld_frames=0,
            gl_backend=render_backend,judge=judge,
            image_stride=args.image_stride,image_frames=len(image_indices),
            collector_source_sha256=hashlib.sha256(collector_source).hexdigest(),
            replay_max_qvel_error=velocity_error,replay_max_contact_force_error=force_error,
            replay_max_qpos_error=error,replay_verified=True,
            train_ready=bool(completed and opened and bilateral and judge.get('success') and not args.no_images),
            model_success_evidence=False)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:report[k] for k in ('frames','peak_open_m','bilateral_contact_5frames','replay_max_qpos_error','train_ready','model_success_evidence')},ensure_ascii=False),flush=True)
    finally:env.close()


if __name__=='__main__':main()
