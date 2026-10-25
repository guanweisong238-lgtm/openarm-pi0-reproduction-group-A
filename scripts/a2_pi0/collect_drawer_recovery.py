"""从真实策略偏离快照重新规划专家恢复示教；不能用作模型成功证据。"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
from drawer_success_training_config import ROOT, MANIFEST
from drawer_simulation import DrawerSimulation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--episode-id',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-images',action='store_true')
    parser.add_argument('--skip-intermediate-dwells',action='store_true',help='采集连续专家动作版本：省去0.5秒中间等待，仍保留末尾拉开保持')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=json.loads(MANIFEST.read_text())['episodes']
    row=next(r for r in rows if r['episode_id']==args.episode_id)
    if row['split']!='train':raise ValueError('只采train状态；拒绝把验证状态移入训练')
    sys.path[:0]=[str(ROOT/'.cache/drawer-deps'),str(ROOT/'.cache/openarm-control')]
    import mujoco
    from contact_drawer_controller import ContactDrawerController, finger_handle_contacts
    from openarm_control.trajectory import quintic_polynomial
    env=DrawerSimulation(ROOT/row['path']);model,data=env.model,env.data
    snap=np.load(args.snapshot)
    state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
    model.eq_data[:]=snap['eq_data']
    mujoco.mj_setState(model,data,snap['integration_state'],state_spec)
    mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(data.qpos,snap['qpos'])
    np.testing.assert_array_equal(data.qvel,snap['qvel'])
    offset=len(snap['applied_prefix']);env.steps=offset
    initial=snap['integration_state'].copy();initial_eq=model.eq_data.copy();start_time=data.time
    records=[];images={key:[] for key in env.provenance['camera_mapping']}
    class Recorder(ContactDrawerController):
        def _park(self,arm):
            # 其他臂由物理位置控制保持；不瞬移qpos/qvel。
            pass

        def _drive(self,active,other,path,dur,grip=None,viewer=None,dt_realtime=False):
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
                if not args.no_images:
                    obs=env.observation()
                    for key in images:images[key].append(obs['observation.images.'+key])
                env.step(action)
            return True
    try:
        expert=Recorder(model,data)
        parameters=env.provenance.get('expert_parameters',{})
        expert.contact_pitch=parameters.get('pitch_degrees',15.)
        expert.pull_distance=parameters.get('nominal_pull_m',.09)
        expert.free_pitch=parameters.get('free_pitch',False)
        expert.return_drawer=False
        completed=bool(expert.open_drawer())
        expert._drive(expert.right,expert.left,[data.qpos[expert.right.king.qpos_indices].copy()],1.,
            grip=float(data.ctrl[expert.right.grip_act]))
        terminal=data.qpos.copy()
        arrays={key:np.asarray([r[key] for r in records]) for key in records[0]}
        arrays.update(initial_integration_state=initial,initial_eq_data=initial_eq,
            terminal_qpos=terminal,actuator_names=np.asarray(env.provenance['actuator_order']))
        for key,value in images.items():
            if value:arrays['images_'+key]=np.asarray(value)
        opening=-arrays['drawer'];contact=(arrays['finger_contact_counts']>0).all(1)&(arrays['finger_normal_forces']>.01).all(1)
        opened=any(np.all(opening[i:i+30]>=.08) for i in range(len(opening)-29))
        bilateral=any(np.all(contact[i:i+5]) for i in range(len(contact)-4))
        # 从同一完整初态仅重放动作，逐帧验证；不灌入任何中间状态。
        model.eq_data[:]=initial_eq;mujoco.mj_setState(model,data,initial,state_spec)
        mujoco.mj_forward(model,data);env.steps=offset;error=0.
        for qpos,action in zip(arrays['qpos'],arrays['action']):
            error=max(error,float(np.max(np.abs(data.qpos-qpos))));env.step(action)
        error=max(error,float(np.max(np.abs(data.qpos-terminal))))
        if error>1e-9:raise RuntimeError(f'动作回放不一致: {error}')
        args.output.mkdir(parents=True)
        np.savez_compressed(args.output/'episode.npz',**arrays)
        report=dict(env.provenance)
        report.pop('stages',None)
        report.pop('grasp_diagnostics',None)
        report['expert_parameters']=dict(parameters,return_drawer=False)
        report.update(frames=len(records),seconds=len(records)/30,initial_control_step_offset=offset,
            initialization=str(snap['origin']) if 'origin' in snap.files else 'actual policy deviation snapshot; expert recovery from identical physical state',
            intermediate_dwells_skipped=args.skip_intermediate_dwells,
            parent_episode_id=args.episode_id,parent_split='train',snapshot=str(args.snapshot.resolve()),
            snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
            source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            expert_returned_success=completed,opened_8cm_held_1s_during_episode=bool(opened),
            success_8cm_held_1s=bool(np.all(opening[-30:]>=.08)),final_open_m=float(-terminal[env.drawer_qpos]),
            bilateral_contact_5frames=bool(bilateral),peak_open_m=float(opening.max()),
            success_open_then_close=False,final_closed_5mm_held_1s=False,
            initial_state_injections=1,intermediate_state_injections=0,weld_frames=0,
            replay_max_qpos_error=error,replay_verified=True,
            train_ready=bool(completed and opened and bilateral and not args.no_images),
            model_success_evidence=False)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:report[k] for k in ('frames','peak_open_m','bilateral_contact_5frames','replay_max_qpos_error','train_ready','model_success_evidence')},ensure_ascii=False),flush=True)
    finally:env.close()


if __name__=='__main__':main()
