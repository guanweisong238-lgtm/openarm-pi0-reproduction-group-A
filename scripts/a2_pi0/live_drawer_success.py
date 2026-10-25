"""v003模型实时视觉闭环。R重新推理，N切换ready/warm起点，Q退出；不执行示教动作。"""
import os
os.environ['MUJOCO_GL']='glfw'
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME','mesa')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES','/usr/share/glvnd/egl_vendor.d/50_mesa.json')
from drawer_success_training_config import ROOT,MANIFEST,NORM_ROOT,ASSET_ID,get_success_config
import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import threading

import mujoco
import numpy as np
from drawer_simulation import DrawerSimulation
import sys
sys.path[:0]=[str(ROOT/'.cache/drawer-deps'),str(ROOT/'.cache/openarm-control')]
from contact_drawer_controller import finger_handle_contacts


def update_judge(previous,opening,contacts,forces,task='roundtrip'):
    if task not in ('open','roundtrip'):raise ValueError(task)
    both=bool(np.all(contacts>0) and np.all(forces>.01))
    grasp_streak=previous.get('grasp_streak',0)+1 if both else 0
    grasped=previous.get('grasped',False) or grasp_streak>=5
    open_streak=previous.get('open_streak',0)+1 if opening>=.08 else 0
    opened=previous.get('opened',False) or open_streak>=30
    closed_streak=previous.get('closed_streak',0)+1 if opened and abs(opening)<=.005 else 0
    pulling=previous.get('pulling_contact',False) or (both and opening>previous.get('opening',opening)+.00005)
    return dict(opening=opening,grasp_streak=grasp_streak,grasped=grasped,
        open_streak=open_streak,opened=opened,closed_streak=closed_streak,
        pulling_contact=pulling,pull_success=bool(grasped and pulling and opened),
        success=bool(grasped and pulling and opened and (task=='open' or closed_streak>=30)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/pi0_openarm_drawer_success_v003_lora/success13_lora_001/2000')
    parser.add_argument('--seconds',type=float,default=24)
    parser.add_argument('--task',choices=('open','roundtrip'),default='roundtrip',help='open验收真实夹持拉开8cm保持1秒；roundtrip额外要求关闭保持')
    parser.add_argument('--gripper-mode',choices=('continuous','slew_binary'),default='continuous',help='连续目标，或模型闭合意图驱动的限速开闭控制')
    parser.add_argument('--action-chunk',type=int,default=4,choices=range(1,17),help='每次重观测前执行的动作数')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--noise-mode',choices=('fixed','resample','zero'),default='fixed',help='固定噪声、每次推理新噪声，或确定性零噪声对照')
    parser.add_argument('--seeds',type=int,nargs='+',help='同一策略依次运行多个噪声种子')
    parser.add_argument('--phase-diagnostic',action='store_true',help='接近/首次夹持片段评分，不解释为完整任务失败')
    parser.add_argument('--start-diagnostic',action='store_true',help='最多前9个控制步，仅评估起步，不判为完整任务失败')
    parser.add_argument('--start-case',choices=('ready','warm'),default='ready')
    parser.add_argument('--episode-ids',type=int,nargs='+',help='显式指定清单中的轨迹起点，保留原始split标签')
    parser.add_argument('--auto-next',action='store_true',help='到时后自动完成下一条，最后退出')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.seconds<=0 or args.seed<0 or (args.seeds and min(args.seeds)<0):parser.error('时长/seed不合法')
    if args.phase_diagnostic and args.start_diagnostic:parser.error('请选择一种诊断范围')
    if args.start_diagnostic and not 1<=round(args.seconds*30)<=9:parser.error('起步诊断须为1至9控制步')
    protocol=json.loads((args.checkpoint/'input_protocol.json').read_text())
    horizon=int(protocol.get('action_horizon',16))
    if args.action_chunk>horizon:parser.error('执行步数超过模型动作窗口')
    if protocol.get('inference_kind')=='fixed_noise_endpoint':
        if args.noise_mode!='fixed' or any(s!=protocol['fixed_noise_seed'] for s in (args.seeds or [args.seed])):
            parser.error('确定性端点策略须使用--noise-mode fixed及其训练固定seed')
    from drawer_start_fit_config import get_stats_location
    norm_root,asset_id=get_stats_location(protocol)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if protocol['manifest_sha256']!=digest(MANIFEST):raise ValueError('manifest不匹配')
    if protocol['norm_stats_sha256']!=digest(norm_root/'norm_stats.json'):raise ValueError('统计不匹配')
    # 重新保存的JSON浮点格式可变化，统计数值及源版本必须一致。
    if json.loads((args.checkpoint/'assets'/asset_id/'norm_stats.json').read_text())!=json.loads((norm_root/'norm_stats.json').read_text()):
        raise ValueError('检查点资产统计与协议版本不一致')
    manifest=json.loads(MANIFEST.read_text());rows={r['episode_id']:r for r in manifest['episodes']}
    if args.episode_ids and any(eid not in rows for eid in args.episode_ids):
        parser.error('episode ID不存在')
    cases=[rows[eid] for eid in args.episode_ids] if args.episode_ids else [rows[3],rows[6]]
    seeds=args.seeds or [args.seed]
    cases=[dict(row,rollout_seed=seed) for row in cases for seed in seeds]
    if args.prepare_only:
        for row in cases:
            env=DrawerSimulation(ROOT/row['path'])
            print('INITIAL_STATE_PASS',row['episode_id'],row['initialization_group'],flush=True)
            env.close()
        s={}
        for _ in range(30):s=update_judge(s,.09,np.array([1,1]),np.array([2.,2.]))
        # 首帧不计拉动，补一个有双侧接触的拉动事件。
        s=update_judge(s,.091,np.array([1,1]),np.array([2.,2.]))
        for _ in range(30):s=update_judge(s,0.,np.array([1,1]),np.array([2.,2.]))
        assert s['success']
        s={}
        for _ in range(60):s=update_judge(s,.09,np.array([1,0]),np.array([2.,0.]))
        for _ in range(30):s=update_judge(s,0.,np.array([1,0]),np.array([2.,0.]))
        assert not s['success']
        print('PASS checkpoint assets, selected initializations, success and unilateral-failure judge',flush=True)
        return
    import jax
    if jax.default_backend()!='gpu':raise RuntimeError('实时推理要求JAX GPU')
    from openpi.policies import policy_config
    from drawer_start_fit_config import get_policy_config
    print('LOADING_POLICY',args.checkpoint,flush=True)
    policy=policy_config.create_trained_policy(get_policy_config(protocol),args.checkpoint,sample_kwargs={'num_steps':int(protocol.get('sampling_steps',10))})
    policy_diagnostics={}
    if protocol.get('state_representation')=='position_velocity16':
        velocity_kernel=np.asarray(policy._model.state_proj.kernel.value,np.float32)[16:24]
        policy_diagnostics['learned_right_velocity_projection_l2']=float(np.linalg.norm(velocity_kernel))
        if not policy_diagnostics['learned_right_velocity_projection_l2']>0:raise ValueError('速度投影仍为零，未学到新增通道')
        print('VELOCITY_PROJECTION_LOADED '+json.dumps(policy_diagnostics),flush=True)
    import cv2
    import mujoco.viewer
    output=ROOT/'logs/a2_pi0'/datetime.now().strftime('success_live_%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    print('RUN_DIRECTORY='+str(output),flush=True)
    events=deque();case_index=0 if args.episode_ids or args.start_case=='ready' else len(seeds);trial=0;quit_all=False
    while not quit_all:
        row=cases[case_index];trial+=1
        trial_seed=row['rollout_seed'];noise_rng=np.random.default_rng(trial_seed)
        noise=np.zeros((horizon,32),np.float32) if args.noise_mode=='zero' else noise_rng.standard_normal((horizon,32)).astype(np.float32)
        env=DrawerSimulation(ROOT/row['path']);directory=output/f'trial_{trial:02d}_episode_{row["episode_id"]}'
        directory.mkdir();observed=[];blocks=[];times=[];inference_noises=[];state={};saved=False;paused=False;advance=False;restart=False
        writer=cv2.VideoWriter(str(directory/'closed_loop.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30/args.action_chunk,(960,240))
        if not writer.isOpened():raise RuntimeError('无法创建闭环视频')
        initial_time=float(env.data.time)
        left_hold=env.data.ctrl[env.actuator_ids[8:]].copy()
        grip_command=float(env.data.ctrl[env.actuator_ids[7]])
        def finish(reason):
            nonlocal saved
            if saved:return
            # 完成时写入MP4索引，不能等待用户关闭保留展示的窗口。
            writer.release()
            saved=True
            if observed:
                np.savez_compressed(directory/'closed_loop.npz',**{k:np.asarray([r[k] for r in observed]) for k in observed[0]},
                                    action_blocks=np.asarray(blocks),inference_seconds=np.asarray(times),inference_noise=np.asarray(inference_noises))
            summary=dict(checkpoint=str(args.checkpoint),task=args.task,policy_diagnostics=policy_diagnostics,episode_id=row['episode_id'],split=row['split'],
                initialization=row['initialization_group'],noise_seed=trial_seed,steps=len(observed),seconds=len(observed)/30,
                peak_open_mm=max((r['opening'] for r in observed),default=0)*1000,
                final_open_mm=state.get('opening',0)*1000,judge=state,termination=reason,
                teacher_actions_executed=0,intermediate_states_injected=0,action_chunk=args.action_chunk,
                contact_scope='entire handle assembly, excluding panel and wrist',simulation_paused_during_inference=True,
                closed_loop_tested=True,fully_executed=reason in ('timeout','success'))
            summary.update(noise_mode=args.noise_mode,scope='first nine steps diagnostic' if args.start_diagnostic else 'approach and first contact diagnostic' if args.phase_diagnostic else 'full task rollout',
                           gripper_mode=args.gripper_mode,
                           state_representation=protocol.get('state_representation','position16'),
                           mask_left_normalized_state=bool(protocol.get('mask_left_normalized_state')),
                           fixed_left_controller=bool(protocol.get('right8_loss')))
            (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
            print('LIVE_RESULT '+json.dumps(summary,ensure_ascii=False),flush=True)
        viewer_threads=[]
        try:
            previous_threads=set(threading.enumerate())
            with mujoco.viewer.launch_passive(env.model,env.data,key_callback=lambda key:events.append(key),
                    show_left_ui=False,show_right_ui=False) as viewer:
                viewer_threads=[t for t in threading.enumerate() if t not in previous_threads and '_launch_internal' in t.name]
                viewer.cam.lookat[:]=[.40,-.24,.53];viewer.cam.distance=.9;viewer.cam.azimuth=-140;viewer.cam.elevation=-25
                print('GUI_READY LIVE MODEL',row['initialization_group'],'Space=pause R=fresh rollout N=next initialization Q=quit',flush=True)
                resume=time.perf_counter()+3
                while viewer.is_running():
                    while events:
                        key=events.popleft()
                        if key in (ord('Q'),ord('q')):quit_all=True
                        elif key in (ord('N'),ord('n')):advance=True
                        elif key in (ord('R'),ord('r')):restart=True
                        elif key==32:paused=not paused
                    if quit_all or advance or restart:break
                    status='DIAGNOSTIC COMPLETE' if saved and (args.start_diagnostic or args.phase_diagnostic) else 'SUCCESS' if state.get('success') else 'FAILED / timeout' if saved else 'Paused' if paused else 'LIVE model inference'
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f"LIVE MODEL | seed {trial_seed}\nInitialization\nSimulation time\nDrawer\nGrasp / Open hold\nStatus\nSpace / R / N / Q",
                        f"{args.checkpoint.parent.name} / {args.checkpoint.name}\n{row['initialization_group']} ({row['split']} ID {row['episode_id']})\n{env.steps/30:.2f} s\n{state.get('opening',0)*1000:.1f} mm\n{state.get('grasped',False)} / {state.get('opened',False)}\n{status}\nPause / Retry / Next / Quit"))
                    viewer.sync()
                    if saved and args.auto_next:
                        advance=case_index+1<len(cases);quit_all=not advance
                        break
                    if paused or saved or time.perf_counter()<resume:
                        time.sleep(.03);continue
                    with viewer.lock():
                        viewer.perturb.active=0;viewer.perturb.active2=0;env.data.xfrc_applied[:]=0
                        observation=env.observation()
                    inference_noises.append(noise.copy())
                    began=time.perf_counter();block=np.asarray(policy.infer(observation,noise=noise)['actions']);times.append(time.perf_counter()-began)
                    if args.noise_mode=='resample':noise=noise_rng.standard_normal((horizon,32)).astype(np.float32)
                    if block.shape!=(horizon,16) or not np.isfinite(block).all():raise ValueError('模型输出异常')
                    blocks.append(block)
                    if protocol.get('right8_loss'):
                        block=block.copy();block[:,8:]=left_hold
                    frame=cv2.cvtColor(np.concatenate([observation['observation.images.'+n] for n in ('ceiling','wrist_left','wrist_right')],axis=1),cv2.COLOR_RGB2BGR)
                    cv2.putText(frame,f'LIVE MODEL seed={trial_seed} t={env.steps/30:.2f}s',(8,230),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);writer.write(frame)
                    for action in block[:min(args.action_chunk,round(args.seconds*30)-env.steps)]:
                        action=action.copy()
                        if args.gripper_mode=='slew_binary':
                            grip_command=min(0.,grip_command+.007) if action[7]>-.24 else -.25
                            action[7]=grip_command
                        began=time.perf_counter()
                        with viewer.lock():
                            viewer.perturb.active=0;viewer.perturb.active2=0;env.data.xfrc_applied[:]=0
                            before=env.data.qpos.copy();velocity_before=env.data.qvel.copy();applied=env.step(action)
                            contacts,forces=finger_handle_contacts(env.model,env.data,include_stems=True)
                            opening=-float(env.data.qpos[env.drawer_qpos]);state=update_judge(state,opening,contacts,forces,task=args.task)
                            observed.append(dict(qpos_before=before,qpos=env.data.qpos.copy(),qvel_before=velocity_before,qvel=env.data.qvel.copy(),requested=action.copy(),applied=applied,
                                opening=opening,contacts=contacts,forces=forces,time=float(env.data.time-initial_time)))
                        viewer.sync();time.sleep(max(0.,1/30-(time.perf_counter()-began)))
                        if state['success']:break
                    if env.steps%40==0:print(f"LIVE steps={env.steps} opening_mm={opening*1000:.1f} grasp={state.get('grasped')} success={state.get('success')}",flush=True)
                    if state.get('success'):finish('success')
                    elif env.steps>=round(args.seconds*30):finish('timeout')
                finish('user_switch' if advance or restart else 'user_exit')
                # 在桌面上下文销毁前释放离屏相机，避免GLFW退出竞态。
                env.close()
        finally:
            writer.release()
            for thread in viewer_threads:thread.join(timeout=10)
            env.close()
        if advance:case_index=(case_index+1)%len(cases)
        elif not restart:quit_all=True


if __name__=='__main__':main()
