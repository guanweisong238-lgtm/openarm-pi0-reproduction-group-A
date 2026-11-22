#!/usr/bin/env python3
"""Limited feedback recovery experiment in MuJoCo, not a robot controller."""
import os, json, argparse
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco as mj
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
ARM=np.r_[0:7,8:15]

def name(m,kind,i): return mj.mj_id2name(m,kind,int(i)) or ''

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=int,default=0);p.add_argument('--frame',type=int,default=0);p.add_argument('--seed',type=int,default=12);p.add_argument('--max-seconds',type=float,default=12);p.add_argument('--output',type=Path,default=ROOT/'results/real_recovery_sim_pilot_20261122');args=p.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 manifest=json.loads((ROOT/'data/processed/real_training_candidate_v001/manifest.json').read_text());entry=next(x for x in manifest['entries'] if x['episode_id']==args.episode)
 if entry['split']!='train':raise ValueError('prototype reference must come from training split')
 with np.load(ROOT/entry['samples_path'],allow_pickle=False) as z:states=z['state'].copy()
 if not 0<=args.frame<min(60,len(states)):raise ValueError('pilot restricted to initial 60 recorded frames; no task-phase certification')
 m=mj.MjModel.from_xml_path(str(ROOT/'assets/real_replay_scene/articulated_scene.xml'))
 # Remove unrelated door/valve collision geometry; retain table and drawer.
 for i in range(m.ngeom):
  body=int(m.geom_bodyid[i]); chain=[]
  while body:chain.append(name(m,mj.mjtObj.mjOBJ_BODY,body));body=int(m.body_parentid[body])
  if any(x in ('cabinet_door','door','valve_base','valve') for x in chain):m.geom_contype[i]=0;m.geom_conaffinity[i]=0;m.geom_rgba[i,3]=0
 d=mj.MjData(m);probe=mj.MjData(m)
 ids=np.array([m.joint(f'openarm_{s}_joint{i}').id for s in ('right','left') for i in range(1,8)]);adr=m.jnt_qposadr[ids];veladr=m.jnt_dofadr[ids]
 acts=np.array([m.actuator(f'{s}_joint{i}_ctrl').id for s in ('right','left') for i in range(1,8)])
 fingers=np.array([[m.joint(f'openarm_{s}_finger_joint{i}').id for i in (1,2)] for s in ('right','left')]);fadr=m.jnt_qposadr[fingers];fact=np.array([m.actuator(f'{s}_finger1_ctrl').id for s in ('right','left')]);franges=m.actuator_ctrlrange[fact]
 low=np.maximum(m.jnt_range[ids,0],m.actuator_ctrlrange[acts,0]);high=np.minimum(m.jnt_range[ids,1],m.actuator_ctrlrange[acts,1]);goal=states[args.frame,ARM].astype(float);finger=np.clip(states[args.frame,[7,15]],franges[:,0],franges[:,1])
 if np.any(goal<low+.01) or np.any(goal>high-.01):raise ValueError('reference outside joint range margin')
 rng=np.random.default_rng(args.seed);start=np.clip(goal+rng.uniform(-.06,.06,14),low+.01,high-.01)
 d.qpos[adr]=start;d.qpos[fadr]=finger[:,None];d.ctrl[acts]=start;d.ctrl[fact]=finger;mj.mj_forward(m,d)
 def bad_contacts(data):
  bad=[]
  for c in data.contact[:data.ncon]:
   if c.dist>=-0.0005:continue
   b=[name(m,mj.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
   # Equal-hand fingers have intended internal contacts.
   if all('finger' in x for x in b) and (all('left' in x for x in b) or all('right' in x for x in b)):continue
   if any(x.startswith('openarm_') for x in b):bad.append({'bodies':b,'penetration_m':float(-c.dist)})
  return bad
 initial=bad_contacts(d)
 if initial:raise RuntimeError(f'unsafe initial pose in simulation: {initial}')
 # Check frozen-reference path geometrically as a conservative preflight.
 for alpha in np.linspace(0,1,41):
  probe.qpos[:]=d.qpos;probe.qpos[adr]=(1-alpha)*start+alpha*goal;mj.mj_forward(m,probe)
  bad=bad_contacts(probe)
  if bad:raise RuntimeError(f'unsafe interpolated reference path, alpha={alpha:.3f}: {bad}')
 args.output.mkdir(parents=True);(args.output/'images').mkdir();renderer=mj.Renderer(m,height=224,width=224)
 names=('global','chest','wrist_left','wrist_right');records=[];cmd=start.copy();cmdvel=np.zeros(14);stable=0;reason='timeout';frequency=30.;prev=0.
 def capture(frame):
  for view in names:
   cam=mj.MjvCamera();mj.mjv_defaultFreeCamera(m,cam)
   if view=='global':cam.lookat[:]=[.25,0,.75];cam.distance=1.3;cam.azimuth=130;cam.elevation=-20
   elif view=='chest':cam.lookat[:]=[.30,0,.48];cam.distance=.85;cam.azimuth=180;cam.elevation=-30
   else:
    side=view.split('_')[1];cam.lookat[:]=d.body(f'openarm_{side}_ee_base_link').xpos;cam.distance=.32;cam.azimuth=110 if side=='left' else 250;cam.elevation=-35
   renderer.update_scene(d,camera=cam);Image.fromarray(renderer.render()).save(args.output/'images'/f'{frame:04d}_{view}.jpg',quality=85)
 try:
  for step in range(int(args.max_seconds*frequency)):
   # Frame zero records initial state; each target command is computed from measured qpos.
   observed=d.qpos[adr].copy();velocity=d.qvel[veladr].copy();error=goal-observed
   desired=np.clip(1.5*error,-.2,.2);cmdvel+=np.clip(desired-cmdvel,-.8/frequency,.8/frequency);cmd=np.clip(cmd+cmdvel/frequency,low+.01,high-.01)
   q16=np.zeros(16);q16[ARM]=observed;q16[[7,15]]=d.qpos[fadr[:,0]]
   a16=np.zeros(16);a16[ARM]=cmd;a16[[7,15]]=finger
   capture(step);records.append((float(d.time),q16.copy(),a16.copy(),velocity.copy()))
   d.ctrl[acts]=cmd;d.ctrl[fact]=finger
   until=(step+1)/frequency
   while d.time<until-1e-10:
    mj.mj_step(m,d)
    bad=bad_contacts(d)
    if bad:reason='contact_abort';break
    if np.any(d.qpos[adr]<low-.01) or np.any(d.qpos[adr]>high+.01):reason='joint_limit_abort';break
   if reason!='timeout':break
   if np.max(abs(goal-d.qpos[adr]))<.05 and np.max(abs(d.qvel[veladr]))<.2:stable+=1
   else:stable=0
   if stable>=5:reason='recovered';break
 finally:renderer.close()
 times=np.array([r[0] for r in records]);states_out=np.stack([r[1] for r in records]);targets=np.stack([r[2] for r in records]);vel=np.stack([r[3] for r in records]);np.savez_compressed(args.output/'trajectory.npz',timestamp_s=times,state=states_out,action_qpos=targets,arm_qvel=vel,goal_arm=goal,initial_arm=start)
 report={'status':reason,'episode':args.episode,'reference_split':entry['split'],'reference_frame':args.frame,'frames':len(records),'sim_time_seconds':float(d.time),'initial_max_error_rad':float(max(abs(start-goal))),'final_max_error_rad':float(max(abs(d.qpos[adr]-goal))),'max_actual_speed_rad_s':float(np.max(abs(vel))),'max_command_step_rad':float(np.max(abs(np.diff(targets[:,ARM],axis=0)))) if len(targets)>1 else None,'cameras':names,'source':'MuJoCo rendered synthetic images, not real cameras','task_phase':'assumed early noncontact; not manually certified','scene':'drawer and table retained; unrelated valve and door disabled; not calibrated; no cube','reference_selection':'fixed demonstration frame for controlled test, not online phase recognition','recovery_method':'feedback proportional velocity, command speed and acceleration limited','safety_check':'initial and nominal path contact preflight; actual simulation contact and joint limits; does not prove path safety or hardware safety','training_use':'do not use for training until phase, camera mapping, recovery validity, trajectory length, and train/validation handling reviewed','robot_commands_sent':False}
 (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
