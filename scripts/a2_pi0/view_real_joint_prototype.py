#!/usr/bin/env python3
"""Kinematic visualization ONLY: uncalibrated 7-joint replay, no task/physics validation."""
import argparse,time
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/processed/real_alignment_prototype_v001/episode_0/samples.npz'
SCENE=ROOT/'assets/real_replay_scene/empty_recovery_scene.xml'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--headless',action='store_true');p.add_argument('--speed',type=float,default=1.0);p.add_argument('--show-scene',action='store_true',help='Show the historical scene objects instead of clearing the table');a=p.parse_args()
 if a.speed<=0:p.error('speed must be positive')
 with np.load(DATA,allow_pickle=False) as d: states=d['state'].copy(); timestamps=d['timestamp_ns'].copy()
 m=mujoco.MjModel.from_xml_path(str(SCENE)); d=mujoco.MjData(m)
 # Hide historical scene objects visually, keeping the table and robot geometry.
 # The model file and all physical properties remain untouched on disk.
 if not a.show_scene:
  for geom in range(m.ngeom):
   body=int(m.geom_bodyid[geom]); name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body) or ''
   if body and name != 'table' and not name.startswith('openarm_'): m.geom_group[geom]=5
 ids=[m.joint(f'openarm_{side}_joint{i}').id for side in ('right','left') for i in range(1,8)]; adr=m.jnt_qposadr[ids]; vectors=states[:,list(range(7))+list(range(8,15))];low=m.jnt_range[ids,0];high=m.jnt_range[ids,1]
 finger_ids=np.array([[m.joint(f'openarm_{side}_finger_joint{i}').id for i in (1,2)] for side in ('right','left')]); finger_adr=m.jnt_qposadr[finger_ids]
 measured=states[:,[7,15]]; finger_low=m.jnt_range[finger_ids[:,0],0]; finger_high=m.jnt_range[finger_ids[:,0],1]; fingers=np.clip(measured,finger_low,finger_high)
 if not np.isfinite(vectors).all():raise ValueError('nonfinite states')
 exceed=((vectors<low)|(vectors>high)); print('KINEMATIC ILLUSTRATION ONLY; not a calibrated real-robot replay or task/physics simulation.',flush=True)
 print('Gripper illustration: observed motor angle copied to both simulated fingers, clipped to simulated limits; mapping not calibrated.',flush=True)
 print('Clipped gripper samples:',int(np.count_nonzero(fingers!=measured)),'of',measured.size,flush=True)
 print('Drawer and object poses were not recorded; task replay is not supported.',flush=True)
 print('joint values outside historical model limits:',int(exceed.sum()),flush=True)
 def frame(i):
  d.qpos[adr]=vectors[i];d.qpos[finger_adr]=fingers[i,:,None];mujoco.mj_forward(m,d)
 if a.headless:
  for i in range(len(states)):frame(i)
  print('PASS',len(states),'frames; headless kinematic update completed');return
 from mujoco import viewer as mjviewer
 print('Viewer: Space pause/play; R restart; close window to exit. Red/scene placement is not calibrated.',flush=True)
 playing=True; restart=False
 def key(k):
  nonlocal playing,restart
  if k==32:playing=not playing
  elif k in (82,114):restart=True
 with mjviewer.launch_passive(m,d,key_callback=key) as viewer:
  if not a.show_scene: viewer.opt.geomgroup[5]=0
  i=0; last=time.monotonic()
  while viewer.is_running():
   now=time.monotonic()
   if restart:i=0;restart=False;last=now
   if playing and now-last>=1/(30*a.speed):
    i=(i+1)%len(states);last=now
   frame(i);viewer.sync();time.sleep(.01)
if __name__=='__main__':main()
