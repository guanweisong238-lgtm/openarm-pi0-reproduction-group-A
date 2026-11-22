#!/usr/bin/env python3
"""Offline target comparison in an empty scene, not a physics or task rollout."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import cv2,mujoco,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];path=ROOT/'results/real_delta_schedule_1000_ep17_20261122';z=np.load(path/'traces.npz');n=583;arrays=[z['recorded_ik_target'][:n],z['delta_pilot_9'][:n]]
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/real_replay_scene/empty_recovery_scene.xml'));d=mujoco.MjData(m);ids=[m.joint(f'openarm_{s}_joint{i}').id for s in ('right','left') for i in range(1,8)];adr=m.jnt_qposadr[ids];fids=np.array([[m.joint(f'openarm_{s}_finger_joint{i}').id for i in (1,2)] for s in ('right','left')]);fadr=m.jnt_qposadr[fids];lo=m.jnt_range[fids[:,0],0];hi=m.jnt_range[fids[:,0],1];camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,camera);camera.lookat[:]=[.25,0,.85];camera.distance=1.25;camera.azimuth=130;camera.elevation=-20;r=mujoco.Renderer(m,height=270,width=480);w=cv2.VideoWriter(str(path/'delta_1000_vs_demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,330))
if not w.isOpened():raise RuntimeError('video writer failed')
try:
 for i in range(n):
  panel=[]
  for q,title in zip(arrays,('Recorded IK target','Delta model predicted target')):
   d.qpos[adr]=q[i,np.r_[0:7,8:15]];d.qpos[fadr]=np.clip(q[i,[7,15]],lo,hi)[:,None];mujoco.mj_forward(m,d);r.update_scene(d,camera=camera);im=cv2.cvtColor(r.render(),cv2.COLOR_RGB2BGR);cv2.rectangle(im,(0,0),(480,25),(0,0,0),-1);cv2.putText(im,title,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panel.append(im)
  footer=np.zeros((60,960,3),np.uint8);cv2.putText(footer,f'Episode 17 frame {i+1}/{n}: recorded images feed predictions; 300ms assumed delay',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);cv2.putText(footer,'Kinematic target display only; not closed loop. Gripper display clipped to limits.',(6,44),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);w.write(np.vstack((np.hstack(panel),footer)))
finally:w.release();r.close()
print('saved video',n,'frames')
