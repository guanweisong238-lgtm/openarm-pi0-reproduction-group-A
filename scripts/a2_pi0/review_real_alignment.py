#!/usr/bin/env python3
"""Inspect the exact aligned four-camera samples beside illustrative MuJoCo joints."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import cv2
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/processed/real_alignment_prototype_v001/episode_0/samples.npz'
SCENE=ROOT/'assets/real_replay_scene/empty_recovery_scene.xml'
CAMERAS=('global','chest','wrist_left','wrist_right')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--fps',type=float,default=15.0)
 p.add_argument('--check',action='store_true',help='Render three frames without opening a GUI')
 p.add_argument('--screenshot',type=Path,help='Save a representative frame as a JPEG when using --check')
 a=p.parse_args()
 if a.fps<=0:p.error('--fps must be positive')
 with np.load(DATA,allow_pickle=False) as f: data={k:f[k] for k in f.files}
 n=len(data['timestamp_ns']);m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
 for i in range(m.ngeom):
  body=int(m.geom_bodyid[i]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body) or ''
  if body and name!='table' and not name.startswith('openarm_'):m.geom_group[i]=5
 ids=[m.joint(f'openarm_{side}_joint{i}').id for side in ('right','left') for i in range(1,8)]; adr=m.jnt_qposadr[ids]
 fids=np.array([[m.joint(f'openarm_{side}_finger_joint{i}').id for i in (1,2)] for side in ('right','left')]);fadr=m.jnt_qposadr[fids]
 flo=m.jnt_range[fids[:,0],0];fhi=m.jnt_range[fids[:,0],1]
 opts=mujoco.MjvOption();opts.geomgroup[5]=0
 camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,camera);camera.lookat[:]=[0.25,0,0.85];camera.distance=1.25;camera.azimuth=130;camera.elevation=-20
 renderer=mujoco.Renderer(m,height=270,width=480)
 def label(img,text):
  cv2.rectangle(img,(0,0),(img.shape[1],25),(0,0,0),-1)
  cv2.putText(img,text,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
  return img
 def render(i):
  t=int(data['timestamp_ns'][i]);tiles=[]
  for cam in CAMERAS:
   path=ROOT/str(data['image_path_'+cam][i]);im=cv2.imread(str(path))
   if im is None:raise RuntimeError(f'cannot decode {path}')
   im=cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA)
   age=(t-int(data['image_timestamp_ns_'+cam][i]))/1e6
   tiles.append(label(im,f'{cam} | age {age:.1f} ms'))
  q=data['state'][i];d.qpos[adr]=q[np.r_[0:7,8:15]]
  grip=np.clip(q[[7,15]],flo,fhi);d.qpos[fadr]=grip[:,None];mujoco.mj_forward(m,d)
  renderer.update_scene(d,camera=camera,scene_option=opts)
  sim=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
  sim=label(sim,'MuJoCo: illustrative joint pose; NOT calibrated')
  info=np.full((270,480,3),20,np.uint8)
  lines=[f'Episode 0 | sample {i+1}/{n}',f't = {(t-int(data["timestamp_ns"][0]))/1e9:.3f} s | 30 Hz grid',
    'Gripper angles (observed / IK target) rad:',f'Right: {q[7]:+.3f} / {data["actions"][i,0,7]:+.3f}',f'Left:  {q[15]:+.3f} / {data["actions"][i,0,15]:+.3f}',
    'Image age: host receive time, NOT exposure sync', 'Simulation: no object / drawer pose or contact',
    'Space pause  A/D frame  J/L 1 s  R restart  Q quit']
  for j,line in enumerate(lines):cv2.putText(info,line,(8,26+j*29),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1,cv2.LINE_AA)
  return np.vstack([np.hstack([tiles[0],tiles[1],sim]),np.hstack([tiles[2],tiles[3],info])])
 try:
  if a.check:
   for i in (0,n//2,n-1):
    image=render(i)
    if i==n//2 and a.screenshot:
     if not cv2.imwrite(str(a.screenshot),image):raise RuntimeError('screenshot failed')
   print(f'PASS rendered aligned samples 0, {n//2}, {n-1}; four images and MuJoCo pose',flush=True);return
  print('Review uses aligned sample files, not nearest-image preview. Space pause; A/D frame; J/L second; R restart; Q quit.',flush=True)
  cv2.namedWindow('Aligned real images and illustrative robot',cv2.WINDOW_NORMAL);cv2.resizeWindow('Aligned real images and illustrative robot',1440,540)
  i=0;paused=False
  while True:
   cv2.imshow('Aligned real images and illustrative robot',render(i))
   k=cv2.waitKey(0 if paused else max(1,round(1000/a.fps))) & 255
   if k in (ord('q'),27):break
   if k==ord(' '):paused=not paused
   elif k==ord('a'):i=max(0,i-1);paused=True
   elif k==ord('d'):i=min(n-1,i+1);paused=True
   elif k==ord('j'):i=max(0,i-30);paused=True
   elif k==ord('l'):i=min(n-1,i+30);paused=True
   elif k==ord('r'):i=0
   elif not paused:
    if i==n-1:paused=True
    else:i+=1
  cv2.destroyAllWindows()
 finally:renderer.close()
if __name__=='__main__':main()
