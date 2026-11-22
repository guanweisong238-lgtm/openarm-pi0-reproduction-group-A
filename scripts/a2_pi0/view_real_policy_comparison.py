#!/usr/bin/env python3
"""Visualize recorded vs model-predicted IK targets; not a policy rollout or physics test."""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import cv2
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
EVAL=ROOT/'results/real_action_eval_20261121'
SCENE=ROOT/'assets/real_replay_scene/empty_recovery_scene.xml'
CAMERAS=('global','chest','wrist_left','wrist_right')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--probe',type=int,default=0,help='Validation window index 0–39')
 p.add_argument('--checkpoint',type=int,choices=(500,1000),default=1000)
 p.add_argument('--fps',type=float,default=15)
 p.add_argument('--check',action='store_true')
 p.add_argument('--screenshot',type=Path)
 a=p.parse_args()
 if a.fps<=0:p.error('fps must be positive')
 report=json.loads((EVAL/'report.json').read_text());probes=report['probes']
 if not 0<=a.probe<len(probes):p.error(f'probe must be 0..{len(probes)-1}')
 with np.load(EVAL/'predictions.npz',allow_pickle=False) as f:target=f['target'].copy();prediction=f[f'prediction_{a.checkpoint}'].copy()
 if target.shape!=prediction.shape or target.shape!=(len(probes),50,16):raise ValueError('unexpected prediction dimensions')
 m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
 for i in range(m.ngeom):
  body=int(m.geom_bodyid[i]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body) or ''
  if body and name!='table' and not name.startswith('openarm_'):m.geom_group[i]=5
 opts=mujoco.MjvOption();opts.geomgroup[5]=0
 jids=[m.joint(f'openarm_{side}_joint{i}').id for side in ('right','left') for i in range(1,8)];adr=m.jnt_qposadr[jids];jlo=m.jnt_range[jids,0];jhi=m.jnt_range[jids,1]
 fids=np.array([[m.joint(f'openarm_{side}_finger_joint{i}').id for i in (1,2)] for side in ('right','left')]);fadr=m.jnt_qposadr[fids];flo=m.jnt_range[fids[:,0],0];fhi=m.jnt_range[fids[:,0],1]
 renderer=mujoco.Renderer(m,height=270,width=480)
 camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,camera);camera.lookat[:]=[0.25,0,0.85];camera.distance=1.25;camera.azimuth=130;camera.elevation=-20
 cache={}
 def load_episode(eid):
  if eid not in cache:
   path=ROOT/'data/processed/real_alignment_prototype_v001'/f'episode_{eid}'/'samples.npz'
   with np.load(path,allow_pickle=False) as f:cache[eid]={k:f[k] for k in ['timestamp_ns']+['image_path_'+x for x in CAMERAS]}
  return cache[eid]
 def label(img,text):
  cv2.rectangle(img,(0,0),(img.shape[1],26),(0,0,0),-1);cv2.putText(img,text,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA);return img
 def sim(q,title):
  arm=q[np.r_[0:7,8:15]];d.qpos[adr]=arm;d.qpos[fadr]=np.clip(q[[7,15]],flo,fhi)[:,None];mujoco.mj_forward(m,d)
  renderer.update_scene(d,camera=camera,scene_option=opts);image=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
  count=int(np.count_nonzero((arm<jlo)|(arm>jhi)));clipped=int(np.count_nonzero((q[[7,15]]<flo)|(q[[7,15]]>fhi)))
  label(image,title);cv2.putText(image,f'Arm outside sim range: {count} | finger clips: {clipped}',(5,258),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA);return image
 def render(probe,frame):
  meta=probes[probe];eid=meta['episode_id'];index=meta['sample_index']+frame;ep=load_episode(eid)
  if index>=len(ep['timestamp_ns']):raise ValueError('window crossed episode')
  imgs=[]
  for cam in CAMERAS:
   path=ROOT/str(ep['image_path_'+cam][index]);im=cv2.imread(str(path))
   if im is None:raise RuntimeError(f'missing image {path}')
   imgs.append(label(cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA),f'REAL {cam} | episode {eid}'))
  demo=sim(target[probe,frame],f'RECORDED IK TARGET | +{frame/30:.2f}s')
  model=sim(prediction[probe,frame],f'PREDICTED IK TARGET | checkpoint {a.checkpoint}')
  out=np.vstack([np.hstack([demo,model,imgs[0]]),np.hstack([imgs[2],imgs[3],imgs[1]])])
  foot=f'Window {probe+1}/{len(probes)} | ep {eid} sample {meta["sample_index"]} | step {frame+1}/50 | Space pause A/D step J/L window R restart Q quit'
  cv2.rectangle(out,(0,490),(1440,540),(0,0,0),-1)
  cv2.putText(out,'Model predicted once from window START. Real images below are later reference frames. No physics or task success test.',(5,508),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
  cv2.putText(out,foot,(5,532),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
  return out
 try:
  if a.check:
   for frame in (0,25,49):
    out=render(a.probe,frame)
    if frame==25 and a.screenshot and not cv2.imwrite(str(a.screenshot),out):raise RuntimeError('failed screenshot')
   print('PASS: recorded vs predicted IK, four real camera views, 3 frames rendered',flush=True);return
  print('IK target illustration only; both grippers clipped to uncalibrated simulated limits. No physics, object state, or success result.',flush=True)
  print('Space pause; A/D frame; J/L previous/next window; R restart; Q exit',flush=True)
  cv2.namedWindow('Recorded vs predicted IK - not physics',cv2.WINDOW_NORMAL);cv2.resizeWindow('Recorded vs predicted IK - not physics',1440,540)
  probe=a.probe;frame=0;paused=False
  while True:
   cv2.imshow('Recorded vs predicted IK - not physics',render(probe,frame))
   k=cv2.waitKey(0 if paused else max(1,round(1000/a.fps)))&255
   if k in (ord('q'),27):break
   if k==ord(' '):paused=not paused
   elif k==ord('a'):frame=max(0,frame-1);paused=True
   elif k==ord('d'):frame=min(49,frame+1);paused=True
   elif k==ord('j'):probe=(probe-1)%len(probes);frame=0
   elif k==ord('l'):probe=(probe+1)%len(probes);frame=0
   elif k==ord('r'):frame=0
   elif not paused:frame=(frame+1)%50
  cv2.destroyAllWindows()
 finally:renderer.close()
if __name__=='__main__':main()
