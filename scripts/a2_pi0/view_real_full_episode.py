#!/usr/bin/env python3
"""Show a held-out full episode: IK targets under recorded-observation teacher forcing."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import cv2
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'results/real_full_episode_17_offline_20261121'
SCENE=ROOT/'assets/real_replay_scene/empty_recovery_scene.xml'
CAMERAS=('global','chest','wrist_left','wrist_right')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,default=DEFAULT);p.add_argument('--fps',type=float,default=20.0);p.add_argument('--check',action='store_true');p.add_argument('--screenshot',type=Path);p.add_argument('--video',type=Path);a=p.parse_args()
 if a.fps<=0:p.error('fps must be positive')
 report=json.loads((a.result/'report.json').read_text());eid=report['episode_id']
 with np.load(a.result/'full_episode.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
 source=ROOT/report.get('source_samples_path',f'data/processed/real_alignment_prototype_v001/episode_{eid}/samples.npz')
 with np.load(source,allow_pickle=False) as z:episode={k:z[k] for k in ['timestamp_ns']+['image_path_'+x for x in CAMERAS]}
 if not np.array_equal(data['timestamp_ns'],episode['timestamp_ns']):raise ValueError('timestamps differ from source')
 n=len(data['timestamp_ns']);m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
 for i in range(m.ngeom):
  body=int(m.geom_bodyid[i]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body) or ''
  if body and name!='table' and not name.startswith('openarm_'):m.geom_group[i]=5
 opt=mujoco.MjvOption();opt.geomgroup[5]=0
 ids=[m.joint(f'openarm_{side}_joint{i}').id for side in ('right','left') for i in range(1,8)];adr=m.jnt_qposadr[ids]
 fids=np.array([[m.joint(f'openarm_{side}_finger_joint{i}').id for i in (1,2)] for side in ('right','left')]);fadr=m.jnt_qposadr[fids];flo=m.jnt_range[fids[:,0],0];fhi=m.jnt_range[fids[:,0],1]
 camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,camera);camera.lookat[:]=[.25,0,.85];camera.distance=1.25;camera.azimuth=130;camera.elevation=-20
 renderer=mujoco.Renderer(m,height=270,width=480)
 def label(im,text):
  cv2.rectangle(im,(0,0),(480,26),(0,0,0),-1);cv2.putText(im,text,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA);return im
 def sim(q,title):
  d.qpos[adr]=q[np.r_[0:7,8:15]];d.qpos[fadr]=np.clip(q[[7,15]],flo,fhi)[:,None];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera,scene_option=opt)
  return label(cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR),title)
 def render(i):
  imgs=[]
  for cam in CAMERAS:
   path=ROOT/str(episode['image_path_'+cam][i]);im=cv2.imread(str(path))
   if im is None:raise RuntimeError(f'cannot read {path}')
   imgs.append(label(cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA),f'REAL {cam} | episode {eid}'))
  demo=sim(data['recorded_ik_target'][i],'RECORDED IK TARGET (reference)')
  pred=sim(data['predicted_ik_target'][i],report.get('model_display','MODEL IK TARGET (teacher-forced)'))
  out=np.vstack((np.hstack((demo,pred,imgs[0])),np.hstack((imgs[2],imgs[3],imgs[1]))))
  elapsed=(int(data['timestamp_ns'][i])-int(data['timestamp_ns'][0]))/1e9;anchor=int(data['replan_anchor'][i])
  cv2.rectangle(out,(0,490),(1440,540),(0,0,0),-1)
  cv2.putText(out,f'Episode {eid} | {i+1}/{n} | {elapsed:.2f}s | model replan at {anchor/30:.2f}s uses RECORDED real observations',(5,508),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
  cv2.putText(out,'NOT closed loop; no calibrated object/drawer state | Space pause A/D frame J/L 1sec R restart Q quit',(5,532),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
  return out
 try:
  if a.check:
   for i in (0,n//2,n-1):
    img=render(i)
    if i==n//2 and a.screenshot and not cv2.imwrite(str(a.screenshot),img):raise RuntimeError('could not write screenshot')
   print(f'PASS: episode {eid} first, midpoint, final rendered; {n} frames',flush=True);return
  if a.video:
   writer=cv2.VideoWriter(str(a.video),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(1440,540))
   if not writer.isOpened():raise RuntimeError('could not open video writer')
   try:
    for i in range(n):
     writer.write(render(i))
     if (i+1)%100==0:print(f'rendered {i+1}/{n}',flush=True)
   finally:writer.release()
   print(f'saved {a.video}: {n} frames at {a.fps} fps',flush=True);return
  print('IMPORTANT: full held-out episode uses recorded real observations at replans; NOT closed-loop rollout or proof of success',flush=True)
  print('Space pause; A/D frame; J/L jump 1 second; R restart; Q quit.',flush=True)
  title='Full validation episode - teacher-forced, NOT closed loop';cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,1440,540);i=0;paused=False
  while True:
   cv2.imshow(title,render(i));key=cv2.waitKey(0 if paused else max(1,round(1000/a.fps)))&255
   if key in (ord('q'),27):break
   if key==ord(' '):paused=not paused
   elif key==ord('a'):i=max(0,i-1);paused=True
   elif key==ord('d'):i=min(n-1,i+1);paused=True
   elif key==ord('j'):i=max(0,i-30);paused=True
   elif key==ord('l'):i=min(n-1,i+30);paused=True
   elif key==ord('r'):i=0;paused=False
   elif not paused:
    if i==n-1:paused=True
    else:i+=1
  cv2.destroyAllWindows()
 finally:renderer.close()
if __name__=='__main__':main()
