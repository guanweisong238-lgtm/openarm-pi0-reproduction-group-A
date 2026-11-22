#!/usr/bin/env python3
"""Quantify teacher-forced trajectory stitching discontinuities, not robot dynamics."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/real_episode17_jump_analysis_20261122'
A=ROOT/'results/real_full_episode_17_offline_20261121/full_episode.npz'
B=ROOT/'results/real_full_episode_17_fixed_noise_20261122/full_episode.npz'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()

def main():
 x=np.load(A,allow_pickle=False);y=np.load(B,allow_pickle=False)
 for name in ('timestamp_ns','observed_state','recorded_ik_target','anchor_indices'):
  np.testing.assert_array_equal(x[name],y[name])
 anchors=x['anchor_indices'];n=len(x['timestamp_ns']);boundary=np.zeros(n-1,dtype=bool);boundary[anchors[1:]-1]=True
 names=['R1','R2','R3','R4','R5','R6','R7','R gripper','L1','L2','L3','L4','L5','L6','L7','L gripper']
 arms=np.array([*range(7),*range(8,15)])
 def measure(q):
  changes=np.abs(np.diff(q,axis=0));maximum=changes.max(axis=1);arm=changes[:,arms].max(axis=1);top=np.argsort(maximum)[-8:][::-1]
  def stat(v):return {'median_rad':float(np.median(v)),'p95_rad':float(np.percentile(v,95)),'max_rad':float(np.max(v)),'count_over_0_1rad':int((v>.1).sum()),'count':int(len(v))}
  return {'all_boundary':stat(maximum[boundary]),'all_nonboundary':stat(maximum[~boundary]),'arm_boundary':stat(arm[boundary]),'arm_nonboundary':stat(arm[~boundary]),
   'largest_changes':[{'frame':int(i+1),'time_seconds':float((x['timestamp_ns'][i+1]-x['timestamp_ns'][0])/1e9),'joint':names[int(changes[i].argmax())],'delta_rad':float(maximum[i]),'is_replan_boundary':bool(boundary[i])} for i in top]}
 r={'episode_id':17,'sampling_frequency_hz':30,'replan_stride_frames':15,'boundary_count':int(boundary.sum()),'threshold_0_1rad':'descriptive diagnostic only, not safety limit','varying_noise':measure(x['predicted_ik_target']),'fixed_noise':measure(y['predicted_ik_target']),'recorded_ik':measure(x['recorded_ik_target']),'recorded_observation':measure(x['observed_state'])}
 dif=np.abs(x['predicted_windows'][:,0,:]-y['predicted_windows'][:,0,:]);r['same_observation_noise_comparison']={'first_anchor_same_seed_max_abs_rad':float(dif[0].max()),'different_seed_anchor_count':len(anchors)-1,'first_action_max_joint_difference_median_rad':float(np.median(dif[1:].max(axis=1))),'first_action_max_joint_difference_max_rad':float(dif[1:].max()),'limitation':'same observations, one fixed seed and 36 alternative seeds; two independent processes; not a complete noise ablation'}
 clipped=x['predicted_ik_target'].copy();clipped[:,7]=np.clip(clipped[:,7],-.7854,0);clipped[:,15]=np.clip(clipped[:,15],0,.7854);r['as_displayed_gripper_clipped']=measure(clipped)
 r['source_sha256']={str(A.relative_to(ROOT)):sha(A),str(B.relative_to(ROOT)):sha(B)}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'diagnosis.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 time=(x['timestamp_ns'][1:]-x['timestamp_ns'][0])/1e9
 cols=arms
 fig,(a,b)=plt.subplots(2,1,figsize=(13,7),sharex=True,constrained_layout=True)
 for ax,z,title in [(a,x,'Different noise at each replan'),(b,y,'Fixed noise seed at each replan')]:
  predicted=np.abs(np.diff(z['predicted_ik_target'][:,cols],axis=0)).max(axis=1);observed=np.abs(np.diff(z['observed_state'][:,cols],axis=0)).max(axis=1)
  ax.plot(time,predicted,label='Predicted target',color='tab:red',lw=1);ax.plot(time,observed,label='Recorded observed joints',color='tab:blue',lw=1)
  ax.scatter(time[boundary],predicted[boundary],s=13,color='tab:red',zorder=3,label='Replan boundary');ax.grid(alpha=.2);ax.set(ylabel='Max arm-joint step (rad)',title=title);ax.legend(loc='upper right',fontsize=8)
 b.set_xlabel('Episode time (s)');fig.savefig(OUT/'joint_jumps.png',dpi=150)
 print(json.dumps({'varying_noise':r['varying_noise']['arm_boundary'],'fixed_noise':r['fixed_noise']['arm_boundary'],'recorded_observation':r['recorded_observation']['arm_boundary'],'noise_comparison':r['same_observation_noise_comparison']},indent=2))
if __name__=='__main__':main()
