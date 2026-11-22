#!/usr/bin/env python3
"""Collect explicitly synthetic recovery candidates from a training episode, never test split."""
import json,subprocess,sys
from pathlib import Path
import mujoco as mj
import numpy as np
ROOT=Path(__file__).resolve().parents[2];ARM=np.r_[0:7,8:15]
trace=ROOT/'results/real_delta_schedule_1000_ep0_20261122/traces.npz';out=ROOT/'results/real_recovery_candidates_projected_v2_1000_ep0_20261122'
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
with np.load(trace) as z:pred=z['delta_pilot_9']
with np.load(ROOT/'data/processed/real_alignment_prototype_v001/episode_0/samples.npz') as z:ref=z['state']
m=mj.MjModel.from_xml_path(str(ROOT/'assets/real_replay_scene/empty_recovery_scene.xml'));ids=[m.joint(f'openarm_{s}_joint{i}').id for s in ('right','left') for i in range(1,8)];low=m.jnt_range[ids,0]+.02;high=m.jnt_range[ids,1]-.02
n=min(len(pred),len(ref));dev=np.max(abs(pred[:n,ARM]-ref[:n,ARM]),axis=1);gdev=np.max(abs(pred[:n,[7,15]]-ref[:n,[7,15]]),axis=1)
candidates=np.where(np.isfinite(pred[:n]).all(axis=1)&(dev>=.1)&(dev<=.6)&(pred[:n,ARM]>low).all(axis=1)&(pred[:n,ARM]<high).all(axis=1))[0]
selected=[]
for k in candidates:
 if not selected or k-selected[-1]>=10:selected.append(int(k))
 if len(selected)==4:break
records=[]
for k in selected:
 folder=out/f'episode_0_frame_{k}';cmd=[sys.executable,str(ROOT/'scripts/a2_pi0/run_model_deviation_recovery_sim.py'),'--episode','0','--frame',str(k),'--initial-trace',str(trace),'--output',str(folder)]
 proc=subprocess.run(cmd,text=True,capture_output=True);row={'frame':k,'initial_max_arm_deviation_rad':float(dev[k]),'initial_max_gripper_deviation_rad':float(gdev[k]),'exit_code':proc.returncode}
 if proc.returncode==0:row['result']=json.loads((folder/'report.json').read_text())
 else:row['failure']=proc.stderr[-1200:]
 records.append(row)
report={'source_split':'train','episode':0,'candidates_considered':len(candidates),'selected_frames':selected,'cases':records,'limitations':['simulator starts from an offline model-predicted target, not a model-controlled sim state','same-index recorded state supplies oracle reference; task phase not inferred','model gripper initial target is projected onto simulator range, so this is an assumed limited state, not measured actual feedback','four simulated cameras and task-free empty scene differ from real training images; not suitable as real correction data','a candidate is not a validated task recovery demonstration']}
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
