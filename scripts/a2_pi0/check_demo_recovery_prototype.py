#!/usr/bin/env python3
"""Numeric-only recovery feasibility probe; no simulator physics or training labels."""
import json
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
src=ROOT/'results/real_delta_schedule_pilot_20261122/traces.npz'
out=ROOT/'results/real_demo_recovery_probe_20261122';out.mkdir(exist_ok=True)
z=np.load(src);pred=z['delta_pilot_9'];demo=z['recorded_ik_target'];arm=np.r_[0:7,8:15]
# Illustration limits, not certified control limits. Grippers stay fixed.
rate=30.; vmax=np.tile(np.array([2,2,3.3,3.3,6.3,6.3,6.3]),2);cap=np.minimum(vmax/rate,0.02)
rows=[]
for i in (100,200,300,400,500):
 start=pred[i].copy(); goal=demo[i].copy();q=start.copy();path=[q.copy()]
 for k in range(1000):
  delta=goal[arm]-q[arm];q[arm]+=np.clip(delta,-cap,cap);path.append(q.copy())
  if np.max(np.abs(goal[arm]-q[arm]))<1e-5:break
 path=np.asarray(path); np.savez_compressed(out/f'frame_{i}_numeric_only.npz',joint_targets=path,frame=i)
 rows.append({'frame':i,'initial_max_arm_error_rad':float(np.max(np.abs(start[arm]-goal[arm]))),'initial_arm_mae_rad':float(np.mean(np.abs(start[arm]-goal[arm]))),'steps_to_fixed_reference':len(path)-1,'duration_seconds':(len(path)-1)/rate,'final_max_arm_error_rad':float(np.max(np.abs(path[-1,arm]-goal[arm]))),'grippers_unchanged':bool(np.array_equal(path[0,[7,15]],path[-1,[7,15]]))})
report={'episode':17,'kind':'offline numeric illustration only; NOT correction demonstrations','reference':'recorded IK target at same time index, frozen during recovery','starting_state':'model-predicted target only, not observed robot state','maximum_arm_step_rad':0.02,'rate_hz':rate,'cases':rows,'limitations':['No collision, joint-limit, acceleration, grasp or contact check.','No updated images, no object or drawer state, no physics and no expert task success validation.','Same-index correspondence is oracle information from a recorded episode and unavailable in novel online situations.','Outputs must not be used as training samples or robot commands.']}
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
