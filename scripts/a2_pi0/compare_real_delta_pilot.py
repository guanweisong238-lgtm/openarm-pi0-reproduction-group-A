#!/usr/bin/env python3
"""Compare cached offline traces, not a closed-loop or physical simulation."""
import json
from pathlib import Path
import numpy as np
from analyze_real_handoff_blend import ARM,blend_handoffs
from real_training_support import ROOT
old=ROOT/'results/real_prefix_schedule_compare_20261122';new=ROOT/'results/real_delta_schedule_pilot_20261122'
a=np.load(old/'traces.npz');b=np.load(new/'traces.npz');r=json.loads((new/'report.json').read_text());prior=json.loads((old/'report.json').read_text());n=min(r['coverage_frames'],prior['coverage_frames']);s=r['modes']['delta_pilot_9']['switch_frames'];ref=b['recorded_ik_target'][:n];assert np.array_equal(a['timestamp_ns'][:n],b['timestamp_ns'][:n])
result={}
for name,q in [('absolute_prefix_raw',a['prefix_pilot_9'][:n]),('absolute_prefix_blend8',blend_handoffs(a['prefix_pilot_9'][:n],s,8)),('delta_prefix_raw',b['delta_pilot_9'][:n]),('delta_prefix_blend8',blend_handoffs(b['delta_pilot_9'][:n],s,8))]:
 v=np.diff(q[:,ARM],axis=0);acc=np.diff(v,axis=0);peak=np.max(abs(v),axis=1);bidx=np.asarray(s)-1
 result[name]={'arm_target_mae_rad':float(np.abs(q[:,ARM]-ref[:,ARM]).mean()),'boundary_median_rad':float(np.median(peak[bidx])),'boundary_max_rad':float(peak[bidx].max()),'whole_max_step_rad':float(peak.max()),'second_diff_max_joint_median_rad_per_frame':float(np.median(np.max(abs(acc),axis=1))),'second_diff_max_joint_p95_rad_per_frame':float(np.percentile(np.max(abs(acc),axis=1),95)),'finite':bool(np.isfinite(q).all())}
out={'episode':17,'frames':n,'delay_cycles':9,'flow_steps':10,'results':result,'limitations':['teacher-forced recorded observations; no physics or closed-loop task success','delta pilot trained only 100 windows from absolute-model adapter; not a converged comparison','first window from absolute baseline; predicted deltas are accumulated and may drift','8-step blending applied offline to arm joints only']}
(new/'comparison_metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
