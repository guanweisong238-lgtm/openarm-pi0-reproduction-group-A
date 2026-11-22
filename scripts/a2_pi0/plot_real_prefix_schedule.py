#!/usr/bin/env python3
"""Compare offline scheduled action discontinuities; no robot simulation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path('results/real_prefix_schedule_compare_20261122');d=np.load(p/'traces.npz');r=json.loads((p/'report.json').read_text());n=r['coverage_frames'];idx=np.r_[0:7,8:15];time=(d['timestamp_ns'][1:n]-d['timestamp_ns'][0])/1e9
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,constrained_layout=True)
for ax,cycles,title in zip(axes,(6,9),('Ideal 200 ms: switch after 20 actions','Simulated 275 ms: continue old plan for 3 extra actions')):
 for mode,label in [('unconditioned_baseline','Original model'),('prefix_pilot','Seven-step prefix pilot')]:
  key=f'{mode}_{cycles}';q=d[key][:n,idx];delta=np.abs(np.diff(q,axis=0)).max(1);bounds=np.array([i-1 for i in r['modes'][key]['switch_frames'] if 0<i<n]);ax.plot(time,delta,label=label,lw=.9,alpha=.8);ax.scatter(time[bounds],delta[bounds],s=9)
 ax.set(title=title,ylabel='Max arm-joint step (rad)');ax.grid(alpha=.2);ax.legend()
axes[-1].set_xlabel('Recorded episode time (s)');fig.savefig(p/'schedule_boundary_comparison.png',dpi=150);print('saved',p/'schedule_boundary_comparison.png')
