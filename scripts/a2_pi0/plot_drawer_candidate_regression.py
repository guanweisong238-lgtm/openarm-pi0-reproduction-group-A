"""Standalone scientific plot of one paired development regression (system Python)."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/drawer-eval-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/drawer_grasp_improvement_20261102'
rows=json.loads((OUT/'candidate_failure_comparison.json').read_text())['regressions']
case=next(r for r in rows if r['candidate_step']==4000 and r['episode_id']==300)
fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
minimum_opening=0.
for name,key,color in [('020 baseline','baseline_trial','#2475aa'),('022 / 4000','candidate_trial','#d0523d')]:
    with np.load(ROOT/case[key]/'closed_loop.npz') as raw:
        time=np.arange(1,len(raw['opening'])+1)/30
        minimum_opening=min(minimum_opening,float(raw['opening'].min()*1000))
        bilateral=((raw['contacts']>0)&(raw['forces']>.01)).all(1)
        streak=[];n=0
        for hit in bilateral:n=n+1 if hit else 0;streak.append(n)
        axes[0].plot(time,raw['opening']*1000,color=color,label=name,lw=1.8)
        axes[1].step(time,streak,where='post',color=color,label=name,lw=1.8)
axes[0].axhline(80,ls='--',color='#777777',lw=1,label='80 mm threshold')
axes[1].axhline(5,ls='--',color='#777777',lw=1,label='5-step grasp threshold')
axes[0].set(ylabel='Drawer opening (mm)',ylim=(min(-3,minimum_opening-2),108),title='Same development start ID300: baseline succeeds, candidate fails')
axes[1].set(ylabel='Consecutive bilateral-contact steps',xlabel='Simulation time (s)',xlim=(0,8),ylim=(-1,35))
for axis in axes:axis.grid(alpha=.2);axis.legend(loc='upper left',fontsize=9)
fig.savefig(OUT/'development_regression_id300.png',dpi=180)
fig.savefig(OUT/'development_regression_id300.pdf')
plt.close(fig)
print('REGRESSION_PLOT_SAVED')
