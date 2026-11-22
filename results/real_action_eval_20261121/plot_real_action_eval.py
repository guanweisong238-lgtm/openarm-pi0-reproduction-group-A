#!/usr/bin/env python3
"""Plot saved held-out action metrics; no training or inference."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
p=Path('results/real_action_eval_20261121');z=np.load(p/'predictions.npz');y=z['target'];obs=z['observed_state']
a=np.abs(z['prediction_500']-y);b=np.abs(z['prediction_1000']-y);hold=np.abs(obs[:,None,:]-y)
fig,(ax,bx)=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
t=np.arange(50)/30
for x,label in [(a,'500 steps'),(b,'1000 steps'),(hold,'Hold observed state')]:ax.plot(t,x.mean(axis=(0,2)),label=label,lw=2)
ax.set(xlabel='Prediction horizon (s)',ylabel='MAE vs recorded IK target (rad)',title='Mean absolute error across 40 validation windows');ax.grid(alpha=.25);ax.legend()
labels=[f'R{i}' for i in range(1,8)]+['R grip']+[f'L{i}' for i in range(1,8)]+['L grip'];i=np.arange(16);w=.27
for x,off,label in [(a,-w,'500 steps'),(b,0,'1000 steps'),(hold,w,'Hold observed state')]:bx.bar(i+off,x.mean(axis=(0,1)),w,label=label)
bx.set_xticks(i,labels,rotation=40,ha='right');bx.set(ylabel='MAE (rad)',title='Error by motor; observed-state hold is a simple baseline');bx.grid(axis='y',alpha=.25);bx.legend(ncol=3)
fig.savefig(p/'action_error_comparison.png',dpi=150);print('saved',p/'action_error_comparison.png')
