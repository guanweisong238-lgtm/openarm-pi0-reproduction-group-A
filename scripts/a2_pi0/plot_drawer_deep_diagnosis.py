"""用系统Matplotlib绘制只读诊断证据；不启动网页或训练。"""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

directory=ROOT/'results/drawer_deep_diagnosis_20261024'
probe=json.loads((directory/'left_components_repeat_checked.json').read_text())['results'][0]
flow=json.loads((directory/'data_and_objective.json').read_text())['loss_shares']['early60']['train']
force=json.loads((directory/'initial_actuator_force.json').read_text())[0]
fig,axes=plt.subplots(1,3,figsize=(15,4.7),gridspec_kw={'width_ratios':[1.3,1,1]})
labels=[f'Left joint {i}' for i in range(1,8)]+['Left gripper']
changes=[probe['variants'][f'left_mean_dim_{i}']['change_from_original_tip_mm'] for i in range(8)]
axes[0].barh(labels,changes,color=['#94a3b8']*7+['#dc2626'])
axes[0].invert_yaxis();axes[0].set_xlabel('Change in right tip prediction (mm)')
axes[0].set_title('One left state feature set to its train mean',fontsize=10)
for i,value in enumerate(changes):axes[0].text(value+1,i,f'{value:.1f}',va='center',fontsize=9)
axes[0].set_xlim(0,max(changes)*1.18);axes[0].grid(axis='x',alpha=.2)
shares=[flow[k] for k in ('right7_loss_fraction','gripper_loss_fraction','left8_loss_fraction','padding_loss_fraction')]
axes[1].pie(shares,labels=['Right arm','Right grip','Unused left','Padding'],autopct='%1.1f%%',
            colors=['#2563eb','#60a5fa','#ea580c','#cbd5e1'],textprops={'fontsize':9})
axes[1].set_title('Flow error contribution: fixed train anchors',fontsize=10)
limits=np.array(force['force_limits_Nm']);ratio=abs(np.array(force['initial_actuator_force_Nm']))/np.max(abs(limits),axis=1)*100
axes[2].bar(range(1,8),ratio,color=['#dc2626' if r>=99.9 else '#94a3b8' for r in ratio])
axes[2].axhline(100,color='#dc2626',ls='--',lw=1);axes[2].set_ylim(0,113)
axes[2].set_xticks(range(1,8));axes[2].set_xlabel('Right arm joint');axes[2].set_ylabel('Actuator force / limit (%)')
axes[2].set_title('Ready: first predicted command, before stepping',fontsize=10);axes[2].grid(axis='y',alpha=.2)
fig.suptitle('Drawer policy diagnosis | Early60 checkpoint | input ablation is not task performance',fontsize=12)
fig.tight_layout();fig.savefig(directory/'evidence.png',dpi=160);fig.savefig(directory/'evidence.svg')
print(directory/'evidence.png')
