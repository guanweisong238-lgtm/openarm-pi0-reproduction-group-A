"""Plot locked unseen-start distributions and completed outcomes using system matplotlib."""
import csv
import json
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/drawer-eval-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',type=Path,default=ROOT/'data/processed/openarm_drawer_eval_v002/manifest.json')
parser.add_argument('--run',type=Path,default=ROOT/'logs/a2_pi0/drawer_generalization_020_20261102')
parser.add_argument('--output',type=Path,default=ROOT/'results/drawer_generalization_020_20261102')
args=parser.parse_args()
OUT = args.output
manifest = json.loads(args.manifest.read_text())
index_path = args.run/'index.json'
outcomes = {r['episode_id']: r for r in json.loads(index_path.read_text())} if index_path.exists() else {}
rows = []
for row in manifest['episodes']:
    p = row['eval_parameters']
    result = outcomes.get(row['episode_id'], {})
    rows.append(dict(episode_id=row['episode_id'], group=row['evaluation_group'],
                     x_delta_mm=(p['drawer_offset_x']-.1)*1000, y_delta_mm=p['drawer_offset_y']*1000,
                     handle_radius_mm=p['handle_radius']*1000, jitter_half_width_rad=p['right_jitter'],
                     joint_seed=p['seed'], success=result.get('success'),
                     peak_open_mm=result.get('peak_open_mm'), failure_mode=result.get('failure_mode')))
with (OUT / 'starts_and_outcomes.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
for col, group in enumerate(('near', 'expanded')):
    selected = [r for r in rows if r['group'] == group]
    done = sum(r['success'] is not None for r in selected)
    passed = sum(r['success'] is True for r in selected)
    for r in selected:
        color, marker = ('#16804a', 'o') if r['success'] is True else ('#c43632', 'x') if r['success'] is False else ('#90969c', '.')
        axes[0, col].scatter(r['x_delta_mm'], r['y_delta_mm'], c=color, marker=marker, s=35, alpha=.85)
        axes[1, col].scatter(r['handle_radius_mm'], r['jitter_half_width_rad'], c=color, marker=marker, s=35, alpha=.85)
    axes[0, col].set(title=f'{group}: {passed}/{done} success; {len(selected)} locked starts',
                     xlabel='Drawer X change from baseline (mm)', ylabel='Drawer Y change (mm)',
                     xlim=(-22, 22), ylim=(-55, 55))
    axes[0, col].axvline(0, color='#dddddd', lw=.7);axes[0, col].axhline(0, color='#dddddd', lw=.7)
    axes[1, col].set(xlabel='Handle radius (mm)', ylabel='Joint jitter sampling half-width (rad)',
                     xlim=(3.5,14.5), ylim=(0,.085))
    for axis in axes[:, col]:
        axis.grid(alpha=.18);axis.set_axisbelow(True)
fig.suptitle('Frozen pi0 endpoint 020/2000 — unseen start evaluation', fontsize=14)
handles=[Line2D([],[],marker=m,color=c,linestyle='',label=l) for m,c,l in
         [('o','#16804a','Success'),('x','#c43632','Failure'),('.','#90969c','Pending')]]
axes[0,0].legend(handles=handles,loc='upper left',fontsize=8)
fig.savefig(OUT / 'start_distribution.png', dpi=180)
fig.savefig(OUT / 'start_distribution.pdf')
plt.close(fig)
print(f'Plotted {len(rows)} locked starts; {len(outcomes)} completed outcomes')
