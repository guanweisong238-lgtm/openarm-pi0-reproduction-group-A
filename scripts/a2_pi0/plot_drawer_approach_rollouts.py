"""先用venv --prepare-only计算物理距离，再用系统Python绘图。"""
from pathlib import Path
import argparse,json,os
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/approach13_delta_012'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))


def prepare():
    from drawer_simulation import DrawerSimulation
    import mujoco,numpy as np
    phases=json.loads((ROOT/'results/drawer_approach_phases_v001/phases.json').read_text())['episodes']
    drift=json.loads((ROOT/'results/drawer_approach_phases_v001/model012_drift.json').read_text());series=[]
    for name in ['ready','warm']:
        s=json.loads((ROOT/f'results/drawer_approach_phases_v001/model012_{name}_score.json').read_text())
        row=next(r for r in phases if r['episode_id']==s['episode_id'])
        with np.load(ROOT/row['path']/'episode.npz') as e:reference=e['qpos'].copy()
        env=DrawerSimulation(ROOT/row['path']);scratch=mujoco.MjData(env.model);sid=env.model.site('right_ee_control_point').id
        def point(q):
            scratch.qpos[:]=q;mujoco.mj_kinematics(env.model,scratch)
            return scratch.site_xpos[sid]+scratch.site_xmat[sid].reshape(3,3)@np.array([0,0,-.16])
        try:
            target=point(reference[row['first_stable_contact_frame']]);teacher=[float(np.linalg.norm(point(reference[i+1])-target)*1000) for i in range(s['frames'])]
        finally:env.close()
        series.append(dict(score=s,teacher=teacher,drift=next(r['rows'] for r in drift if r['episode_id']==s['episode_id'])))
    OUT.mkdir(exist_ok=True);(OUT/'plot_data.json').write_text(json.dumps(series)+'\n')


def plot():
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series=json.loads((OUT/'plot_data.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,7))
    for col,item in enumerate(series):
        s=item['score'];d=item['drift'];t=np.arange(1,s['frames']+1)/30
        axes[0,col].plot(t,item['teacher'],label='Original demonstration',color='#2563eb')
        axes[0,col].plot(t,[v['distance_to_reference_contact_mm'] for v in s['trajectory']],label='Actual model rollout',color='#ea580c')
        axes[0,col].set_title(f"Train ID{s['episode_id']} | {s['initialization']}")
        axes[1,col].plot(t,[v['goal_position_error_mm'] for v in d],label='Requested goal vs reference goal',color='#7c3aed')
        axes[1,col].plot(t,[v['physical_position_error_mm'] for v in d],label='Actual vs reference state',color='#ea580c')
        axes[1,col].set_xlabel('Simulation seconds')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_ylim(bottom=0)
    axes[0,0].set_ylabel('Distance to demonstrated contact point (mm)')
    axes[1,0].set_ylabel('Same-time deviation (mm; sensitive to pace)')
    fig.suptitle('Approach 012/1200 | real physics | seed42, resampled noise, chunk4 | no stable contact')
    fig.tight_layout();fig.savefig(OUT/'closed_loop_drift.png',dpi=160);fig.savefig(OUT/'closed_loop_drift.svg')
    print(OUT/'closed_loop_drift.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:prepare()
    else:plot()
