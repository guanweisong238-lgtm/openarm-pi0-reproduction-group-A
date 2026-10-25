"""本地绘图：分别呈现起步、接近、首次夹持的固定检查误差。"""
import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();records=[json.loads(s) for s in (args.run_dir/'evaluation.jsonl').read_text().splitlines()]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey='row')
    for col,(stage,title) in enumerate([('start_anchor','Start'),('approach_anchor','Approach'),('grasp_anchor','First bilateral contact')]):
        for split,color in [('train','#2563eb'),('validation','#ea580c')]:
            for row,key in enumerate(['first_action_position_mm','first_action_attitude_deg']):
                axes[row,col].plot([r['step'] for r in records],[r['splits'][split]['slices'][stage+'/all'][key] for r in records],marker='o',color=color,label=split)
        axes[0,col].set_title(title);axes[1,col].set_xlabel('Training step')
        for row in range(2):axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
    axes[0,0].set_ylabel('Mean first-action position error (mm)');axes[1,0].set_ylabel('Mean first-action attitude error (deg)')
    fig.suptitle('13 episodes | fixed observations, three inference seeds | offline diagnostic')
    fig.tight_layout();fig.savefig(args.run_dir/'phase_evaluation.png',dpi=160);fig.savefig(args.run_dir/'phase_evaluation.svg')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['worst_position_mm','worst_attitude_deg'],['ID12 start9 worst position (mm)','ID12 start9 worst attitude (deg)']):
        ax.plot([r['step'] for r in records],[r['splits']['train']['retained_id12_start_gate'][key] for r in records],marker='o')
        ax.axhline(2,color='gray',ls='--');ax.set_title(title);ax.set_xlabel('Training step');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.run_dir/'start_retention.png',dpi=160);fig.savefig(args.run_dir/'start_retention.svg')
    print(args.run_dir/'phase_evaluation.png')


if __name__=='__main__':main()
