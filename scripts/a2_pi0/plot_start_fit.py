"""系统Python：绘制起步拟合的固定帧指标，区别于随机样本训练loss。"""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summaries=[]
    for directory in args.results:summaries.extend(json.loads((directory/'summary.json').read_text()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    fields=[('train','mean_position_mm','Training frames: mean position error (mm)'),
            ('train','worst_position_mm','Training frames: worst position error (mm)'),
            ('train','worst_attitude_deg','Training frames: worst attitude error (deg)'),
            ('validation','mean_position_mm','Held-out start frames: mean position error (mm)')]
    for summary in summaries:
        protocol=json.loads((Path(summary['run_directory'])/'protocol.json').read_text())
        representation=protocol.get('action_representation','absolute')
        label=f"{summary['variant']} | {representation}"
        label+=f" | aug={protocol.get('augmentation',True)} | first4={protocol.get('first_four_weight',1):g}"
        label+=f" | t={protocol.get('time_sampling','beta')}"
        records=[json.loads(line) for line in (Path(summary['run_directory'])/'evaluation.jsonl').read_text().splitlines()]
        for axis,(split,key,title) in zip(axes.flat,fields):
            axis.plot([r['step'] for r in records],[r['splits'][split][key] for r in records],marker='o',label=label)
            axis.set_title(title,fontsize=10);axis.set_yscale('log');axis.grid(alpha=.2)
    for axis in axes.flat:
        axis.axhline(2.,color='#64748b',lw=1,ls='--',label='2 mm / deg diagnostic target')
        axis.set_xlabel('Additional training steps');axis.legend(fontsize=8)
    fig.suptitle('Start-frame fitting diagnostic | original 16-step targets | three inference seeds')
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160);fig.savefig(args.output.with_suffix('.svg'))
    print(args.output)


if __name__=='__main__':main()
