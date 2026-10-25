"""读取配对实验日志，校验可比性并导出指标表和独立图像。使用系统Python运行。"""
import argparse
import json
import os
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--early',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    runs=[args.control,args.early]
    protocols=[json.loads((run/'protocol.json').read_text()) for run in runs]
    evaluations=[read_lines(run/'evaluation.jsonl') for run in runs]
    metrics=[read_lines(run/'metrics.jsonl') for run in runs]
    configs=[re.sub(r"exp_name='[^']+'", "exp_name='<paired>'",(run/'config.txt').read_text()) for run in runs]
    if configs[0]!=configs[1]:raise ValueError('除实验名外的训练配置不同')
    for run in runs:
        if not (run/'summary.json').exists():raise ValueError('实验尚未完成：'+str(run))
    for key in ('source_checkpoint','manifest_sha256','norm_stats_sha256','anchors'):
        if protocols[0][key]!=protocols[1][key]:raise ValueError('配对协议不一致：'+key)
    if len(metrics[0])!=len(metrics[1]):raise ValueError('两组更新次数不同')
    if [r['episode_id'] for r in metrics[0]]!=[r['episode_id'] for r in metrics[1]]:
        raise ValueError('两组逐步抽取的轨迹不一致')
    for control,early in zip(*metrics):
        if not early['forced_early'] and early['local_frame']!=control['local_frame']:
            raise ValueError('未替换的帧不匹配')
    initial_max_difference=0.
    initial_differences_by_metric={}
    comparisons=[]
    for split in ('train','validation'):
        before=[e[0]['splits'][split]['slices'] for e in evaluations]
        after=[e[-1]['splits'][split]['slices'] for e in evaluations]
        for name in before[0]:
            for key in before[0][name]:
                difference=abs(before[0][name][key]-before[1][name][key])
                initial_max_difference=max(initial_max_difference,difference)
                initial_differences_by_metric[key]=max(initial_differences_by_metric.get(key,0.),difference)
            for key in ('first_action_position_mm','right_arm_mae_rad','first_action_attitude_deg'):
                c,w=after[0][name][key],after[1][name][key]
                comparisons.append(dict(split=split,subset=name,metric=key,
                    baseline=before[0][name][key],baseline_early=before[1][name][key],control=c,early60=w,
                    difference_in_changes=(w-before[1][name][key])-(c-before[0][name][key]),
                    early_relative_to_control_percent=(w/c-1)*100 if c else None))
    result=dict(control=str(args.control),early=str(args.early),updates_each=len(metrics[0]),
        initial_max_absolute_metric_difference=initial_max_difference,
        initial_max_difference_by_metric=initial_differences_by_metric,
        initial_metrics_bitwise_equal=initial_max_difference==0.,
        paired_episode_schedule_verified=True,
        sampling=[p['sampling_details'] for p in protocols],comparisons=comparisons)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    selected=('start_anchor/script_ready_start','start_anchor/human_warm_start',
              'early_anchor/script_ready_start','early_anchor/human_warm_start','legacy_anchor/all')
    lines=['# 早期采样配对实验离线结果','',
        '两组从同一2000步权重出发，各更新500步。负百分比表示early60比uniform对照的误差更低。',
        '权重来源、训练配置、逐步轨迹选择、未替换帧一致性已校验。离线指标不是闭环任务成功率。',
        '同权重的两次初始评估存在数值差异，表中同时列出；原因未确认，较小变化不宜过度解释。','',
        '| split / subset | 训练前 control / early | uniform +500 | early60 +500 | 相对uniform变化 | 变化量之差mm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        if row['metric']=='first_action_position_mm' and row['subset'] in selected:
            lines.append(f"| {row['split']} / {row['subset']} | {row['baseline']:.3f} / {row['baseline_early']:.3f} | {row['control']:.3f} | {row['early60']:.3f} | {row['early_relative_to_control_percent']:+.1f}% | {row['difference_in_changes']:+.3f} |")
    (args.output/'offline_results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,subset,title in zip(axes,(selected[0],selected[1],selected[-1]),
            ('Ready: first observation','Human warm: first observation','Original four anchors / episode')):
        rows=[next(r for r in comparisons if r['split']==split and r['subset']==subset and r['metric']=='first_action_position_mm') for split in ('train','validation')]
        for offset,key,label,color in ((-.24,'baseline','Original 2000 (mean of two evals)','#94a3b8'),(0,'control','Uniform +500','#2563eb'),(.24,'early60','Early60 +500','#ea580c')):
            values=[(r['baseline']+r['baseline_early'])/2 if key=='baseline' else r[key] for r in rows]
            errors=[abs(r['baseline']-r['baseline_early'])/2 for r in rows] if key=='baseline' else None
            bars=ax.bar(np.arange(2)+offset,values,yerr=errors,capsize=2,width=.23,label=label,color=color)
            ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
        ax.set_xticks([0,1],['Train','Validation']);ax.set_title(title,fontsize=10)
        ax.set_ylabel('First-action target position error (mm)');ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,max(r[k] for r in rows for k in ('baseline','control','early60'))*1.22)
    axes[0].legend(fontsize=8)
    fig.suptitle('Paired 500-step experiment | three inference seeds | lower is better')
    fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=160);fig.savefig(args.output/'comparison.svg')
    profile,profile_axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for axis,(split,group) in zip(profile_axes.flat,((s,g) for s in ('train','validation') for g in ('script_ready_start','human_warm_start'))):
        for source,label,color in ((evaluations[0][0],'Original 2000 (control eval)','#94a3b8'),
                                  (evaluations[0][-1],'Uniform +500','#2563eb'),
                                  (evaluations[1][-1],'Early60 +500','#ea580c')):
            points={}
            for anchor in source['splits'][split]['anchors']:
                if anchor['initialization']==group and anchor['local']<75:
                    points.setdefault(anchor['local'],[]).extend(v['first_action_position_mm'] for v in anchor['by_seed'].values())
            xs=sorted(points)
            axis.plot(xs,[np.mean(points[x]) for x in xs],marker='o',lw=1.5,label=label,color=color)
        axis.set_title(split+' / '+group,fontsize=10);axis.grid(alpha=.2)
        axis.set_ylabel('Target position error (mm)');axis.set_ylim(bottom=0)
        axis.set_xlabel('Observation frame within episode (30 Hz)')
    profile_axes[0,0].legend(fontsize=8)
    profile.suptitle('Early observation profile | episode/seed means | offline fixed observations')
    profile.tight_layout();profile.savefig(args.output/'early_profile.png',dpi=160);profile.savefig(args.output/'early_profile.svg')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
