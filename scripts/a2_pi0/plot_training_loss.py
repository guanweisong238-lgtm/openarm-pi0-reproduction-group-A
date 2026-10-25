"""Native Matplotlib live loss figure; read-only log monitor, refresh every N steps.

Use system Python if Matplotlib is unavailable in the training venv.
--once writes the same PNG without opening a desktop window.
"""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))


def read_records(path):
    if not path.exists():return []
    records=[]
    # A writer may currently be appending the final line; wait for its newline.
    for number,line in enumerate(path.read_text().splitlines(keepends=True),1):
        if not line.endswith('\n'):continue
        try:records.append(json.loads(line))
        except json.JSONDecodeError as error:raise ValueError(f'Invalid complete JSON line {path}:{number}') from error
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--every',type=int,default=100)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--label',default='OpenArm v003 LoRA',help='本地窗口及图像标题，便于区分配对实验')
    parser.add_argument('--loss-label',default='Train flow loss (all32)',help='训练损失的实际口径')
    parser.add_argument('--fixed-field',default='flow_all32')
    parser.add_argument('--fixed-label',default='Fixed-anchor flow loss (all32)')
    args=parser.parse_args()
    if args.every<1:parser.error('--every must be positive')
    if not args.run_dir.is_dir():parser.error('run directory does not exist')
    import matplotlib
    matplotlib.use('Agg' if args.once else 'TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    fig,(train_ax,eval_ax)=plt.subplots(2,1,figsize=(10.5,7),sharex=True)
    if not args.once:fig.canvas.manager.set_window_title(args.label+' — local live loss')
    fig.subplots_adjust(left=.10,right=.97,top=.88,bottom=.12,hspace=.32)
    title=fig.suptitle('OpenArm v003 LoRA: training loss',fontsize=14)
    footer=fig.text(.10,.035,'',fontsize=9)
    raw_line,=train_ax.plot([],[],color='#94a3b8',alpha=.5,lw=.7,label='Per-step train loss (random samples)')
    smooth_line,=train_ax.plot([],[],color='#2563eb',lw=2,label=f'Trailing {args.every}-step mean')
    fixed_train,=eval_ax.plot([],[],color='#2563eb',marker='o',lw=1.6,label='Fixed training anchors (no augmentation)')
    fixed_val,=eval_ax.plot([],[],color='#ea580c',marker='o',lw=1.8,label='Fixed held-out validation anchors (no augmentation)')
    train_ax.set_ylabel(args.loss_label)
    eval_ax.set_ylabel(args.fixed_label)
    eval_ax.set_xlabel('Training step')
    for ax in (train_ax,eval_ax):
        ax.grid(alpha=.22);ax.legend(loc='upper right',fontsize=8);ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    last_signature=None
    output=args.run_dir/'loss_curve.png'

    def refresh():
        nonlocal last_signature
        metrics=read_records(args.run_dir/'metrics.jsonl')
        evaluations=read_records(args.run_dir/'evaluation.jsonl')
        latest=max((r['step'] for r in metrics),default=0)
        finished=(args.run_dir/'summary.json').exists()
        boundary=latest if finished else latest//args.every*args.every
        visible=[r for r in metrics if r['step']<=boundary]
        evaluated=[r for r in evaluations if r['step']<=boundary]
        signature=(boundary,len(evaluated),finished)
        if signature==last_signature:return
        if not visible and not evaluated:return
        steps=np.array([r['step'] for r in visible]);loss=np.array([r['loss'] for r in visible],float)
        if not np.isfinite(loss).all():raise ValueError('Nonfinite training loss in logs')
        raw_line.set_data(steps,loss)
        if len(loss)>=args.every:
            # Cumulative sum gives an exact, trailing window without zero-padding bias.
            sums=np.r_[0.,np.cumsum(loss)]
            smoothed=(sums[args.every:]-sums[:-args.every])/args.every
            smooth_line.set_data(steps[args.every-1:],smoothed)
            mean_text=f'{args.every}-step mean: {smoothed[-1]:.6f}'
        else:
            smooth_line.set_data([],[]);mean_text=f'Waiting for {args.every} training steps'
        xs=[r['step'] for r in evaluated]
        fixed_train.set_data(xs,[r['splits']['train'][args.fixed_field] for r in evaluated])
        fixed_val.set_data(xs,[r['splits']['validation'][args.fixed_field] for r in evaluated])
        for ax in (train_ax,eval_ax):
            ax.relim();ax.autoscale_view();ax.set_ylim(bottom=0)
        eval_ax.set_xlim(0,max(args.every,boundary)*1.03)
        state='Finished' if finished else 'Live'
        title.set_text(f'{args.label} — {state} — step {boundary}')
        footer.set_text(f'{mean_text}\n{args.run_dir.name} | refresh every {args.every} steps; validation retains its actual evaluation interval')
        fig.canvas.draw_idle()
        temporary=output.with_name('loss_curve.pending.png')
        fig.savefig(temporary,dpi=150)
        temporary.replace(output)
        if finished:fig.savefig(output.with_suffix('.svg'))
        last_signature=signature
        print(f'PLOT_UPDATED step={boundary} evaluations={len(evaluated)} status={state} PNG={output}',flush=True)

    refresh()
    if args.once:
        plt.close(fig)
        return
    timer=fig.canvas.new_timer(interval=1000)
    timer.add_callback(refresh)
    timer.start()
    print('PLOT_WINDOW_READY: refresh every '+str(args.every)+' steps; closing this window does not stop training',flush=True)
    try:plt.show()
    finally:timer.stop()


if __name__=='__main__':main()
