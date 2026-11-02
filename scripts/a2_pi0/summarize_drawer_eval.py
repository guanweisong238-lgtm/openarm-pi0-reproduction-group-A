"""汇总批量评估结果：按协议分组出成功率、耗时、延迟与失败模式分布。

只做后处理：读 `run_drawer_eval_batch.py` 产出的 `index.json`（以及
`run_config.json` 记录的后端与固定配置），不重算判据、不改阈值。

分组口径见 `docs/drawer-eval-protocol.md` 第 8.1 节：
- train / validation：示教清单原有划分
- eval_only：新增扰动起点，唯一的泛化数字

产物：`metrics.json`（机器可读）与 `metrics.md`（人可读表格）。
"""
from pathlib import Path
import argparse
import json
import statistics
from collections import Counter
from datetime import datetime


def describe(rows):
    """一个分组内的统计量；空组返回 None。"""
    if not rows:
        return None
    successes = [row for row in rows if row['success']]
    peaks = [row['peak_open_mm'] for row in rows if row.get('peak_open_mm') is not None]
    first_steps = [row['first_success_step'] for row in successes if row.get('first_success_step')]
    sim_seconds = [row['simulation_seconds'] for row in successes if row.get('simulation_seconds') is not None]
    p50 = [row['inference_p50_ms'] for row in rows if row.get('inference_p50_ms') is not None]
    p95 = [row['inference_p95_ms'] for row in rows if row.get('inference_p95_ms') is not None]
    return dict(
        trials=len(rows), success=len(successes), success_rate=len(successes) / len(rows),
        peak_open_mm_median=statistics.median(peaks) if peaks else None,
        peak_open_mm_min=min(peaks) if peaks else None,
        peak_open_mm_max=max(peaks) if peaks else None,
        first_success_step_median=statistics.median(first_steps) if first_steps else None,
        simulation_seconds_median_on_success=statistics.median(sim_seconds) if sim_seconds else None,
        inference_p50_ms_median=statistics.median(p50) if p50 else None,
        inference_p95_ms_median=statistics.median(p95) if p95 else None,
        failure_modes=dict(Counter(row['failure_mode'] for row in rows if row.get('failure_mode'))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='批量评估输出目录（含 index.json）')
    parser.add_argument('--output', type=Path, help='默认写到 --run 目录')
    args = parser.parse_args()
    index = json.loads((args.run / 'index.json').read_text())
    config = json.loads((args.run / 'run_config.json').read_text())
    output = args.output or args.run

    groups, repeated = {}, {}
    for row in index:
        if row.get('repeat', 1) > 1:
            repeated.setdefault(row['episode_id'], []).append(row)
        key = 'eval_only' if row.get('eval_only') else row['split']
        groups.setdefault(key, []).append(row)

    order = [key for key in ('train', 'validation', 'eval_only') if key in groups]
    summary = {key: describe(groups[key]) for key in order}
    stability = {episode: describe(rows) for episode, rows in repeated.items()}
    total = describe(index)

    payload = dict(run_directory=str(args.run), generated_at=datetime.now().astimezone().isoformat(),
                   checkpoint=config['checkpoint'],
                   checkpoint_protocol_sha256=config['checkpoint_protocol_sha256'],
                   gl_backend=config['run'].get('gl_backend'),
                   runtime=config['run'], protocol_snapshot=config.get('protocol'),
                   groups=summary, overall=total, repeated_start_stability=stability,
                   scope='成功率按协议第 8.1 节分组；eval_only 为唯一泛化数字',
                   caveats=[
                       '仿真时间在推理期暂停，不代表硬件实时性能',
                       '确定性策略：同起点重复不计独立样本，只作稳定性参考',
                       '本表不改判据；判据见 docs/drawer-eval-protocol.md'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')

    def pct(value):
        return '—' if value is None else f'{value * 100:.1f}%'

    def num(value, digits=1):
        return '—' if value is None else f'{value:.{digits}f}'

    lines = ['# 抽屉拉取批量评估结果', '',
             f'运行目录：`{args.run}`',
             f'检查点：`{config["checkpoint"]}`',
             f'渲染后端：`{config["run"].get("gl_backend")}`（跨批次比较必须一致）',
             f'固定配置：{config["run"]}', '',
             '## 分组结果', '',
             '| 分组 | 次数 | 成功 | 成功率 | 峰值开度中位数 | 首次成功步中位数 | 成功时仿真耗时中位数 | 推理 p50 / p95 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key in order:
        row = summary[key]
        lines.append(f'| {key} | {row["trials"]} | {row["success"]} | {pct(row["success_rate"])} | '
                     f'{num(row["peak_open_mm_median"])} mm | {num(row["first_success_step_median"], 0)} | '
                     f'{num(row["simulation_seconds_median_on_success"], 2)} s | '
                     f'{num(row["inference_p50_ms_median"])} / {num(row["inference_p95_ms_median"])} ms |')
    lines += ['', '## 失败模式分布', '']
    for key in order:
        modes = summary[key]['failure_modes']
        lines.append(f'- **{key}**：' + ('无失败' if not modes else
                                           '，'.join(f'`{name}` {count}' for name, count in sorted(modes.items()))))
    lines += ['', '## 逐条明细', '', '| 起点 | 分组 | 判定 | 峰值开度 | 首次成功步 | 仿真耗时 | 失败模式 |',
              '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for row in index:
        verdict = '成功' if row['success'] else '失败'
        lines.append(f'| {row["episode_id"]} | {"eval_only" if row.get("eval_only") else row["split"]} | '
                     f'{verdict} | {num(row["peak_open_mm"])} mm | {num(row.get("first_success_step"), 0)} | '
                     f'{num(row["simulation_seconds"], 2)} s | {row.get("failure_mode") or "—"} |')
    if stability:
        lines += ['', '## 同起点重复（不计独立样本）', '']
        for episode, row in sorted(stability.items()):
            lines.append(f'- 起点 {episode}：{row["success"]}/{row["trials"]} 成功')
    lines += ['', '## 口径声明', ''] + [f'- {item}' for item in payload['caveats']] + ['']
    (output / 'metrics.md').write_text('\n'.join(lines))
    print(f'PASS 汇总 {len(index)} 条 -> {output / "metrics.json"} , {output / "metrics.md"}', flush=True)
    for key in order:
        row = summary[key]
        print(f'  {key}: {row["success"]}/{row["trials"]} = {pct(row["success_rate"])}', flush=True)


if __name__ == '__main__':
    main()
