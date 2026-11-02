"""第 3 步自检判定：检查同实现重复性，并与 10-25 历史基线并列对照。

判定口径（2026-11-01 修订，见 `docs/drawer-eval-protocol.md` 变更记录）：
- **通过条件**：同一份实现、同一个起点重复运行时，峰值开度与成败判定一致（自洽）。
- **不通过条件**：重复运行结果不同，说明存在隐藏随机源，成功率无意义。
- 与 10-25 live 基线的一致性**只作参考展示，不作为通过条件**：批量无窗口实现与交互式
  脚本存在不可消除的结构差异，接触任务下差异大小与成败无单调关系（实测 ID2：差 6.47e-4
  失败，差 1.29e-3 反而成功）。

用法：
    python check_drawer_selfcheck.py --run logs/a2_pi0/drawer_eval_selfcheck_glfw_20261101 \
        --repeat-run logs/a2_pi0/drawer_eval_repeat_id2_20261101
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'results/drawer_pull_success_20261025/physical_scores.json'


def load_index(run):
    path = Path(run) / 'index.json'
    if not path.exists():
        raise SystemExit(f'缺少 {path}')
    return json.loads(path.read_text())


def load_baseline():
    """10-25 的 live 单次记录：只作历史观察，不是判定依据。"""
    if not BASELINE.exists():
        return {}
    payload = json.loads(BASELINE.read_text())
    rows = {}
    for item in payload.get('frozen_configuration_trials', []):
        rows[item['episode_id']] = dict(success=bool(item['model_pull_success']),
                                        peak_open_mm=float(item['peak_open_mm']),
                                        implementation='live_1025')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='主批次目录')
    parser.add_argument('--repeat-run', type=Path, nargs='*', default=[],
                        help='重复运行批次目录（同一实现）；用于自洽判定')
    parser.add_argument('--peak-tolerance-mm', type=float, default=1e-6,
                        help='重复运行峰值开度允许差（默认要求完全一致）')
    args = parser.parse_args()

    main_rows = load_index(args.run)
    baseline = load_baseline()

    by_episode = {}
    for row in main_rows:
        by_episode.setdefault(row['episode_id'], []).append(dict(row, source=str(args.run)))
    for run in args.repeat_run:
        for row in load_index(run):
            by_episode.setdefault(row['episode_id'], []).append(dict(row, source=str(run)))

    print(f'主批次 {args.run}')
    print(f'重复批次 {args.repeat_run or "（未提供）"}')
    print(f'检查起点 {sorted(by_episode)}\n')

    print(f'{"起点":<6}{"批次内次数":>10}{"成败":>16}{"峰值开度(mm)":>26}{"自洽":>8}{"live基线":>10}')
    repeatable, inconsistencies = True, []
    for episode in sorted(by_episode):
        rows = by_episode[episode]
        peaks = [row['peak_open_mm'] for row in rows]
        verdicts = [bool(row['success']) for row in rows]
        same_peak = (max(peaks) - min(peaks)) <= args.peak_tolerance_mm
        same_verdict = len(set(verdicts)) == 1
        consistent = same_peak and same_verdict
        if len(rows) > 1 and not consistent:
            repeatable = False
            inconsistencies.append(episode)
        verdict_text = '/'.join('成功' if v else '失败' for v in verdicts)
        peak_text = '/'.join(f'{p:.6f}' for p in peaks)
        reference = baseline.get(episode)
        reference_text = '—' if reference is None else (
            f'{"成功" if reference["success"] else "失败"} {reference["peak_open_mm"]:.1f}')
        print(f'{episode:<6}{len(rows):>10}{verdict_text:>16}{peak_text:>26}'
              f'{("一致" if consistent else "不一致"):>8}{reference_text:>10}')

    print()
    has_repeat = bool(args.repeat_run) or any(len(rows) > 1 for rows in by_episode.values())
    if not has_repeat:
        print('REPEAT_PENDING 未提供重复批次，无法判定自洽性。请用 --repeat 2 补跑。')
    elif repeatable:
        print('PASS 同一实现在所有起点上重复运行结果一致，确定性成立。')
        print('     10-25 live 基线仅作历史对照；批量实现需自建基线后再跑全量。')
    else:
        print(f'FAIL 以下起点重复运行结果不一致：{inconsistencies}')
        print('     存在隐藏随机源，成功率无意义；先修工具再跑全量。')

    print('\n参考：与 live 基线的对照（不是判定条件）')
    for episode in sorted(by_episode):
        reference = baseline.get(episode)
        if reference is None:
            continue
        batch = by_episode[episode][0]
        flag = '一致' if bool(reference['success']) == bool(batch['success']) else '不一致'
        print(f'  起点 {episode}: live={"成功" if reference["success"] else "失败"} '
              f'({reference["peak_open_mm"]:.2f}mm) vs 批量={"成功" if batch["success"] else "失败"} '
              f'({batch["peak_open_mm"]:.2f}mm) -> {flag}')
    return 0 if (has_repeat and repeatable) else 1


if __name__ == '__main__':
    sys.exit(main())
