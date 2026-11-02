"""逐 trial 独立进程的批量评估驱动：彻底消除进程内状态污染与批次依赖。

为什么需要它：
2026-11-01 实测发现同一批脚本里，同一起点的结果会随"批次里还有哪些起点"而改变
（ID2 在 `--episode-ids 12 2 6` 下为 20.20mm，在 `--episode-ids 2 --repeat 2` 下为
-0.0mm）。这类跨 trial 状态污染（GL 上下文 / JAX 运行时）无法靠调参消除，只能用
进程隔离切断。

本脚本不改判定逻辑，只负责按 trial 逐个调用 `run_drawer_eval_batch.py`。代价是每个
trial 都要重新加载策略（约 20 秒），33 次约多 11 分钟，换取"结果只依赖该 trial 自身"。

起点清单解析：
默认加载主示教清单，并**自动加入**项目内固定的 eval-only 起点集
（`data/processed/openarm_drawer_eval_v001/manifest.json`，编号 100-119）。
因此 `--episode-ids 100` 这类写法无需再显式传 `--extra-manifest`。

用法：
    python run_drawer_eval_isolated.py --episode-ids 100 101 \
        --output logs/a2_pi0/drawer_eval_isolated_smoke_20261101
    python run_drawer_eval_isolated.py \
        --output logs/a2_pi0/drawer_eval_isolated_full_20261101
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BATCH = Path(__file__).parent / 'run_drawer_eval_batch.py'
DEFAULT_MANIFEST = ROOT / 'data/processed/openarm_drawer_success_v003/manifest.json'
DEFAULT_EVAL_MANIFEST = ROOT / 'data/processed/openarm_drawer_eval_v001/manifest.json'


def episode_ids(manifests):
    ids = []
    for path in manifests:
        ids += [row['episode_id'] for row in json.loads(Path(path).read_text())['episodes']]
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='输出目录')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--weight-precision',choices=('auto','bfloat16','checkpoint'),default='auto')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST, help='主示教清单')
    parser.add_argument('--extra-manifest', type=Path, nargs='*', default=[],
                        help='额外起点清单；省略时自动加入项目内 eval-only 清单')
    parser.add_argument('--no-eval-manifest', action='store_true', help='不自动加入 eval-only 清单')
    parser.add_argument('--episode-ids', type=int, nargs='+')
    parser.add_argument('--seconds', type=float, default=24.0)
    parser.add_argument('--task', choices=('open', 'roundtrip'), default='open')
    parser.add_argument('--action-chunk', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='只打印将执行的子命令，不真正运行')
    args = parser.parse_args()

    manifests = [args.manifest, *args.extra_manifest]
    if not args.no_eval_manifest and DEFAULT_EVAL_MANIFEST.exists() and \
            DEFAULT_EVAL_MANIFEST not in manifests:
        manifests.append(DEFAULT_EVAL_MANIFEST)

    available = episode_ids(manifests)
    ids = args.episode_ids or available
    missing = [eid for eid in ids if eid not in available]
    if missing:
        raise SystemExit(f'起点不存在于任何清单：{missing}（可用：{sorted(available)}）')
    if args.repeat < 1:
        raise SystemExit('repeat 必须 >= 1')

    print(f'ISOLATED_PLAN trials={len(ids) * args.repeat} episodes={ids} repeat={args.repeat}',
          flush=True)
    print(f'ISOLATED_MANIFESTS {[str(m) for m in manifests]}', flush=True)

    offset = 0
    for episode in ids:
        for _ in range(args.repeat):
            command = [sys.executable, str(BATCH),
                       '--output', str(args.output), '--append',
                       '--trial-offset', str(offset), '--repeat', '1',
                       '--episode-ids', str(episode),
                       '--seconds', str(args.seconds), '--task', args.task,
                       '--action-chunk', str(args.action_chunk), '--seed', str(args.seed),
                       '--manifest', str(args.manifest)]
            command += ['--weight-precision',args.weight_precision]
            if len(manifests) > 1:
                command += ['--extra-manifest', *map(str, manifests[1:])]
            if args.checkpoint:
                command += ['--checkpoint', str(args.checkpoint)]
            if args.video:
                command += ['--video']
            print(f'ISOLATED_TRIAL offset={offset} episode={episode}', flush=True)
            if args.dry_run:
                print('DRY_RUN ' + ' '.join(command), flush=True)
                offset += 1
                continue
            subprocess.run(command, check=True)
            offset += 1

    if args.dry_run:
        print(f'DRY_RUN_DONE 将执行 {offset} 个 trial，未真正运行', flush=True)
        return

    index_path = args.output / 'index.json'
    if not index_path.exists():
        raise SystemExit(f'未产出 {index_path}，请检查子进程输出')
    index = json.loads(index_path.read_text())
    successes = sum(1 for item in index if item['success'])
    print(f'ISOLATED_DONE trials={len(index)} success={successes} output={args.output}', flush=True)


if __name__ == '__main__':
    main()
