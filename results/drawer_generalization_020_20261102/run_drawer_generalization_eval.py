"""Lock 100 unseen starts, evaluate frozen 020/2000 once, and independently replay all trials.

No training, expert labels, checkpoint selection, or outcome-dependent start filtering.
Run with --stage prepare, then --stage run (requires host GPU/GLFW access).
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).parent
RESULT = ROOT / 'results/drawer_generalization_020_20261102'
DATA = ROOT / 'data/processed/openarm_drawer_eval_v002'
CONFIG = ROOT / 'configs/drawer-eval-starts-002.json'
RUN = ROOT / 'logs/a2_pi0/drawer_generalization_020_20261102'
OLD_RESULT = ROOT / 'results/drawer_eval_020_20261102'
CHECKPOINT = ROOT / 'checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint():
    frozen = read(OLD_RESULT / 'frozen_checkpoint_020_2000.json')
    for row in frozen['files']:
        path = CHECKPOINT / row['path']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Frozen checkpoint changed: {path}')
    return dict(verified_at=now(), files=frozen['files'], checkpoint=str(CHECKPOINT))


def state_key(path, scene):
    with np.load(path / 'episode.npz', allow_pickle=False) as raw:
        state = raw['initial_integration_state']
        eq = raw['initial_eq_data']
    geometry = [scene.get('drawer_offset_x_m', 0), scene.get('drawer_offset_y_m', 0),
                scene.get('drawer_handle_radius_m')]
    return hashlib.sha256(state.tobytes() + eq.tobytes() + json.dumps(geometry).encode()).hexdigest()


def prepare():
    if any(path.exists() for path in (RESULT, DATA, CONFIG, RUN)):
        raise FileExistsError('Do not overwrite a locked experiment; use a new version')
    RESULT.mkdir(parents=True)
    write(RESULT / 'checkpoint_before.json', verify_checkpoint())
    protocol = dict(
        created_at=now(), seed=2026110202, sample_counts=dict(near=70, expanded=30),
        episode_ids=list(range(200, 300)), checkpoint=str(CHECKPOINT),
        near=dict(x_m=[.09, .11], y_m=[-.03, .03], radius_m=[.005, .012], jitter_half_width_rad=[.01, .04]),
        expanded=dict(x_m=[.08, .12], y_m=[-.05, .05], radius_m=[.004, .014], jitter_half_width_rad=[.04, .08]),
        sampling='Independent continuous uniform draws; expanded jitter bound exceeds near bound. Joint deltas sampled within joint/control limits.',
        allowed_rejections='Only initial nonfinite state, active weld, illegal joint range, penetration >2mm, or exact duplicate scene+state; record all rejected candidates. No reachability/expert/policy success filtering.',
        runtime=dict(seconds=24, task='open', seed=42, action_chunk=4, weight_precision='checkpoint', gl_backend='glfw', isolated_process_per_trial=True),
        criterion='Original strict bilateral >.01N for5 control steps, contact pulling, opening >=.08m held30 steps, exact independent replay validity.',
        exclusion='All100 starts are eval_only and never used for gradients, expert labels, or checkpoint selection. No training in this experiment.',
        interpretation='New combinations from the same simulator/task; near and expanded populations reported separately. Expanded is not a pure single-factor ablation. No real-world claim.',
        generator_sha256=digest(Path(__file__)))
    write(RESULT / 'preregistered_protocol.json', protocol)
    (RESULT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    # The protocol above is written before drawing any starts or running a policy.
    from build_drawer_eval_starts import build_start, REFERENCE
    with np.load(REFERENCE / 'episode.npz', allow_pickle=False) as raw:
        reference = {key: raw[key].copy() for key in raw.files}
    rng = np.random.default_rng(protocol['seed'])
    starts, rejected = [], []
    for group, count in protocol['sample_counts'].items():
        ranges = protocol[group]
        while sum(p['evaluation_group'] == group for p in starts) < count:
            profile = dict(name=f'{group}_{len(starts):03d}', evaluation_group=group,
                           drawer_offset_x=float(rng.uniform(*ranges['x_m'])),
                           drawer_offset_y=float(rng.uniform(*ranges['y_m'])),
                           handle_radius=float(rng.uniform(*ranges['radius_m'])),
                           right_jitter=float(rng.uniform(*ranges['jitter_half_width_rad'])),
                           seed=int(rng.integers(1_000_000, 2_000_000_000)))
            try:
                build_start(profile, reference)
            except ValueError as exc:
                rejected.append(dict(profile=profile, reason=str(exc)))
                write(RESULT / 'physical_rejections.json', rejected)
                if len(rejected) > 1000:
                    raise RuntimeError('Too many physical rejections') from exc
                continue
            starts.append(profile)
    config = dict(name='drawer_eval_starts_002_unseen100', episode_id_start=200,
                  seed=protocol['seed'], purpose=protocol['exclusion'], starts=starts)
    write(CONFIG, config)
    write(RESULT / 'physical_rejections.json', rejected)
    subprocess.run([sys.executable, str(SCRIPTS / 'build_drawer_eval_starts.py'),
                    '--config', str(CONFIG), '--output', str(DATA)], check=True, cwd=ROOT)
    manifest = read(DATA / 'manifest.json')
    old_keys = set()
    for name in ('openarm_drawer_success_v003', 'openarm_drawer_eval_v001'):
        for row in read(ROOT / f'data/processed/{name}/manifest.json')['episodes']:
            path = ROOT / row['path']
            old_keys.add(state_key(path, read(path / 'report.json').get('scene_overrides', {})))
    new_keys = set()
    for row, profile in zip(manifest['episodes'], starts, strict=True):
        row['evaluation_group'] = profile['evaluation_group']
        path = ROOT / row['path']
        report = read(path / 'report.json')
        # Remove inherited reference outcomes: these starts have no demonstration.
        for key in ('stages', 'contact_metrics', 'success_open_then_close', 'opened_8cm_held_1s_during_episode',
                    'success_8cm_held_1s', 'final_closed_5mm_held_1s', 'final_open_m', 'peak_open_m',
                    'replay_max_qpos_error', 'replay_verified', 'upstream_returned_success'):
            report.pop(key, None)
        report['evaluation_group'] = profile['evaluation_group']
        write(path / 'report.json', report)
        key = state_key(path, row['scene_overrides'])
        if key in old_keys or key in new_keys:
            raise ValueError('Exact duplicate start; do not evaluate')
        new_keys.add(key)
    write(DATA / 'manifest.json', manifest)
    files = [CONFIG, DATA / 'manifest.json', RESULT / 'preregistered_protocol.json']
    files += sorted(DATA.glob('episodes/*/report.json')) + sorted(DATA.glob('episodes/*/episode.npz'))
    write(RESULT / 'locked_inputs.json', dict(locked_at=now(), starts=100,
          groups=dict(Counter(p['evaluation_group'] for p in starts)), exact_duplicates_with_original33=0,
          files={str(p.relative_to(ROOT)): digest(p) for p in files}))
    print('LOCKED_100_STARTS', flush=True)


def verify_inputs():
    for name, expected in read(RESULT / 'locked_inputs.json')['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Locked input changed: {name}')


def report():
    verify_inputs()
    write(RESULT / 'checkpoint_after.json', verify_checkpoint())
    import assess_drawer_open as assessor
    assessor.MANIFEST = DATA / 'manifest.json'
    rows = read(RUN / 'index.json')
    if len(rows) != 100 or {r['episode_id'] for r in rows} != set(range(200, 300)):
        raise ValueError('Expected exactly100 unique completed trials')
    cfg = read(RUN / 'run_config.json')
    for name, expected in cfg['runtime_source_sha256'].items():
        if digest(SCRIPTS / name) != expected:
            raise ValueError(f'Runtime source changed: {name}')
    audited = []
    for row in rows:
        trial = RUN / row['trial']
        summary = read(trial / 'summary.json')
        if Path(summary['checkpoint']).resolve() != CHECKPOINT or summary['weight_precision'] != 'checkpoint':
            raise ValueError('Unexpected checkpoint/precision')
        if summary['noise_seed'] != 42 or summary['noise_mode'] != 'fixed' or summary['action_chunk'] != 4 or summary['task'] != 'open':
            raise ValueError('Unexpected deployment settings')
        result = assessor.assess(trial)
        if result['model_pull_success'] != row['success']:
            raise ValueError('Independent replay disagrees')
        audited.append(result)
        write(RESULT / 'independent_replay.json', dict(complete=len(audited)==100, results=audited))
        print('INDEPENDENT_REPLAY', row['episode_id'], result['model_pull_success'], flush=True)
    from summarize_drawer_eval import describe
    groups = {p['episode_id']: p['evaluation_group'] for p in read(DATA / 'manifest.json')['episodes']}
    metrics = dict(checkpoint=str(CHECKPOINT), completed_at=now(), trials=100,
                   evidence_valid=sum(r['evidence_valid'] for r in audited),
                   frozen_weights_verified=True, locked_inputs_verified=True,
                   overall=describe(rows), groups={g:describe([r for r in rows if groups[r['episode_id']]==g]) for g in ('near','expanded')})
    for key in ('overall',):
        metrics[key]['wilson95'] = wilson(metrics[key]['success'], metrics[key]['trials'])
    for value in metrics['groups'].values():
        value['wilson95'] = wilson(value['success'], value['trials'])
    write(RESULT / 'metrics.json', metrics)
    for name in ('index.json', 'run_config.json'):
        (RESULT / name).write_bytes((RUN / name).read_bytes())
    lines = ['# 冻结020模型：全新100起点评估', '', f'完成时间：{metrics["completed_at"]}', '',
             '同一份020/2000权重，评估前锁定70个原范围组合起点和30个扩大范围起点。没有训练、专家标签或按结果筛选起点。', '',
             '| 分组 | 成功 | 成功率 | Wilson 95%区间 |', '| --- | ---: | ---: | --- |']
    for label, value in [('原范围组合70',metrics['groups']['near']),('扩大范围30',metrics['groups']['expanded']),('合计100',metrics['overall'])]:
        lo, hi = value['wilson95']
        lines.append(f'| {label} | {value["success"]}/{value["trials"]} | {value["success_rate"]:.1%} | {lo:.1%}–{hi:.1%} |')
    lines += ['', f'独立重放证据有效：{metrics["evidence_valid"]}/100；权重、输入清单与运行源码指纹已核对。', '',
              '## 运行条件与边界', '',
              '沿用原严格判据：双指各>.01N持续5步、有接触拉动、开度≥80mm保持30步；open任务，24秒上限，固定噪声42，H4，continuous夹爪，checkpoint原精度，GLFW软件渲染，每条独立进程。推理时仿真暂停。', '',
              'near：X偏移.09–.11m、Y±.03m、把手半径.005–.012m、关节扰动半宽.01–.04rad。',
              'expanded：X偏移.08–.12m、Y±.05m、把手半径.004–.014m、关节扰动半宽.04–.08rad。X基准偏移为.10m；实际关节扰动受物理限位约束。', '',
              '每个起点同时随机组合多个因素；expanded不是单因素消融。70/30的合并率只代表预定混合比例。新起点共享原模拟器和ready姿态来源，不能推断真机或任意场景成功率。', '',
              '## 失败分布', '']
    for group, value in metrics['groups'].items():
        lines.append(f'- {group}: {json.dumps(value["failure_modes"],ensure_ascii=False)}')
    lines += ['', '## 逐条结果', '', '| ID | 组别 | 成功 | 峰值开度mm | 失败类型 |', '| --- | --- | --- | ---: | --- |']
    for row in rows:
        lines.append(f'| {row["episode_id"]} | {groups[row["episode_id"]]} | {row["success"]} | {row["peak_open_mm"]:.2f} | {row["failure_mode"] or "—"} |')
    lines += ['', '原始证据：`logs/a2_pi0/drawer_generalization_020_20261102/`；起点：`data/processed/openarm_drawer_eval_v002/manifest.json`。', '']
    (RESULT / 'README.md').write_text('\n'.join(lines))
    write(RESULT / 'progress.json', dict(status='complete', completed_at=now(), metrics=metrics))
    print('NEW100_COMPLETE', json.dumps(metrics, ensure_ascii=False), flush=True)


def wilson(k, n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half, center+half]


def run():
    verify_inputs()
    verify_checkpoint()
    if RUN.exists():
        raise FileExistsError('Existing output cannot be overwritten')
    write(RESULT / 'progress.json', dict(status='running', started_at=now(), run=str(RUN)))
    subprocess.run([sys.executable, str(SCRIPTS / 'run_drawer_eval_isolated.py'),
                    '--checkpoint', str(CHECKPOINT), '--weight-precision', 'checkpoint',
                    '--manifest', str(DATA / 'manifest.json'), '--no-eval-manifest',
                    '--output', str(RUN)], check=True, cwd=ROOT)
    report()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare','run','report'), required=True)
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'report':report}[args.stage]()
