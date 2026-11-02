"""无窗口批量闭环评估：按评估协议跑满起点集，逐 trial 落盘并汇总。

职责：
- 复用 `live_drawer_success.update_judge` 的成功判据与 `DrawerSimulation` 物理环境，不引入新阈值；
- 去掉窗口、键盘、暂停与实时节流，改为顺序批量执行；
- 每个 trial 产出与交互式脚本同构的 `summary.json` 与 `closed_loop.npz`，便于共用复核口径；
- 汇总 `index.json` 供后续生成指标表，并在 `run_config.json` 记录本次运行的全部固定条件。

协议与判据见 `docs/drawer-eval-protocol.md`。本脚本不改动检查点与数据。
"""
import os
# 渲染后端必须与证据生成器（live_drawer_success.py）一致，否则像素不同会改变确定性策略的动作。
# 2026-11-01 首次自检实测：用 egl 后端时 ID12/ID6 的成败判定相对 10-25 记录全部翻转。
_GL_BACKEND = os.environ.get('DRAWER_EVAL_GL', 'glfw')
os.environ['MUJOCO_GL'] = _GL_BACKEND
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'mesa')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/50_mesa.json')

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime

import numpy as np
import mujoco

from drawer_success_training_config import ROOT, MANIFEST
from drawer_simulation import DrawerSimulation

# 项目内依赖：接触判定与关节速度索引都在这里。
sys.path[:0] = [str(ROOT / '.cache/drawer-deps'), str(ROOT / '.cache/openarm-control')]

# 交互式脚本顶部会把 MUJOCO_GL 强制设为 glfw（它是窗口程序），导入后恢复本脚本选定的后端。
from live_drawer_success import update_judge  # noqa: E402
os.environ['MUJOCO_GL'] = _GL_BACKEND

from contact_drawer_controller import finger_handle_contacts  # noqa: E402

CHECKPOINT_DEFAULT = ROOT / 'checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000'
CAMERAS = ('ceiling', 'wrist_left', 'wrist_right')
# 预热用固定文本，与示教 report.json 中的 prompt 一致；不从这里读批次内容。
WARMUP_PROMPT = 'Open the drawer, then close it.'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(manifests, episode_ids):
    """合并多个清单的起点；episode 编号在不同清单间必须唯一。"""
    rows, seen = [], set()
    for path in manifests:
        payload = json.loads(Path(path).read_text())
        for row in payload['episodes']:
            if row['episode_id'] in seen:
                raise ValueError(f'起点编号重复：{row["episode_id"]}')
            seen.add(row['episode_id'])
            rows.append({**row, 'manifest_path': str(path)})
    if episode_ids:
        missing = [eid for eid in episode_ids if eid not in {r['episode_id'] for r in rows}]
        if missing:
            raise ValueError(f'起点不存在：{missing}')
        rows = [next(r for r in rows if r['episode_id'] == eid) for eid in episode_ids]
    if not rows:
        raise ValueError('没有可运行的起点')
    return rows


def make_noise(horizon, seed):
    """确定性端点策略的固定查询噪声：与训练期同一构造方式。"""
    return np.random.default_rng(seed).standard_normal((horizon, 32)).astype(np.float32)


def check_protocol(checkpoint, args):
    """冻结校验：换 seed 或换采样方式一律拒绝，与交互式脚本同口径。"""
    protocol = json.loads((checkpoint / 'input_protocol.json').read_text())
    if protocol['manifest_sha256'] != digest(MANIFEST):
        raise ValueError('manifest 与检查点协议不一致')
    horizon = int(protocol.get('action_horizon', 16))
    if args.action_chunk > horizon:
        raise ValueError('执行步数超过模型动作窗口')
    if protocol.get('inference_kind') == 'fixed_noise_endpoint':
        if args.seed != int(protocol['fixed_noise_seed']):
            raise ValueError(f'确定性端点策略只能使用训练固定噪声 {protocol["fixed_noise_seed"]}，'
                             f'不能把换 seed 当作多次独立测试')
    return protocol, horizon


def classify_failure(summary, rows):
    """按协议第 7 节优先级分类；成功返回 None。"""
    if summary['termination'] == 'error':
        return 'simulation_error'
    judge = summary['judge']
    if not rows:
        return 'simulation_error'
    contacts = np.asarray([r['contacts'] for r in rows])
    forces = np.asarray([r['forces'] for r in rows])
    any_contact = bool(np.any(contacts > 0))
    if not any_contact:
        return 'no_contact'
    if not judge.get('grasped'):
        return 'single_finger'
    peak = summary['peak_open_mm']
    if peak < 80.0:
        if int(judge.get('grasp_streak', 0)) == 0:
            return 'grasp_slip'
        if judge.get('pulling_contact'):
            return 'insufficient_open'
        return 'grasp_slip'
    if not judge.get('opened'):
        return 'hold_fail'
    return 'timeout_other'


def run_trial(row, policy, protocol, args, horizon, directory, limit):
    """单次 rollout：取观测→推理→执行 chunk→重新观测，成功或到步数上限结束。"""
    directory.mkdir(parents=True, exist_ok=False)
    noise = make_noise(horizon, args.seed)
    env = DrawerSimulation(ROOT / row['path'])
    observed, blocks, times = [], [], []
    state = {}
    termination = 'timeout'
    error = None
    writer = None
    camera = None
    if args.video:
        import cv2 as camera
    try:
        initial_time = float(env.data.time)
        left_hold = env.data.ctrl[env.actuator_ids[8:]].copy()
        if camera is not None:
            writer = camera.VideoWriter(str(directory / 'closed_loop.mp4'),
                                        camera.VideoWriter_fourcc(*'mp4v'),
                                        30 / args.action_chunk, (960, 240))
        while True:
            env.data.xfrc_applied[:] = 0
            observation = env.observation()
            began = time.perf_counter()
            block = np.asarray(policy.infer(observation, noise=noise)['actions'])
            times.append(time.perf_counter() - began)
            if block.shape != (horizon, 16) or not np.isfinite(block).all():
                raise ValueError(f'模型输出异常：{block.shape}')
            blocks.append(block)
            if protocol.get('right8_loss'):
                block = block.copy()
                block[:, 8:] = left_hold
            if writer is not None:
                frame = camera.cvtColor(np.concatenate(
                    [observation['observation.images.' + n] for n in CAMERAS], axis=1), camera.COLOR_RGB2BGR)
                camera.putText(frame, f'EVAL ep={row["episode_id"]} t={env.steps/30:.2f}s',
                               (8, 230), camera.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
                writer.write(frame)
            remaining = limit - env.steps
            for action in block[:min(args.action_chunk, remaining)]:
                action = action.copy()
                before = env.data.qpos.copy()
                velocity_before = env.data.qvel.copy()
                applied = env.step(action)
                contacts, forces = finger_handle_contacts(env.model, env.data, include_stems=True)
                opening = -float(env.data.qpos[env.drawer_qpos])
                state = update_judge(state, opening, contacts, forces, task=args.task)
                observed.append(dict(qpos_before=before, qpos=env.data.qpos.copy(),
                                     qvel_before=velocity_before, qvel=env.data.qvel.copy(),
                                     requested=action.copy(), applied=applied, opening=opening,
                                     contacts=contacts, forces=forces,
                                     time=float(env.data.time - initial_time)))
                if state['success']:
                    break
            if state.get('success'):
                termination = 'success'
                break
            if env.steps >= limit:
                termination = 'timeout'
                break
    except Exception as exc:  # 单条失败不能让整批中断，但必须如实记录
        termination = 'error'
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if writer is not None:
            writer.release()
        env.close()

    summary = dict(checkpoint=str(args.checkpoint), task=args.task,
                   episode_id=row['episode_id'], split=row.get('split', 'unknown'),
                   initialization=row.get('initialization_group', 'unknown'),
                   noise_seed=args.seed, noise_mode='fixed', action_chunk=args.action_chunk,
                   gripper_mode=args.gripper_mode,
                   steps=len(observed), seconds=len(observed) / 30,
                   peak_open_mm=max((r['opening'] for r in observed), default=0.0) * 1000,
                   inference_count=len(times),
                   inference_p50_ms=float(np.percentile(np.asarray(times) * 1000, 50)) if times else None,
                   inference_p95_ms=float(np.percentile(np.asarray(times) * 1000, 95)) if times else None,
                   final_open_mm=state.get('opening', 0.0) * 1000,
                   judge=state, termination=termination, error=error,
                   teacher_actions_executed=0, intermediate_states_injected=0,
                   contact_scope='entire handle assembly, excluding panel and wrist',
                   simulation_paused_during_inference=True,
                   closed_loop_tested=True, fully_executed=termination in ('timeout', 'success'),
                   state_representation=protocol.get('state_representation', 'position16'),
                   mask_left_normalized_state=bool(protocol.get('mask_left_normalized_state')),
                   fixed_left_controller=bool(protocol.get('right8_loss')),
                   scope='full task rollout (batch, no window)',
                   gl_backend=_GL_BACKEND,
                   episode_manifest=row.get('manifest_path'), eval_only=bool(row.get('eval_only', False)))
    if observed:
        np.savez_compressed(directory / 'closed_loop.npz',
                            **{k: np.asarray([r[k] for r in observed]) for k in observed[0]},
                            action_blocks=np.asarray(blocks),
                            inference_seconds=np.asarray(times),
                            inference_noise=np.asarray([noise] * len(blocks)))
    summary['failure_mode'] = None if summary['judge'].get('success') else classify_failure(summary, observed)
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def verify_replay(row, directory, summary):
    """独立重放：只重放实际下发动作，逐帧核对位置/速度/接触力与接触计数。"""
    npz = directory / 'closed_loop.npz'
    if not npz.exists():
        return dict(evidence_valid=False, replay_reason='缺少 closed_loop.npz')
    with np.load(npz) as raw:
        data = {k: raw[k].copy() for k in raw.files}
    env = DrawerSimulation(ROOT / row['path'])
    position_error = velocity_error = contact_error = 0.0
    first_success = None
    state = {}
    try:
        for i, action in enumerate(data['applied']):
            if 'qpos_before' in data:
                position_error = max(position_error, float(np.max(np.abs(env.data.qpos - data['qpos_before'][i]))))
            if 'qvel_before' in data:
                velocity_error = max(velocity_error, float(np.max(np.abs(env.data.qvel - data['qvel_before'][i]))))
            env.step(action)
            position_error = max(position_error, float(np.max(np.abs(env.data.qpos - data['qpos'][i]))))
            if 'qvel' in data:
                velocity_error = max(velocity_error, float(np.max(np.abs(env.data.qvel - data['qvel'][i]))))
            counts, forces = finger_handle_contacts(env.model, env.data, include_stems=True)
            if not np.array_equal(counts, data['contacts'][i]):
                raise ValueError('接触计数重放不一致')
            contact_error = max(contact_error, float(np.max(np.abs(forces - data['forces'][i]))))
            state = update_judge(state, -float(env.data.qpos[env.drawer_qpos]), counts, forces,
                                 task=summary['task'])
            if state['success'] and first_success is None:
                first_success = i + 1
    finally:
        env.close()
    velocity_required = summary.get('state_representation') == 'position_velocity16'
    velocity_recorded = 'qvel' in data and 'qvel_before' in data
    permitted = summary['teacher_actions_executed'] == 0 and summary['intermediate_states_injected'] == 0
    valid = (permitted and position_error < 1e-9 and contact_error < 1e-7
             and (not velocity_recorded or velocity_error < 1e-9)
             and (not velocity_required or velocity_recorded))
    return dict(evidence_valid=bool(valid),
                replay_max_qpos_error=position_error,
                replay_max_qvel_error=velocity_error if velocity_recorded else None,
                replay_max_contact_force_error=contact_error,
                replay_success=bool(state.get('success')),
                replay_first_success_step=first_success)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument('--weight-precision', choices=('auto','bfloat16','checkpoint'),default='auto')
    parser.add_argument('--output', type=Path, help='默认 logs/a2_pi0/drawer_eval_<时间戳>')
    parser.add_argument('--manifest', type=Path, default=MANIFEST, help='主示教清单')
    parser.add_argument('--extra-manifest', type=Path, nargs='*', default=[],
                        help='追加起点清单，例如 eval-only 集；编号不得与主清单重复')
    parser.add_argument('--episode-ids', type=int, nargs='+', help='只跑指定起点')
    parser.add_argument('--seconds', type=float, default=24.0)
    parser.add_argument('--task', choices=('open', 'roundtrip'), default='open')
    parser.add_argument('--gripper-mode', choices=('continuous',), default='continuous')
    parser.add_argument('--action-chunk', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat', type=int, default=1, help='同起点重复次数，仅用于稳定性检查')
    parser.add_argument('--video', action='store_true', help='保存三路拼接录像（默认关闭）')
    parser.add_argument('--no-warmup', action='store_true', help='跳过丢弃首次推理的预热')
    parser.add_argument('--append', action='store_true', help='向已有输出目录追加，供逐 trial 隔离驱动使用')
    parser.add_argument('--trial-offset', type=int, default=0, help='trial 编号偏移，供逐 trial 隔离驱动使用')
    parser.add_argument('--prepare-only', action='store_true', help='只校验协议、初态与判据，不加载策略')
    args = parser.parse_args()

    if args.seconds <= 0 or args.seed < 0 or args.repeat < 1:
        parser.error('时长/seed/重复次数不合法')
    protocol, horizon = check_protocol(args.checkpoint, args)
    if args.weight_precision=='auto':args.weight_precision=protocol.get('recommended_weight_precision','bfloat16')
    rows = load_cases([args.manifest, *args.extra_manifest], args.episode_ids)
    limit = round(args.seconds * 30)
    print(f'CASES {len(rows)} limit_steps={limit} checkpoint={args.checkpoint}', flush=True)

    if args.prepare_only:
        for row in rows:
            env = DrawerSimulation(ROOT / row['path'])
            print('INITIAL_STATE_PASS', row['episode_id'], row.get('initialization_group'), flush=True)
            env.close()
        state = {}
        for _ in range(30):
            state = update_judge(state, .09, np.array([1, 1]), np.array([2., 2.]), task='open')
        state = update_judge(state, .091, np.array([1, 1]), np.array([2., 2.]), task='open')
        for _ in range(30):
            state = update_judge(state, 0., np.array([1, 1]), np.array([2., 2.]), task='open')
        assert state['success']
        unilateral = {}
        for _ in range(60):
            unilateral = update_judge(unilateral, .09, np.array([1, 0]), np.array([2., 0.]), task='open')
        for _ in range(30):
            unilateral = update_judge(unilateral, 0., np.array([1, 0]), np.array([2., 0.]), task='open')
        assert not unilateral['success']
        print('PASS protocol, initializations, success and unilateral-failure judge', flush=True)
        return

    import jax
    if jax.default_backend() != 'gpu':
        raise RuntimeError('批量评估要求 JAX GPU')
    from openpi.policies import policy_config
    from drawer_start_fit_config import get_policy_config, get_stats_location

    norm_root, asset_id = get_stats_location(protocol)
    if protocol['norm_stats_sha256'] != digest(norm_root / 'norm_stats.json'):
        raise ValueError('归一化统计与协议不一致')
    if json.loads((args.checkpoint / 'assets' / asset_id / 'norm_stats.json').read_text()) != \
            json.loads((norm_root / 'norm_stats.json').read_text()):
        raise ValueError('检查点资产统计与协议版本不一致')

    print('LOADING_POLICY', args.checkpoint, flush=True)
    if args.weight_precision=='checkpoint':
        from drawer_policy_loader import create_checkpoint_precision_policy
        policy=create_checkpoint_precision_policy(get_policy_config(protocol),args.checkpoint,
            sample_kwargs={'num_steps':int(protocol.get('sampling_steps',10))})
    else:
        policy = policy_config.create_trained_policy(
            get_policy_config(protocol), args.checkpoint,
            sample_kwargs={'num_steps': int(protocol.get('sampling_steps', 10))})
    diagnostics = {}
    if protocol.get('state_representation') == 'position_velocity16':
        kernel = np.asarray(policy._model.state_proj.kernel.value, np.float32)[16:24]
        diagnostics['learned_right_velocity_projection_l2'] = float(np.linalg.norm(kernel))
        if not diagnostics['learned_right_velocity_projection_l2'] > 0:
            raise ValueError('速度投影仍为零，未学到新增通道')

    output = args.output or ROOT / 'logs/a2_pi0' / datetime.now().strftime('drawer_eval_%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=args.append)
    runtime_names=['run_drawer_eval_batch.py','run_drawer_eval_isolated.py','drawer_simulation.py',
                   'live_drawer_success.py','assess_drawer_open.py','contact_drawer_controller.py',
                   'drawer_policy_loader.py','drawer_start_fit_config.py','drawer_velocity_config.py',
                   'drawer_success_training_config.py','openarm_policy.py']
    runtime_sources={name:(Path(__file__).parent/name).read_bytes() for name in runtime_names}
    runtime_hashes={name:hashlib.sha256(source).hexdigest() for name,source in runtime_sources.items()}
    manifest_hashes={str(path):digest(path) for path in [args.manifest,*args.extra_manifest]}
    if (output/'run_config.json').exists():
        prior=json.loads((output/'run_config.json').read_text())
        if prior.get('runtime_source_sha256',runtime_hashes)!=runtime_hashes or prior.get('manifest_sha256',manifest_hashes)!=manifest_hashes:
            raise RuntimeError('本批次源码或起点清单已变化，必须使用新目录重新评估')
    if not (output / 'run_config.json').exists():
        snapshot=output/'runtime_source';snapshot.mkdir()
        for name,source in runtime_sources.items():(snapshot/name).write_bytes(source)
        (output / 'run_config.json').write_text(json.dumps(dict(
            checkpoint=str(args.checkpoint), checkpoint_protocol_sha256=digest(args.checkpoint / 'input_protocol.json'),
            protocol=protocol, policy_diagnostics=diagnostics,
            runtime_source_sha256=runtime_hashes,manifest_sha256=manifest_hashes,
            manifests=[str(args.manifest), *map(str, args.extra_manifest)],
            run=dict(seconds=args.seconds, task=args.task, seed=args.seed, action_chunk=args.action_chunk,
                     gripper_mode=args.gripper_mode, repeat=args.repeat, step_limit=limit,
                     weight_precision=args.weight_precision,
                     gl_backend=_GL_BACKEND),
            notes='成功判据与复核阈值取自 docs/drawer-eval-protocol.md；本脚本不引入新阈值'), indent=2) + '\n')
    print('RUN_DIRECTORY=' + str(output), flush=True)

    if not args.no_warmup:
        # 预热只用固定假观测，**不创建场景**：2026-11-01 实测，预热复用批次首个起点会让
        # 结果依赖批次构成（ID2 在 `12 2 6` 下 20.20mm，在 `2 --repeat 2` 下 -0.0mm）。
        dummy = {'observation.state': np.zeros(16, np.float32),
                 'observation.velocity': np.zeros(16, np.float32),
                 'prompt': WARMUP_PROMPT,
                 **{f'observation.images.{name}': np.zeros((240, 320, 3), np.uint8) for name in CAMERAS}}
        began = time.perf_counter()
        policy.infer(dummy, noise=make_noise(horizon, args.seed))
        print(f'WARMUP discarded {time.perf_counter()-began:.2f}s (固定假观测，不依赖场景，不计入延迟统计)',
              flush=True)

    index_path = output / 'index.json'
    index = json.loads(index_path.read_text()) if (args.append and index_path.exists()) else []
    trial = args.trial_offset
    started = time.perf_counter()
    for row in rows:
        for repeat in range(args.repeat):
            trial += 1
            name = f'trial_{trial:03d}_episode_{row["episode_id"]:03d}'
            if args.repeat > 1:
                name += f'_r{repeat + 1}'
            summary = run_trial(row, policy, protocol, args, horizon, output / name, limit)
            replay = verify_replay(row, output / name, summary)
            summary.update(replay)
            summary['model_pull_success'] = bool(summary['evidence_valid'] and summary['judge'].get('success'))
            if summary['model_pull_success']:
                summary['failure_mode'] = None
            (output / name / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
            index.append(dict(trial=name, episode_id=row['episode_id'], split=row.get('split', 'unknown'),
                              eval_only=bool(row.get('eval_only', False)), repeat=repeat + 1,
                              success=summary['model_pull_success'], evidence_valid=summary['evidence_valid'],
                              peak_open_mm=summary['peak_open_mm'],
                              first_success_step=summary.get('replay_first_success_step'),
                              simulation_seconds=summary['seconds'], termination=summary['termination'],
                              failure_mode=summary['failure_mode'],
                              inference_count=summary['inference_count'],
                              inference_p50_ms=summary['inference_p50_ms'],
                              inference_p95_ms=summary['inference_p95_ms']))
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
            print('EVAL_RESULT ' + json.dumps(index[-1], ensure_ascii=False), flush=True)
            print(f'PROGRESS {trial} elapsed={time.perf_counter()-started:.0f}s', flush=True)

    successes = sum(1 for item in index if item['success'])
    print(f'BATCH_DONE trials={len(index)} success={successes} output={output}', flush=True)


if __name__ == '__main__':
    main()
