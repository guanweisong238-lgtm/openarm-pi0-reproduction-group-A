"""生成 eval-only 抽屉评估起点：只构造合法初态，不采示教、不进训练。

为什么不用 `collect_drawer_trial.py` 采完整示教：
评估只需要 `DrawerSimulation` 能复现的初始物理状态（`report.json` +
`episode.npz` 的 `initial_integration_state` / `initial_eq_data` /
`actuator_names`），不需要专家动作与图像。跑完整示教对评估集是浪费，
而且会让"考卷"带上专家路径的痕迹。

同源性：初态继承参考 episode（默认 ID12）的 ready 姿态积分状态，
只改动柜体几何与右臂初始扰动。柜体是 body 平移、不占 qpos，
因此直接复用参考状态不会引入不一致。

产物：
- `data/processed/openarm_drawer_eval_v001/episodes/<name>/{report.json,episode.npz}`
- `data/processed/openarm_drawer_eval_v001/manifest.json`（每条标 `eval_only: true`）

协议见 `docs/drawer-eval-protocol.md` 第 8.2 节。
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/50_mesa.json')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')

from pathlib import Path
import argparse
import hashlib
import json
import sys
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '.cache/openarm-control'
sys.path[:0] = [str(ROOT / '.cache/drawer-deps'), str(SOURCE), str(Path(__file__).parent)]

import mujoco  # noqa: E402
from openarm_control.config import ARTICULATED_SCENE  # noqa: E402

ACTUATORS = [name for side in ('right', 'left') for name in (
    *(f'{side}_joint{i}_ctrl' for i in range(1, 8)), f'{side}_finger1_ctrl')]
CAMERAS = {'ceiling': 'camera_ceiling', 'wrist_left': 'camera_wrist_left',
           'wrist_right': 'camera_wrist_right'}
REFERENCE = ROOT / 'data/processed/openarm_drawer_success_v003/episodes/episode_012'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_model(profile):
    """与采集脚本同一套场景构造：柜体平移、把手粗细、顶部与腕部相机。"""
    spec = mujoco.MjSpec.from_file(ARTICULATED_SCENE)
    drawer = next(body for body in spec.bodies if body.name == 'cabinet_drawer')
    drawer.pos[0] += profile['drawer_offset_x']
    drawer.pos[1] += profile['drawer_offset_y']
    if profile.get('handle_radius') is not None:
        next(geom for geom in spec.geoms if geom.name == 'drawer_handle').size[0] = profile['handle_radius']
    top = next(camera for camera in spec.cameras if camera.name == 'tablecam')
    ceiling = spec.worldbody.add_camera(name='camera_ceiling')
    ceiling.pos, ceiling.quat, ceiling.fovy = top.pos, top.quat, top.fovy
    model = spec.compile()
    # 腕部相机固定在腕部局部坐标系，与采集中 --wrist-view side 完全一致。
    position = np.array([0.07, 0.10, -0.04])
    target = np.array([0.0, 0.0, -0.14])
    z_axis = position - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.column_stack([x_axis, y_axis, z_axis]).ravel())
    for name in ('camera_wrist_left', 'camera_wrist_right'):
        cid = model.camera(name).id
        model.cam_pos[cid], model.cam_quat[cid], model.cam_fovy[cid] = position, quaternion, 85
    model.vis.quality.shadowsize = 1024
    model.vis.global_.offwidth = max(320, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(240, model.vis.global_.offheight)
    return model


def build_start(profile, reference):
    """在参考初态上施加几何与右臂扰动，返回可直接被 DrawerSimulation 加载的数据。"""
    model = compile_model(profile)
    data = mujoco.MjData(model)
    actuator_ids = np.array([model.actuator(name).id for name in reference['actuator_names'].tolist()])
    joint_ids = model.actuator_trnid[actuator_ids, 0]
    qpos_ids = model.jnt_qposadr[joint_ids]
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    model.eq_data[:] = reference['initial_eq_data']
    mujoco.mj_resetData(model, data)
    mujoco.mj_setState(model, data, reference['initial_integration_state'], spec)
    mujoco.mj_forward(model, data)

    low, high = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if model.jnt_limited[jid]:
            low[aid] = max(low[aid], model.jnt_range[jid, 0])
            high[aid] = min(high[aid], model.jnt_range[jid, 1])

    applied_delta = np.zeros(7)
    jitter = profile.get('right_jitter', 0.0)
    if jitter:
        right_qpos = qpos_ids[:7]
        original = data.qpos[right_qpos].copy()
        # 与采集脚本同一采样口径：先取扰动窗口与限位交集，再均匀采样，不做事后裁剪。
        delta_low = np.maximum(-jitter, low[actuator_ids[:7]] - original)
        delta_high = np.minimum(jitter, high[actuator_ids[:7]] - original)
        if np.any(delta_low >= delta_high):
            raise ValueError('初始关节扰动与限位没有可采样交集')
        delta = np.random.default_rng(profile['seed']).uniform(delta_low, delta_high)
        perturbed = original + delta
        if np.any(perturbed < low[actuator_ids[:7]]) or np.any(perturbed > high[actuator_ids[:7]]):
            raise ValueError('扰动后的初始右臂位置超限')
        data.qpos[right_qpos] = perturbed
        data.ctrl[actuator_ids[:7]] = perturbed
        applied_delta = delta
        mujoco.mj_forward(model, data)

    # 物理有效性：不能有活动 weld、状态必须有限、接触不能有超过 2mm 的穿透。
    weld_mask = model.eq_type == mujoco.mjtEq.mjEQ_WELD
    if np.any(data.eq_active[weld_mask]):
        raise ValueError('初始状态包含活动抓取绑定')
    if not np.isfinite(data.qpos).all():
        raise ValueError('初始状态非有限')
    penetration = min((contact.dist for contact in data.contact), default=0.0)
    if penetration < -0.002:
        raise ValueError(f'初始状态几何穿透 {-penetration*1000:.2f}mm 过大')

    state = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    poses = {name: {'position': model.cam_pos[model.camera(name).id].tolist(),
                    'quaternion_wxyz': model.cam_quat[model.camera(name).id].tolist(),
                    'fovy_degrees': float(model.cam_fovy[model.camera(name).id])}
             for name in CAMERAS.values()}
    return dict(state=state, eq_data=model.eq_data.copy(), actuator_names=np.asarray(ACTUATORS),
                poses=poses, applied_delta=applied_delta, penetration=float(penetration))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/drawer-eval-starts-001.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/openarm_drawer_eval_v001')
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} 已存在；评估集不可覆盖，请换版本号')
    config = json.loads(args.config.read_text())
    reference_report = json.loads((args.reference / 'report.json').read_text())
    with np.load(args.reference / 'episode.npz', allow_pickle=False) as raw:
        reference = {key: raw[key].copy() for key in raw.files}
    if reference_report['mujoco_version'] != mujoco.__version__:
        raise ValueError(f'MuJoCo 版本不一致：{reference_report["mujoco_version"]} vs {mujoco.__version__}')

    args.output.mkdir(parents=True)
    (args.output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    episodes = []
    for index, profile in enumerate(config['starts']):
        episode_id = config['episode_id_start'] + index
        directory = args.output / 'episodes' / f'eval_{episode_id:03d}'
        directory.mkdir(parents=True, exist_ok=False)
        built = build_start(profile, reference)
        np.savez_compressed(directory / 'episode.npz',
                            initial_integration_state=built['state'],
                            initial_eq_data=built['eq_data'],
                            actuator_names=built['actuator_names'])
        report = {**reference_report,
                  'collector_sha256': digest(Path(__file__)),
                  'start_only': True,
                  'source_episode': str(args.reference.relative_to(ROOT)),
                  'frames': 0,
                  'scene_overrides': {'drawer_handle_radius_m': profile.get('handle_radius'),
                                      'drawer_offset_y_m': profile['drawer_offset_y'],
                                      'drawer_offset_x_m': profile['drawer_offset_x']},
                  'wrist_view': 'side',
                  'camera_local_poses': built['poses'],
                  'camera_mapping': CAMERAS,
                  'actuator_order': ACTUATORS,
                  'initial_control_step_offset': 0,
                  'grasp_diagnostics': [],
                  'expert_parameters': None,
                  'expert_returned_success': None,
                  'initial_right_randomization': {
                      'seed': profile.get('seed'),
                      'uniform_half_width_rad': profile.get('right_jitter', 0.0),
                      'applied_delta_rad': built['applied_delta'].tolist(),
                      'sampling': 'uniform over intersection of +/-half-width and joint/control limits',
                      'method': 'reset right qpos and ctrl before recording; no intermediate state injection'},
                  'note': 'eval-only 起点：只含初态，不含示教动作与图像；不得用于训练'}
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        episodes.append({'episode_id': episode_id, 'path': str(directory.relative_to(ROOT)),
                         'split': 'eval_only', 'eval_only': True, 'frames': 0,
                         'initialization_group': 'eval_scripted_start',
                         'scene_overrides': report['scene_overrides'],
                         'eval_parameters': {'drawer_offset_x': profile['drawer_offset_x'],
                                             'drawer_offset_y': profile['drawer_offset_y'],
                                             'handle_radius': profile.get('handle_radius'),
                                             'right_jitter': profile.get('right_jitter', 0.0),
                                             'seed': profile.get('seed')},
                         'initial_max_penetration_m': built['penetration'],
                         'episode_sha256': digest(directory / 'episode.npz')})
        print('START_BUILT', episode_id, profile['name'], report['scene_overrides'],
              f'penetration={built["penetration"]*1000:.2f}mm', flush=True)

    manifest = {'name': config['name'], 'schema_version': 1,
                'created_at': datetime.now().astimezone().isoformat(),
                'purpose': 'eval-only 起点集：用于评估成功率，不参与任何训练',
                'training_excluded': True,
                'reference_episode': str(args.reference.relative_to(ROOT)),
                'config': str(args.config.relative_to(ROOT)),
                'note': '这些条目只有初态；expert_parameters 为空，not 示教轨迹',
                'episodes': episodes}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'PASS 生成 {len(episodes)} 条 eval-only 起点 -> {args.output}', flush=True)
    print('MANIFEST', (args.output / 'manifest.json').relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
