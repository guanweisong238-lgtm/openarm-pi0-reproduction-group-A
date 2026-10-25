"""重新审核全部抽屉采集；不改原始数据，不信任旧报告的任务/接触结论。"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import mujoco
import numpy as np

from collect_drawer_trial import ROOT, UPSTREAM_COMMIT
sys.path[:0] = [str(ROOT/'.cache/drawer-deps'), str(ROOT/'.cache/openarm-control')]
from contact_drawer_controller import finger_handle_contacts


def longest(mask):
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def audit(directory):
    provenance = json.loads((directory/'report.json').read_text())
    row = {'path':str(directory.relative_to(ROOT)), 'issues':[], 'warnings':[],
           'human':bool(provenance.get('human_recording')),
           'selfcheck':directory.name.startswith('selfcheck_'),
           'reported_success':provenance.get('success_open_then_close'),
           'contact_scope':'whole handle assembly, excluding panel and wrist'}
    scene = ROOT/'.cache/openarm-control'/provenance['source_scene']
    if provenance['source_commit'] != UPSTREAM_COMMIT or hashlib.sha256(scene.read_bytes()).hexdigest() != provenance['scene_sha256']:
        raise ValueError('source or scene mismatch')
    if provenance['mujoco_version'] != mujoco.__version__:
        raise ValueError('MuJoCo version mismatch')
    spec = mujoco.MjSpec.from_file(str(scene))
    overrides = provenance.get('scene_overrides', {})
    cabinet = next(b for b in spec.bodies if b.name=='cabinet_drawer')
    cabinet.pos[0] += overrides.get('drawer_offset_x_m',0)
    cabinet.pos[1] += overrides.get('drawer_offset_y_m',0)
    if overrides.get('drawer_handle_radius_m') is not None:
        next(g for g in spec.geoms if g.name=='drawer_handle').size[0] = overrides['drawer_handle_radius_m']
    model = spec.compile()
    data = mujoco.MjData(model)
    with np.load(directory/'episode.npz', allow_pickle=False) as archive:
        keys = ['state','qpos','action','native_action','substeps','eq_data','eq_active',
                'timestamp','drawer','initial_integration_state','initial_eq_data','terminal_qpos','actuator_names']
        arrays = {k:archive[k] for k in keys}
        n = len(arrays['action'])
        if n == 0 or arrays['state'].shape != (n,16) or arrays['action'].shape != (n,16):
            raise ValueError('invalid numeric dimensions')
        for k in keys[:-1]:
            if not np.isfinite(arrays[k]).all(): row['issues'].append('nonfinite '+k)
        if n != provenance['frames']: row['issues'].append('report frame mismatch')
        np.testing.assert_allclose(arrays['timestamp']-arrays['timestamp'][0],np.arange(n)/30,atol=.00101,rtol=0)
        # 每个相机单独解压检查，避免长人工示教占用多GB峰值内存。
        row['cameras'] = {}
        for name in ('ceiling','wrist_left','wrist_right'):
            key = 'images_'+name
            if key not in archive.files: continue
            images = archive[key]
            valid = images.shape==(n,240,320,3) and images.dtype==np.uint8
            mean = float(images.mean())
            black = int(np.sum(~np.any(images,axis=(1,2,3))))
            row['cameras'][name] = {'shape':list(images.shape),'dtype':str(images.dtype),
                                   'mean_brightness':mean,'fully_black_frames':black}
            if not valid or black: row['issues'].append('invalid RGB '+name)
            del images
        row['rgb_complete'] = len(row['cameras'])==3
        row['rgb_status_snapshot'] = provenance.get('rgb_export_status')
        if row['cameras'] and not row['rgb_complete']: row['issues'].append('partial RGB cameras')
    ids = np.array([model.actuator(str(name)).id for name in arrays['actuator_names']])
    qids = model.jnt_qposadr[model.actuator_trnid[ids,0]]
    np.testing.assert_allclose(arrays['qpos'][:,qids],arrays['state'],atol=1e-6,rtol=0)
    np.testing.assert_array_equal(arrays['native_action'][:,ids],arrays['action'])
    # 检查未来16步窗口在每条轨迹末尾正确填充；不跨轨迹。
    for frame in (0,max(0,n-16),max(0,n-2),n-1):
        window = arrays['action'][np.minimum(frame+np.arange(16),n-1)]
        assert window.shape==(16,16)
    model.eq_data[:] = arrays['initial_eq_data']
    mujoco.mj_setState(model,data,arrays['initial_integration_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model,data)
    arm_joints = [model.joint(f'openarm_{side}_joint{i}').id for side in ('right','left') for i in range(1,8)]
    dofs = model.jnt_dofadr[arm_joints]
    drawer = model.joint('drawer_slide').qposadr[0]
    contacts, forces = [], []
    error = 0.
    weld = model.eq_type==mujoco.mjtEq.mjEQ_WELD
    for i in range(n):
        error = max(error,float(np.max(np.abs(data.qpos-arrays['qpos'][i]))))
        c,f = finger_handle_contacts(model,data,include_stems=True)
        contacts.append(c);forces.append(f)
        model.eq_data[:] = arrays['eq_data'][i]
        data.eq_active[:] = arrays['eq_active'][i]
        data.ctrl[:] = arrays['native_action'][i]
        for _ in range(int(arrays['substeps'][i])):
            mujoco.mj_forward(model,data)
            data.qfrc_applied[:] = 0
            data.qfrc_applied[dofs] = data.qfrc_bias[dofs]
            mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
    error = max(error,float(np.max(np.abs(data.qpos-arrays['terminal_qpos']))))
    counts, force = np.array(contacts), np.array(forces)
    both = np.all(counts>0,axis=1)&np.all(force>.01,axis=1)
    opening = -arrays['drawer']*1000
    increasing = np.r_[False,np.diff(opening)>.05]
    decreasing = np.r_[False,np.diff(opening)<-.05]
    final = -float(data.qpos[drawer])*1000
    closed_hold = n>=30 and bool(np.all(np.abs(opening[-30:])<=5)) and abs(final)<=5
    active_welds = int(np.any(arrays['eq_active'][:,weld],axis=1).sum())
    row.update(frames=n,seconds=n/30,initial_open_mm=float(opening[0]),peak_open_mm=float(opening.max()),
        final_open_mm=final,open_hold_seconds=longest(opening>=80)/30,closed_hold_passed=closed_hold,
        bilateral_frames=int(both.sum()),longest_bilateral_seconds=longest(both)/30,
        bilateral_pull_frames=int((both&increasing).sum()),bilateral_push_frames=int((both&decreasing).sum()),
        peak_finger_forces_N=force.max(axis=0).tolist(),actual_weld_frames=active_welds,
        replay_error=error,replay_verified=error<1e-6,
        warmup_frames=provenance.get('warmup_frames',0),
        action_sha256=hashlib.sha256(arrays['action'].tobytes()).hexdigest(),
        initial_condition_sha256=hashlib.sha256(arrays['qpos'][0].tobytes()+json.dumps(overrides,sort_keys=True).encode()).hexdigest())
    if error>=1e-6:row['issues'].append('replay mismatch')
    if provenance.get('weld_frames') != active_welds:row['warnings'].append('reported weld count incorrect')
    success = (abs(opening[0])<=5 and row['open_hold_seconds']>=1 and closed_hold
               and both.any() and row['bilateral_pull_frames']>0 and not active_welds
               and provenance.get('mode')=='contact' and not row['issues'])
    row['numeric_success'] = bool(success)
    if row['selfcheck']:row['classification']='selfcheck_excluded'
    elif row['issues']:row['classification']='data_issue'
    elif active_welds or provenance.get('mode')!='contact':row['classification']='assisted_excluded'
    elif not success:row['classification']='task_failed_or_different_task'
    elif not row['rgb_complete']:row['classification']='numeric_success_rgb_pending' if row['rgb_status_snapshot']=='pending' else 'numeric_success_only'
    else:row['classification']='success_rgb_candidate'
    if row['reported_success'] is not None and bool(row['reported_success']) != bool(success):
        row['warnings'].append('legacy task flag differs from recomputed assembly-contact result')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','-C',str(ROOT/'.cache/openarm-control'),'rev-parse','HEAD'],text=True).strip()
    if commit!=UPSTREAM_COMMIT:raise ValueError('upstream version changed')
    paths=sorted(p.parent for group in ('drawer_trial','drawer_expansion','drawer_human')
                 for p in (ROOT/'data'/group).rglob('report.json') if (p.parent/'episode.npz').exists())
    result={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'scope':'all drawer episode.npz with report.json in drawer_trial/expansion/human; numeric/RGB replicas kept separate',
            'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'episodes':[]}
    for index,path in enumerate(paths,1):
        try:row=audit(path)
        except Exception as error:row={'path':str(path.relative_to(ROOT)),'classification':'audit_error','error':repr(error)}
        result['episodes'].append(row)
        (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(f"{index}/{len(paths)} {row['path']} {row['classification']}",flush=True)
    groups={}
    for row in result['episodes']:
        if 'action_sha256' in row:groups.setdefault(row['action_sha256'],[]).append(row['path'])
    result['action_duplicate_groups']=[v for v in groups.values() if len(v)>1]
    result['counts']=dict(Counter(row['classification'] for row in result['episodes']))
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 抽屉采集数据统一审核', '', '扫描时间：'+result['time'], '',
           '口径：整个把手组件双侧接触、打开80mm保持1秒、终末关闭5mm内保持1秒、无抓取绑定、独立动作回放。', '',
           'RGB检查覆盖尺寸、dtype和全黑帧；未对全部图像逐帧重渲染做像素比对。状态为扫描快照，后台导出可能继续更新。', '',
           '| 分类 | 数量 |','|---|---:|']
    lines += [f'| {key} | {value} |' for key,value in result['counts'].items()]
    lines += ['', '## 人工录制', '', '| 记录 | 最大开度mm | 打开保持秒 | 最终开度mm | 双侧接触帧 | 分类 |',
              '|---|---:|---:|---:|---:|---|']
    for row in result['episodes']:
        if '/session_' in row['path']:
            lines.append(f"| {row['path'].replace('data/drawer_human/','')} | {row.get('peak_open_mm',0):.2f} | {row.get('open_hold_seconds',0):.2f} | {row.get('final_open_mm',0):.2f} | {row.get('bilateral_frames',0)} | {row['classification']} |")
    lines += ['', '## 数据使用边界', '',
              '- success_rgb_candidate仅表示可进入训练前整理，不自动加入现有训练/验证集。',
              '- 人工warm-start和脚本ready-start条件不同，需明确任务初始化及采样策略。',
              '- 数值/RGB副本和动作重复组不能重复计为独立示教；失败、辅助绑定、自检排除。',
              '- 本检查未修改原始数据、旧归一化统计或已有训练划分。', '',
              '## 动作重复组', '']
    for group in result['action_duplicate_groups']:
        lines.append('- '+'；'.join(group))
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    candidates=[{'path':row['path'],'frames':row['frames'],'warmup_frames':row.get('warmup_frames',0),
                 'action_sha256':row['action_sha256']} for row in result['episodes']
                if row['classification']=='success_rgb_candidate']
    (args.output/'rgb_candidates.json').write_text(json.dumps({'note':'Audit candidates only; not a training split or authorization to overwrite existing datasets.',
        'episodes':candidates},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['counts'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
