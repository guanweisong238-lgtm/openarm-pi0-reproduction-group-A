"""Read-only failure diagnosis; held-out states never become training snapshots or labels."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from drawer_simulation import DrawerSimulation
from drawer_success_training_config import ROOT


def main():
    run = ROOT/'logs/a2_pi0/drawer_generalization_020_20261102'
    out = ROOT/'results/drawer_generalization_failure_diagnosis_20261102'
    out.mkdir(exist_ok=False)
    manifest = ROOT/'data/processed/openarm_drawer_eval_v002/manifest.json'
    cases = {r['episode_id']:r for r in json.loads(manifest.read_text())['episodes']}
    index = json.loads((run/'index.json').read_text())
    records = []
    for row in index:
        trial = run/row['trial']
        with np.load(trial/'closed_loop.npz') as raw:
            contacts=raw['contacts'];forces=raw['forces'];applied=raw['applied'];requested=raw['requested']
            both=((contacts>0)&(forces>.01)).all(1);streak=maximum=0
            for flag in both:
                streak=streak+1 if flag else 0;maximum=max(maximum,streak)
            clipped=np.abs(requested[:,:8]-applied[:,:8])>1e-9
            touched=np.flatnonzero((contacts>0).any(1))
            item=dict(episode_id=row['episode_id'],group=cases[row['episode_id']]['evaluation_group'],
                      success=row['success'],failure_mode=row['failure_mode'],peak_open_mm=row['peak_open_mm'],
                      control_steps=len(applied),bilateral_steps=int(both.sum()),max_bilateral_streak=maximum,
                      first_contact_step=int(touched[0]+1) if len(touched) else None,
                      contact_steps_per_finger=(contacts>0).sum(0).tolist(),max_force_N=forces.max(0).tolist(),
                      clipped_steps_per_right_joint_and_gripper=clipped.sum(0).tolist(),
                      clip_steps_first150=clipped[:150].sum(0).tolist(),
                      max_clip_magnitude_rad=np.abs(requested[:,:8]-applied[:,:8]).max(0).tolist(),
                      trial=str(trial.relative_to(ROOT)),source_sha256=hashlib.sha256((trial/'closed_loop.npz').read_bytes()).hexdigest())
            if not row['success']:
                env=DrawerSimulation(ROOT/cases[row['episode_id']]['path'])
                try:
                    scratch=mujoco.MjData(env.model);site=env.model.site('right_ee_control_point').id
                    handle=env.model.geom('drawer_handle').id;xyz=[];rot=[]
                    for q in raw['qpos']:
                        scratch.qpos[:]=q;mujoco.mj_kinematics(env.model,scratch)
                        rotation=scratch.site_xmat[site].reshape(3,3)
                        xyz.append(scratch.site_xpos[site]+rotation@np.array([0,0,-.16])-scratch.geom_xpos[handle])
                        rot.append(rotation.copy())
                    xyz=np.asarray(xyz);near=int(np.linalg.norm(xyz,axis=1).argmin())
                    item.update(closest_nominal_grasp_point_step=near+1,closest_nominal_grasp_point_distance_mm=float(np.linalg.norm(xyz[near])*1000),
                                closest_grasp_point_minus_handle_mm=(xyz[near]*1000).tolist(),
                                closest_gripper_target_rad=float(applied[near,7]),
                                closest_rotation_world=rot[near].tolist())
                    np.savez_compressed(out/f'geometry_{row["episode_id"]}.npz',grasp_point_minus_handle=xyz)
                finally:env.close()
            records.append(item)
    payload=dict(scope='Read-only diagnosis of observed test failures; no held-out training snapshots, expert labels, or gradients.',
                 records=records,failures=sum(not r['success'] for r in records))
    (out/'diagnosis.json').write_text(json.dumps(payload,indent=2)+'\n')
    bad=[r for r in records if not r['success']]
    lines=['# 新100起点失败诊断（2026-11-02）','','只读取现有评估记录，未把评估状态制作成训练快照或专家标签。', '',
           '| ID | 失败类型 | 双指最长持续步 | 峰值开度mm | 名义夹持点最近距离mm |', '| --- | --- | ---: | ---: | ---: |']
    for r in bad:lines.append(f'| {r["episode_id"]} | {r["failure_mode"]} | {r["max_bilateral_streak"]} | {r["peak_open_mm"]:.2f} | {r["closest_nominal_grasp_point_distance_mm"]:.2f} |')
    lines += ['', '14条未满足抓取条件的轨迹双指最长有效接触仅0–4步；237达到过7步但未持续拉开。6条开度超过80mm仍未通过抓取条件。', '',
              '关节限位、接触位置偏差与夹爪闭合记录见diagnosis.json。名义夹持点是工具局部偏移[0,0,-.16]，不是实际指片接触点；最近距离和关节裁剪只能作为诊断线索，不能单独证明根因。', '',
              '下一轮采用独立扩范围训练场景、增加接触前后反馈样本，并保留原成功样本；训练与最终验收起点另外生成并隔离。', '']
    (out/'README.md').write_text('\n'.join(lines))
    print('DIAGNOSIS_COMPLETE',len(records),'failures',len(bad),flush=True)


if __name__=='__main__':main()
