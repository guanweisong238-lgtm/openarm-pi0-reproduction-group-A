"""Read-only paired diagnosis on DEVELOPMENT starts; no final-test states or labels."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from drawer_simulation import DrawerSimulation
from drawer_success_training_config import ROOT


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result=ROOT/'results/drawer_grasp_improvement_20261102'
    logs=ROOT/'logs/a2_pi0/grasp_improvement_20261102'
    manifest=ROOT/'data/processed/openarm_drawer_eval_v003_selection/manifest.json'
    cases={r['episode_id']:r for r in json.loads(manifest.read_text())['episodes']}
    base=logs/'baseline_selection16'
    baseline={r['episode_id']:r for r in json.loads((base/'index.json').read_text())}
    regressions=[];comparisons=[]
    def load(trial):
        with np.load(trial/'closed_loop.npz') as raw:
            return {k:raw[k].copy() for k in ('qpos','qpos_before','qvel_before','requested','applied','contacts','forces','opening','inference_noise')}
    for step in (2000,4000):
        run=logs/f'feedback_endpoint_022_contact_expanded_{step}_selection16'
        for row in json.loads((run/'index.json').read_text()):
            eid=row['episode_id'];old=baseline[eid]
            comparisons.append(dict(step=step,episode_id=eid,baseline_success=old['success'],candidate_success=row['success']))
            if not old['success'] or row['success']:continue
            bt=base/old['trial'];ct=run/row['trial'];b=load(bt);c=load(ct)
            assert np.array_equal(b['qpos_before'][0],c['qpos_before'][0])
            assert np.array_equal(b['qvel_before'][0],c['qvel_before'][0])
            assert np.array_equal(b['inference_noise'][0],c['inference_noise'][0])
            env=DrawerSimulation(ROOT/cases[eid]['path'])
            try:
                scratch=mujoco.MjData(env.model);sid=env.model.site('right_ee_control_point').id;hid=env.model.geom('drawer_handle').id
                def geometry(raw):
                    points=[];relative=[]
                    for q in raw['qpos']:
                        scratch.qpos[:]=q;mujoco.mj_kinematics(env.model,scratch)
                        point=scratch.site_xpos[sid]+scratch.site_xmat[sid].reshape(3,3)@np.array([0,0,-.16])
                        points.append(point.copy());relative.append(point-scratch.geom_xpos[hid])
                    return np.asarray(points),np.asarray(relative)
                bp,br=geometry(b);cp,cr=geometry(c)
                def summary(raw,relative):
                    both=((raw['contacts']>0)&(raw['forces']>.01)).all(1);maximum=streak=0;first_grasp=None
                    for i,x in enumerate(both):
                        streak=streak+1 if x else 0;maximum=max(maximum,streak)
                        if streak>=5 and first_grasp is None:first_grasp=i+1
                    touch=np.flatnonzero((raw['contacts']>0).any(1))
                    clip=np.abs(raw['requested'][:150,:8]-raw['applied'][:150,:8])>1e-9
                    return dict(first_contact_step=int(touch[0]+1) if len(touch) else None,
                        first_grasp_step=first_grasp,max_bilateral_streak=maximum,
                        closest_nominal_grasp_point_first150_mm=float(np.linalg.norm(relative[:150],axis=1).min()*1000),
                        peak_open_mm=float(raw['opening'].max()*1000),clipped_steps_first150_per_right_dof=clip.sum(0).tolist())
                delta=c['requested'][0,:8]-b['requested'][0,:8]
                common=min(len(bp),len(cp),30)
                entry=dict(candidate_step=step,episode_id=eid,group=cases[eid]['evaluation_group'],
                    baseline_trial=str(bt.relative_to(ROOT)),candidate_trial=str(ct.relative_to(ROOT)),
                    initial_qpos_qvel_and_noise_equal=True,first_requested_right8_delta_rad=delta.tolist(),
                    first_right_arm_action_max_delta_rad=float(np.abs(delta[:7]).max()),
                    grasp_point_world_difference_at30_or_last_mm=float(np.linalg.norm(cp[common-1]-bp[common-1])*1000),
                    baseline=summary(b,br),candidate=summary(c,cr),
                    raw_sha256=dict(baseline=sha(bt/'closed_loop.npz'),candidate=sha(ct/'closed_loop.npz')))
                regressions.append(entry)
            finally:env.close()
    payload=dict(scope='Read-only paired development-case comparison; final-test trajectories are not read; no training labels or parameter changes.',
        comparisons=comparisons,regressions=regressions,
        caution='Nominal grasp point is not an actual finger contact point. Output/trajectory differences are observations, not a single-parameter causal ablation.')
    (result/'candidate_failure_comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# 022相对020的开发集回归诊断','','同一起点、相同初始位置/速度和固定噪声对照；只读取开发集，不读取最终100。', '',
        '| 候选步数 | ID | 首个右臂动作最大差rad | 第30步夹持点世界位置差mm | 双指最长持续步：020→022 | 峰值开度：020→022 mm |',
        '| --- | --- | ---: | ---: | --- | --- |']
    for r in regressions:
        lines.append(f'| {r["candidate_step"]} | {r["episode_id"]} | {r["first_right_arm_action_max_delta_rad"]:.6f} | {r["grasp_point_world_difference_at30_or_last_mm"]:.2f} | '
                     f'{r["baseline"]["max_bilateral_streak"]}→{r["candidate"]["max_bilateral_streak"]} | {r["baseline"]["peak_open_mm"]:.2f}→{r["candidate"]["peak_open_mm"]:.2f} |')
    lines+=['','这些样本在020成功、022失败；具体首次接触、抓稳时刻和关节裁剪统计见JSON。轨迹差异不能单独归因于某一关节或某条标签。',
        '旧四起点通过不等于新起点通过；离线误差改善没有保证完整闭环改善。最终模型选择已冻结，本诊断不用于重新调参。','']
    (result/'candidate_failure_comparison.md').write_text('\n'.join(lines))
    print('PAIRED_DEVELOPMENT_REGRESSIONS',[(r['candidate_step'],r['episode_id'],round(r['grasp_point_world_difference_at30_or_last_mm'],2)) for r in regressions])


if __name__=='__main__':main()
