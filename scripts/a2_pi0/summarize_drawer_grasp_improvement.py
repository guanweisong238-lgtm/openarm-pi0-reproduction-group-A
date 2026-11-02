"""Package completed grasp experiments without changing scores, inputs, or checkpoints."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from summarize_drawer_eval import describe

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/drawer_grasp_improvement_20261102'
RUN=ROOT/'logs/a2_pi0/grasp_improvement_20261102/baseline020_final100'


def read(p):return json.loads(p.read_text())
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def wilson(k,n):
    z=1.959963984540054;v=k/n;den=1+z*z/n
    center=(v+z*z/(2*n))/den;half=z*math.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def main():
    final=read(OUT/'final_comparison.json');assert final['complete']
    rows=read(RUN/'index.json');assert len(rows)==100 and {r['episode_id'] for r in rows}==set(range(400,500))
    replay=read(OUT/'baseline020_final100_independent_replay.json');proof=replay['results']
    assert replay['complete'] and len(proof)==100 and all(r['evidence_valid'] for r in proof)
    assert {r['episode_id'] for r in proof}==set(range(400,500))
    verdicts={r['episode_id']:r['model_pull_success'] for r in proof}
    assert all(verdicts[r['episode_id']]==r['success'] for r in rows)
    assert all(r['teacher_actions_executed']==r['intermediate_states_injected']==0 for r in proof)
    assert all(r['noise_mode']=='fixed' and r['noise_seed']==42 and r['action_chunk']==4 for r in proof)
    cfg=read(RUN/'run_config.json');checkpoint=Path(cfg['checkpoint'])
    expected=dict(seconds=24.,task='open',seed=42,action_chunk=4,gripper_mode='continuous',repeat=1,step_limit=720,weight_precision='checkpoint',gl_backend='glfw')
    assert cfg['run']==expected
    frozen=read(OUT/'frozen_final_models.json');expected_files=frozen['models']['baseline020']['files']
    actual={name:digest(checkpoint/name) for name in expected_files};assert actual==expected_files
    write(OUT/'checkpoint_after_final100.json',dict(checkpoint=str(checkpoint),files=actual,identical_to_frozen_before_test=True))
    for name,h in read(OUT/'locked_evaluation_inputs.json')['files'].items():assert digest(ROOT/name)==h
    for name,h in cfg['runtime_source_sha256'].items():assert digest(ROOT/'scripts/a2_pi0'/name)==h
    for name,h in cfg['manifest_sha256'].items():assert digest(Path(name))==h
    manifest=read(ROOT/'data/processed/openarm_drawer_eval_v004_final/manifest.json')
    cases={r['episode_id']:r for r in manifest['episodes']}
    groups={name:describe([r for r in rows if cases[r['episode_id']]['evaluation_group']==name]) for name in ('near','expanded')}
    metrics=dict(completed_at=final['completed_at'],checkpoint=str(checkpoint),model_upgraded=False,
        overall=describe(rows),groups=groups,scope='Unseen100 for unchanged020; final set never selects weights; separate from earlier IDs200..299.')
    for item in [metrics['overall'],*groups.values()]:item['wilson95']=wilson(item['success'],item['trials'])
    inference=[];raw_hashes={}
    for r in rows:
        path=RUN/r['trial']/'closed_loop.npz'
        with np.load(path) as raw:inference.extend(raw['inference_seconds'].tolist())
        raw_hashes[r['trial']]=digest(path)
    metrics['pooled_inference_ms']=dict(count=len(inference),p50=float(np.percentile(inference,50)*1000),p95=float(np.percentile(inference,95)*1000))
    metrics['max_simulation_seconds_on_success']=max(r['simulation_seconds'] for r in rows if r['success'])
    write(OUT/'metrics.json',metrics)
    with (OUT/'starts_and_outcomes.csv').open('w') as stream:
        fields=['episode_id','group','x_delta_mm','y_delta_mm','handle_radius_mm','jitter_half_width_rad','joint_seed','success','peak_open_mm','failure_mode']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            case=cases[r['episode_id']];p=case['eval_parameters']
            writer.writerow(dict(episode_id=r['episode_id'],group=case['evaluation_group'],x_delta_mm=(p['drawer_offset_x']-.1)*1000,
                y_delta_mm=p['drawer_offset_y']*1000,handle_radius_mm=p['handle_radius']*1000,jitter_half_width_rad=p['right_jitter'],
                joint_seed=p['seed'],success=r['success'],peak_open_mm=r['peak_open_mm'],failure_mode=r['failure_mode']))
    for name in ('index.json','run_config.json'):shutil.copyfile(RUN/name,OUT/name)
    original_prefix=hashlib.sha256((ROOT/'optimization_log.md').read_bytes()[:19915]).hexdigest()
    assert original_prefix=='03bd60097df4d027192efed15412acf916f1a4c5a52eb12d07c97a7230e6e21a'
    audit=dict(complete=True,trials=100,success=metrics['overall']['success'],evidence_valid=100,model_upgraded=False,
        checkpoint_before_after_identical=True,checkpoint_inference_files=len(actual),locked_evaluation_inputs_unchanged=True,
        runtime_source_unchanged=True,teacher_actions_executed=0,intermediate_states_injected=0,
        max_replay_errors={key:max(r[key] for r in proof) for key in ('replay_max_qpos_error','replay_max_qvel_error','replay_max_contact_force_error')},
        original_optimization_log_prefix_bytes=19915,original_optimization_log_prefix_sha256=original_prefix,
        raw_trial_sha256=raw_hashes,model_selection_used_final_results=False)
    write(OUT/'completion_audit.json',audit)
    driver_report=OUT/'driver_report.md'
    if not driver_report.exists():shutil.copyfile(OUT/'README.md',driver_report)
    lines=['# 扩范围抓取修复与最终独立验收（2026-11-02）','',
        '**本轮训练没有取得合格升级，继续推荐020/2000。** 已完成清理、失败诊断、补充训练数据、021/022训练与开发筛选、最终独立100起点测试及重放复核。','',
        '## 训练与筛选结果','',
        '| 权重 | 原train12/10/9/11 | 新near8 | 新expanded8 | 筛选合计16 | 决定 |',
        '| --- | --- | --- | --- | --- | --- |',
        '| 020/2000 | 历史同权重已通过，本轮不重复 | 8/8 | 7/8 | 15/16 | 保留 |',
        '| 021/2000 | 2/4 | 未执行 | 未执行 | 未执行 | 不进入新筛选 |',
        '| 021/4000 | 0/4 | 未执行 | 未执行 | 未执行 | 不进入新筛选 |',
        '| 022/2000 | 4/4 | 5/8 | 6/8 | 11/16 | 未达到升级标准 |',
        '| 022/4000 | 4/4 | 7/8 | 6/8 | 13/16 | 未达到升级标准 |','',
        '021采用30个新训练场景增加接触阶段反馈，共7647查询；原train四起点回归后，022加入020在全部10个train起点上的325条成功预测保留标签，样本采样权重12，合计7972查询。两轮均从020开始、学习率3e-6、4000步、保存2000/4000步，cached端点、夹爪权重4、局部位置权重0.5。最初计划的022学习率1e-5未执行；修订见[amendment022.json](amendment022.json)，原协议保留。','',
        '新增专家场景尝试48个，30个完整专家与反馈专家通过；专家失败不证明物理不可达。训练参考验收仅筛选训练数据，开发/最终起点没有按模型或专家成功过滤。7972查询不是7972条独立轨迹。所有标签和梯度来自原10个train父起点，原物理validation/eval_only、开发16和最终100均不用于训练。','',
        '022恢复了四个旧起点的通过，但没有在开发16上超过020，未执行新候选的完整33回归或最终100。4/4不能写成33/33，未执行不计失败。020历史完整33/33仍有效，原020/4000和/6000未在本轮评估。','',
        '## 020最终新100起点结果','',f'完成时间：{final["completed_at"]}。起点400—499，seed2026110207；开发起点300—315，seed2026110206。最终100在模型训练前锁定，模型选择在首次最终测试推理前冻结。','',
        '| 分组 | 成功 | 成功率 | Wilson 95%区间 |','| --- | ---: | ---: | --- |']
    for label,item in [('原范围新组合',groups['near']),('扩大范围新组合',groups['expanded']),('预定70/30混合',metrics['overall'])]:
        lo,hi=item['wilson95'];lines.append(f'| {label} | {item["success"]}/{item["trials"]} | {item["success_rate"]:.1%} | {lo:.1%}—{hi:.1%} |')
    lines+=['','上一批起点200—299的结果为84/100（66/70、18/30）。本批为85/100（66/70、19/30），**权重相同，不能把分数差异当作本轮训练带来的提升**。合计率只代表预定70/30混合；两组范围、起点与置信区间分别保留。','',
        f'成功时仿真耗时中位数{metrics["overall"]["simulation_seconds_median_on_success"]:.2f}秒，最长{metrics["max_simulation_seconds_on_success"]:.2f}秒；仿真在推理期暂停，不代表真机实时性能。实际{len(inference)}次推理合并p50/p95为{metrics["pooled_inference_ms"]["p50"]:.2f}/{metrics["pooled_inference_ms"]["p95"]:.2f}ms，区别于各trial分位数的中位数。','',
        f'失败分类：{json.dumps(metrics["overall"]["failure_modes"],ensure_ascii=False)}。single_finger表示未满足连续5步双指各>.01N，并非全过程只有一指接触。峰值超过80mm也不能替代完整判据。','',
        '100条独立重放全部证据有效，位置、速度、接触力最大误差见[一致性审计](completion_audit.json)。权重21个推理文件前后一致，运行源码及锁定清单检查通过；专家运行动作与中途状态注入均为0。最终测试后未再调参或换权重。','',
        '## 运行条件与范围','',
        '沿用双指各>.01N连续5步、接触拉动、开度≥80mm保持30控制步；open任务、24秒上限、固定噪声42、H4、continuous夹爪、checkpoint原精度、GLFW软件渲染、每条独立进程。','',
        'near：X相对基准±10mm，Y±30mm，把手半径5—12mm，关节扰动采样半宽.01—.04rad。expanded：X±20mm，Y±50mm，把手半径4—14mm，采样半宽.04—.08rad。X基准场景偏移.10m；实际关节扰动受关节/控制限位约束。多个因素同时变化，不能做单因素归因。','',
        '同一模拟器、任务、相机和ready姿态来源，不能推断任意场景、真实机器人或关闭往返成功率；当前策略仍是π0骨干一步确定性改造版，原多步flow未验收。','',
        '## 失败对比与证据','',
        '[开发集成对诊断](candidate_failure_comparison.md)显示，部分022回归样本在相同初态和噪声下，初始目标仅差几毫弧度，随后发生毫米至厘米级轨迹差异；这不证明单一关节差异是唯一原因。','',
        '![开发集ID300的开度与接触对比](development_regression_id300.png)','',
        '[逐条参数与结果CSV](starts_and_outcomes.csv)、[指标JSON](metrics.json)、[独立重放](baseline020_final100_independent_replay.json)、[候选记录](candidates.json)、[初始协议](protocol.json)、[冻结模型指纹](frozen_final_models.json)。','',
        '数据审计：[021数据](training_data_full_audit.json)、[022数据](022_training_data_full_audit.json)、[021梯度](021_gradient_audit.json)、[022梯度](022_gradient_audit.json)、[训练场景覆盖](training_reference_coverage.json)。','',
        '[清理报告](../project_cleanup_20261102/README.md)：释放9.70GiB，原11份展开模型、原始数据和评估证据保留；新候选另外保存4份，当前共15份完整检查点。原始预处理缓存需要时可重建。优化日志原始19915字节前缀校验一致，其后按时间追加新记录。','',
        '原始最终轨迹：`logs/a2_pi0/grasp_improvement_20261102/baseline020_final100/`。模型推荐未升级，后续若继续训练需另留新的独立最终起点，不把本批再次称为未见过的测试集。','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print('EXPERIMENT_PACKAGED',json.dumps(dict(success=metrics['overall']['success'],groups=groups,max_replay_errors=audit['max_replay_errors']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
