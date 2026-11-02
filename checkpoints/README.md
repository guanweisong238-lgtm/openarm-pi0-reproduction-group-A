# 当前保留的模型检查点

更新：2026-11-02晚间清理后，共7份完整检查点。推荐020/2000：固定33起点全部通过；两批新100分别84/100、85/100。021/022训练未达到升级标准，推荐权重没有改变。

| 路径（相对此目录） | 用途与结果 |
| --- | --- |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000` | **当前推荐**；train 10/10、validation 3/3、eval_only 20/20，完整33/33及独立重放均通过 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/4000` | 同课程中间权重，未做完整闭环验收；不能因训练步数更多而替代2000步推荐模型 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/6000` | 同课程末步权重，未做完整闭环验收 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000` | 历史成功基线，固定批量评估13/33；历史交互单次ID12、ID2通过，ID6失败 |
| `pi0_openarm_drawer_velocity_v001/pull12_velocity_015/2000` | 最终016训练的直接源模型，保留以便重训016；自身闭环未通过 |
| `pi0_openarm_drawer_start_fit_delta_v001/start9_lowmix_refine_011_io_mask/450` | 局部起步里程碑，只通过同训练初态前9步，不代表完整任务 |
| `pi0_openarm_drawer_success_v003_lora/success13_lora_001/2000` | 原13条数据正式训练基线，完整闭环未通过 |

020部署使用fixed noise42、action chunk4、continuous夹爪、task open、glfw，以及`--weight-precision checkpoint`。命令见[当前交接](../docs/project-handoff.md)，证据见[33/33验收报告](../results/drawer_eval_020_20261102/README.md)和[新100最终报告](../results/drawer_grasp_improvement_20261102/README.md)。加载020/2000无需同时加载016或015；上述7份保留模型均包含完整参数、优化器及原始资产。

## 已删除的失败候选

以下8份于2026-11-02晚间删除权重，表中成绩是删除前实验结果；路径不能直接加载。保留协议、配置、统计、逐文件清单、训练日志与评估轨迹，不含完整权重备份，需要重训才能重新推理。详见[本次清理报告](../results/project_cleanup_failed_candidates_20261102/README.md)。

| 原路径（已删除权重） | 历史结果 |
| --- | --- |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_021_contact_expanded/2000` | 扩范围接触反馈微调；原train12/10/9/11为2/4，未进入新筛选16；不推荐 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_021_contact_expanded/4000` | 同课程末步；原train四起点0/4，未进入新筛选16；不推荐 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_022_contact_expanded/2000` | 增加020全部10个train起点成功预测保留样本；原train四起点4/4，新筛选16为11/16；低于020的15/16，不推荐 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_022_contact_expanded/4000` | 同课程末步；原train四起点4/4，新筛选16为13/16；低于020的15/16，不推荐 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_019_feedback/1000` | 状态反馈修复候选，ID12/10筛查0/2 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_019_feedback/2000` | 状态反馈修复候选，两种加载精度下ID12/10均未通过 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_018_parity/500` | cached路径对照，4个训练起点仅ID9通过 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_017/2000` | 新旧监督混合退步案例，历史bfloat16完整评估0/33；原始精度对照仍未通过筛查 |

## 更早的清理

2026-11-02较早删除逐文件验证与展开目录一致的016/2000.zip和可重建原始预处理缓存，释放约9.70GiB，当时11份展开模型全部保留，后新增4份021/022候选。见[早间清理报告](../results/project_cleanup_20261102/README.md)。晚间再清理上述8份后剩7份；所有训练数据和评估证据保留。

2026-10-25曾删除11份已被替代的历史检查点，归档不含权重备份，需重训。见[当日删除清单](../results/project_cleanup_20261025_103927/README.md)、[当时检查点说明](../docs/archive/checkpoints-before-cleanup-20261025.md)。更早记录：`results/project_cleanup_20261024_213130/`、`results/project_cleanup_20261025_recovery_014/`。
