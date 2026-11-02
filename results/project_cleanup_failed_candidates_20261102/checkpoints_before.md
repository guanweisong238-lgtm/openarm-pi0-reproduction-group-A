# 当前保留的模型检查点

更新：2026-11-02。当前共15份带协议的完整检查点。推荐020/2000：固定33起点全部通过，独立重放33/33通过；021/022本轮训练未达到升级标准，未替换020，其他修复候选和历史失败均保留。

| 路径（相对此目录） | 用途与结果 |
| --- | --- |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000` | **当前推荐**；train 10/10、validation 3/3、eval_only 20/20，完整33/33及独立重放均通过 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_021_contact_expanded/2000` | 扩范围接触反馈微调；原train12/10/9/11为2/4，未进入新筛选16；不推荐 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_021_contact_expanded/4000` | 同课程末步；原train四起点0/4，未进入新筛选16；不推荐 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_022_contact_expanded/2000` | 增加020全部10个train起点成功预测保留样本；原train四起点4/4，新筛选16为11/16；低于020的15/16，不推荐 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_022_contact_expanded/4000` | 同课程末步；原train四起点4/4，新筛选16为13/16；低于020的15/16，不推荐 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/4000` | 同课程中间权重，未做完整闭环验收；不能因训练步数更多而替代2000步推荐模型 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/6000` | 同课程末步权重，未做完整闭环验收 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_019_feedback/1000` | 状态反馈修复候选，ID12/10筛查0/2 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_019_feedback/2000` | 状态反馈修复候选，两种加载精度下ID12/10均未通过 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_018_parity/500` | cached路径对照，4个训练起点仅ID9通过 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_017/2000` | 新旧监督混合退步案例，历史bfloat16完整评估0/33；原始精度对照仍未通过筛查 |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000` | 历史成功基线，固定批量评估13/33；历史交互单次ID12、ID2通过，ID6失败 |
| `pi0_openarm_drawer_velocity_v001/pull12_velocity_015/2000` | 最终016训练的直接源模型，保留以便重训016；自身闭环未通过 |
| `pi0_openarm_drawer_start_fit_delta_v001/start9_lowmix_refine_011_io_mask/450` | 局部起步里程碑，只通过同训练初态前9步，不代表完整任务 |
| `pi0_openarm_drawer_success_v003_lora/success13_lora_001/2000` | 原13条数据正式训练基线，完整闭环未通过 |

020部署使用fixed noise42、action chunk4、continuous夹爪、task open、glfw，以及 `--weight-precision checkpoint`。命令见[当前交接](../docs/project-handoff.md)，证据见[33/33验收报告](../results/drawer_eval_020_20261102/README.md)。020的原训练协议保持快照，后验结果另存 `validation_report.json`、`deployment_config.json`。旧016报告见[历史成功报告](../results/drawer_pull_success_20261025/README.md)。

2026-10-25曾删除11份已被替代的历史检查点，保留其协议、配置、统计及文件清单，不含权重备份。本次自主修复没有删除失败权重。历史报告出现的被删模型路径是当时实验位置，无法直接加载；需重训。

- [本次删除清单与元数据](../results/project_cleanup_20261025_103927/README.md)
- [清理前检查点说明快照](../docs/archive/checkpoints-before-cleanup-20261025.md)
- 更早清理记录：`results/project_cleanup_20261024_213130/`、`results/project_cleanup_20261025_recovery_014/`。

完整保留的目录均包含参数、优化器及其原始资产，本轮没有拆分模型内部文件。加载016不需要加载其源015；保留015是为了重训来源一致。

同样，加载推荐的020/2000无需同时加载016；020包含完整参数和归一化资产。保留源模型与失败候选用于重训和审计。

2026-11-02清理删除了已逐文件验证与展开目录完全一致的016/2000.zip，以及可重建的原始预处理缓存，释放约9.70GiB；全部11份原展开模型保留，之后新增上述4份021/022候选。见[清理报告](../results/project_cleanup_20261102/README.md)。本轮训练和筛选记录见[流程状态](../results/drawer_grasp_improvement_20261102/progress.json)，原33基准与本轮新筛选16不可混为同一组成绩。
