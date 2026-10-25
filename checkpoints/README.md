# 当前保留的模型检查点

更新：2026-10-25。清理后保留4份完整检查点，约20.7GiB；下表是当前实际存在的权重。

| 路径（相对此目录） | 用途与结果 |
| --- | --- |
| `pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000` | 当前成功模型，π0骨干一步确定性改造版；ID12及ID2+5mm通过，warm-start ID6未通过 |
| `pi0_openarm_drawer_velocity_v001/pull12_velocity_015/2000` | 最终016训练的直接源模型，保留以便重训016；自身闭环未通过 |
| `pi0_openarm_drawer_start_fit_delta_v001/start9_lowmix_refine_011_io_mask/450` | 局部起步里程碑，只通过同训练初态前9步，不代表完整任务 |
| `pi0_openarm_drawer_success_v003_lora/success13_lora_001/2000` | 原13条数据正式训练基线，完整闭环未通过 |

016部署必须使用fixed noise42、action chunk4、continuous夹爪与task open。命令见[当前交接](../docs/project-handoff.md)，证据见[最终报告](../results/drawer_pull_success_20261025/README.md)。

本轮删除11份已被替代的历史检查点，保留其协议、配置、统计及文件清单，不含权重备份。历史报告出现的被删模型路径是当时实验位置，无法直接加载；需重训。

- [本次删除清单与元数据](../results/project_cleanup_20261025_103927/README.md)
- [清理前检查点说明快照](../docs/archive/checkpoints-before-cleanup-20261025.md)
- 更早清理记录：`results/project_cleanup_20261024_213130/`、`results/project_cleanup_20261025_recovery_014/`。

完整保留的目录均包含参数、优化器及其原始资产，本轮没有拆分模型内部文件。加载016不需要加载其源015；保留015是为了重训来源一致。
