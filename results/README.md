# 实验结果索引

更新：2026-10-25。完整教学复盘见[optimization_log.md](../optimization_log.md)，当前运行命令见[项目交接](../docs/project-handoff.md)。

## 当前结果

- [最终模型报告与原始证据路径](drawer_pull_success_20261025/README.md)：π0骨干一步确定性改造版，标准起点及Y+5mm场景通过，warm-start失败。
- [初次标准起点评分](feedback_endpoint_016_nominal_score.json)、[冻结模型复测及验证评分](feedback_endpoint_016_frozen_physical_scores.json)。
- [现场展示评分](drawer_pull_success_20261025/live_showcase_095417_score.json)：开度101.56mm、17.07秒仿真，独立重放位置/速度/力误差0。
- 小型MLP成功基线：`drawer_goal_mlp_close0_v003/`、`drawer_goal_mlp_frozen_validation_v004/`，明确不属于π0。

成功依据是真实双指接触、接触拉动与>=80mm保持1秒；离线误差、专家成功或瞬时开度不替代该判据。未完成原多步flow复现、关闭往返或广泛泛化验证。

## 历史排查

| 阶段 | 报告 |
| --- | --- |
| 原13条训练的起步诊断 | [深入诊断](drawer_deep_diagnosis_20261024/README.md)、[早期采样对照](early_sampling_control_20261024/README.md) |
| 残差与首9帧拟合 | [工作流](start9_workflow_20261024/README.md)、[增强与动作权重](start9_aug_objective_20261024/README.md)、[时间采样](start9_time_sampling_20261024/README.md) |
| 局部物理起步011 | [报告](start9_lowmix_refine_011/README.md) |
| 扩大到接近012 | [报告与漂移诊断](approach13_delta_012/README.md) |
| 完整路径集中拟合013b | [报告](pull12_fit_013b/README.md) |
| 恢复数据014 | [报告](pull12_recovery_014/README.md) |
| 速度输入015 | [报告](pull12_velocity_015/README.md)，该文件部分叙述是训练时快照，最终闭环未通过 |

历史报告与日志保留，但部分权重已清理，旧路径不保证能加载。未将失败改写为成功。

## 清理审计

- [2026-10-25本次整理](project_cleanup_20261025_103927/README.md)：删除11份旧检查点及可重建恢复缓存，约57.26GiB；保留4份完整权重。
- [2026-10-25上一轮](project_cleanup_20261025_recovery_014/README.md)。
- [2026-10-24清理](project_cleanup_20261024_213130/README.md)。

清理元数据不含已删除权重的备份，精确重新运行旧模型需要重训。原始示教、反馈查询、物理轨迹、评估JSON和视频继续保留。
