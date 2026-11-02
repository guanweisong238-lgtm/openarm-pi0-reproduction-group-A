# 文档索引

更新：2026-11-02。先读当前状态，再按需查历史实验。历史文档中的“最新”“训练中”和旧模型路径只描述当时，不覆盖当前交接。

## 当前使用

| 文档 | 内容 |
| --- | --- |
| [项目交接](project-handoff.md) | 当前模型、启动命令、数据资产、环境及下一步 |
| [评估协议](drawer-eval-protocol.md) | 冻结的成功判据、固定运行配置、指标定义与测试集规则 |
| [评估运行手册](eval-runbook.md) | 冒烟测试、全量跑法、中断续跑、出表与常见错误 |
| [当前33/33验收报告](../results/drawer_eval_020_20261102/README.md) | 020/2000完整评估、独立重放、数据隔离审计与实测展示命令 |
| [自主修复过程](../results/drawer_repair_20261101/README.md) | 017退步后的诊断、独立随机训练数据、020课程与筛查 |
| [修复评估协议](drawer-repair-eval-protocol.md) | 新候选的权重精度、逐级筛查、同一权重33条严格验收 |
| [优化复盘](../optimization_log.md) | 从起步错误到拉取成功的教学式解释与证据 |
| [最终模型报告](../results/drawer_pull_success_20261025/README.md) | 训练协议、成功与失败边界、录像和物理评分 |
| [检查点清单](../checkpoints/README.md) | 当前11份权重及其验收状态 |
| [结果索引](../results/README.md) | 各阶段证据入口与清理审计 |
| [接口约定](interfaces.md) | 基础模块接口；当前模型增量协议以checkpoint和交接为准 |
| [人工采集操作](drawer-human-teleop.md) | 鼠标控制及录制流程 |

## 历史训练方案

下列方案保留用于理解和复查，不应直接作为当前运行入口；其中部分检查点已删除。

- 单轨迹：[初始过拟合](drawer-overfit.md)、[无损输入](drawer-lossless-overfit.md)、[阶段拟合](drawer-stage-fit.md)、[关节加权](drawer-joint-weighting.md)。
- 多轨迹：[13条成功数据训练](drawer-success-training.md)、[训练诊断](drawer-training-diagnosis.md)、[早期采样对照](drawer-early-sampling-control.md)。
- 分阶段：[首9帧拟合](drawer-start-fit.md)、[接近与夹持](drawer-approach-training.md)。
- 数据准备：[试采](drawer-data-trial.md)、[扩充](drawer-data-expansion.md)。
- [原周计划](weekly-progress.md)记录项目计划与历史进展，不是当前验收表。

## 历史入口快照

[archive/](archive/)保存本次整理前的交接、检查点说明、项目计划和结果模板。快照不再更新；其中相对路径按原文件位置理解，当前有效链接请使用本索引。

文档体积很小，保留研究过程有助于复现。节省空间主要通过删除被替代权重和可重建缓存完成，原始数据及失败证据保留。
