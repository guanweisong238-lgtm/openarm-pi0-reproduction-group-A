# 项目清理与文档整理（2026-10-25）

已完成删除并验证保留模型文件未改变。释放文件分配空间约57.257GiB，操作前可用27.025GiB，删除后可用84.283GiB。文件系统实时可用量可能随其他程序写入略有变化。

## 删除清单

| 路径 | 释放GiB | 原因 |
| --- | ---: | --- |
| `checkpoints/pi0_openarm_drawer_approach_v001/approach13_delta_012/1200` | 5.179 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_approach_v001/pull12_fit_013b/2000` | 5.179 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_approach_v001/pull12_recovery_014/3000` | 5.179 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_lora_lossless/stage_fit_lossless_002/3500` | 5.178 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_lora_lossless/stage_fit_right8_key147_005/4500` | 5.178 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_delta_004_io_mask/600` | 5.180 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_delta_refine_005_io_mask/1200` | 5.180 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_time_lowmix_010_io_mask/300` | 5.180 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_start_fit_v001/start9_cleanup_002_io_mask/200` | 5.180 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_start_fit_v001/start9_refine_003_io_mask/600` | 5.180 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `checkpoints/pi0_openarm_drawer_success_v003_lora/success13_early60_500_003/500` | 5.178 | 已完成且被后续模型替代的历史对照；评估与源码快照保留 |
| `.cache/drawer_recovery_inputs` | 0.285 | 可从保留的恢复轨迹和源码重新生成的预处理缓存 |

本轮删除11份旧检查点及一个缓存目录。删除前已归档小型配置、协议、统计、索引和文件大小清单，见[manifest.json](manifest.json)与[metadata/](metadata/)。这些归档不包含大模型权重或优化器分片的备份，无法直接恢复已删除模型，需要重训。

`.cache/drawer_recovery_inputs/`可以从保留的原始恢复轨迹、清单与`drawer_recovery_cache.py`重新构建，首次使用相应历史训练入口会增加预处理时间。

## 保留

当前[4份完整检查点](../../checkpoints/README.md)及其参数/优化器/资产；其中015是最终016重训的直接源模型。保留文件大小和修改时间逐一核对一致。

原始人工示教、原13条成功数据、恢复轨迹、反馈查询、底座权重、虚拟环境、原始预处理缓存、速度缓存、源码、实验日志、视频、成功和失败评分均保留。本轮没有删除历史实验源码或修改训练/推理算法。

## 文档整理

- 根[README](../../README.md)改为当前成果、有效命令和文档入口。
- [当前交接](../../docs/project-handoff.md)精简为当前模型、依赖、限制与下一步，消除过期状态叠加。
- 新增[文档索引](../../docs/README.md)，区分当前使用、历史方案和入口快照。
- [结果索引](../README.md)与[检查点清单](../../checkpoints/README.md)更新到实际文件状态。
- 整理前入口保存于[docs/archive/](../../docs/archive/)，教学复盘与各阶段原始报告保留。
- A2说明将旧环境验收表明确标注为历史快照，优化日志注明旧权重删除状态。

## 验证

验证结果见[verification.json](verification.json)。只执行输入/资产预检，没有重新训练或更改成功模型。
