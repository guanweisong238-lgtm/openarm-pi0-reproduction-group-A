# 项目交接：当前有效状态

更新：2026-10-25。本文件只记录当前状态，历史过程见[优化复盘](../optimization_log.md)与[清理前交接快照](archive/project-handoff-before-cleanup-20261025.md)。历史文件中的“当前”、待执行计划及权重路径不作为现状。

## 已完成与限制

当前成功模型为`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000`，是π0骨干+LoRA+三IO投影的一步确定性改造版。原多步flow版本尚未通过。

- 标准起点ID12多次完成真实双指拉取，最近展示开度101.56毫米、仿真时间17.07秒，中途停滞后自主恢复。
- validation ID2柜体Y+5毫米通过；validation ID6 human warm-start未通过保持门槛。
- 只验证拉开保持，未验证关闭/完整往返，不宣称广泛泛化或稳定成功率。
- 原13条成功示教10 train/3 validation，原数据与统计未覆盖。最终监督查询1200个，均相关于train ID12；1080训练/120同父场景扰动留出，不是1200条成功轨迹。
- 模型输入为当前三路RGB、实际位置16维与速度16维；左状态屏蔽。右7残差、夹爪绝对目标，左臂保持初始目标。
- 运行中没有专家动作/参考轨迹查询、中途状态注入或抓取weld。推理期间仿真暂停，仿真秒数不是硬件实时性能。
- 评分要求双指各>.01N连续5步、接触拉动、开度>=80毫米保持30控制步；已记录成功的独立动作重放位置、速度、接触力最大差0。

完整证据：[最终报告](../results/drawer_pull_success_20261025/README.md)、[现场展示评分](../results/drawer_pull_success_20261025/live_showcase_095417_score.json)。

## 直接展示

在项目根目录运行原生MuJoCo窗口：

```bash
.venv/bin/python scripts/a2_pi0/live_drawer_success.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000 \
  --task open --seconds 24 --episode-ids 12 \
  --seed 42 --noise-mode fixed --action-chunk 4 --gripper-mode continuous
```

空格暂停、R重新运行、Q退出；完成后保留窗口，自动退出加`--auto-next`。固定code42/H4/一步采样是训练条件，不要任意更换噪声种子、重采样或夹爪解码。slew_binary对照失败，未采用。

录像回放：

```bash
/usr/bin/python3 scripts/a2_pi0/show_closed_loop_video.py \
  --video logs/a2_pi0/success_live_20261025-095417-291279/trial_01_episode_12/closed_loop.mp4
```

任务完成时立即释放VideoWriter写入MP4索引，避免窗口保留时视频无法播放；本次视频128帧已完整解码验证。

## 训练与关键资产

训练入口`scripts/a2_pi0/train_drawer_feedback_endpoint.py`，默认从保留的015/2000加载。2000步、峰值lr1e-5、warmup100、cosine至1e-6，更新50,087,968参数，无图像增强；固定noise42/t1、right8端点MSE、窗口4。

| 内容 | 路径 |
| --- | --- |
| 原13条成功数据清单 | `data/processed/openarm_drawer_success_v003/manifest.json` |
| 最终反馈监督查询 | `data/drawer_feedback_queries/v001/` |
| 原始人工/试采/恢复数据 | `data/drawer_human/`、`data/drawer_trial/`、`data/drawer_recovery/` |
| 速度统计 | `assets/pi0_openarm_drawer_velocity_v001/openarm_drawer_success_v003_right7_delta_velocity/` |
| 原始预处理缓存 | `.cache/drawer_success_inputs/45865b20503a36b52717/` |
| 速度缓存 | `.cache/drawer_velocity_v001/6a20f3a46f2c84bbfacab/` |
| 最终训练日志、曲线、源码快照 | `logs/a2_pi0/drawer_feedback_20261025-031627-720249/` |
| 当前模型、源模型与基线 | [检查点清单](../checkpoints/README.md) |

缓存身份包含部分源码，修改`drawer_success_training_config.py`、`openarm_policy.py`、`drawer_success_cache.py`可能触发约4.6GiB缓存重建。速度资产身份还依赖`prepare_drawer_velocity.py`与`DrawerSimulation`源码，修改需显式版本化，不能覆盖旧统计。

训练时协议`input_protocol.json`保留原快照；后验验证见检查点的`validation_report.json`和`deployment_config.json`，不要重写历史训练状态。

## 环境

RTX5060Ti 16GB，主训练/推理使用`.venv/bin/python`。系统`/usr/bin/python3`有Matplotlib/Tk，用于原生曲线及回放；项目venv有JAX/MuJoCo，不要混用依赖。GPU与桌面需要宿主访问；工具沙箱不可用不等于主机不可用。

## 下一步

先补train warm-start的恢复、夹持和保持数据，冻结配置后再测试隔离的validation起点；再扩大柜体、把手和初始姿态范围。如目标是原论文复现，需单独解决原多步flow，不能将当前改造结果替代。

小型MLP反馈基线也有成功记录，但它不是π0结果。所有失败日志和评分保留，不删除失败证据。

## 2026-10-25清理结果

保留4份完整检查点：016成功模型、015重训源、011局部起步模型、原13条训练基线。删除11份被替代的历史权重与可重建恢复预处理缓存，释放约57.26GiB。原始数据、底座权重、环境、源码、日志与评估均保留。归档配置不含权重备份，已删除实验需重训。

清单：[本次清理报告](../results/project_cleanup_20261025_103927/README.md)。其他文档按[文档索引](README.md)阅读。
