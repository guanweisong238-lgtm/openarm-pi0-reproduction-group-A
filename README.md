# OpenArm + MuJoCo + π0 / OpenPI 复现项目（A组）

更新：2026-10-25。当前处于仿真复现阶段，已训练出能在部分起点自主拉取抽屉的π0骨干一步确定性改造版。

## 当前成果与边界

标准起点ID12多次成功，柜体Y偏移5毫米的validation ID2也通过；human warm-start ID6未通过保持门槛。成功标准是真实双指接触、接触拉动、开度>=80毫米保持1秒。原多步flow版本、关闭往返与广泛泛化尚未完成，不能将当前改造结果称为原论文完整复现。

当前模型：`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000`。

## 展示模型

在项目根目录执行，使用本机GPU与原生MuJoCo窗口：

```bash
.venv/bin/python scripts/a2_pi0/live_drawer_success.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000 \
  --task open --seconds 24 --episode-ids 12 \
  --seed 42 --noise-mode fixed --action-chunk 4 --gripper-mode continuous
```

空格暂停、R重新运行、Q退出。每4个控制步重新观测，推理时仿真暂停；仿真耗时不能当成真机实时性能。固定噪声42属于训练条件，不能任意换种子或解码配置。

## 阅读入口

| 文档 | 用途 |
| --- | --- |
| [当前项目交接](docs/project-handoff.md) | 模型、数据、环境与下一步 |
| [文档索引](docs/README.md) | 当前使用文档与历史方案分类 |
| [教学式优化复盘](optimization_log.md) | 从起步偏离到成功的排查过程 |
| [最终模型报告](results/drawer_pull_success_20261025/README.md) | 训练协议、视频、真实物理证据与失败边界 |
| [检查点清单](checkpoints/README.md) | 清理后保留的4份完整权重 |
| [实验结果索引](results/README.md) | 各阶段结果与清理审计 |
| [数据说明](data/README.md) | 数据来源与管理 |
| [A2环境与脚本](scripts/a2_pi0/README.md) | 环境安装和历史基础链路 |

## 文件与管理

`scripts/`保存仿真、训练与评估代码，`configs/`和`assets/`保存配置及归一化资产；`data/`保存原始和处理后数据，`weights/`为底座，`checkpoints/`为实验模型，`logs/`和`results/`保存实验记录与证据。

原13条成功数据按10 train/3 validation隔离，最终1200个反馈监督查询相关于train ID12，不代表1200条独立成功轨迹。原始数据、统计、失败案例和运行时源码快照保留。

2026-10-25整理删除11份被替代权重及可重建缓存，释放约57.26GiB，详见[清理报告](results/project_cleanup_20261025_103927/README.md)。历史文档中的部分模型路径已不可加载，需重训。

大型数据、环境、缓存和权重不提交Git；模型、数据版本及实验结果按协议和实际证据记录。当前工作区含尚未提交的开发改动，本次整理没有提交或回滚代码。

项目初始成员分工与周计划保存在[原计划快照](docs/archive/project-plan-before-cleanup-20261025.md)。

## 上游

- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [OpenArm控制参考项目](https://github.com/Manas-arumalla/openarm-control)
