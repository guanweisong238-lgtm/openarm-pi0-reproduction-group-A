# OpenArm + MuJoCo + π0 / OpenPI 复现项目（A组）

更新：2026-11-02。当前π0骨干一步确定性改造版已在固定33个仿真起点全部完成拉开并保持，独立动作重放33/33通过。

## 当前成果与边界

固定起点集：train 10/10、validation 3/3、eval_only 20/20。成功标准是真实双指接触、接触拉动、开度>=80毫米保持1秒，判据未放宽。原多步flow版本、关闭往返和真机未完成；固定基准的33/33不代表广泛泛化或原论文完整复现。

当前模型：`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000`。

冻结该模型后的全新100起点评估已完成：原范围随机组合66/70（94.3%），扩大范围18/30（60.0%），总计84/100；独立重放100/100证据有效。全部起点在运行前锁定且不参与训练，权重前后指纹一致。扩大范围后的抓取稳定性仍有不足，见[新测试报告与分布图](results/drawer_generalization_020_20261102/README.md)。

后续021/022两轮训练未达到升级标准，继续使用020。保留的同一020在另一批训练前锁定的新100起点（400—499）上得到85/100：near66/70、expanded19/30，独立重放100/100有效，位置/速度/接触力最大误差0。两批分数不同不代表训练提升，见[本轮训练比较与最终验收](results/drawer_grasp_improvement_20261102/README.md)。此前清理释放9.70GiB；晚间再清理8份失败候选权重，释放约41.43GiB，保留7份完整模型、全部数据和评估证据，见[清理报告](results/project_cleanup_failed_candidates_20261102/README.md)。

## 展示模型

在项目根目录执行，使用本机GPU与原生MuJoCo窗口：

```bash
.venv/bin/python scripts/a2_pi0/live_drawer_success.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000 \
  --task open --seconds 24 --episode-ids 12 \
  --seed 42 --noise-mode fixed --action-chunk 4 --gripper-mode continuous \
  --weight-precision checkpoint
```

空格暂停、R重新运行、Q退出。每4个控制步重新观测，推理时仿真暂停；仿真耗时不能当成真机实时性能。固定噪声42属于训练条件，不能任意换种子或解码配置。

## 阅读入口

| 文档 | 用途 |
| --- | --- |
| [当前项目交接](docs/project-handoff.md) | 模型、数据、环境与下一步 |
| [文档索引](docs/README.md) | 当前使用文档与历史方案分类 |
| [教学式优化复盘](optimization_log.md) | 从起步偏离到成功的排查过程 |
| [当前模型33/33验收报告](results/drawer_eval_020_20261102/README.md) | 训练协议、视频、真实物理证据与失败边界 |
| [检查点清单](checkpoints/README.md) | 当前推荐模型、修复候选与历史基线 |
| [实验结果索引](results/README.md) | 各阶段结果与清理审计 |
| [数据说明](data/README.md) | 数据来源与管理 |
| [A2环境与脚本](scripts/a2_pi0/README.md) | 环境安装和历史基础链路 |

## 文件与管理

`scripts/`保存仿真、训练与评估代码，`configs/`和`assets/`保存配置及归一化资产；`data/`保存原始和处理后数据，`weights/`为底座，`checkpoints/`为实验模型，`logs/`和`results/`保存实验记录与证据。

原13条成功数据保持10 train/3 validation隔离。020课程使用4767个相关查询：4192专家反馈、575条旧策略成功训练轨迹输出标签，含独立随机训练场景；验证/评估状态未进入训练或专家标签。当前推荐的是课程中保存的2000步权重。原始数据、统计、失败案例和源码快照保留。

2026-10-25整理删除11份被替代权重及可重建缓存，释放约57.26GiB，详见[清理报告](results/project_cleanup_20261025_103927/README.md)。历史文档中的部分模型路径已不可加载，需重训。

大型数据、环境、缓存和权重不提交Git；模型、数据版本及实验结果按协议和实际证据记录。当前工作区含尚未提交的开发改动，本次整理没有提交或回滚代码。

项目初始成员分工与周计划保存在[原计划快照](docs/archive/project-plan-before-cleanup-20261025.md)。

## 上游

- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [OpenArm控制参考项目](https://github.com/Manas-arumalla/openarm-control)
