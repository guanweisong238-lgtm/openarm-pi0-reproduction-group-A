# OpenArm + π0：任务 C 真机双臂操作

更新：2026-11-22。目标是左臂拉开抽屉，右臂抓取方块并放入、释放，最后左臂推合抽屉；策略目标为真机自主完成。

新采集批次调整了相机位置并更换了方块，共有 49 条轨迹，四路图像和双臂记录已通过结构与时间检查。用户复核时发现从 episode 35 起改为左夹爪全程夹住把手，待右臂放入物块后直接推合，其余未发现问题。新批次待生成训练样本。[新数据与审核命令](data/README.md#2026-11-22-新采集批次)

已有检查点使用 11 月 21 日的实采演示。模型使用四路相机、30 Hz 对齐与 50 步动作窗口，已试验绝对角度和角度变化量两种标签。最新变化量模型在有演示前缀的短窗中误差有所下降，连续接续时仍会漂移。真机闭环效果待验证。[任务状态](docs/task-real-cube-to-drawer.md)

## 项目入口

- [任务定义与当前状态](docs/task-real-cube-to-drawer.md)
- [数据说明](data/README.md)、[视觉缺陷](data/real_data_20261121_193807_数据缺陷与审核.md)、[动作语义](data/real_data_20261121_193807_动作语义核验.md)
- [训练样本协议](docs/real-training-sample-contract.md)与[脚本使用说明](scripts/a2_pi0/README.md)
- [真机部署接线与联调](scripts/a2_pi0/deploy/README.md)：策略输出如何接到机械臂（未上机验证）
- [项目交接](docs/project-handoff.md)、[检查点](checkpoints/README.md)、[结果索引](results/README.md)
- [任务 C 优化记录](optimization_log_taskC.md)
- [历史优化记录](optimization_log.md)：按用户要求保留原文；其中指向已清理任务 A、B 资产的路径可能失效。

原始真机数据保持只读。任务 C 的模型底座、四相机代码、训练数据、检查点、验证记录和用于真实轨迹示意回放的 MuJoCo 必要场景资源均保留。
