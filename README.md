# OpenArm + π0：任务 C 真机双臂操作

任务依次是左臂拉开抽屉，右臂抓取方块、放入并释放，最后左臂推合抽屉。策略目标为真机自主完成。

相机调整和更换方块后的批次包含 49 条实采轨迹，已生成 30 Hz、50 步动作窗口，并用于四相机绝对关节目标模型的训练。从 episode 35 起，示教改为左夹爪全程夹住把手，待右臂放入物块后直接推合。 [数据与审核记录](data/README.md#2026-11-22-新采集批次)

当前联调使用累计 6500 步的模型。**真机目前卡在示教起点：两侧 J4 保持时仍差约 5°，超过 2.86° 的到位判据，完整示教回放尚未开始。** 无补偿保持已复测；重力补偿只做了离线验证，尚未下发。模型换段跳变、夹爪早合仍未解决；夹爪限幅只在离线消除了越界。**模型目标尚未下发真机。** [最新交接](docs/project-handoff.md#2026-11-25-j4-保持偏差与重力补偿) · [现场放行条件](docs/on-site-verification.md#j4-补偿试验放行)

## 项目入口

- [任务定义与当前状态](docs/task-real-cube-to-drawer.md)
- [数据说明](data/README.md)、[视觉缺陷](data/real_data_20261121_193807_数据缺陷与审核.md)、[动作语义](data/real_data_20261121_193807_动作语义核验.md)
- [训练样本协议](docs/real-training-sample-contract.md)与[脚本使用说明](scripts/a2_pi0/README.md)
- [真机部署接线与联调](scripts/a2_pi0/deploy/README.md)包含相机端口、双臂摆位、右 J7 双编码器实测和模拟复现命令。
- [项目交接](docs/project-handoff.md)、[检查点](checkpoints/README.md)、[结果索引](results/README.md)
- [OpenArm V2 官方资料本地归档](references/openarm-v2/index.html)
- [任务 C 优化记录](optimization_log_taskC.md)
- [历史优化记录](optimization_log.md)：按用户要求保留原文；其中指向已清理任务 A、B 资产的路径可能失效。

原始真机数据保持只读。任务 C 的模型底座、四相机代码、训练数据、检查点、验证记录和用于真实轨迹示意回放的 MuJoCo 必要场景资源均保留。
