# OpenArm + π0：任务 C 真机双臂操作

任务依次是左臂拉开抽屉，右臂抓取方块、放入并释放，最后左臂推合抽屉。策略目标为真机自主完成。

本轮 π0 仅使用 `data/real_20261128_202032/`，从底座完成 1,000 步 LoRA 重训。新采集任务为左臂拉开抽屉，右臂将方块放入并释放。数据划分为 18 条训练、4 条验证。 [数据与审核记录](data/README.md#2026-11-28-重训批次) · [重训结果](results/real_20261128_202032_pi0_lora_1000_v001/summary.json)

已完成 2,000 步续训（累计 3,000 步），最终检查点的固定验证损失为 0.0645。 [续训结果与验证](results/README.md#2026-11-28-关键阶段续训)

**11 月 28 日批次的第 0 条已在真机完成首帧摆位。** 双臂到位后，独立读回确认 16 轴失能。短段与整条关节轨迹均已离线生成，真机播放待验证。 [真机结果与待办](docs/project-handoff.md#2026-11-28-第-0-条示范首帧真机摆位) · [示范回放说明](results/real_20261128_demo0_replay_preflight/README.md)

π0 工作台入口为 `scripts/a2_pi0/deploy/run_pi0.sh`，可加 `--check` 做离线检查。默认配置已切换至新批次续训第 2,000 步模型和第 0 条示范起点，π0 真机闭环待验收。 [部署说明](scripts/a2_pi0/deploy/README.md) · [电脑端核验](docs/on-site-verification.md#2026-11-28-电脑端连接检查)

## 项目入口

- [任务定义与当前状态](docs/task-real-cube-to-drawer.md)
- [数据说明](data/README.md)、[视觉缺陷](data/real_data_20261121_193807_数据缺陷与审核.md)、[动作语义](data/real_data_20261121_193807_动作语义核验.md)
- [训练样本协议](docs/real-training-sample-contract.md)与[脚本使用说明](scripts/a2_pi0/README.md)
- [真机部署接线与联调](scripts/a2_pi0/deploy/README.md)包含新工作台入口、运行环境及历史真机联调记录。
- [项目交接](docs/project-handoff.md)、[检查点](checkpoints/README.md)、[结果索引](results/README.md)
- [OpenArm V2 官方资料本地归档](references/openarm-v2/index.html)
- [任务 C 优化记录](optimization_log_taskC.md)
- [历史优化记录](optimization_log.md)：按用户要求保留原文；其中指向已清理任务 A、B 资产的路径可能失效。

原始真机数据保持只读。任务 C 的模型底座、四相机代码、训练数据、检查点、验证记录和用于真实轨迹示意回放的 MuJoCo 必要场景资源均保留。
