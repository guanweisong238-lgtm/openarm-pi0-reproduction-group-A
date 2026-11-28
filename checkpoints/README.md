# 任务 C 检查点

## 关键阶段续训

已完成 2,000 步续训（累计 3,000 步），固定验证损失为 0.0645。工作台默认权重位于 `real_20261128_202032_pi0_lora_phase_continue2000_v001/step_2000/`，使用同批次第 0 条示范起点，真机闭环待验收。 [训练验证](../results/README.md#2026-11-28-关键阶段续训)

新批次 `real_20261128_202032` 的首轮 π0 LoRA 检查点位于 `real_20261128_202032_pi0_lora_1000_v001/step_1000/`，并保留 `step_0500/`。加载需要本仓库的 `pi0_base`、训练代码及对应检查点内的归一化统计。 [训练配置](../results/real_20261128_202032_pi0_lora_1000_v001/config.json) · [训练汇总](../results/real_20261128_202032_pi0_lora_1000_v001/summary.json)

历史联调使用累计 6500 步的 `real_camera_adjusted_lora_continue5000_20261122_v002/step_3000`。目录名记录该轮续训的 3000 步，前序累计 3500 步。加载时使用对应训练配置、归一化统计和 `pi0_base` 底座。 [训练配置](../results/real_camera_adjusted_lora_continue5000_20261122_v002/config.json) · [历史联调](../docs/project-handoff.md#2026-11-25-示教摆位与-6500-步模型联调)

`real_pi0_four_camera_lora_001/step_0500/` 和 `step_1000/` 是探索性训练得到的 LoRA 适配器；需要 `weights/openpi/openpi-assets/checkpoints/pi0_base/params`、对应四路模型代码及检查点内归一化统计。它们不是已验证可部署的完整机器人策略，也不包含可用于续训的优化器状态。

`real_pi0_four_camera_smoke_*` 和 `real_pi0_four_camera_cleanup_check_001` 是任务 C 链路检查产物，不代表完成任务的模型。训练与重载记录见 [训练报告](../results/real_pi0_four_camera_lora_001/README.md)。
