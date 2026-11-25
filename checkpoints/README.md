# 任务 C 检查点

当前部署联调使用累计 6500 步的 `real_camera_adjusted_lora_continue5000_20261122_v002/step_3000`。目录名记录本轮续训的 3000 步，前序累计 3500 步。加载时使用对应训练配置、归一化统计和 `pi0_base` 底座。 [训练配置](../results/real_camera_adjusted_lora_continue5000_20261122_v002/config.json) · [最新部署交接](../docs/project-handoff.md#2026-11-25-示教摆位与-6500-步模型联调)

`real_pi0_four_camera_lora_001/step_0500/` 和 `step_1000/` 是探索性训练得到的 LoRA 适配器；需要 `weights/openpi/openpi-assets/checkpoints/pi0_base/params`、对应四路模型代码及检查点内归一化统计。它们不是已验证可部署的完整机器人策略，也不包含可用于续训的优化器状态。

`real_pi0_four_camera_smoke_*` 和 `real_pi0_four_camera_cleanup_check_001` 是任务 C 链路检查产物，不代表完成任务的模型。训练与重载记录见 [训练报告](../results/real_pi0_four_camera_lora_001/README.md)。
