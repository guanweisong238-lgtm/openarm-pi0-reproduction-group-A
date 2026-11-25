# 任务 C 结果

## 2026-11-25 离线验证

- 从顶部录像定位方块和右指尖，并在第 34 条回放中区分左臂目标误差与推理延迟。[合爪几何、起步和整段复测](real_taskC_geometry_and_startup_20261125/README.md)
- 对 10 条录像和 3 个随机种子测试夹爪限幅、腕部画面和顶部全局画面，并用原保护逻辑复查停发。[夹爪复测与顶部画面对照](real_taskC_step6500_gripper_followup_20261125/README.md)
- 核对 49 条原始示教的夹爪指令和实测角度，用两个随机起点完整回放 10 条验证录像，比较左右夹爪越界和右夹爪合拢时机。[逐条结果](real_taskC_step6500_gripper_diagnosis_20261125/summary.json) · [三条开场误合的画面对照](real_taskC_step6500_gripper_diagnosis_20261125/early-closure.jpg)
- [连续录像，每 20 帧重算一次](real_taskC_step6500_stride20_offline_20261125/summary.json)：第 5、38 条示教的合爪时机、目标越界和新旧动作跳变。
- [合爪输入对照与同一时刻的计划差异](real_taskC_step6500_diagnosis_20261125/diagnosis.json)：分别替换画面或关节读数，并把新旧计划对齐到录像中的同一帧。
- [让新计划参考旧计划](real_taskC_step6500_guidance_comparison_20261125/summary.json)：手臂与夹爪分组比较；含[第 5 条](real_taskC_step6500_guidance_comparison_20261125/episode_5/grasp_comparison_half_speed.mp4)和[第 38 条](real_taskC_step6500_guidance_comparison_20261125/episode_38/grasp_comparison_half_speed.mp4)四路半速短片。
- 只给手臂做 4、8、12 帧过渡，按整段过渡统计速度超限与示教偏差。目录内有第 5、38 条录像的四路合爪截图，目视对照已记入交接。[手臂过渡结果](real_taskC_step6500_arm_blend_20261125/summary.json) · [合爪画面对照](../docs/project-handoff.md#2026-11-25-连续录像离线验证)

以上均基于录制观测进行离线验证，真机任务成功率待验收。[最新交接](../docs/project-handoff.md#2026-11-25-两侧夹爪离线排查)

## 历史结果

- [14/20 窗口与七步前缀离线对照](real_prefix_schedule_compare_20261122/README.md)

- [1,000 步探索性训练与验证](real_pi0_four_camera_lora_001/README.md)
- [验证轨迹的 50 步动作预测评估](real_action_eval_20261121/README.md)
- [四路模型合成样本测试](real_four_camera_smoke_20261121/README.md)
- [真实训练样本链路测试](real_four_camera_real_batch_20261121/README.md)
- `real_data_review_20261121/`：逐条人工审核表；用户后来指示本轮默认全部通过，见数据目录的 `review_decision.json`。
- `real_task_c_cleanup_20261121/cleanup.json`：任务 A、B 工作区清理清单及保留文件哈希。
- `real_pi0_four_camera_smoke_*`、`real_pi0_four_camera_cleanup_check_001`：任务 C 的短程链路验证；`smoke_001` 的检查点保存曾失败。
