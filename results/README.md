# 任务 C 结果

## 2026-11-28 历史观测对照

从同一个 `step_2000` 出发，三组按同样的顺序各训练 200 步。任务指令、当前四路画面和当前关节角保持一致，只改变是否提供过去约 100 ms 的信息。用未参与训练的第 13、19、20、24 条录像验证。 [完整数据](real_20261128_temporal_inputs200_v001/comparison.json) · [实验与绘图脚本](../scripts/a2_pi0/eval_real_temporal_inputs.py)

| 额外输入 | 约 0.4 秒后的手臂平均角度误差 | 相邻目标变化超过 10° | 第 24 条首次触发保护 |
| --- | ---: | ---: | ---: |
| 不加历史 | 3.045° | 14 次 | 8.2 秒 |
| 过去四路画面 | 3.150° | 18 次 | 11.8 秒 |
| 过去两臂关节位移 | 3.067° | 14 次 | 9.0 秒 |

角度误差比较的是预测第 12–17 帧和对应的示范目标。大跳次数取固定 10 Hz 回放中三组都通过保护的相邻时刻，首次拒绝之后的回放仅用于离线诊断。旧画面组在第 24 条更晚触发保护，但总体没有让旧预测更准或切换更平滑。旧画面特征经过 4×4 空间压缩，且反向、减速样本较少，不据此否定视觉历史。三组均未部署，默认模型、`50/20` 节奏和保护未改。

折线图中，黑线是示范关节**目标**，不是实测反馈；彩线是三组模型对同一时刻的预测。全关节图覆盖两臂各 7 个关节与夹爪，彩线断开处未绘制该时段的预测切片。

- [第 13 条左 J5 放大图](real_20261128_temporal_inputs200_v001/episode_13_left_j5_detail.png)
- 全关节图：[第 13 条](real_20261128_temporal_inputs200_v001/episode_13_joint_angles.png) · [第 19 条](real_20261128_temporal_inputs200_v001/episode_19_joint_angles.png) · [第 20 条](real_20261128_temporal_inputs200_v001/episode_20_joint_angles.png) · [第 24 条](real_20261128_temporal_inputs200_v001/episode_24_joint_angles.png)

## 2026-11-28 关键阶段续训

已完成 2,000 步续训，最终检查点的固定验证损失为 0.0645。 从首轮 1,000 步 LoRA 权重热启动，使用新初始化的优化器。续训只使用同一新批次，排除首轮用过的 1,000 个窗口。2,000 个新窗口中，拉抽屉、接近方块、抓取和持物共占 70.1%。 [训练配置](real_20261128_202032_pi0_lora_phase_continue2000_v001/config.json) · [逐次固定验证](real_20261128_202032_pi0_lora_phase_continue2000_v001/validation.jsonl) · [独立重载](real_20261128_202032_pi0_lora_phase_continue2000_v001/independent_restore_2000.json)

| 续训步数 | 累计步数 | 固定验证损失 |
| ---: | ---: | ---: |
| 0 | 1000 | 0.0896 |
| 500 | 1500 | 0.0723 |
| 1000 | 2000 | 0.0732 |
| 1500 | 2500 | 0.0653 |
| 2000 | 3000 | 0.0645 |

已有基于录制反馈的[历史观测离线对照](#2026-11-28-历史观测对照)；真机闭环验收仍待完成。

按第 13 条示范调整过的仿真场景已完成示范指令回放，抽屉最大打开 95.9 mm，方块最大抬升 273.3 mm，末半秒落在抽屉底板上。 [示范回放](real_20261128_202032_pi0_mujoco_reference_fit_demo/mujoco_rollout.mp4) · [测量记录](real_20261128_202032_pi0_mujoco_reference_fit_demo/summary.json)

## 2026-11-28 新批次 π0 重训

仅使用 `real_20261128_202032`，从预训练 π0 底座完成 1,000 步 LoRA。18 条训练、4 条验证，8 个固定验证窗口的平均流匹配损失从 0.3866 降至 0.0896。 [训练汇总](real_20261128_202032_pi0_lora_1000_v001/summary.json) · [训练配置](real_20261128_202032_pi0_lora_1000_v001/config.json) · [候选清单](real_20261128_202032_pi0_lora_1000_v001/manifest.json)

最终权重独立重载的逐窗口损失最大差值为 0.000561，在 0.001 诊断容差内。对首个验证窗口以 5 个种子推理，均返回有限的 50×16 动作块，与示范的动作 MAE 为 0.081–0.095 rad。新模型尚未部署到真机。 [独立重载](real_20261128_202032_pi0_lora_1000_v001/independent_restore_1000.json)

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
