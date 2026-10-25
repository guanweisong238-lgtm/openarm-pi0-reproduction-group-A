# 抽屉采集数据统一审核

扫描时间：2026-10-24T15:58:26.405744+08:00

口径：整个把手组件双侧接触、打开80mm保持1秒、终末关闭5mm内保持1秒、无抓取绑定、独立动作回放。

RGB检查覆盖尺寸、dtype和全黑帧；未对全部图像逐帧重渲染做像素比对。状态为扫描快照，后台导出可能继续更新。

| 分类 | 数量 |
|---|---:|
| task_failed_or_different_task | 29 |
| numeric_success_only | 9 |
| success_rgb_candidate | 10 |
| selfcheck_excluded | 2 |
| numeric_success_rgb_pending | 3 |
| assisted_excluded | 3 |

## 人工录制

| 记录 | 最大开度mm | 打开保持秒 | 最终开度mm | 双侧接触帧 | 分类 |
|---|---:|---:|---:|---:|---|
| session_20261024_152741/episode_001 | 100.29 | 1.53 | 0.14 | 0 | task_failed_or_different_task |
| session_20261024_152741/episode_002 | 98.95 | 1.23 | 0.53 | 426 | success_rgb_candidate |
| session_20261024_154326/episode_001 | 94.49 | 1.17 | 0.00 | 610 | success_rgb_candidate |
| session_20261024_154326/episode_002 | 93.82 | 0.87 | 0.00 | 403 | task_failed_or_different_task |
| session_20261024_154326/episode_003 | 91.64 | 1.53 | 0.00 | 483 | success_rgb_candidate |
| session_20261024_154326/episode_004 | 96.54 | 1.47 | 0.00 | 497 | numeric_success_rgb_pending |
| session_20261024_154326/episode_005 | 96.42 | 0.97 | 0.00 | 394 | task_failed_or_different_task |
| session_20261024_154326/episode_006 | 91.66 | 1.40 | 0.26 | 366 | numeric_success_rgb_pending |
| session_20261024_154326/episode_007 | 92.50 | 0.87 | 0.00 | 364 | task_failed_or_different_task |
| session_20261024_154326/episode_008 | 100.51 | 2.20 | 0.00 | 365 | numeric_success_rgb_pending |

## 数据使用边界

- success_rgb_candidate仅表示可进入训练前整理，不自动加入现有训练/验证集。
- 人工warm-start和脚本ready-start条件不同，需明确任务初始化及采样策略。
- 数值/RGB副本和动作重复组不能重复计为独立示教；失败、辅助绑定、自检排除。
- 本检查未修改原始数据、旧归一化统计或已有训练划分。

## 动作重复组

- data/drawer_expansion/diverse_numeric/approach_side_neg；data/drawer_expansion/diverse_rgb/approach_side_neg
- data/drawer_expansion/diverse_numeric/approach_side_pos；data/drawer_expansion/diverse_rgb/approach_side_pos
- data/drawer_expansion/diverse_numeric/cabinet_y_pos005；data/drawer_expansion/diverse_rgb/cabinet_y_pos005
- data/drawer_expansion/diverse_rgb/initial_seed108；data/drawer_expansion/pilot_numeric/initial_seed108
- data/drawer_expansion/pilot_numeric/pitch12；data/drawer_expansion/pilot_rgb/pitch12
- data/drawer_expansion/pilot_numeric/pitch18；data/drawer_expansion/pilot_rgb/pitch18
- data/drawer_human/selfcheck_20261024_152649_052017；data/drawer_human/selfcheck_20261024_153547_117525
- data/drawer_trial/assisted_numeric_001；data/drawer_trial/assisted_rgb_001；data/drawer_trial/assisted_rgb_side_001
- data/drawer_trial/contact_force_stop_001；data/drawer_trial/contact_force_stop_rgb_001
- data/drawer_trial/contact_reachable_002；data/drawer_trial/contact_reachable_rgb_001
- data/drawer_trial/contact_roundtrip_001；data/drawer_trial/contact_roundtrip_rgb_001
