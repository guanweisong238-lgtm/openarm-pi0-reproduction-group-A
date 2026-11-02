# 抽屉拉取批量评估结果

运行目录：`/home/song03/scientific_research/openarm-pi0-reproduction-group-A/logs/a2_pi0/repair_screen_020_2000/full_33`
检查点：`/home/song03/scientific_research/openarm-pi0-reproduction-group-A/checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000`
渲染后端：`glfw`（跨批次比较必须一致）
固定配置：{'seconds': 24.0, 'task': 'open', 'seed': 42, 'action_chunk': 4, 'gripper_mode': 'continuous', 'repeat': 1, 'step_limit': 720, 'weight_precision': 'checkpoint', 'gl_backend': 'glfw'}

## 分组结果

| 分组 | 次数 | 成功 | 成功率 | 峰值开度中位数 | 首次成功步中位数 | 成功时仿真耗时中位数 | 推理 p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 10 | 10 | 100.0% | 102.0 mm | 136 | 4.53 s | 138.4 / 139.4 ms |
| validation | 3 | 3 | 100.0% | 101.9 mm | 142 | 4.73 s | 138.4 / 139.7 ms |
| eval_only | 20 | 20 | 100.0% | 101.3 mm | 145 | 4.83 s | 138.4 / 139.1 ms |

## 失败模式分布

- **train**：无失败
- **validation**：无失败
- **eval_only**：无失败

## 逐条明细

| 起点 | 分组 | 判定 | 峰值开度 | 首次成功步 | 仿真耗时 | 失败模式 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | train | 成功 | 101.6 mm | 134 | 4.47 s | — |
| 1 | train | 成功 | 101.7 mm | 138 | 4.60 s | — |
| 2 | validation | 成功 | 101.4 mm | 145 | 4.83 s | — |
| 3 | validation | 成功 | 101.9 mm | 142 | 4.73 s | — |
| 4 | train | 成功 | 102.0 mm | 141 | 4.70 s | — |
| 5 | train | 成功 | 102.0 mm | 141 | 4.70 s | — |
| 6 | validation | 成功 | 102.1 mm | 102 | 3.40 s | — |
| 7 | train | 成功 | 102.1 mm | 111 | 3.70 s | — |
| 8 | train | 成功 | 101.5 mm | 100 | 3.33 s | — |
| 9 | train | 成功 | 102.0 mm | 98 | 3.27 s | — |
| 10 | train | 成功 | 101.9 mm | 142 | 4.73 s | — |
| 11 | train | 成功 | 100.3 mm | 133 | 4.43 s | — |
| 12 | train | 成功 | 102.0 mm | 141 | 4.70 s | — |
| 100 | eval_only | 成功 | 101.0 mm | 181 | 6.03 s | — |
| 101 | eval_only | 成功 | 101.2 mm | 158 | 5.27 s | — |
| 102 | eval_only | 成功 | 96.2 mm | 147 | 4.90 s | — |
| 103 | eval_only | 成功 | 101.6 mm | 136 | 4.53 s | — |
| 104 | eval_only | 成功 | 101.5 mm | 146 | 4.87 s | — |
| 105 | eval_only | 成功 | 101.2 mm | 151 | 5.03 s | — |
| 106 | eval_only | 成功 | 101.0 mm | 141 | 4.70 s | — |
| 107 | eval_only | 成功 | 101.1 mm | 135 | 4.50 s | — |
| 108 | eval_only | 成功 | 101.7 mm | 147 | 4.90 s | — |
| 109 | eval_only | 成功 | 100.4 mm | 151 | 5.03 s | — |
| 110 | eval_only | 成功 | 101.2 mm | 144 | 4.80 s | — |
| 111 | eval_only | 成功 | 102.1 mm | 144 | 4.80 s | — |
| 112 | eval_only | 成功 | 102.5 mm | 467 | 15.57 s | — |
| 113 | eval_only | 成功 | 101.9 mm | 139 | 4.63 s | — |
| 114 | eval_only | 成功 | 101.7 mm | 144 | 4.80 s | — |
| 115 | eval_only | 成功 | 102.1 mm | 142 | 4.73 s | — |
| 116 | eval_only | 成功 | 102.0 mm | 137 | 4.57 s | — |
| 117 | eval_only | 成功 | 101.5 mm | 149 | 4.97 s | — |
| 118 | eval_only | 成功 | 101.0 mm | 130 | 4.33 s | — |
| 119 | eval_only | 成功 | 101.2 mm | 149 | 4.97 s | — |

## 口径声明

- 仿真时间在推理期暂停，不代表硬件实时性能
- 确定性策略：同起点重复不计独立样本，只作稳定性参考
- 本表不改判据；判据见 docs/drawer-eval-protocol.md
