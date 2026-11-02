# 抽屉拉取评估协议（冻结版）

建立：2026-11-01。本文件是抽屉拉取任务**唯一有效的评估口径**。任何成功率、耗时、延迟数字，只有按本协议跑出来才可引用；不按本协议的运行只能写成探索性观察。

本协议不改动任何训练状态。检查点内的 `input_protocol.json` 是训练时快照，后验验收以下文与 `deployment_config.json` 为准。

## 1. 为什么要先冻结口径

模型的成功判据在 `scripts/a2_pi0/live_drawer_success.py` 的 `update_judge()` 里已经写死，但此前只存在于代码中。批量测试会出现大量边界情况（差 0.5 毫米到 80、保持 29 步），没有事先写死的规则时，容易在看到结果之后放宽标准。本协议的作用是：**看完结果不得修改判据**。

## 2. 固定资产

| 项目 | 值 |
| --- | --- |
| 检查点 | `checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000` |
| 协议快照 | 同名 `input_protocol.json`，sha256 = `917e311c4e4981428b639d6ef297de155d582dfa2c5cb3a8f9a6f884903499fa` |
| 示教清单 | `data/processed/openarm_drawer_success_v003/manifest.json`，sha256 = `428d1e96cc772018194b8d6ecc95e8e0ef52a2023794fee6245bb8bb7ef7afd2` |
| 归一化统计 | `assets/pi0_openarm_drawer_velocity_v001/openarm_drawer_success_v003_right7_delta_velocity`，sha256 = `fdc5f2fd3be53e6155c94a9709a9001bfa253e0049ce238bb9d8fe1a8dc01bb3` |
| 模型性质 | π0 骨干的一步确定性改造版（`inference_kind=fixed_noise_endpoint`），**不是**原多步 flow 复现 |
| 可训练参数 | LoRA + `state_proj`/`action_in_proj`/`action_out_proj`，共 50,087,968 |

`manifest` 与 `norm_stats` 的哈希由运行脚本在加载时自动比对（哈希不符直接抛异常）。执行方在开跑前只需核对上表 `input_protocol.json` 的 sha256。

**更换检查点必须新建一份协议文件**，不得在本文件上覆盖成功率历史。

## 3. 固定运行配置（训练条件，不可调）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `--task` | `open` | 只验收"拉开并保持"，不含关闭往返 |
| `--seconds` | `24`（720 控制步） | 超时上限，非目标时长 |
| `--noise-mode` | `fixed` | 确定性策略，强制项 |
| `--seed` | `42` | 即 `fixed_noise_seed`，属训练条件 |
| `--action-chunk` | `4` | 每 4 个控制步重观测一次 |
| `--gripper-mode` | `continuous` | 直接用模型连续目标 |
| 左臂 | 固定初始目标（`left_hold`） | 模型左 8 维输出被替换，不执行 |
| 控制频率 | 30 Hz | 物理步长 1 ms，按 33/34 子步调度，误差不累积 |
| 推理期间仿真 | 暂停 | 仿真时间与墙钟不是同一量 |
| 抓取绑定（weld） | 全程禁止 | 出现即抛异常，运行作废 |
| 渲染后端 | `MUJOCO_GL=glfw`（mesa 软件渲染） | 必须与证据生成器 `live_drawer_success.py` 一致。**换成 `egl` 会渲染出不同像素，确定性策略动作随之改变**：2026-11-01 实测导致 ID12 由成功翻为失败、ID6 由失败翻为成功 |

**噪声种子 42 不是可换的评估参数。** `live_drawer_success.py:63-65` 会校验协议里的 `fixed_noise_seed`，换 seed 直接 `parser.error` 退出。把 seed 改大跑多次再报"多次成功"的写法一律不成立。

## 4. 成功判据

一次 rollout 记为成功，必须以下三条**同时**成立（`update_judge()`，逐控制步累积状态）：

1. **夹住**：左右指片各自与把手组件（`drawer_handle` + `drawer_handle_stem1/2`）产生法向力 > 0.01 N，且连续保持 **5 个控制步**（`grasp_streak >= 5`）。
2. **真拉动**：双指都接触的前提下，抽屉开度比此前记录值增加 > 0.00005 m（0.05 毫米），产生一次 `pulling_contact` 事件。
3. **拉开并保持**：开度 >= 0.08 m，连续 **30 个控制步**（1 秒仿真时间）。

`--task roundtrip` 另需"关回并保持"：`abs(opening) <= 0.005` 连续 30 步。本协议当前只验收 `open`。

判据中的开度取 `-qpos[drawer_slide]`（向外为正）。接触计数与法向力由 `finger_handle_contacts(..., include_stems=True)` 计算，与训练期评分函数一致。

**峰值开度不构成成功。** 曾超过 80 mm 但未保持 30 步的，判失败（已知案例：validation ID6 峰值 87.78 mm、最终 76.94 mm）。

## 5. 证据有效性（独立重放复核）

`summary.json` 只有同时满足下列条件才产出 `evidence_valid=true`，也只有这时才可计为 `model_pull_success`：

| 检查 | 阈值 | 含义 |
| --- | --- | --- |
| `teacher_actions_executed` | 必须为 0 | 运行期没有任何专家动作替代 |
| `intermediate_states_injected` | 必须为 0 | 没有中途灌入示教状态 |
| `replay_max_qpos_error` | < 1e-9 | 从完整初态只重放实际下发动作，关节位置一致 |
| `replay_max_contact_force_error` | < 1e-7 | 接触力一致 |
| 接触计数 | 完全相等 | 逐帧 `array_equal` |
| `replay_max_qvel_error` | < 1e-9 | 速度一致（速度协议下必须有速度记录） |

复核由 `scripts/a2_pi0/assess_drawer_open.py` 完成，不接受其他评分方式替代。

## 6. 指标定义

一次 rollout 的评估单元为：

```
(episode_id, noise_seed=42, action_chunk=4, gripper_mode=continuous, task=open, seconds=24)
```

必须输出以下指标，缺项即视为该批次无效：

| 指标 | 定义 | 口径注意 |
| --- | --- | --- |
| 成功率 | 该分组内 `model_pull_success=true` 的占比 | 分子分母都要写出次数 |
| 首次成功步 | `first_success_step`（重放中的 1-based 控制步），未成功记 null | 反映快慢，不参与成败 |
| 峰值开度 | `max(opening) * 1000`（毫米） | 与 80 毫米门槛比较 |
| 仿真时间 | `steps / 30`（秒） | **推理期仿真暂停**，非硬件实时 |
| 推理延迟 | `inference_seconds` 的 p50 / p95（毫秒，墙钟） | 必须丢弃首次推理（JAX 编译） |
| 失败模式 | 见第 7 节分类 | 每条失败都要归类，不许只写"未通过" |

**仿真时间与墙钟延迟必须分两列记录**，任何把两者相加或互相解释的写法都不成立。

## 7. 失败模式分类

每条失败 rollout 归入且仅归入一类，按优先级自上而下取第一个命中的：

| 代码 | 判据 |
| --- | --- |
| `simulation_error` | 仿真异常、状态非有限、动作异常中断 |
| `no_contact` | 全程未出现任何指片与把手的接触 |
| `single_finger` | 有接触，但从未出现双指同时 > 0.01 N 达 5 步 |
| `grasp_slip` | 曾 `grasped=true`，但夹持中断且峰值开度 < 80 mm |
| `insufficient_open` | 有 `pulling_contact`，峰值开度 < 80 mm |
| `hold_fail` | 峰值开度 >= 80 mm，但 `open_streak` 未达 30 步 |
| `timeout_other` | 以上均不命中且到达 720 步上限 |

## 8. 测试集构造

### 8.1 分组

| 分组 | 来源 | 用途 |
| --- | --- | --- |
| `train` | manifest 中 `split=train`（ID 0,1,4,5,7,8,9,10,11,12） | 训练见过，成功率天然偏高，仅供参考 |
| `validation` | manifest 中 `split=validation`（ID 2,3,6） | 同源未见起点，主要参考 |
| `eval_only` | 新增扰动起点集 | **唯一的泛化数字** |

三组必须分开列，不得只报总数。

### 8.2 eval-only 起点规则

- 存放在 `data/processed/openarm_drawer_eval_v001/`，manifest 每条标 `eval_only: true`。
- **永不参与训练**，包括后续 017 及以后任何一轮。混入训练即该组数字作废。
- 允许变化的维度与范围：柜体 `drawer_offset_x` ∈ ±0.1 m 内取小范围、`drawer_offset_y` ∈ ±0.03 m、`handle_radius` ∈ [0.005, 0.02] m、`initial_right_jitter` <= 0.05 rad。
- 每条必须保存 `report.json`（含 overrides 与 jitter seed）与 `episode.npz`，使 `DrawerSimulation` 能复现初态。

### 8.3 样本量

验收要求每个模型至少 30 次独立测试。当前可用起点：train 10 + validation 3 + eval_only 约 20 = 33。

**同一起点的重复运行不计为独立样本**，只能在报告中作为"同起点稳定性"单独列出。

## 9. 确定性声明

本模型为确定性策略（固定噪声码 42、`t=1`、`num_steps=1`），仿真亦为确定性。因此：

- **同一实现、同一批次配置内**，同一 `episode_id` 重复运行，成败判定与峰值开度一致。2026-11-01 实测：ID2 在 `--repeat 2` 下两次均为失败、峰值 `-0.000000 mm`、失败模式 `no_contact`，判定 `PASS`。
- **跨实现不一致是结构性事实，不是 bug。** 批量无窗口脚本与交互式 `live_drawer_success.py` 的首个动作块必然相差 6e-4 量级（ID2 实测 6.47e-4），而接触任务下该差异足以翻转成败（差 6.47e-4 失败、差 1.29e-3 反而成功）。**因此不得用「复现历史单次数字」作为工具校准标准**；校准标准应为上一条的「同实现重复一致」。
- **结果不得依赖批次构成。** 预热必须使用与场景无关的固定输入，跨 trial 必须进程隔离（`run_drawer_eval_isolated.py`）。2026-11-01 实测：同一 ID2 在 `--episode-ids 12 2 6` 下为 20.20 mm，在 `--episode-ids 2 --repeat 2` 下为 `-0.000000 mm`，差异来自预热复用了批次首个起点的场景。
- 重复运行之间的差别来自墙钟调度与浮点路径，**不是随机性**，不能据此构造置信区间；
- 统计量只能来自**起点与几何的差异**，不能来自换噪声种子；
- 已知实测：ID12 两次运行的 `first_success_step` 分别为 140 与 469，峰值开度 101.15 mm 与 101.64 mm，但成败判定一致。这说明**成败稳定、步数不稳定**，报告中应如实写出这个波动。
- **渲染后端属于固定条件，不是可调参数。** 2026-11-01 用 `egl` 后端跑批量脚本时，ID12 由成功变为失败（73.42 mm，`grasp_slip`）、ID6 由失败变为成功（87.99 mm），只有 ID2 保持一致。原因是不同 GL 后端渲染出的 RGB 存在像素差异，而本模型对输入高度敏感（参见 `optimization_log.md` 第 3.2 节：左夹爪读数改动 0.001 即导致预测位置偏移 87.9 mm）。因此**跨批次比较必须核对 `run_config.json` 里的 `gl_backend` 字段**，后端不同则结果不可比。

## 10. 已知基线（2026-10-25，本协议建立前）

来源：`results/drawer_pull_success_20261025/physical_scores.json`，按同一判据与同一重放复核得出，可作本协议的对照基线。

| 起点 | 分组 | 峰值开度 | 首次成功步 | 判定 |
| --- | --- | ---: | ---: | --- |
| ID12 | train | 101.15 mm | 140 | 通过 |
| ID12（冻结配置复测） | train | 101.64 mm | 469 | 通过 |
| ID2（柜体 Y+5 mm） | validation | 90.78 mm | 141 | 通过 |
| ID6（human warm-start） | validation | 87.78 mm | — | 未通过（`hold_fail`） |

新批次结果必须与上表可比：判据、阈值、运行配置一致，落盘格式一致（`summary.json` + `closed_loop.npz`）。

## 11. 配套脚本与落盘

| 脚本 | 职责 | 状态 |
| --- | --- | --- |
| `scripts/a2_pi0/live_drawer_success.py` | 交互式证据生成（保留窗口、录像） | 已有，不改 |
| `scripts/a2_pi0/assess_drawer_open.py` | 独立重放复核与评分 | 已有，不改 |
| `scripts/a2_pi0/run_drawer_eval_batch.py` | 无窗口批量评估 | 待建（本协议后的第 2 步） |

批量脚本的每个 trial 目录必须与现有格式完全一致：`summary.json`（含 `episode_id`、`noise_seed`、`action_chunk`、`gripper_mode`、`teacher_actions_executed`、`intermediate_states_injected`、`state_representation`、`termination`、`judge`）与 `closed_loop.npz`（含 `applied`、`qpos_before`、`qpos`、`qvel_before`、`qvel`、`opening`、`contacts`、`forces`、`inference_seconds`）。格式对齐后可直接复用 `assess_drawer_open.py` 评分，无需重写判据。

结果目录命名：`results/drawer_eval_<YYYYMMDD>/`，内含 `metrics.json`（机器可读）、`metrics.md`（人可读表格）、`trails/`（每 trial 索引）与 `README.md`（口径声明与失败分析）。

## 12. 变更控制

修改本文件任何判据、阈值、指标定义或固定配置，必须在文末追加一条变更记录（日期、内容、原因、影响哪些既有数字）。**在已有结果上修改判据等于作废该结果**，需重跑并保留旧表。

## 变更记录

- 2026-11-01 建立本协议。判据、固定配置与指标定义取自 `live_drawer_success.py`、`assess_drawer_open.py` 与 016 检查点的 `deployment_config.json`，未引入新阈值。
- 2026-11-01 补充「渲染后端」为固定条件（`MUJOCO_GL=glfw`），并新增第 9 节关于后端影响的说明。原因：首次批量自检使用 `egl` 后端时 ID12 由成功翻为失败、ID6 由失败翻为成功。影响：所有 `egl` 批次结果作废，不得与 10-25 基线或后续 `glfw` 批次并列比较。
