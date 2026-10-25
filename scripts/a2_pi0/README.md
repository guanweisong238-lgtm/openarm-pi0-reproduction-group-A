# A2：π0 模型与 LoRA 训练

## 2026-10-25已验证的抽屉拉取入口

最新成功模型为Pi0骨干的4步一步确定性改造版，
`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000`。
原始起点/柜体Y+5mm通过，warm-start未通过，不能混称原多步flow全部复现成功。
训练入口`train_drawer_feedback_endpoint.py`，监督查询采集`collect_drawer_feedback_queries.py`；
原生闭环入口`live_drawer_success.py`，独立评分`assess_drawer_open.py`。
复现命令、成功视频和教学记录见[完整报告](../../results/drawer_pull_success_20261025/README.md)。

## 历史环境与基础链路（2026-10-23快照）

以下验收表、旧命令和待办描述当时状态，不覆盖上面的成功入口。部分旧权重已删除；当前模型与文件以[项目交接](../../docs/project-handoff.md)和[检查点清单](../../checkpoints/README.md)为准。环境安装与基础适配细节保留供复查。

本文件负责说明 A2 模块的职责、运行顺序、接口约定和实际验收结果。更新日期：2026-10-23。
所有命令在小组项目根目录执行，使用本项目的 `.venv`、数据、缓存和权重，不依赖旁边的 `openpi/` 工作副本。

## 1. 当前验收状态

- [x] 独立环境、NVIDIA GPU 运算及官方预训练模型推理验证。
- [x] 官方 `pi0_base` 的 33 个文件通过大小和 MD5 校验。
- [x] OpenArm 数据 v3.0 → v2.1 转换，全部 30 条轨迹读取及末端动作边界检查。
- [x] 三路图像、16 维输入输出适配、内部 32 维补齐和训练批次验证。
- [x] 仅训练集归一化统计计算、保存及 OpenPI 回读。
- [x] LoRA 初始化、冻结范围检查、10 步真实更新及 checkpoint 保存。
- [x] 新进程重载 checkpoint，对验证轨迹首帧推理检查通过。
- [ ] 补全公开数据来源链接、固定版本、许可和采集端动作定义。
- [ ] 固定验证集的训练前后基线比较、更长训练及训练/验证曲线。
- [ ] 与 A1/A3 完成标定、相机、时序和公开示教回放联调。
- [ ] 抓取闭环及任务成功率评估。

当前完成的是“数据 → LoRA 更新 → 保存 → 重载推理”的最小链路。10 步训练和单帧验证输入推理不代表策略已经学会抓取或仿真闭环成功。

## 2. 文件职责

| 文件 | 负责内容 |
| --- | --- |
| `convert_openarm_v3_to_v21.py` | 按轨迹和视频时间偏移转换格式，保留三路相机和数据划分 |
| `openarm_policy.py` | 图像格式统一、模型输入字段映射及输出裁剪至 16 维 |
| `compute_openarm_norm_stats.py` | 仅计算训练集状态/动作统计，保存统计来源 |
| `training_config.py` | 模型、LoRA 冻结规则、变换、路径和试训练超参数 |
| `local_data_loader.py` | 显式选择本地训练/验证轨迹，生成模型输入批次 |
| `download_pi0_base.py` | 匿名下载基础权重，逐文件校验大小和 MD5，重试失败文件 |
| `check_lora_init.py` | 参数加载、可训练集合、冻结精度与初始化显存检查 |
| `train_smoke.py` | 10 步更新，记录损失/梯度/实际更新量，保存完整训练产物 |
| `check_trained_policy.py` | 重载保存的策略，使用验证轨迹 24 首帧检查重复推理 |

代码文件开头用中文说明职责；主要类和函数说明输入输出、异常及边界，关键流程按模块注释。

## 3. 独立环境

已验证设备：RTX 5060 Ti（16 GB），Python 3.11.17，JAX 0.5.3。
依赖入口：[configs/a2-requirements.txt](../../configs/a2-requirements.txt)。

- OpenPI / openpi-client：`215abfb217dbac7d5f1273282331b9b1866c0479`。
- LeRobot：`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，使用 v2.1 读写接口。
- NumPy 1.26.4、PyArrow 21.0.0。
- datasets 3.6.0、fsspec 2025.3.0、gcsfs 2025.3.0。

首次建立环境：

```bash
uv venv --python 3.11 .venv
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python --torch-backend=cu128 -r configs/a2-requirements.txt
uv pip check --python .venv/bin/python
```

这些兼容性固定避免 PyArrow/NumPy 运行时冲突及新版 datasets 的 `Column`、`List` 与旧 LeRobot 接口不兼容。新版环境生成的 Parquet 不能仅靠降级依赖或修改版本标签修复，需要用兼容环境重新转换。A1 仿真独立维护环境，不直接合入其 NumPy 2 依赖。

## 4. 数据与预处理

| 项目 | 当前约定 |
| --- | --- |
| 原始数据 | `data/openarm_pickup/`，LeRobot v3.0 |
| 转换数据 | `data/processed/openarm_pickup_v21/`，LeRobot v2.1 |
| 机器人标识 | `openarm_bimanual`；采集硬件版本仍需来源佐证 |
| 任务 | `Pick up the cube.` |
| 总量 | 30 条轨迹、5,405 帧、30 Hz |
| 训练集 | 轨迹 0–23，共 4,548 帧 |
| 验证集 | 轨迹 24–29，共 857 帧 |
| 状态/动作顺序 | 右臂 7 关节＋右夹爪，再接左臂 7 关节＋左夹爪 |
| 训练相机 | `ceiling`、`wrist_left`、`wrist_right` |

两路头部图像保留在原始数据，但不用于当前训练。转换不改变原始状态和动作数值，不做关节翻转、单位换算或增量动作转换。

统计和训练均显式选择训练轨迹；只在元数据声明 `splits` 不会保证实际筛选。每个原始训练帧对统计只计一次权重，状态和动作采用均值/标准差归一化，然后补齐至 32 维。第 15 维左夹爪动作恒定，标准差下限为 `1e-6`；这份数据不能提供左夹爪开合的学习变化。

来源、许可、数据版本和处理记录需补入 [data/README.md](../../data/README.md)。本地目录名不是已核验的公开数据集名称。

## 5. 运行顺序

以下命令用于复现。转换拒绝覆盖已有输出，已验收步骤无需重复执行。

```bash
# 1. 单轨迹试转换，再转换全部轨迹（--limit 0）
.venv/bin/python scripts/a2_pi0/convert_openarm_v3_to_v21.py --limit 1 --output data/processed/openarm_pickup_smoke_v21
.venv/bin/python scripts/a2_pi0/convert_openarm_v3_to_v21.py --limit 0 --output data/processed/openarm_pickup_v21

# 2. 训练集统计、配置和输入批次
.venv/bin/python scripts/a2_pi0/compute_openarm_norm_stats.py
.venv/bin/python scripts/a2_pi0/training_config.py
.venv/bin/python scripts/a2_pi0/local_data_loader.py

# 3. 基础权重和 LoRA 初始化
.venv/bin/python scripts/a2_pi0/download_pi0_base.py
.venv/bin/python scripts/a2_pi0/check_lora_init.py

# 4. 真实更新、保存、新进程重载
PYTHONUNBUFFERED=1 .venv/bin/python scripts/a2_pi0/train_smoke.py
PYTHONUNBUFFERED=1 .venv/bin/python scripts/a2_pi0/check_trained_policy.py
```

当前配置为 `pi0_openarm_lora` / `smoke_001`：batch size 1、预测长度 16、训练 10 步、随机种子 42。两个语言模型分支启用 LoRA，冻结全部非 LoRA 参数（包括图像编码器），不维护 EMA。

`train_smoke.py` 不覆盖已有实验、不支持自动续训；重复运行已完成实验会报目录已存在。新实验需修改配置中的实验名。重载脚本目前固定读取 `smoke_001/10`，新实验还需同步调整读取路径，并保持模型结构匹配。

加载器无限迭代，由训练步数控制停止；验证也必须限制批次数。当前 `drop_last=True` 在 batch size 大于 1 时会丢弃不足一批的尾部，正式评估需另行处理。

## 6. 产物与已验证结果

| 产物 | 本项目内路径 |
| --- | --- |
| 基础权重 | `weights/openpi/openpi-assets/checkpoints/pi0_base/` |
| 归一化统计、来源清单 | `assets/pi0_openarm_lora/openarm_pickup/` |
| 下载、初始化日志 | `logs/a2_pi0/download_pi0_base_*.log`、`check_lora_init_*.log` |
| 训练日志、逐步指标 | `logs/a2_pi0/train_20261023-201709-084338/` 下的 `run.log`、`metrics.jsonl` |
| 训练 checkpoint | `checkpoints/pi0_openarm_lora/smoke_001/10/` |
| 重载推理证据 | `logs/a2_pi0/reload_20261023-202002-296482/` 下的日志、`summary.json`、`actions.npy` |

checkpoint 的 `params/` 保存完整模型（含冻结基础参数），不是仅保存小型 LoRA 适配器；`optimizer/` 保存步数和优化器状态，`assets/openarm_pickup/` 保存本次统计，`config.txt` 保存配置表示。当前脚本尚未实现从优化器状态恢复训练。

| 验收项目 | 2026-10-23 实测 |
| --- | --- |
| 基础权重校验 | 33 个文件、11.19 GiB，大小和 MD5 全部通过 |
| 总参数 / 可训练参数 | 3,288,036,112 / 49,987,584（1.5203%） |
| 初始化显存 | JAX 报告约 6.44 GiB |
| 10 步训练 | 损失、梯度均有限，每步实际参数更新非零，退出码 0 |
| 训练显存 | JAX 报告当前约 6.82 GiB、峰值约 7.32 GiB，不是整机监控值 |
| 保存与重载 | 保存成功；验证轨迹 24 首帧输出 `(16, 16)`，有限值及固定噪声重复性检查通过 |
| 重载调用耗时 | 首次含编译约 10.760 秒，第二次约 0.205 秒，仅为单次观测 |

以上不是正式性能或泛化评估。完整验证集指标、曲线和抓取成功率尚未完成。

## 7. A1/A3 接口约定（第一版，待联合验收）

| 项目 | 模型侧要求 / 分工 |
| --- | --- |
| 状态输入 | `observation.state`：16 维原始位置，右臂在前；使用A1包装器时已完成重排 |
| 图像输入 | `observation.images.ceiling`、`observation.images.wrist_left`、`observation.images.wrist_right` |
| 图像表示 | RGB，HWC uint8 或 CHW/HWC 的 [0, 1] 浮点数组；A2 缩放到 224×224 |
| 任务文本 | 字符串 `prompt`，或读取器提供的 `task` |
| 动作输出 | 已反归一化的 `actions`，形状 `(16, 16)`，右臂在前 |
| 时间语义 | 16 个未来时间步，数据间隔为 1/30 秒 |
| A2 负责 | 格式映射、归一化/反归一化、分词和模型内部补齐 |
| A1 负责 | 实际关节状态、位置控制接口、同一时刻图像、时间戳及可重复复位 |
| A3 负责 | 顺序映射、时间调度、合法性检查、限幅反馈和闭环调用 |

A1 原生环境默认20Hz，新增 `OpenArmPolicyEnv` 包装器提供名义30Hz和顶部/双腕三路图像，并已按执行器名完成双向顺序转换；A3使用该包装器时不要重复交换左右臂。包装器保留1ms物理步长，每周期推进33/34步，300周期约10秒；它不负责模型推理的墙钟调度。A1原生接口仍需A3自行适配。头部两侧相机不是当前必需输入。

2026-10-23，本机A1观测快照通过A2现有输入适配、归一化、分词和维度补齐。当前联调改用CELL场景，保留原生三路相机，并额外定位episode 0首帧。该条示教回放已由用户目视确认抓放成功；这不等于相机标定完成或模型自主任务成功。

采集端动作单位、关节零点/方向、夹爪含义和超出仿真限位的动作，仍需来源说明和公开示教回放核实。维度一致不能证明可直接执行。

## 8. Git 与文件管理

代码、依赖配置、README 和小型实验报告纳入 Git。根目录 [.gitignore](../../.gitignore) 排除原始/转换数据、权重、checkpoint、生成的统计、缓存和日志；获取与生成方法应记录在文档中。

`results/` 下的小型 Markdown、JSON、CSV 报告可提交，动作数组和视频忽略。数据说明保存在 `data/*.md`，需要纳入版本管理的小型来源清单放在 `data/manifests/`。不使用全局 `*.lock` 忽略规则，保留依赖锁文件。

空 `.lock` 仅是下载互斥锁；目录或进度条存在不能证明下载完整。以文件校验、最终 PASS 和进程退出码 0 作为验收依据。

## 9. CELL仿真快照推理

`infer_sim_snapshot.py`读取A1快照，加载10步LoRA检查点，两次固定噪声推理，保存动作及限位诊断；不连接仿真或执行动作。A1回放脚本定位后生成 `policy_observation.npz`。

调用：`.venv/bin/python scripts/a2_pi0/infer_sim_snapshot.py --observation /绝对路径/policy_observation.npz`

默认检查点为 `checkpoints/pi0_openarm_lora/smoke_001/10`；结果在 `logs/a2_pi0/sim_infer_时间戳/`。`actions.npy`是原始反归一化预测，`clipped_actions.npy`仅用于诊断，两者均保持右臂在前；执行时仍需仿真包装器检查和限幅。接口约定见 `docs/interfaces.md`。

## 10. 首轮LoRA训练与离线验证

`train_lora.py`复用已验证的LoRA初始化和更新步骤，默认从pi0_base重新初始化，运行1000步，实验名lora_001，批次大小1，100步学习率预热，峰值2.5e-5，余弦衰减至2.5e-6。每100步验证，每500步保存，禁止覆盖已有实验；当前不支持恢复中断训练。

验证使用轨迹24至29各两帧，共12个固定观测，每次复用相同随机数。报告的是包含32维补齐和末端填充的flow matching损失，不是完整验证集动作误差或仿真成功率。1000步只作为第一轮训练实验，约0.22个训练集遍历，效果需实测。

```bash
.venv/bin/python scripts/a2_pi0/train_lora.py --dry-run
.venv/bin/python scripts/a2_pi0/train_lora.py --steps 1000 --exp-name lora_001
```

日志位于logs/a2_pi0/lora_时间戳，包含训练metrics.jsonl、validation.jsonl、validation_samples.json、config.txt及完成后的summary.json；检查点位于checkpoints/pi0_openarm_lora/lora_001/500和1000。保留原smoke_001/10供闭环对照，不自动替换常驻推理模型。

常驻闭环基线已归档在results/a2_resident_closed_loop_baseline_20261023.json：5轮20步，单轮推理约0.186至0.189秒，物理时间在推理期间暂停，任务成功尚未验证。

`evaluate_actions.py --checkpoint /检查点目录`使用相同的12帧验证观测计算原始16维动作MAE/RMSE，剔除跨出轨迹末端的填充目标，分别报告手臂与夹爪。结果在logs/a2_pi0/action_eval_时间戳/summary.json；不把标签输入策略，也不宣称仿真成功。

旧版LeRobot在非零起始轨迹子集上查询未来动作存在索引兼容问题。本项目local_data_loader.py的EpisodeSubsetDataset仅在动作边界查询时将原始轨迹号映射到子集局部编号，视频仍使用原始轨迹号。训练24条和验证6条轨迹的末帧16步填充与边界均已实测通过；未修改site-packages。

### 2026-10-23首轮训练结果

lora_001已完成1000步，从pi0_base重新初始化，未覆盖smoke_001。固定12帧验证flow损失从0.472746降至0.172219。相同12帧原始16维动作MAE：smoke_001/10为0.218136，lora_001/500为0.167640，lora_001/1000为0.111568；相对10步基线降低约48.9%。各维及手臂/夹爪分项见results/a2_lora_001_training_evaluation.json。

所有动作评估均从保存的检查点重新加载，覆盖验证轨迹24至29，剔除轨迹末端填充，使用相同噪声。该指标不证明模型自主抓放成功。下一轮可视化测试建议使用lora_001/1000；当前resident_inference.py仍固定使用smoke_001/10，尚未切换。

## 11. 顺序诊断实验

1. CELL外观透明度消融：A1 configs/a2_dataset_render.json选择cell_visual_alpha=0.2，默认配置不变；固定首帧状态的16步预测MAE从0.080939降至0.067242。只改变cell_vis非碰撞视觉网格，不代表全部相机已标定或任务成功。
2. evaluate_actions.py新增--frames-per-episode 0评估完整划分；--split train --episode 0用于单轨迹训练误差检查。完整验证857帧MAE=0.110004，前20%时间段误差最高；摘要见results/a2_full_validation_20261023.json。
3. train_lora.py --train-episode 0 --steps 1000 --exp-name overfit_ep0_001：仅训练episode 0，仍保持验证集隔离，原归一化统计不变。
4. 待单轨迹预测及闭环结果确认后，再决定全数据集训练遍历次数；不因离线loss下降自动进入长训练。

顺序诊断实测：单轨迹1000步模型在episode 0全263帧MAE=0.028870，原模型为0.073385；但相同渲染下15秒闭环仍未抬起方块。动作重建与记录的关节状态最大误差为0，方块最终约[0.45,0,1.024993]，托盘原点[0.47,0.15,1.005]。第四步扩大训练暂未启动；先排查单轨迹离线拟合与闭环执行的差异。完整记录见results/a2_sequential_diagnosis_20261023.json。

`check_action_trends.py`使用原始示教输入，在13个起点和3个固定噪声种子上比较完整16步动作方向及幅度，保存原始预测NPZ和JSON。2026-10-23实测首帧右关节3位移幅度仅为真实动作的约16%（seed42）、62%（seed0）、25%（seed1）；结论为动作趋势仍未拟合充分，不能把换种子当作问题修复。摘要见results/a2_overfit_action_trends_20261023.json。

`check_gripper_transition.py`诊断真实示教闭合阶段和实际闭环之间的差别，只做离线输入替换。2026-10-24第120帧结果：真实示教输入的前4步右夹爪目标均值0.0129rad；仅将当前右夹爪状态替换为实际张开值后，预测为-0.8361rad。反向仅用示教夹爪状态替换实际输入后为0.0257rad。说明当前开合量强烈影响预测；不是向仿真注入假状态的修复方案。右臂控制点同时存在约2.6至3.1cm偏差，仍需纠偏数据与闭合过渡训练验证。详见results/a2_gripper_transition_diagnosis_20261024.json。
