# 项目背景交接（截至 2026-10-24 13:10）

## 已完成的抽屉拉取验证（2026-10-25，最高优先级）

本轮已训练并验证标准起点的自主抽屉拉取，当前没有训练进程在运行。
最终模型：`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000`。
这是Pi0骨干的一步确定性改造版，非原多步flow复现成功。原先012～015均未通过。
模型预测right8动作，现有32维输入加入实际qvel，动作窗口4、固定noise42、t1一步采样，
左臂固定初始目标，右7残差/绝对夹爪，夹爪保持原连续输出continuous。
训练1200个真实物理分支反馈监督查询（1080 train/120同父场景扰动留出），
来源均为原train ID12，不含validation起点；原13条数据及划分/统计未覆盖。
查询包含成功反馈专家和真实失败策略访问状态，逐条执行16步专家并独立重放0误差，模型只学前4步。
源015/2000，2000步lr1e-5，LoRA+三IO共50,087,968参数，right8端点MSE。
训练源快照/曲线：`logs/a2_pi0/drawer_feedback_20261025-031627-720249/`。

真实模型闭环：ID12原始起点首次4.67s/峰值101.15mm通过；跨进程复测15.63s/101.64mm通过；
validation ID2柜体Y+5mm为4.70s/90.78mm通过。
validation ID6 human warm-start未通过，24s峰值87.78mm但最终76.94mm，未保持80mm达1s。
每条独立重放实际动作，qpos/qvel/接触力最大差0；无专家动作/中间状态灌入/抓取weld。
严格门槛为双指整个把手组件各>.01N连续5步、真实接触拉动、>=80mm保持30帧。
两次原始起点完成时间不同，不能把固定seed等同跨进程逐比特轨迹复现；差异没有单独定位。
推理期间仿真暂停，仿真秒数不是硬件实时性能。只验证拉取保持，未完成关闭/往返。

运行：`live_drawer_success.py --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000
--task open --seconds 24 --episode-ids 12 --seed 42 --noise-mode fixed --action-chunk 4 --gripper-mode continuous --auto-next`。
固定code42是训练条件，不能把换7/99当有效随机种子测试；应测不同真实初态。
完整结果、视频、失败边界、逐步教学与复制命令：`results/drawer_pull_success_20261025/README.md`。
检查点后验验证另在validation_report.json；input_protocol.json保留训练时快照，不重写成假历史。
另有约7万参数MLP关节目标反馈基线，固定权重/闭合目标0在ID12/2/6都通过，明确非Pi0结果。
slew_binary夹爪对照在train ID12/8均失败，未采用，日志保留。不要把基线的控制调法混成Pi0成功。

后续若扩大目标，优先补train warm-start下的恢复与保持，再扩展场景，保留验证隔离；
现有成功属于固定柜体/小幅位置变化的局部原型，不宣称所有13条起点或广泛泛化都已通过。

## 当前授权与执行状态（2026-10-25，优先于下文）

**015历史训练记录（已完成，闭环未通过）。** 源014/3000，2000步lr1e-5、原/恢复1002/998次，
数据与14相同，新增实际当前速度信息：state32=位置16+qvel16，左右顺序一致；左位置/速度屏蔽。
使用原网络已有32维槽，不增加网络参数。新state_proj后16行置零，GPU同噪声检查源函数最大差0，
再开始学习；不是专家阶段/未来信息。13条示教只重放动作提取速度，逐帧qpos误差0，
速度统计仅原10条train，std floor0.1rad/s；原位置/动作统计不动，独立新资产：
`assets/pi0_openarm_drawer_velocity_v001/openarm_drawer_success_v003_right7_delta_velocity/`。
速度缓存key`6a20f3a46f2c84bbfacab`，约0.7MB；不要随意修改prepare_drawer_velocity.py/DrawerSimulation，
其源码参与速度资产身份校验，变化需要显式新版本，不能覆盖已有来源。
新入口`train_drawer_velocity.py`，配置`drawer_velocity_config.py`，实时仿真按协议自动选择。
日志`logs/a2_pi0/drawer_success_20261025-022715-553653/`，报告`results/pull12_velocity_015/README.md`。
计划检查点`checkpoints/pi0_openarm_drawer_velocity_v001/pull12_velocity_015/2000`。
新live同时记录qvel_before/qvel；独立评分器对速度输入模型要求速度记录且重放一致。
已经验证缺失/被篡改速度拒绝、真实速度重放通过（仅/tmp测试fixture，不是模型成功）。
015结束后必须继续完整初态seed42 resample/chunk4真实闭环；成功后7/99和原验证起点，失败继续解决。

014已完成3000步，固定validation首动作平均1.745mm，但完整24秒ID12 seed42闭环仍无接触/无开度，
独立重放qpos和接触力误差0，`results/pull12_recovery_014/physical_score.json`。
CPU留一相关轨迹核回归发现，step120仅q预测动作MAE0.032999rad，q+实际v为0.003915rad；
支持进一步检验动态可观测性，不是Pi0成功或唯一因果证明。
重现脚本`probe_drawer_velocity_value.py`，原数值`results/pull12_recovery_014/velocity_probe.json`。
备用连续专家（省去中间0.5秒等待，末尾保持保留）已从原始起点及013b step60/120试采，
目录`data/drawer_recovery/continuous_pilot_ready`、`continuous_v003/`；未加入015，属于专家证据。
当前尚未有自主拉取成功的Pi0模型，用户要求持续执行至成功，不能以训练结束为完成。

用户要求持续自主推进，直到训练模型在真实物理仿真中完成抽屉拉取；遇到问题自行解决，
过程需教学式讲解。当前优先验收拉开，不把关闭作为本次拉取门槛，也不把专家成功冒充模型成功。
真实双指组件接触持续5步、接触拉动、开度>=80mm保持30帧；策略读取实际RGB/关节观测，
禁止专家动作替代或中途灌入示教状态。`live_drawer_success.py --task open`已增加独立拉取评分，
默认roundtrip保留。`assess_drawer_open.py`从初态只重放实际动作，独立重算接触/力和事件评分。

012阶段训练1200步完成，但ID12/8较长闭环均无双侧接触。固定离线误差下降不等于闭环通过。
同噪声逐推理观测对照发现：原示教观测最坏首动作位置27.57/16.07mm，实际观测273.40/455.54mm；
存在稀疏评估漏掉的欠拟合，也有闭环偏离放大，尚不能只归因为示教少或唯一归因为分布偏移。
完整记录`results/approach13_delta_012/README.md`。

013b已完成2000步训练：源012/1200，仅train ID12集中拟合从起点到拉开保持的0～301帧，
未来16步标签原样，原13条数据/10-3划分/残差统计不变。
LoRA+三IO、右7残差、左状态屏蔽、无增强、low_mix、首4权重4；2000步lr1e-5，结束评估/保存。
日志`logs/a2_pi0/drawer_success_20261025-012805-022077/`，原生loss每100步刷新。
`results/pull12_fit_013b/README.md`记录预定闭环：ID12、seed42、resample/chunk4，
预定通过后冻结配置检验7/99；实际未通过，未扩展测试。不得把单训练初态成功称作泛化。
结束固定检查train/validation平均位置2.178/1.832mm，但最坏接近帧108仍14.09mm。
实际seed42的resample+chunk4、resample+chunk16、zero+chunk4三轮24秒均无双侧接触/无开度；
独立物理重放qpos与接触力最大差均0。用原示教前76步带到标准预接近状态再交给模型，
后续12秒也失败，仅用于定位，不可作为完整起点成功证据。记录`results/pull12_fit_013b/`。
原013因初始评估抽帧过密，在任何梯度更新前终止，日志保留，无新权重。

独立恢复采集`collect_drawer_recovery.py`从012实际偏离快照重新规划专家动作，
不瞬移其他臂，不灌中间状态，保留源快照与独立动作重放检查。
`data/drawer_recovery/v001/`3条来自012实际11/12/15步，v002再加013b实际40/60/120步3条；
均专家拉开约93mm/保持1秒、独立回放误差0，这是专家成功，不是模型成功。
v002清单合并6条，每条只抽原帧0～59与75～89，共450观测；未来16动作仍原连续索引，不跳等待段。
6条均相关于train ID12，不能称为6种场景泛化。13条原数据/划分/统计不变。

**014历史训练配方**：源013b/2000，3000步lr3e-6，1526次原片段+1474次恢复；
原片段移除0.5秒中间等待帧60～74、135～149、177～191的采样，末尾拉开保持保留。
双项调整共同检验控制效果，不能唯一归因。源脚本已按运行时拷贝并SHA存档。
日志`logs/a2_pi0/drawer_success_20261025-015807-867565/`，报告`results/pull12_recovery_014/README.md`。
准备结束后同一真实完整初态resample/chunk4/seed42，成功后7/99和原验证起点；失败继续修复。
备用连续专家采集只省去0.5秒中间等待，step120数值试采已成功，不属于014数据，也不是模型成功。

2026-10-25又归档元数据并删除006/007/008/009各300步与012/600权重，释放25.899GiB，
磁盘约44GiB空闲；保留010/011/012-1200/013b及旧关键基线。没有权重备份。
完整清单`results/project_cleanup_20261025_recovery_014/`。
采集新的起点可保留原30Hz的2ms子步奇偶相位：report的`initial_control_step_offset`，
DrawerSimulation默认0，老数据行为保持。恢复缓存独立命名空间，不修改原v003缓存。


## 最新里程碑：起点拟合与局部物理闭环（2026-10-24 23:20）

011从010/300小学习率续训，计划600步、lr1e-6，400/450连续通过后450步自动停止。
独立重载native最坏1.557mm/0.929°，BF16最坏1.579mm/0.961°，首次通过train首9帧×3种子门槛；
validation最坏约2.053mm仍未通过2mm，不能声称泛化过关。
随后真正运行ID12首9步/0.3秒物理闭环：fixed noise+chunk4为2/3通过，
fixed+chunk1仅1/3通过；同一模型改为每次推理新噪声序列、chunk4后原3种子通过，
冻结配置新增0～4也全部通过（局部8/8，最坏1.988mm/1.056°）。
仅说明该训练初态短起步片段；完整接近、抓握、开关往返未训练/验证。
14次物理实验从完整初态重放applied动作，逐步qpos最大差0；8份新噪声序列逐元素核对一致。
当前候选`checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_lowmix_refine_011_io_mask/450`。
完整报告与曲线`results/start9_lowmix_refine_011/README.md`。
新实时参数`--seeds`、`--action-chunk`、`--noise-mode`、`--start-diagnostic`。
后续实验显式指定`--noise-mode resample --action-chunk 4`；默认仍fixed用于兼容历史。
下一步扩大训练至接近和抓握，保留13条数据10 train/3 validation隔离；先学逐阶段再完整往返。


用于新对话快速接手。所有数字均可从 `results/`、`logs/` 与下述脚本复核。

## 磁盘清理更新（2026-10-24 21:31）

已按用户要求清理10份已完成/被替代的检查点，释放51.788GiB，可用空间约64.7GiB。
保留8份完整模型，清单与用途见`checkpoints/README.md`。原始数据、13条成功数据、
底座权重、环境、缓存、日志和结果全部保留。删除模型的配置/协议/统计已归档，
**不包含权重备份**；下文历史路径中已删除的权重需重训，不能直接加载。
完整记录`results/project_cleanup_20261024_213130/README.md`。


## 顺序验收进展（2026-10-24 晚，优先于下文历史记录）

已执行首9帧拟合诊断，入口 `docs/drawer-start-fit.md`。未进入新的完整任务闭环。
四组同源200步控制实验：LoRA / +三IO层 / +左状态屏蔽 / +right8损失，
train首动作平均位置误差分别15.318 / 14.876 / 8.686 / 9.705mm。
三IO层实际权重更新已验证；屏蔽左状态改善较大，删除左动作监督没有进一步改善。
从相同屏蔽组200步权重，各续训600步比较绝对与右7残差动作，
平均/最坏位置误差分别4.917/11.494mm与2.323/4.532mm。
残差定义 `action[t+k,:7]-state[t,:7]`，夹爪保持绝对目标；独立train统计与解码器，
原数据/统计不改。新 `drawer_delta_config.py` 与 `drawer_start_fit_config.py` 负责配置恢复，
`live_drawer_success.py` 已支持新语义。残差模型独立重载10/50步、native/BF16均未通过
事先设定的训练9帧×3种子最坏位置<2mm、最坏姿态<2°门槛；不能宣称启动问题已解决。
结果 `results/start9_delta_004/`、`results/start9_cleanup_002/`。
残差续训 `start9_delta_refine_005` 已完成1200步，每100步检查，仍未通过。
最终train平均/最坏位置1.465/3.568mm、最坏姿态2.308°；
完整结果 `results/start9_workflow_20261024/README.md`。
只有通过保存后独立重载才执行覆盖范围内的短闭环，之后才能扩展到抓握和完整往返。

## 最新实验状态（2026-10-24 下午，优先于下文历史记录）

13条成功v003数据已完成训练：10 train/3 validation，原实验`success13_lora_001/2000`。
真实闭环显示train与validation起点均有早期偏离，首帧图像/状态已复核匹配，原示教动作可重现。
随后从同一2000权重做uniform/early60配对训练，各500步、lr3e-6、fresh optimizer，
只改变最初75帧采样概率。加密前75帧并保留原四帧口径，三种子离线评估改善，
但两个模型的train ID12/8、seed42、各12秒四次真实闭环均无把手接触/无拉开。
early60的ready前4步实际偏离仍约93mm；不能宣称已解决任务。
完整方案`docs/drawer-early-sampling-control.md`，结果入口
`results/early_sampling_control_20261024/README.md`；下一步建议首0～8帧单起点拟合诊断。
新增`train_drawer_success.py --checkpoint/--early-probability/--dense-approach-eval`，
实时仿真`live_drawer_success.py --episode-ids ... --auto-next`。
GPU/桌面在默认工具沙箱不可用，但本次经授权的沙箱外执行已成功运行训练与原生MuJoCo窗口。
用户要求教学式讲解；原生loss图每100步刷新，不使用网页。

随后已完成只读深入诊断（无新训练）：固定输入消融发现，ready首帧仅将左夹爪读数
0→训练均值0.001069rad，右夹持点预测变化87.91mm，同进程原输入复测差异0。
early60固定train诊断损失中左8维占41.64%，padding仅0.81%。当前所有非LoRA参数冻结，
包含state/action三投影层（共100,384参数）；上游模型LoRA过滤规则允许其更新。
错误首目标使右2/4/7执行器立即饱和。建议先做首0～8帧小样本拟合及接口层解冻消融，
再逐项检验无关左状态屏蔽、right8目标和必要的残差动作；这些方案已于晚间按上述顺序逐项执行，结果以上述最新进展为准。
完整证据、限制和验收顺序见`results/drawer_deep_diagnosis_20261024/README.md`。

## 后续方向变更（2026-10-24，本对话）

用户已要求先扩充脚本专家示教数据，撤销“必须单轨迹拟合达标后才能扩充”的顺序限制。
以 `docs/drawer-data-expansion.md` 为新数据工作的入口；原实验结论仍保留，
不能据此认定失败主因已确定为数据量。用户还要求所有操作进行教学式讲解。
扩充试采位于 `data/drawer_expansion/`，原单轨迹及归一化统计不改。
训练代码目前仍绑定原单轨迹，新采数据不会自动被读取。
扩充第二批已完成：`data/drawer_expansion/dataset_v002.json` 为7条无损RGB、3234帧的
清单（train 5条、validation 2条），包括抓握角、侧向接近、初态和柜体Y小幅变化。
`drawer_multi_episode_dataset.py` 已提供读取接口并验证跨轨迹边界；训练入口仍未接入，
旧归一化统计保持不变。偏低/偏高试采未通过，不列入成功训练数据。
全采集审核见 `results/collection_audit_20261024/summary.md` 与 `audit.json`：
扫描56份抽屉记录，均数值有限且独立回放误差0；截至本轮RGB补查，成功RGB候选10条，
成功待RGB 3条，成功数值版9份（含RGB副本），未达往返/其他任务29份，辅助绑定3份，自检2份。
人工10条中6条通过数值任务验收：旧会话152741/002，新会话154326/001、003、004、006、008。
新会话002、005、007打开保持仅0.87/0.97/0.87秒；旧会话001无双侧组件夹持。
人工接触应计整个把手组件，不能沿用横杆口径拒绝支柱夹持。训练清单仍为原7条，
未自动加入人工数据；人工warm-start条件与脚本ready-start须单独处理。

**随后已按用户要求整理13条成功数据为独立训练数据版本 v003**：
`data/processed/openarm_drawer_success_v003/manifest.json`，13条10700帧（train 10条8632帧，
validation 3条2068帧），全部RGB已完成。保留原两条脚本验证轨迹，并将旧人工会话002整体
留作验证，新人工会话成功轨迹均用于train。NPZ独立拷贝且源/目标SHA256一致，未裁剪或拼接。
统计：`assets/pi0_openarm_drawer_success_v003_lora/openarm_drawer_success_v003/`，
仅用train计算mean/std；不覆盖旧统计。
新配置及预处理入口：`scripts/a2_pi0/drawer_success_training_config.py`，
`get_success_config()` / `transformed_dataset(cfg, split)`；CPU验证覆盖13条边界、26个模型输入
观测、32维补齐和训练统计独立重算，记录 `results/drawer_success_v003_validation.json`。
旧单轨迹训练脚本没有自动切换，本次未启动GPU训练。数据README记录用法和warm-start边界。
随后已新增多轨迹训练入口 `scripts/a2_pi0/train_drawer_success.py`，配方、短测及正式命令见
`docs/drawer-success-training.md`。从pi0_base开始，均匀选train轨迹再选帧，保持all32目标，
lr峰值1e-5。固定train40/validation12个检查帧，三种子实际动作与FK诊断。
10700帧uint8磁盘缓存及CPU输入/轨迹边界检查已通过；尚未启动GPU短测，需用户终端执行。
随后用户已完成 `success13_smoke_001` 的20步GPU短测：更新有限非零，三种子诊断与检查点保存通过，
无OOM（XLA训练图7.11GiB）。validation臂MAE 0.201080→0.195536，位置72.430→72.050mm，
属于小幅早期信号，不是收敛。日志 `logs/a2_pi0/drawer_success_20261024-161702-707695/`。
下一步用户终端执行文档中的 `success13_lora_001` 2000步，从base重新开始。
随后该2000步实验已完成，1000/2000检查点已保存，最终loss图PNG/SVG已生成。
验证固定12帧三种子均值：臂MAE 0.201158→0.055792，位置72.413→17.191mm，姿态31.634→6.752度。
明显组间差异：验证ready起点位置20.93mm，人工预接近起点9.72mm；ready首帧误差41～53mm。
详细结果和下一步诊断见 `docs/drawer-success-training.md`，未验证模型闭环成功。

## 1. 项目是什么

OpenArm 双臂机器人 + MuJoCo 仿真上复现 π0 / OpenPI 视觉-动作策略。阶段 I（仿真复现），
周期 2026-10-20 至 11-19。目标：公开数据适配 → 官方推理 → LoRA 微调 → 仿真闭环 → 定量评估。
不要求新算法或真机部署。

本仓库只存 A 组自己的代码、配置、文档与实验记录，不复制上游仓库。

## 2. 任务定义

单一任务：`Open the drawer, then close it.`（拉开抽屉再推回关上）。

- 机器人：OpenArm，右臂 7 关节 + 右夹爪，左臂 7 关节 + 左夹爪。
- 动作：16 维绝对位置目标（右臂7、右夹爪、左臂7、左夹爪），弧度。
- 模型内部动作维度 32 维：前 16 维真实，后 16 维补齐。
- 观测：三路 320×240 RGB（ceiling / wrist_left / wrist_right）+ 16 维关节状态 + 任务文本。
- 控制：30 Hz，每次执行模型输出前 4 步再重观测，最多 24 秒。
- 成功判据：真实接触抓握，拉出 ≥80 mm 保持 1 秒，随后推回至 5 mm 内保持 1 秒。

## 3. 环境与硬约束

| 项 | 值 |
|---|---|
| 项目根 | `/home/song03/scientific_research/openarm-pi0-reproduction-group-A` |
| Python | `.venv/bin/python`（3.11.17） |
| jax / flax / openpi | 0.5.3（cuda12）/ 0.10.2 / 0.1.0 |
| GPU | RTX 5060 Ti 16 GB，驱动 580.178.04 |
| 训练显存 | XLA 报 7.11 GiB，无 OOM |
| 上游仿真源码 | `.cache/openarm-control`，固定 commit `14ea59fa…`，Apache-2.0 |
| 磁盘 | `/dev/nvme0n1p6`，394 G，当前约 41 G 可用；**每个检查点 5.2 G** |

**重要约束**：AI agent 的执行 shell 看不到 `/dev/nvidiactl`、`CapEff` 为 0，无法用 GPU，
也无法 `mknod`。**训练与闭环必须在用户终端执行**，agent 只能做 CPU 侧验证（数据链路、
MuJoCo 正运动学、纯函数测试）。agent 可以直接读用户跑出来的产物文件。

## 4. 数据

`data/drawer_trial/contact_roundtrip_rgb_001/`：单条示教，462 帧，15.4 秒，30 Hz。

- `episode.npz` 保存原始 uint8 RGB、state、action、`qpos`（全物理状态）、
  `initial_integration_state`、`actuator_names`、`terminal_qpos`。
- `lerobot_v21/` 为格式检查用的镜像，视频是有损编码。
- 场景为上游 pedestal，柜体沿世界 X 前移 100 mm 以便拉出超过 80 mm；14 mm 原把手。
- 关键阶段边界（后续所有分析共用）：

| 阶段 | 帧 | 内容 |
|---|---|---|
| approach | 0–149 | 接近把手 |
| grasp | 150–191 | 闭合夹爪 |
| pull | 192–281 | 拉出 |
| open_hold | 282–311 | 保持打开 |
| push | 312–401 | 推回 |
| closed_hold | 402–461 | 关闭保持 |

归一化统计在 `assets/pi0_openarm_drawer_lora/openarm_drawer_roundtrip/norm_stats.json`，
mean/std，std 下限 1e-6。**左臂动作 std 只有 1e-6~6e-5（物理静止），右臂 0.046~0.511。**

## 5. 模型与训练配方

- π0 base 重新初始化；`gemma_2b_lora`（rank 16）+ `gemma_300m_lora`（rank 32），其余冻结。
- 可训练 49,987,584 / 全部 3,288,036,112。冻结投影层 3,248,160。
- batch 1，预测 16 步，无 EMA，无 wandb。
- 图像增强沿用上游默认；无损实验指**原始 npz uint8 RGB 直接读入**，不经过视频编解码。
- 优化器 `optax.chain(clip_by_global_norm(1.0), adamw)`。**clip 在 Adam 之前**，
  所以 `grad_norm > 1.0` 会同比压低有效步长。

## 6. 实验链条（时间顺序）

| 实验 | 关键设置 | 结果 |
|---|---|---|
| `overfit_roundtrip_001` | 有损视频，2000 步，lr 2.5e-5 余弦 | 全 462 帧 MAE 0.011，接近段右臂 0.0464；闭环 0 开度 |
| `overfit_roundtrip_lossless_001` | 原始 RGB，同配方 | 拟合与任务均无改善；闭环 0 开度，最近 63.9 mm |
| 训练诊断 | 8 帧短程对照 | 见第 7 节 |
| `stage_fit_lossless_002` | 从无损 2000 续训，阶段采样 50/20/30，lr 恒 1e-5 | 3500 为最佳：approach 右臂 0.0397、位置 9.25 mm、姿态 2.48° |
| 闭环评估（3500） | 三噪声 | **三 seed 全败**，开度 0 |
| `stage_fit_right8_003` | 只对右臂7维+右夹爪算损失（right8），lr 1e-5，250 步检查 | **500 步触发止损，见第 8 节** |

## 7. 已证实的结论

1. **误差高度集中**（approach 0–149，三噪声均值，MuJoCo 扰动法归因）：

| 关节 | MAE rad | 敏感度 mm/rad | 占位置误差 |
|---|---:|---:|---:|
| joint2 | 0.01812 | 214.4 | 34.2% |
| joint7 | 0.01579 | 160.0 | 22.2% |
| joint1 | 0.00635 | 361.3 | 20.2% |
| joint4 | 0.00562 | 328.3 | 16.2% |
| joint3/5/6 | ≤0.00164 | 122–325 | 7.2% |
| 右夹爪 | 0.00071 | **0.0** | 0% |

   前四关节合计 92.8%。joint7 是腕部末端旋转，0.0158 rad ≈ 0.9°，直接对应 2.48° 姿态偏差。
   右夹爪对**位置**零贡献但对抓握关键，不能从损失剔除。
   来源：`results/a2_approach_joint_attribution_20261024.json`。

2. **闭环失败机制已定位**（seed 42）：
   - 跟踪误差均值仅 **4.24 mm**（机械臂跟得很好）；
   - **命令夹持点**离把手最近仍有 **27.3 mm** → 问题在模型输出，不在控制；
   - 夹持点从第 10 秒起稳定停在 Z≈467 mm，比把手（505 mm）**低 38 mm**，14 秒不再修正；
   - 接触始终是 `[n, 0]`：263 帧单侧接触、双侧 0 帧、最大力 15.46 N → 一侧指片搭在把手下沿；
   - 上一轮同口径为 −66.63 mm，本轮 −38 mm，方向对了但差一个量级。

3. **采样噪声必须三种子看**：同一帧 approach 右臂 MAE 为 0.1524 / 0.0665 / 0.0603（42/7/99）。
   单看一个种子会得出完全不同的结论。**三个种子同时变差 = 真退化；只有一个变差 = 采样波动。**

4. **32 维平均损失掩盖分组残差**：第 90 帧 all32 = 0.0619，右臂 7 维 = 0.2283，补齐 16 维 = 0.0008；
   第 0 帧夹爪 = 3.846 而 all32 仅 0.2175。`flow_squared_errors` 与原 `compute_loss`
   数值对齐检查通过（rtol 1e-5），数学实现无误，问题是用它当验收证据不充分。

5. **左臂占 12–18% 损失预算**，而其动作 MAE 仅 8.2e-10 rad。原因是 flow matching 在归一化
   空间对"恒零动作"仍需预测 O(1) 噪声（`u_t = noise - actions_norm`，`actions_norm ≈ 0`）。

6. **过拟合信号**：训练 loss 下降而采样质量下降即"背答案"。3500→4000 训练 loss 0.0037→0.0034，
   但三噪声一致恶化（arm +53%、姿态 +115%）。

## 8. 已证伪 / 失败的路线

| 路线 | 证据 |
|---|---|
| 无损图像单独重训 | 拟合与任务均无改善（flow 0.04303 vs 0.03984） |
| 关闭图像增强 | 短测右臂 0.03266→0.03222，收益 1.3%，不显著 |
| 放开投影层 | 短测 0.03460，比只训 LoRA 更差；且同时改精度与可训练性，非单因素 |
| 增加去噪步数 10→50 | 0.06658→0.07579，无效（不改起点噪声，走再细也到同一处） |
| 继续训练到 4000 步 | 全面退化 |
| **right8 损失（首轮）** | **500 步即触发止损，见下** |

### right8 首轮结果（`stage_fit_right8_003`）

| 步 | arm MAE | 位置mm | 姿态° | 夹爪MAE | objective | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 3500 起点 | 0.0397 | **9.255** | **2.48** | 0.0208 | 0.0959 | — |
| 3750 | 0.0657 | 8.239 | 4.29 | 0.0236 | 0.1193 | regressed, stalls=1 |
| 4000 | 0.0337 | **10.255** | 3.34 | 0.0263 | **0.1345** | regressed, stalls=2 → STOP |

五项指标里四项恶化：位置 +11%、姿态 +35%、夹爪 +26%、训练目标损失 +40%，仅 arm MAE 改善 15%。

**关键异常**：`grad_norm` 冲到 **2.11 / 2.69**，远超 clip 1.0。因 clip 在 Adam 之前，
有效步长被压到 37–47%。所以这一轮的失败**不能归因于"损失维度"假设本身**——它至少混合了
"梯度尺度过大被裁剪"这一因素。只训了 500 步，样本也不足。

**下一步若要重试 right8**：先把 lr 降到 2e-6~5e-6（使 grad_norm 回到 1.0 以内）或把
`clip_gradient_norm` 提到 3.0，只改一项，再观察 1000 步以上。

## 9. 当前卡点与两层问题

- 第一层：**轨迹内精度不足**。approach 首动作位置 9.25 mm，门槛 2 mm；右臂 MAE 0.0397，门槛 0.01。
- 第二层：**偏离后无纠偏**。模型在示教轨迹附近能预测，一旦偏离就按记忆输出，不会回升。

用户方案明确：只有第一层足够准、闭环仍偏离时，才转入第二层（补采偏低/偏高/偏前的纠偏示教）。
**现在不满足该前提，不应跳步。**

## 10. 已知的坑（容易误判）

1. `clipped_control_steps` 必须按关节拆。seed 42 的 462 步**全部在左夹爪**（无关）；
   seed 99 才有一个真信号（右臂 joint6 顶 0.785 上限 511 步）。
2. 空间误差分两种口径：`position_error_mm_first_action_*`（首动作，与门槛同口径）
   和 `position_error_mm_worst_step_*`（最差预测步，仅诊断）。用后者对 2 mm 门槛永远不达标。
3. 离线 MAE 与闭环无关。上一轮离线 0.011 rad 但开度 0。**判据是闭环成败，不是离线数字**。
4. 不同实验的 approach 检查帧不同（8 帧手工 / 21 帧 anchor / 本轮 0,1,2,4,8,49,99,149），
   **不同轮次的绝对值不可直接比较**，只能各自与自己的起点比。
5. 判定规则不能用"任一指标改善"（会被另一项严重退化掩盖）。现有规则含
   `REGRESSION_TOLERANCE = 0.20`：任一关键指标恶化 >20% 即记停滞。用真实数据回归测试过。
6. `best_approach_check` 只按 arm MAE 选，会掩盖位置退化（4000 就是例子）。
7. `libEGL warning: failed to create dri2 screen` 是软件渲染回退提示，不是错误。
   已用视频首帧验证：非黑像素 100%、亮度 132.9、三 seed 一致。

## 11. 关键文件索引

**文档**
- `docs/drawer-data-trial.md` — 数据采集、真实接触问题、往返轨迹
- `docs/drawer-overfit.md` — 有损视频首轮过拟合与闭环、对齐诊断
- `docs/drawer-lossless-overfit.md` — 无损图像重训
- `docs/drawer-training-diagnosis.md` — 采样敏感、损失分解、短程对照（必读）
- `docs/drawer-stage-fit.md` — 阶段加权续训、三噪声闭环、损失维度对照（必读，含全部最新数字）

**脚本**（均在 `scripts/a2_pi0/`）
- `train_drawer_stage_fit.py` — 主训练入口，支持 `--loss-dims` / `--check-interval` / `--save-every` / `--dry-run` / `--check-sampling`
- `evaluate_drawer_overfit.py` — 462 帧离线 + 三噪声闭环，`--noise-seeds 42,7,99`
- `drawer_lossless_dataset.py` — 原始 npz RGB 数据集
- `drawer_training_objective.py` — `flow_squared_errors`，含可选 `dims` 掩码
- `drawer_simulation.py` — MuJoCo 环境（FK、动作回放）
- `diagnose_drawer_alignment.py` — 夹持点 FK 诊断、图像消融
- `audit_drawer_training.py` — 短程对照实验

**测试**（`tests/`，纯 CPU，无 pytest，直接 `python` 运行）
- `test_drawer_stage_judge.py` — 停滞/回退判定 8 项
- `test_drawer_loss_dims.py` — 损失掩码归属 4 项

**检查点**（`checkpoints/pi0_openarm_drawer_lora_lossless/`）
- `overfit_roundtrip_lossless_001/2000` — 无损基线，2000 是历史续训起点；1000已清理
- `stage_fit_lossless_002/3500` — **第一轮最佳**（approach 臂 0.0397 / 位置 9.25 / 姿态 2.48）
- `stage_fit_right8_003/4000` — 已止损，权重已于2026-10-24清理，诊断日志仍保留
- `stage_fit_right8_lr3e6_004/4500` — 低学习率right8对照；4000已清理
- `stage_fit_right8_key147_005/4500` — 关节加权候选，待三种子空间评估；4000已清理

2026-10-24磁盘清理共删除12个检查点，释放约62.14 GiB，保留6个。
另保留旧抓方块 `pi0_openarm_lora/overfit_ep0_5000_001/5000` 与有损抽屉
`pi0_openarm_drawer_lora/overfit_roundtrip_001/2000` 作为历史对照。
完整清单及被删检查点的配置归档见 `results/checkpoint_cleanup_20261024/`；归档不含权重，
旧文档引用已删除路径的命令不能直接运行。数据、基础权重、日志和评估报告均未清理。

**结果**（`results/`）
- `a2_drawer_training_audit_20261024.json` — 采样/损失/短程对照
- `a2_drawer_lossless_overfit_001.json` — 无损训练 + 闭环
- `a2_approach_joint_attribution_20261024.json` — 关节级归因

## 12. 未决问题

1. right8 是否有效：需在 `grad_norm ≤ 1.0` 的条件下重试（降 lr 或提 clip）。
2. 第 2 步（按 joint2/joint7/joint1/joint4 加权）与第 3 步（LoRA 秩 32→64）均未开始。
3. 2 mm 门槛是否过严：物理上夹住 14 mm 把手可能需要 5 mm 级即可，但尚无实测依据。
4. 纠偏示教何时启动：前提是第一层达标，当前未达标。
5. 单轨迹无验证集，所有结论仅限训练轨迹拟合，不能称泛化。

最终检查点独立重载已完成：10步native平均1.462mm、最坏3.589mm、最坏姿态2.312°；
全BF16平均1.449mm、最坏3.531mm、最坏姿态2.332°，均未通过。
记录`results/start9_delta_refine_005/reload.json`。本轮未运行新短闭环、抓握或往返。


2026-10-24 晚续：图像增强/首4步权重的三组配对实验已完成（006/007/008，各300步、lr3e-6）。
三组同源005/1200，固定相同采样顺序；关闭增强将平均/最坏位置1.470/3.868降为0.971/3.093mm，
加权首4步后0.929/2.971mm。独立重载008：native平均0.924、最坏2.996mm、姿态1.521°；
BF16平均0.926、最坏3.064mm、姿态1.518°。均未通过最坏<2mm门槛，未执行新物理闭环。
完整记录`results/start9_aug_objective_20261024/README.md`，新增训练选项`--no-augmentation`、`--first-four-weight`。
训练数据/统计不改，loss窗口每100步刷新；新增3份5.2GiB检查点，目前都保留供复查。


后续只读去噪诊断已完成：008/300超标集中在ID12第7/8帧seed42（约2.24/3.02mm），
记录器与原始sample_actions同进程输出最大差0。逐关节FK消融表明贡献有抵消，
不能直接将角度误差最大关节当作末端误差最大来源。正确标签插值轨迹上仍存在后段速度偏差；
低噪声时间采样加密是下一项待验证假设，尚未训练。完整证据及限制见
`results/start9_denoising_bridge_20261024/README.md`；未运行新物理闭环。


2026-10-24 低噪声时间采样对照009/010已完成：同源008/300，各300步、lr3e-6、
关闭增强、首4步权重4。原Beta组train平均/最坏位置0.835/2.764mm；
50%Beta+50%Uniform(.05,.3)组0.746/2.484mm。独立重载混合组native平均0.739、
最坏2.448mm、姿态1.401°；BF16平均0.744、最坏2.489mm、姿态1.406°。
均未通过最坏<2mm门槛，没有运行新闭环。实际采样t已逐步记录并检查，
完整协议、曲线及限制见`results/start9_time_sampling_20261024/README.md`。
新增`--time-sampling beta/low_mix`，默认Beta逐元素等于旧时间采样已验证。
