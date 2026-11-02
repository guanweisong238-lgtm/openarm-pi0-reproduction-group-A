# 项目交接：当前有效状态

更新：2026-11-02。本文件只记录当前状态，历史过程见[优化复盘](../optimization_log.md)与[清理前交接快照](archive/project-handoff-before-cleanup-20261025.md)。历史文件中的“当前”、待执行计划及权重路径不作为现状。

## 已完成与限制

当前推荐模型为`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000`，是π0骨干+LoRA+三IO投影的一步确定性改造版。原多步flow版本尚未通过。

- 固定33起点全部通过：train 10/10、validation 3/3、eval_only 20/20；原独立评分函数重放33/33通过，位置、速度、接触力误差全部0。
- 冻结020/2000后的全新100起点评估：原范围随机组合66/70（94.3%），扩大范围18/30（60.0%），合计84/100。100条独立重放有效且位置/速度/接触力误差为0，权重前后指纹一致。16条失败保留，主要未满足持续双指有效抓取；没有针对本轮评估继续训练或挑选权重。起点集`data/processed/openarm_drawer_eval_v002/`继续禁止进入训练/专家标签生成，详见[完整新测试报告](../results/drawer_generalization_020_20261102/README.md)。
- 原生MuJoCo展示ID12已实测通过：139步、4.63秒仿真、峰值102.44毫米，录像35帧完整解码。
- 只验证拉开保持，未验证关闭/完整往返，不宣称广泛泛化或稳定成功率。
- 原13条成功示教保持10 train/3 validation，原数据与统计未覆盖。020使用4767个相关查询（4192专家反馈+575条旧策略train预测保留标签），覆盖原10个train父起点和独立随机训练场景；验证/评估状态未进训练或标签生成。
- 模型输入为当前三路RGB、实际位置16维与速度16维；左状态屏蔽。右7残差、夹爪绝对目标，左臂保持初始目标。
- 运行中没有专家动作/参考轨迹查询、中途状态注入或抓取weld。推理期间仿真暂停，仿真秒数不是硬件实时性能。
- 评分要求双指各>.01N连续5步、接触拉动、开度>=80毫米保持30控制步；已记录成功的独立动作重放位置、速度、接触力最大差0。

完整证据：[33/33验收报告](../results/drawer_eval_020_20261102/README.md)、[独立重放](../results/drawer_eval_020_20261102/independent_replay_020_2000.json)、[原生展示评分](../results/drawer_eval_020_20261102/native_demo_score.json)。历史016证据仍保留。

## 直接展示

在项目根目录运行原生MuJoCo窗口：

```bash
.venv/bin/python scripts/a2_pi0/live_drawer_success.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000 \
  --task open --seconds 24 --episode-ids 12 \
  --seed 42 --noise-mode fixed --action-chunk 4 --gripper-mode continuous \
  --weight-precision checkpoint
```

空格暂停、R重新运行、Q退出；完成后保留窗口，自动退出加`--auto-next`。固定code42/H4/一步采样是训练条件，不要任意更换噪声种子、重采样或夹爪解码。slew_binary对照失败，未采用。

录像回放：

```bash
/usr/bin/python3 scripts/a2_pi0/show_closed_loop_video.py \
  --video results/drawer_eval_020_20261102/native_demo_id12.mp4
```

任务完成时立即释放VideoWriter写入MP4索引，避免窗口保留时视频无法播放；本次020视频35帧已完整解码验证。

## 训练与关键资产

训练入口`scripts/a2_pi0/train_drawer_feedback_endpoint.py`。020从016/2000训练6000步课程，保存2000/4000/6000步，按闭环验收选中2000步。lr1e-5、warmup100、cosine至1e-6；更新50,087,968参数，无图像增强；cached固定noise42/t1、窗口4，夹爪权重4、局部线性夹持点位置权重0.5，旧策略保留查询采样权重4。原始精度加载已修复并验证。

| 内容 | 路径 |
| --- | --- |
| 原13条成功数据清单 | `data/processed/openarm_drawer_success_v003/manifest.json` |
| 当前混合监督查询 | `data/drawer_feedback_queries/v005_randomized_rehearsal/` |
| 原始人工/试采/恢复数据 | `data/drawer_human/`、`data/drawer_trial/`、`data/drawer_recovery/` |
| 速度统计 | `assets/pi0_openarm_drawer_velocity_v001/openarm_drawer_success_v003_right7_delta_velocity/` |
| 原始预处理缓存（2026-11-02已清理，可重建） | `.cache/drawer_success_inputs/45865b20503a36b52717/`；清理元数据见`results/project_cleanup_20261102/` |
| 速度缓存 | `.cache/drawer_velocity_v001/6a20f3a46f2c84bbfacab/` |
| 当前训练日志、指标、源码快照 | `logs/a2_pi0/drawer_feedback_20261102-004233-380521/` |
| 当前模型、源模型与基线 | [检查点清单](../checkpoints/README.md) |

缓存身份包含部分源码，修改`drawer_success_training_config.py`、`openarm_policy.py`、`drawer_success_cache.py`可能触发约4.6GiB缓存重建。速度资产身份还依赖`prepare_drawer_velocity.py`与`DrawerSimulation`源码，修改需显式版本化，不能覆盖旧统计。

训练时协议`input_protocol.json`保留原快照；后验验证见检查点的`validation_report.json`和`deployment_config.json`，不要重写历史训练状态。

## 环境

RTX5060Ti 16GB，主训练/推理使用`.venv/bin/python`。系统`/usr/bin/python3`有Matplotlib/Tk，用于原生曲线及回放；项目venv有JAX/MuJoCo，不要混用依赖。GPU与桌面需要宿主访问；工具沙箱不可用不等于主机不可用。

## 评估体系（2026-11-01 建立）

抽屉任务已从“单次演示成功”转入“可统计评估”阶段。判据、运行条件与指标全部冻结在 [评估协议](drawer-eval-protocol.md)，操作步骤见 [评估运行手册](eval-runbook.md)。

| 能力 | 入口 | 状态 |
| --- | --- | --- |
| 无窗口批量闭环 | `scripts/a2_pi0/run_drawer_eval_batch.py` | 已验证（单进程多 trial） |
| 逐 trial 进程隔离 | `scripts/a2_pi0/run_drawer_eval_isolated.py` | 冒烟通过（ID 100/101） |
| eval-only 起点生成 | `scripts/a2_pi0/build_drawer_eval_starts.py` | 已产出 20 条（ID 100–119） |
| 自检判定 | `scripts/a2_pi0/check_drawer_selfcheck.py` | ID2 `--repeat 2` 两次一致，PASS |
| 分组出表 | `scripts/a2_pi0/summarize_drawer_eval.py` | 格式已验证 |
| 后端像素对比 | `scripts/a2_pi0/compare_render_backends.py` | 已验证 |

**016历史工具校准的四条口径结论**（实测条件以当时记录为准；020保留配置与精度见修复协议）：

1. **渲染后端是固定实验条件**：必须 `MUJOCO_GL=glfw`。改用 `egl` 时 ID12 由成功翻为失败（73.42 mm）、ID6 由失败翻为成功（87.99 mm）。
2. **不能用「复现历史单次数字」做校准标准**：批量无窗口实现与 `live_drawer_success.py` 的首个动作块必然差 6e-4 量级（ID2 实测 6.47e-4），而接触任务下该差异足以翻转成败（差 6.47e-4 失败、差 1.29e-3 反而成功）。校准标准是「同实现重复运行一致」。
3. **结果不得依赖批次构成**：预热必须用与场景无关的固定输入，跨 trial 必须进程隔离。实测同一 ID2 在 `--episode-ids 12 2 6` 下为 20.20 mm、在 `--episode-ids 2 --repeat 2` 下为 -0.000000 mm。
4. **仿真时间 ≠ 实时性能**：推理期仿真暂停，延迟按墙钟 p50/p95 另记（当前 RTX 5060 Ti 上约 p50 143 ms / p95 148 ms）。

## 下一步

33条全量隔离评估已在宿主完成，train 6/10、validation 0/3、eval_only 7/20，见[结果表](../logs/a2_pi0/drawer_eval_isolated_full_20261101/metrics.md)。20条失败中17条未接触或未形成稳定双指夹持。

已完成 train ID9/10/11 的失败动作精确重放、几何诊断和定向恢复补采：10条带图像专家尝试中8条通过严格判据，失败保留。新版 `data/drawer_feedback_queries/v002_targeted_20261101/` 含原1200查询和新增568查询，共1768，训练入口 `--prepare-only` 已通过。原数据、统计与016权重未覆盖，validation/eval_only 未加入训练。详见[本次报告](../results/drawer_train_failure_diagnosis_20261101/README.md)。

017已完成2000步训练，但同配置全量评估0/33，原成功能力退步。用户随后授权自主修复，直到同一权重通过全部33个起点；验证/评估起点继续隔离，成功判据保持不变。017失败证据见 `logs/a2_pi0/drawer_eval_017_full_20261101/metrics.md`。

修复进展（2026-11-02）：

- 新增与部署相同的KV-cache端点训练路径，初态及后续探针数值一致；018路径对照500步在train ID9通过，但ID12/10/11未通过。
- 纠正新旧标签动作幅度差异，生成 `v003_feedback_repair` 2400查询；019/1000和/2000均未通过ID12/10筛查。
- 发现上游部署加载器将50,087,968个训练时float32参数统一舍入为bfloat16。已提供原始精度加载器并记录运行精度；017、019的原始精度对照仍未通过，说明精度不是唯一原因。旧016/017的bfloat16历史结果不覆盖。
- 新数据 `data/drawer_feedback_queries/v005_randomized_rehearsal/` 共4767查询：4192专家反馈、575条016成功训练轨迹上的策略输出保留标签。覆盖全部10个原train父起点及独立连续随机场景，含明确的混合监督来源；不含validation/eval_only监督。
- 020已完成6000步课程；第一份2000步候选通过4个训练起点筛查后，完整33/33及独立重放33/33均通过，未继续评估4000/6000步。原生展示ID12也通过。训练日志 `logs/a2_pi0/drawer_feedback_20261102-004233-380521/`，最终调度状态 `results/drawer_repair_20261101/round020.json`。

当前33/33目标已达成，推荐020/2000；016作为历史基线保留。正式证据见[验收报告](../results/drawer_eval_020_20261102/README.md)，修复口径见[协议](drawer-repair-eval-protocol.md)。原训练 `input_protocol.json` 保持快照，后验33/33结果另存检查点的 `validation_report.json` 和 `deployment_config.json`。

技术债与边界不变：原多步 flow 未复现（当前是一步确定性改造版），关闭往返与真机尚未验收，A3目录仍为空模板。如目标是原论文复现，需单独解决原多步 flow 目标与采样，不能将当前改造结果替代。

2026-11-02后续抓取修复已完成：磁盘清理释放9.70GiB，随后完成失败诊断、30个独立训练场景的反馈查询及021/022两轮训练。021/2000与/4000的原train四起点筛查分别2/4、0/4；加入020全10个train起点的325条成功预测保留标签后，022两份权重均为4/4，但新筛选16仅11/16、13/16，低于020的15/16。因此本轮未升级推荐模型。新增查询`v008_expanded_contact_rehearsal020`共7972条，属于未推荐候选的训练数据，不能写成020已验收模型的训练集。保留的同一020在训练前锁定的最终新100起点400—499上得到85/100（near66/70、expanded19/30），独立重放100/100有效，位置/速度/接触力最大误差0，权重前后指纹一致。与上一批84/100的差异不是训练提升，见[本轮正式报告](../results/drawer_grasp_improvement_20261102/README.md)。新候选未执行完整33回归，4/4不代表33/33。

小型MLP反馈基线也有成功记录，但它不是π0结果。所有失败日志和评分保留，不删除失败证据。

## 2026-10-25清理结果

当时保留4份完整检查点：016成功模型、015重训源、011局部起步模型、原13条训练基线。020验收时增至11份，本轮021/022再新增4份，曾为15份；晚间清理8份已失败候选后，目前保留7份完整检查点，见最新清单。当时删除11份被替代的历史权重与可重建恢复预处理缓存，释放约57.26GiB。原始数据、底座权重、环境、源码、日志与评估均保留。归档配置不含权重备份，已删除实验需重训。2026-11-02另清理重复ZIP与可重建原始预处理缓存，释放9.70GiB，见[新清理报告](../results/project_cleanup_20261102/README.md)。

清单：[本次清理报告](../results/project_cleanup_20261025_103927/README.md)。其他文档按[文档索引](README.md)阅读。

2026-11-02晚间按用户要求再次清理，删除017/018/019/021/022共8份失败候选权重，释放约41.43GiB，可用空间约59.13GiB。推荐020逐文件SHA256一致，全部数据及评估证据保留。被删候选仅留配置和元数据，重新推理需重训。见[晚间清理报告](../results/project_cleanup_failed_candidates_20261102/README.md)。
