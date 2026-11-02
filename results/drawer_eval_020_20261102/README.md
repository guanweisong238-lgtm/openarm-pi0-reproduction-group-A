# 020/2000：固定33起点全部通过

2026-11-02。选定检查点：`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000`。同一份冻结权重在完整33起点评估中全部通过，另用原独立评分函数重新重放33条，全部通过。

| 分组 | 016历史批量基线 | 本次020/2000 |
| --- | ---: | ---: |
| train | 6/10 | **10/10** |
| validation | 0/3 | **3/3** |
| eval_only | 7/20 | **20/20** |
| 合计 | 13/33 | **33/33** |

016历史基线使用上游统一bfloat16加载；020保留检查点参数原始精度。因此本表比较整体方案，不是训练方法的单因素消融。

## 实际验收与证据

- [逐条结果和分组表](metrics.md)、[机器可读指标](metrics.json)、[33条索引](index.json)。原始动作、状态、接触力及源码快照在 `logs/a2_pi0/repair_screen_020_2000/full_33/`。
- 每条均满足既定判据：双指各>0.01N连续5步、有接触拉动事件、开度≥80mm保持30控制步。判据没有放宽。
- [独立重放33条](independent_replay_020_2000.json)：位置、速度、接触力最大差全部0，接触计数逐步相等。原评分函数在审计进程中使用两份冻结清单的合并副本；原代码和试验均未修改。
- [完成证据审计](completion_evidence_audit_020_2000.json)：33个ID唯一且完整、每条均同一检查点、没有专家动作或中途状态注入、固定噪声42、运行源码与冻结快照一致。训练筛查的ID9/10/11/12与完整评估中的重复运行物理轨迹逐项相等。
- [数据隔离审计](completion_data_audit_020_2000.json)：实际前2000次梯度更新只使用原10个train父起点，未用validation ID2/3/6或eval_only ID100–119，也未用本阶段查询留出索引。
- [冻结权重清单](frozen_checkpoint_020_2000.json)：21份推理相关文件、5,262,189,356字节，均早于全量评估且前后SHA256和修改时间一致。

峰值开度范围96.18–102.46mm，成功仿真耗时中位数4.73秒。ID112（较粗把手）用了15.57秒，其余起点最长6.03秒；全部在24秒上限内。不能把通过解释为每条都同样流畅。

合并全部1241次实际推理记录，墙钟延迟p50=138.40ms、p95=139.51ms。`metrics.md`的分组延迟列沿用原汇总器，表示各试次分位数的组中位数。推理期间仿真暂停，仿真耗时不是硬件实时性能。

## 模型与训练

模型仍是π0骨干+LoRA+三IO投影的一步确定性改造版；输入当前三RGB、实际关节位置/速度，左状态屏蔽，右7位置残差及夹爪绝对目标，左臂保持初始目标。运行时没有专家、参考路径查询或抓取weld。

从016重新训练一轮6000步，保留2000/4000/6000步；按实际闭环筛查选择2000步，后两份没有再用于完整评估。cached训练路径与部署一致，lr1e-5、warmup100、cosine至1e-6，无图像增强，夹爪损失权重4、局部线性夹持点位置损失权重0.5。训练与数据修复过程见[过程报告](../drawer_repair_20261101/README.md)。

训练集 `data/drawer_feedback_queries/v005_randomized_rehearsal/` 共4767查询：4192专家反馈、575条016成功train轨迹上记录的模型预测（旧能力保留标签，采样权重4）。覆盖原10个train父起点及22个独立连续随机训练场景。数据、统计和原训练/验证划分未覆盖。查询之间相关，不能当作4767条独立成功轨迹；随机查询留出也不是独立场景测试。

同时修复了上游加载器统一舍入float32训练参数的问题。需要 `weight_precision=checkpoint`；新权重已在训练协议中推荐该模式，默认auto可识别。历史检查点仍保留原有加载行为。多个改动联合取得结果，尚未隔离每个因素的因果贡献。

## 运行与展示

项目根目录、GPU及桌面显示可用的终端：

```bash
.venv/bin/python scripts/a2_pi0/live_drawer_success.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000 \
  --task open --seconds 24 --episode-ids 12 \
  --seed 42 --noise-mode fixed --action-chunk 4 \
  --gripper-mode continuous --weight-precision checkpoint
```

空格暂停、R重新运行、Q退出；自动退出加 `--auto-next`。此命令已实测：原生窗口ID12成功，139步、4.63秒仿真、峰值102.44mm；[实际模型录像](native_demo_id12.mp4)（35帧完整解码）、[独立评分](native_demo_score.json)。交互式实现与批量实现的轨迹不要求逐比特相同。

全量复测须使用新输出目录：

```bash
.venv/bin/python scripts/a2_pi0/run_drawer_eval_isolated.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000 \
  --weight-precision checkpoint \
  --output logs/a2_pi0/drawer_eval_020_reproduce_001
```

运行口径见[修复评估协议](../../docs/drawer-repair-eval-protocol.md)。不要切换EGL或噪声种子。

## 结果边界

这是固定33个起点、固定配置下的33/33观察。验证/评估状态未进入训练或专家标签生成，但多轮用同一基准挑选模型属于开发集模型选择，不能宣称新独立测试集100%泛化。原多步flow、关闭往返、真机以及实时部署尚未完成。
