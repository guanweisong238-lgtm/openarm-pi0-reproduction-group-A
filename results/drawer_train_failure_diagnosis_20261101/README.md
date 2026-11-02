# ID9/10/11 失败定位与纠正监督

2026-11-01。完成失败诊断和补数据；尚未训练 017，不能把专家恢复通过记成模型性能提高。

## 016 全量评估基线

33 个不同起点：train 6/10，validation 0/3，eval_only 7/20。20 条失败中 single_finger 11、no_contact 6、grasp_slip 2、hold_fail 1。固定 glfw、code42、H4、continuous、24 秒；[完整表](../../logs/a2_pi0/drawer_eval_isolated_full_20261101/metrics.md)。历史 live 单次成功数字不与本批次混算。

## 定位结果

仅分析 train ID9/10/11；三者都是人工 warm-start。逐步只重放实际下发动作，qpos/qvel 与原评估日志最大差均为 0。完整物理快照来自这一重放，包含积分状态、eq_data、动作前缀和真实控制步偏移。

| 起点 | 首次任一指接触（控制步） | 双指有效接触步数 | 最近夹持点距把手中心 | 最近时夹持点相对把手 XYZ | 右关节5目标被限幅步数 |
| --- | ---: | ---: | ---: | --- | ---: |
| ID9 | 42 | 0 | 13.61 mm | (-9.01, -10.20, 0.32) mm | 178 |
| ID10 | 无 | 0 | 22.98 mm | (-18.18, -13.96, -1.55) mm | 48 |
| ID11 | 271 | 0 | 13.89 mm | (-13.35, 0.46, 3.81) mm | 177 |

夹持点定义与历史诊断一致：right_ee_control_point + R·(0,0,-0.16)，不是任一指片的表面距离。最近距离不能替代接触判定。原成功示教在有效双指接触时的夹持点相对把手 X 中位数为 +15.00/+10.39/+9.66 mm；失败策略最近时仍处于负 X，且 ID9/10 有明显 Y 偏差。ID9 在第30步靠近后，第60步又退到约61.94 mm，夹爪目标同时由 -0.206 变到 -0.075，说明至少该段存在接近/闭合配合不当。ID10 第60步仍距把手70.32 mm；ID11 为71.78 mm。

三者的关节限幅都集中于右关节5，最大超限约0.0112–0.0136 rad；这是一条额外诊断线索，尚无消融证据证明它是失败主因。当前证据支持“接近及夹持未完成”，不能唯一归因于视觉、闭合时机或某个关节。几何比较也不等于已定位神经网络首次数值偏离的因果来源。

详见 [diagnosis.json](diagnosis.json) 和每个 episode 子目录中的几何数组、完整物理快照。

## 已采集的数据

原始数据：[targeted_20261101](../../data/drawer_recovery/targeted_20261101/manifest.json)。从各训练起点的初态、执行30步后的实际偏离态、最接近把手的失败态分别规划专家恢复。

- 9 个快照先用显式15°恢复姿态试采：7条通过、2条失败。
- ID11 初态改用12°，先数值试采再采图像，通过；最终10条带图像尝试中8条接纳，2条失败保留。
- 合格轨迹来自 train ID9/10/11，分别2/3/3条，共2376个控制步。全部有真实三RGB、实际位置/速度及专家动作。
- 图像通常每4步采集，`image_frame_indices` 显式对应全速物理记录；首条 ID9/30 试采保留逐步图像。稀疏图像须用本次专用准备脚本，不能按旧版逐帧图像缓存读取。
- 8条全部满足真实双指接触、接触拉动和≥80mm持续30步；峰值83.65–83.97mm。成功按过程事件判定，并不保证额外末尾等待时仍高于80mm。
- 独立动作重放的位置、速度、接触力最大误差均为0，接触计数逐步相等；无抓取weld、无中途状态注入。初态加载只用于离线生成监督。
- 额外重放审计中全部查询及未来16步标签的接触穿透均小于4mm，所有合格轨迹最深约0.675mm；见原始数据目录的 `penetration_audit.json`。
- 失败的 ID9/607 与 ID11/0（15°）只拉到约48.5/48.7mm，不进入训练清单。失败日志和轨迹均保留。

[验证记录](../../data/drawer_recovery/targeted_20261101/validation_report.json)。这些是相关的专家分支恢复数据，不是8个独立泛化场景，更不是016模型自主成功记录。

## 可直接读取的新版查询数据

`data/drawer_feedback_queries/v002_targeted_20261101/`：保留原1200查询，加入568个纠正查询，共1768。每条新增查询来自合格恢复轨迹的真实观测，保存未来16步专家动作，模型输入只监督前4步；不填充缺失的未来动作。

沿用原归一化统计、32维位置/速度输入、归一化后屏蔽左状态、右7位置残差及夹爪绝对目标。新增查询逐条验证动作反归一化和速度编码。原数据和原统计均未覆盖；validation ID2/3/6、eval_only ID100–119 没有用于补采或标签生成。

## 复现与边界

```bash
.venv/bin/python scripts/a2_pi0/diagnose_drawer_train_failures.py \
  --run logs/a2_pi0/drawer_eval_isolated_full_20261101 --output results/<新诊断目录>
MUJOCO_GL=glfw .venv/bin/python scripts/a2_pi0/collect_targeted_drawer_recovery.py \
  --diagnosis results/drawer_train_failure_diagnosis_20261101/diagnosis.json \
  --output data/drawer_recovery/<新目录>
```

批量采集默认15°，额外的ID11初态12°版本需用 `collect_drawer_recovery.py --pitch-degrees 12 --image-stride 4` 单独采集并验证，再向批量采集命令传 `--extra-recovery <该恢复目录>` 纳入清单；原失败版本保留。首次ID9/30逐帧图像试采未保存启动源码快照（报告中已明确标记），后续采集均保存 `collector_source.py`。

```bash
.venv/bin/python scripts/a2_pi0/prepare_targeted_drawer_queries.py \
  --manifest data/drawer_recovery/targeted_20261101/manifest.json \
  --output data/drawer_feedback_queries/<新版本>
.venv/bin/python scripts/a2_pi0/train_drawer_feedback_endpoint.py \
  --data data/drawer_feedback_queries/v002_targeted_20261101 \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000 \
  --prepare-only
```

本次只执行准备校验，不启动训练。训练入口已改为从实际训练查询读取监督来源9/10/11/12，并按数据目录记录新版身份；不再写死 `[12]`。随机查询留出仍是同父场景插值检查，不是独立场景测试。是否提高成功率须在训练后按同一评估协议重新测量；测试集不得并入训练。
