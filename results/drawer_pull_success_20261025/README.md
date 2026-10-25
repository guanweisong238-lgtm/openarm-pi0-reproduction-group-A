# 抽屉拉取：已验证的Pi0骨干一步确定性模型

## 已完成什么

检查点：`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000`。
它是Pi0骨干+LoRA+三IO投影的一步确定性改造版，原多步flow版本尚未通过。
模型独立读取实际三RGB、关节位置/速度并输出右臂/夹爪动作，左臂保持初始目标；
运行时没有专家轨迹查询、专家动作替代或中途物理状态灌入，也没有抓取weld。

|起点|仿真时间|峰值开度mm|真实双指接触+拉动+80mm保持1s|
|---|---:|---:|---|
|原始train ID12，首次|4.67s|101.15|通过|
|同模型跨进程复测ID12|15.63s|101.64|通过|
|validation ID2，柜体Y偏移5mm|4.70s|90.78|通过|
|validation ID6，human warm-start|24s|87.78|未通过：未保持80mm达1s，最终76.94mm|

这些是固定柜体、小幅位置变化的局部验证，不是广泛场景泛化率。
两个ID12实际轨迹完成时间不同，差异没有单独定位；固定噪声不能替代端到端实际记录。
每条记录均独立从完整初态只重放实际动作，逐步关节位置、速度、接触力最大差均0。
全部保留成功和失败记录，见physical_scores.json；warm-start没有因峰值越过80mm而改判成功。

## 直接运行

在项目根目录执行，原生MuJoCo窗口，不使用网页：

```bash
.venv/bin/python scripts/a2_pi0/live_drawer_success.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000 \
  --task open --seconds 24 --episode-ids 12 \
  --seed 42 --noise-mode fixed --action-chunk 4 --gripper-mode continuous --auto-next
```

保留`--noise-mode fixed --seed 42`，这是训练使用的固定查询噪声码，不是可随意更换的随机采样策略。
动作窗口4步，每4个30Hz控制步重新观测，推理期间仿真暂停；仿真耗时不等同真实硬件实时性能。
默认夹爪模式continuous为上表配置。后续slew_binary仅为独立控制对照，不能混入本表。
本轮只验收拉开保持，不代表已完成关闭/往返或原论文完整复现。

## 训练安排与可复查材料

原13条示教、10/3划分及已有统计均保留。源015/2000，实际位置+速度填入原32维槽，
右7残差动作/绝对夹爪，左位置/速度屏蔽；LoRA+三IO共50,087,968参数更新。
输入速度来自实际qvel，与图像/位置同一控制时刻；速度统计仅原10条train，独立资产不覆盖旧版。

`data/drawer_feedback_queries/v001/`含1200个监督查询，来源为train ID12的成功反馈专家
和真实失败策略访问状态，加小幅位置/速度扰动；75个穿透/接触不合格候选被拒绝。
每个查询都从完整状态真实执行专家16步并逐条独立重放一致，模型使用前4步标签；
输入只取查询起点，不包含专家进度、未来状态或抽屉真值。查询不是1200条成功长示教。
专家仅在离线生成标签时使用固定参考路径，模型部署不访问它。

2000步，lr1e-5、warmup100、cosine到1e-6、fresh optimizer，无图像增强；
固定噪声42、t=1，只监督right8动作，预测/执行窗口4，一步推理。
1080个查询训练、120个查询留出，属于同父场景扰动插值留出，不是独立任务泛化测试。
固定留出动作MSE从1.149降至0.07245，首动作目标位置平均误差8.73→4.83mm。
这个口径与旧成功轨迹稀疏anchors不同，不能直接拿数值跨实验排名。

训练日志及运行时源码快照：`logs/a2_pi0/drawer_feedback_20261025-031627-720249/`。
每100步更新的原生loss最终图：training_loss.png/svg。训练入口train_drawer_feedback_endpoint.py。
闭环视频：pi0_endpoint_original_start.mp4；原始动作/状态/力在对应logs/a2_pi0/success_live目录。

## 教学解释

离线低误差没有保证闭环：原成功轨迹主要覆盖正常路径，策略自己的小偏差会改变后续输入。
本轮为这些状态生成实际对应的纠偏标签，保留当前速度，并让一步策略学习短窗口反馈动作。
同一批组合调整最终实现了实际拉取；没有做完所有单变量因果消融，不能宣称某一项是唯一原因。

另有约7万参数的MLP关节目标反馈基线，冻结权重后在ID12、ID2、ID6均通过。
它使用通用位置/速度稳定控制和模型预测的夹爪闭合意图，明确不属于Pi0结果。
这帮助确认物理任务和监督控制可行，不能替代对Pi0自身的验收。

后续扩大目标时，优先修复Pi0 warm-start下的夹持/保持并补充相应train恢复状态，
再扩大柜体/把手/初始姿态变化；验证起点保持隔离。

## 额外夹爪控制对照（未采用）

冻结相同权重，尝试slew_binary解码，仅测试train ID12/8：
ID12峰值31.82mm未形成持续双指夹持；ID8无开度，两轮均失败。
因此部署配置仍continuous，不把MLP基线的夹爪调法直接迁移当作Pi0成功。
记录logs/a2_pi0/success_live_20261025-034957-737591/，全部失败证据保留。
checkpoint的input_protocol.json为训练时快照，后验验收另见validation_report.json；
deployment_config.json固定了上表实际通过的配置。
