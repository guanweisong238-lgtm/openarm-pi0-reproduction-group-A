# 起步采样的配对训练对照

问题：success13_lora_001/2000在train和validation起点都出现早期轨迹偏离。
前两次诊断见`results/live_start_audit_20261024/`和`results/train_start_control_20261024/`。

本轮只检验增加最初75帧观测采样的效果，不改变图像、状态、动作表示、损失维度或数据划分。
最初75帧是固定2.5秒的起步窗口，不等同于每条人工轨迹完整的接近阶段。

两组均从`success13_lora_001/2000`完整权重开始，优化器重新初始化；
各500步，batch1，all32 flow损失，峰值3e-6、warmup50、cosine末值3e-7。
checkpoint步数500表示本轮更新数，来源2000另存input_protocol.json。
随机种子42，两组逐步选中的train轨迹相同。

- `success13_uniform500_control_002`：均匀抽轨迹、均匀抽帧，沿用旧版随机流。
- `success13_early60_500_003`：60%概率强制从轨迹前75帧抽样，40%从全轨迹抽样。
  后者也可能落入前75帧，因此总早期比例超过60%。额外选择使用独立随机流。

实际预定500步：control前15帧7次/前75帧59次；early60前15帧59次/前75帧329次。
两组轨迹抽样计数完全一致，为[52,49,45,34,62,54,45,56,45,58]。
抽样次数只用于检查干预发生，不是因果结论。

评估：保留每轨迹原四个时间均匀帧，另并入0、1、4、8、15、30、45、60、74帧。
共120 train/36 validation anchors，每个均使用seed42/7/99。
分别报告原四帧、早期帧、首帧，按ready/human初始化分组。
扩充后的总平均不能直接与旧四帧均值相比；使用legacy_anchor/all保持旧口径。
所有验证均为固定观测离线评估，无梯度、无验证集统计重算。

```bash
.venv/bin/python scripts/a2_pi0/train_drawer_success.py \
  --checkpoint checkpoints/pi0_openarm_drawer_success_v003_lora/success13_lora_001/2000 \
  --exp-name success13_uniform500_control_002 --steps 500 --lr 3e-6 \
  --eval-every 500 --save-every 500 --dense-approach-eval --early-probability 0

.venv/bin/python scripts/a2_pi0/train_drawer_success.py \
  --checkpoint checkpoints/pi0_openarm_drawer_success_v003_lora/success13_lora_001/2000 \
  --exp-name success13_early60_500_003 --steps 500 --lr 3e-6 \
  --eval-every 500 --save-every 500 --dense-approach-eval --early-probability 0.6
```

日志独立保存在各自RUN_DIRECTORY，原生loss窗口每100步刷新。
离线对比及图像生成入口`compare_early_sampling.py`，最终结果保存至
`results/early_sampling_control_20261024/`。
随后使用相同train ID12/8、seed42、每次执行4步、每条12秒的真实闭环比较。
12秒是早期行为诊断，不能代替完整任务、多种子成功率测试。
若离线改善而短闭环仍失败，应如实报告；不能仅凭loss宣布修复。

## 执行结果（2026-10-24，已完成）

两组各500次更新正常，原生loss每100步刷新。early60相对uniform的验证ready首帧位置
误差45.72→40.31mm，human首帧14.29→10.51mm，原四帧总体17.22→15.69mm。
同一初始权重两次评估有最多0.962mm的分组位置均值差异，原因未确认，详表同时保留两次训练前值。
四次12秒闭环均无把手接触、无拉开。加权模型ready执行4步偏离同刻示教92.77mm，
与原2000模型93.29mm接近；human为14.32mm但第30步已偏离210.90mm。
本轮支持离线预测改善，未解决早期闭环偏离。下一步建议首0～8帧单起点拟合诊断。
完整报告、曲线和数值：[实验报告](../results/early_sampling_control_20261024/README.md)。
