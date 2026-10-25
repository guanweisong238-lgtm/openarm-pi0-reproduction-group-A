# 低噪声时间采样配对实验

## 配置与验证

两组从008/300同源权重开始，各300步、lr3e-6、每50步固定检查，fresh optimizer。关闭图像增强、首4步权重4、LoRA及三IO层更新、train ID12首9帧、validation ID3帧0/4/8、采样顺序/统计/三个推理种子完全一致。以上协议字段已自动比较。原始数据、统计与验证划分不改。

对照t=.999*Beta(1.5,1)+.001；实验为50%原分布+50%Uniform(.05,.3)。噪声和原始Beta随机键不变，选择/均匀分量用fold_in独立键。默认采样逐元素等于旧实现已在CPU验证；10万样本P(t<.3)为16.385%与58.237%。未用重要性权重修正，实验有意改变不同t的损失权重。

## 最终训练检查

|分布|train平均位置mm|train最坏位置mm|train最坏姿态°|validation平均位置mm|通过|
|---|---:|---:|---:|---:|---|
|beta|0.8346|2.7639|1.3096|1.0041|False|
|low_mix|0.7456|2.4842|1.4382|1.0448|False|

## 实际采样

|分布|更新次数|实际t<.3比例|实际t<.1比例|
|---|---:|---:|---:|
|beta|300|17.000%|1.333%|
|low_mix|300|53.667%|9.667%|

## 结论和限制

低噪声加密组最终train平均和最坏位置误差略低，但仍没有一次完整通过train门槛（全部9帧×3种子最坏位置<2mm、姿态<2°）。实验组250步最坏约2.22mm，300步回升约2.48mm；该瞬时最佳权重未单独保存，不能把它当作最终模型。validation偶尔通过不代表全部检查通过。

只有一个训练种子的配对结果，收益还未建立统计稳定性，也未复查所有去噪阶段或完整任务；不能据此断言后段速度偏差已解决。固定评估仍原始10步推理/不增强/标准flow检查，跨分布训练loss不可直接横比。未执行短物理闭环、抓握和完整往返。

[固定帧误差对照图](../start9_time_lowmix_010/time_comparison.png)；两组原生loss每100步更新并已保存。

## 保存路径

- 权重：`/home/song03/scientific_research/openarm-pi0-reproduction-group-A/checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_time_beta_009_io_mask/300`
- 记录：`/home/song03/scientific_research/openarm-pi0-reproduction-group-A/logs/a2_pi0/drawer_start_fit_20261024-221002-446313`
- 权重：`/home/song03/scientific_research/openarm-pi0-reproduction-group-A/checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_time_lowmix_010_io_mask/300`
- 记录：`/home/song03/scientific_research/openarm-pi0-reproduction-group-A/logs/a2_pi0/drawer_start_fit_20261024-221454-361117`

## 独立重载

|精度|平均位置mm|最坏位置mm|最坏姿态°|通过|
|---|---:|---:|---:|---|
|native|0.7387|2.4477|1.4014|False|
|bfloat16|0.7444|2.4890|1.4063|False|

native超标项：frame 8 seed 42=2.448mm。

bfloat16超标项：frame 8 seed 42=2.489mm。

[重载逐帧结果](../start9_time_lowmix_010/reload.json)。两组训练及重载均正常退出，仍未进入后续物理闭环。
