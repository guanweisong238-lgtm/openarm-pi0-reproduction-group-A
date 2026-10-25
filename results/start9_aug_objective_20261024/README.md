# 图像增强与首4步损失配对实验

三组从同一`start9_delta_refine_005_io_mask/1200`独立开始；各300步，lr3e-6，fresh optimizer；train ID12首9帧、真实未来16步，validation ID3首0/4/8只监控。每50步检查9×3种子，每100步本地loss刷新。三组采样顺序、统计SHA、源检查点、学习率和步数已自动核对一致。

|组|train平均位置mm|train最坏位置mm|train最坏姿态°|validation平均位置mm|通过|
|---|---:|---:|---:|---:|---|
|增强开启|1.4698|3.8677|2.1155|1.8266|False|
|增强关闭|0.9714|3.0925|1.6327|1.2373|False|
|增强关闭+首4步权重4|0.9289|2.9709|1.5098|1.1702|False|

门槛仍为全部训练检查的最坏位置<2mm、最坏姿态<2°。三个实验均无一次训练检查完整通过。尚未进入短闭环、抓握、完整往返和验证起点成功率测试。

## 解释与限制

关闭图像增强后平均及最坏误差改善，但不能从9帧拟合推断全任务应永久关闭增强。流匹配噪声/时间采样仍随机。首4步权重4仅改变时间步权重，后12步权重保持1，不改变任何标签或原始数据。全序列损失仍覆盖32维。加权导致loss口径变化，跨组比较统一采用物理单位的固定帧误差。

三组使用相同的flow_squared_errors训练实现；推理仍10步。每组只有一个训练种子，三个推理种子只检测输出噪声变化，不能替代多个独立训练复验。加权的小幅收益暂不足以认定普遍有效。

[三组曲线](../start9_noaug_first4_008/paired_comparison.png)

## 检查点及日志

- `/home/song03/scientific_research/openarm-pi0-reproduction-group-A/checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_aug_control_006_io_mask/300`
  日志：`/home/song03/scientific_research/openarm-pi0-reproduction-group-A/logs/a2_pi0/drawer_start_fit_20261024-213623-480141`。
- `/home/song03/scientific_research/openarm-pi0-reproduction-group-A/checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_noaug_007_io_mask/300`
  日志：`/home/song03/scientific_research/openarm-pi0-reproduction-group-A/logs/a2_pi0/drawer_start_fit_20261024-214151-122240`。
- `/home/song03/scientific_research/openarm-pi0-reproduction-group-A/checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_noaug_first4_008_io_mask/300`
  日志：`/home/song03/scientific_research/openarm-pi0-reproduction-group-A/logs/a2_pi0/drawer_start_fit_20261024-214647-840055`。

## 独立重载结果

|精度|平均位置mm|最坏位置mm|最坏姿态°|通过|
|---|---:|---:|---:|---|
|native|0.9244|2.9958|1.5210|False|
|bfloat16|0.9255|3.0635|1.5177|False|

[独立重载逐帧结果](../start9_noaug_first4_008/reload.json)。训练和重载正常完成，原生loss已保存；门槛未过，后续物理闭环未执行。进一步须检查具体超标帧/关节和流匹配去噪轨迹，再决定优化改动，避免无限叠加训练步数。
