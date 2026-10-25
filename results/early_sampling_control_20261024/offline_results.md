# 早期采样配对实验离线结果

两组从同一2000步权重出发，各更新500步。负百分比表示early60比uniform对照的误差更低。
权重来源、训练配置、逐步轨迹选择、未替换帧一致性已校验。离线指标不是闭环任务成功率。
同权重的两次初始评估存在数值差异，表中同时列出；原因未确认，较小变化不宜过度解释。

| split / subset | 训练前 control / early | uniform +500 | early60 +500 | 相对uniform变化 | 变化量之差mm |
|---|---:|---:|---:|---:|---:|---:|
| train / legacy_anchor/all | 16.820 / 16.756 | 16.857 | 15.162 | -10.1% | -1.632 |
| train / early_anchor/script_ready_start | 34.079 / 34.020 | 31.774 | 29.260 | -7.9% | -2.455 |
| train / early_anchor/human_warm_start | 13.496 / 13.504 | 13.111 | 10.942 | -16.5% | -2.178 |
| train / start_anchor/script_ready_start | 51.799 / 51.187 | 54.587 | 46.834 | -14.2% | -7.140 |
| train / start_anchor/human_warm_start | 14.531 / 14.471 | 14.238 | 11.776 | -17.3% | -2.403 |
| validation / legacy_anchor/all | 17.291 / 17.416 | 17.219 | 15.689 | -8.9% | -1.655 |
| validation / early_anchor/script_ready_start | 35.247 / 35.350 | 32.532 | 30.174 | -7.2% | -2.462 |
| validation / early_anchor/human_warm_start | 16.788 / 16.826 | 16.022 | 12.234 | -23.6% | -3.826 |
| validation / start_anchor/script_ready_start | 44.540 / 45.502 | 45.721 | 40.306 | -11.8% | -6.377 |
| validation / start_anchor/human_warm_start | 14.689 / 15.007 | 14.288 | 10.506 | -26.5% | -4.101 |
