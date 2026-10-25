# 项目磁盘清理记录

已删除10份完整检查点，释放51.788GiB。
文件系统剩余空间由12.897GiB增至64.686GiB。

## 保留的8份检查点

|路径|保留用途|
|---|---|
|`checkpoints/pi0_openarm_drawer_lora_lossless/stage_fit_lossless_002/3500`|旧单轨迹关键基线|
|`checkpoints/pi0_openarm_drawer_lora_lossless/stage_fit_right8_key147_005/4500`|旧单轨迹key147对照|
|`checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_delta_004_io_mask/600`|残差600步对照及续训起点|
|`checkpoints/pi0_openarm_drawer_start_fit_delta_v001/start9_delta_refine_005_io_mask/1200`|当前最新残差模型；起步拟合门槛仍未通过|
|`checkpoints/pi0_openarm_drawer_start_fit_v001/start9_cleanup_002_io_mask/200`|绝对/残差600步对照的共同起点|
|`checkpoints/pi0_openarm_drawer_start_fit_v001/start9_refine_003_io_mask/600`|绝对动作600步对照|
|`checkpoints/pi0_openarm_drawer_success_v003_lora/success13_early60_500_003/500`|起步拟合系列的公共权重起点|
|`checkpoints/pi0_openarm_drawer_success_v003_lora/success13_lora_001/2000`|13条数据正式训练基线；已知完整闭环未成功|

## 删除项目

|路径|删除依据|
|---|---|
|`checkpoints/pi0_openarm_lora/overfit_ep0_5000_001/5000`|早期pickup拟合，当前已转为13条抽屉数据|
|`checkpoints/pi0_openarm_drawer_lora/overfit_roundtrip_001/2000`|早期有损单轨迹模型，已有无损基线与多轨迹模型|
|`checkpoints/pi0_openarm_drawer_lora_lossless/overfit_roundtrip_lossless_001/2000`|旧单轨迹初版，保留后续3500基线及key147版本|
|`checkpoints/pi0_openarm_drawer_lora_lossless/stage_fit_right8_lr3e6_004/4500`|旧版right8分支，保留后续key147版本及评估记录|
|`checkpoints/pi0_openarm_drawer_success_v003_lora/success13_smoke_001/20`|20步流程短测，正式2000步模型已保存|
|`checkpoints/pi0_openarm_drawer_success_v003_lora/success13_lora_001/1000`|正式训练中间检查点，保留最终2000步|
|`checkpoints/pi0_openarm_drawer_success_v003_lora/success13_uniform500_control_002/500`|已完成的uniform对照分支，保留报告与early60公共起点|
|`checkpoints/pi0_openarm_drawer_start_fit_v001/start9_pair_001_lora/200`|已完成的LoRA-only消融分支，保留指标及参数更新证据|
|`checkpoints/pi0_openarm_drawer_start_fit_v001/start9_pair_001_io/200`|已完成的IO消融分支，保留指标及参数更新证据|
|`checkpoints/pi0_openarm_drawer_start_fit_v001/start9_cleanup_002_io_mask_right8/200`|未采用的right8分支，保留其评估及训练曲线|

## 验证与恢复边界

保留检查点的完整文件清单及文件大小与删除前一致。原始示教、13条成功数据、底座权重、虚拟环境、归一化统计、训练缓存、所有日志、结果和图像均保留。

每份删除模型的根目录配置、输入协议、数据清单和assets统计已复制至`metadata/`，复制前后SHA256一致。归档不含参数和优化器权重；要重新运行已删除模型，须按配置重新训练。历史文档中的被删除检查点路径是历史记录，不再是可直接加载的模型。

目前已有足够训练余量，保留4.6GiB现用预处理缓存，避免下次训练重新解码全部图像。

[完整清单](manifest.json)
