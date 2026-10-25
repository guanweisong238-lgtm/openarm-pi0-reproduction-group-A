# 当前保留的模型检查点

## 已验证拉取模型（2026-10-25最新）

`pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000`为Pi0骨干一步确定性改造版，
原始起点两次和柜体Y+5mm验证场景均完成真实双指拉取>=80mm保持1s；warm-start验证未通过。
固定noise42/H4/continuous夹爪，独立重放位置、速度、接触力误差0。
部署参数/后验验证在该检查点deployment_config.json及validation_report.json；训练协议保留原快照。
成功视频与完整边界：`results/drawer_pull_success_20261025/README.md`。
下方“没有成功模型”等内容是当时历史状态，由本节覆盖。

## 2026-10-25更新（优先于下方历史记录）

为持续训练腾出空间，已归档配置并删除006/007/008/009的300步权重，以及012/600中间权重，
释放25.899GiB。删除清单与元数据见`results/project_cleanup_20261025_recovery_014/`，无权重备份。
010/300、011/450、012/1200、013b/2000及旧关键基线保留；014正在恢复训练。
011仅通过局部9步，012和013b完整闭环均失败，当前没有完成拉取的模型。
下方“保留8份”“当前最新”等字样是2026-10-24历史记录，已被本节及项目handoff最新状态覆盖。

2026-10-24清理后保留以下8份完整检查点。
归档配置及删除清单：`results/project_cleanup_20261024_213130/README.md`。

|检查点|用途|
|---|---|
|`pi0_openarm_drawer_lora_lossless/stage_fit_lossless_002/3500`|旧单轨迹关键基线|
|`pi0_openarm_drawer_lora_lossless/stage_fit_right8_key147_005/4500`|旧单轨迹key147对照|
|`pi0_openarm_drawer_start_fit_delta_v001/start9_delta_004_io_mask/600`|残差600步对照及续训起点|
|`pi0_openarm_drawer_start_fit_delta_v001/start9_delta_refine_005_io_mask/1200`|当前最新残差模型；起步拟合门槛仍未通过|
|`pi0_openarm_drawer_start_fit_v001/start9_cleanup_002_io_mask/200`|绝对/残差600步对照的共同起点|
|`pi0_openarm_drawer_start_fit_v001/start9_refine_003_io_mask/600`|绝对动作600步对照|
|`pi0_openarm_drawer_success_v003_lora/success13_early60_500_003/500`|起步拟合系列的公共权重起点|
|`pi0_openarm_drawer_success_v003_lora/success13_lora_001/2000`|13条数据正式训练基线；已知完整闭环未成功|

被清理的历史分支保留日志和评估记录，权重需重新训练。当前残差模型仍处于起步拟合诊断阶段。


2026-10-24 晚续：图像增强/首4步权重的三组配对实验已完成（006/007/008，各300步、lr3e-6）。
三组同源005/1200，固定相同采样顺序；关闭增强将平均/最坏位置1.470/3.868降为0.971/3.093mm，
加权首4步后0.929/2.971mm。独立重载008：native平均0.924、最坏2.996mm、姿态1.521°；
BF16平均0.926、最坏3.064mm、姿态1.518°。均未通过最坏<2mm门槛，未执行新物理闭环。
完整记录`results/start9_aug_objective_20261024/README.md`，新增训练选项`--no-augmentation`、`--first-four-weight`。
训练数据/统计不改，loss窗口每100步刷新；新增3份5.2GiB检查点，目前都保留供复查。


2026-10-24 低噪声时间采样对照009/010已完成：同源008/300，各300步、lr3e-6、
关闭增强、首4步权重4。原Beta组train平均/最坏位置0.835/2.764mm；
50%Beta+50%Uniform(.05,.3)组0.746/2.484mm。独立重载混合组native平均0.739、
最坏2.448mm、姿态1.401°；BF16平均0.744、最坏2.489mm、姿态1.406°。
均未通过最坏<2mm门槛，没有运行新闭环。实际采样t已逐步记录并检查，
完整协议、曲线及限制见`results/start9_time_sampling_20261024/README.md`。
新增`--time-sampling beta/low_mix`，默认Beta逐元素等于旧时间采样已验证。


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
