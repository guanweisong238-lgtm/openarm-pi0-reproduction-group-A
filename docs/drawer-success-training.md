# v003多轨迹训练

数据：13条成功RGB，10条train/3条validation。用户终端GPU运行，agent只完成CPU预检。
入口：`scripts/a2_pi0/train_drawer_success.py`，默认从pi0_base重新初始化LoRA和优化器。
新统计与旧3500检查点的动作/状态空间含义不同，不提供静默续训旧检查点选项。

## 本轮配方

- batch=1、16步预测、LoRA-only冻结、无EMA，保持既有模型和图像增强。
- 优化目标保持上游all32 flow均值；不叠加right8或关键关节加权。
- 默认峰值lr=1e-5，warmup为min(100,steps/10)，余弦降至1e-6；这是首轮实验设置，不是已证明最优。
- 每步等概率选择10条train之一，再在该轨迹内均匀抽帧，validation不参与更新。
  该方案平衡轨迹长度，但不消除轨迹内部的长停留偏置。
- 归一化仍用已经冻结的train-only逐帧统计；不因更换采样策略重算或混入验证数据。
- 图像预处理缓存为uint8 memmap，浮点输入float32。缓存不保存随机增强。
  缓存身份包含manifest、统计、模型结构和相关适配代码；旧缓存不冒充新版本。
- 默认每500步诊断、1000步保存（约5.2GiB/份），不覆盖同名实验。

## 命令

CPU缓存/边界/输入/采样隔离检查（已通过，不加载模型）：

```bash
.venv/bin/python scripts/a2_pi0/train_drawer_success.py --prepare-only
```

用户终端先执行20步GPU链路检查：

```bash
.venv/bin/python scripts/a2_pi0/train_drawer_success.py \
  --exp-name success13_smoke_001 --steps 20 --lr 1e-5 \
  --eval-every 20 --save-every 20 \
  2>&1 | tee logs/success13_smoke_001_console.log
```

确认更新有限非零、诊断与保存均完成后，执行首轮2000步：

```bash
.venv/bin/python scripts/a2_pi0/train_drawer_success.py \
  --exp-name success13_lora_001 --steps 2000 --lr 1e-5 \
  --eval-every 500 --save-every 1000 \
  2>&1 | tee logs/success13_lora_001_console.log
```

两次独立从base初始化；短测权重不作为正式起点。短测warmup只有2步，不能与正式配方
作精度比较。不能同时启动两次GPU训练。完整检查点保存的optimizer当前未实现恢复入口。

## 判读

每条轨迹均匀取4帧，train共40帧、validation共12帧；固定噪声42/7/99。
各自记录分组flow、右臂/夹爪/左臂实际采样MAE和首动作夹持点位置、姿态误差。
动作MAE排除末尾填充。空间指标三种子均计算；用各轨迹自己的场景和qpos做FK。
这些是少量固定检查帧，不是全轨迹误差或闭环成功率。不能与旧单轨迹8帧的数值直接比较。

日志：`logs/a2_pi0/drawer_success_*/` 下的`protocol.json`、`metrics.jsonl`、`evaluation.jsonl`、`summary.json`。
权重：`checkpoints/pi0_openarm_drawer_success_v003_lora/<exp-name>/<step>`，附数据manifest及新统计资产。
训练完成只说明程序执行完成；评估需结合held-out动作与空间精度，再验证对应初始条件下的闭环。
旧单轨迹评估脚本使用旧统计/资产ID，不应直接用于新权重。

CPU侧已通过：全10700帧缓存构建、13条首末动作与原数据反归一化一致、split隔离、
episode-local索引和缓存模型输入。GPU数值更新、完整模型诊断和保存仍待用户终端短测。

## 20步GPU短测（已完成）

`success13_smoke_001`，日志 `logs/a2_pi0/drawer_success_20261024-161702-707695/`。
20次更新均有限且实际update_norm非零（0.00244～0.03956），grad_norm最大1.408。
训练图XLA报告7.11GiB，无OOM。检查点20的params/optimizer/assets/protocol/manifest均落盘；
尚未独立验证重载后的模型输出。
固定检查帧的三种子均值：train臂MAE 0.182182→0.177560，validation 0.201080→0.195536rad；
train位置70.181→69.691mm，validation 72.430→72.050mm。
三个种子的右臂MAE和位置分别均小幅下降；这只支持链路正常和早期学习信号，不证明收敛或泛化。
下一步执行2000步独立实验，不从短测20步续训；正式训练warmup100，短测warmup2。

## 本地实时loss曲线

已为正在运行的首轮实验打开独立Matplotlib窗口，每100个训练step刷新并覆盖保存
`logs/a2_pi0/drawer_success_20261024-162209-781293/loss_curve.png`。
上图：逐step训练loss和最近100step滑动均值；下图：固定训练/验证检查帧的flow loss。
验证仍按训练脚本的500步间隔计算，绘图不会伪造每100步验证点。
日志读取独立于训练，不需要重启训练，关闭绘图窗口不停止训练。
训练完成后保存最终PNG和SVG，窗口仍保留供查看。

```bash
# 在另一个终端运行；使用系统Python的Matplotlib，不改变训练venv依赖。
/usr/bin/python3 scripts/a2_pi0/plot_training_loss.py \
  --run-dir logs/a2_pi0/drawer_success_20261024-162209-781293 --every 100

# 只生成PNG，不打开窗口。
/usr/bin/python3 scripts/a2_pi0/plot_training_loss.py \
  --run-dir logs/a2_pi0/drawer_success_20261024-162209-781293 --every 100 --once
```

后续实验替换`--run-dir`为其实际RUN_DIRECTORY。绘图工具已实际观察到1100→1200步自动刷新。

## 首轮2000步结果（已完成）

`success13_lora_001` 日志 `logs/a2_pi0/drawer_success_20261024-162209-781293/`，
保存1000、2000。2000次更新有限且非零，无OOM。最终PNG/SVG已保存，尚未闭环或独立重载验证。

| 三种子均值 | step0 | step2000 |
|---|---:|---:|
| train臂MAE rad | 0.182202 | 0.046639 |
| validation臂MAE rad | 0.201158 | 0.055792 |
| train首动作位置mm | 70.154 | 16.941 |
| validation首动作位置mm | 72.413 | 17.191 |
| validation姿态度 | 31.634 | 6.752 |

1500→2000验证臂MAE仅改善0.23%，位置改善7.03%；flow 0.139613→0.140166略升。
这不足以认定过拟合，后期学习率下降也会影响进度。
关键分组：验证script_ready_start臂MAE0.07364、位置20.93mm；human_warm_start为
0.02010、9.72mm。脚本原轨迹ID12首帧三种子平均位置52.79mm，验证ID2首帧41.12mm，
其余三个检查帧多在9～17mm。错误集中在ready起点，不宜只依据总均值追加训练。
每条仅4个时间均匀检查帧，尚不能代表整段接近或抓握质量。
组级分析：`results/success13_lora001_group_analysis_20261024.json`。
下一步建议加密接近/抓握检查帧、分别报告两种起点，再决定是否增加早期帧采样或做闭环诊断。

## 原生窗口实时闭环

```bash
.venv/bin/python scripts/a2_pi0/live_drawer_success.py
```

默认加载2000步检查点及v003归一化统计，先运行验证ID3的ready起点；N切换验证ID6的
human warm起点，R重新运行当前起点，Space暂停，Q退出。每轮最多24秒仿真时间。
模型每次读取当前三路图像和关节状态，预测16步，只执行前4步后重新观察；推理时暂停物理时间。
仅初始化读取示教的完整初始状态，不执行任何示教动作或注入中途状态。
评分包含把手及其支柱的双侧真实接触、拉开80mm保持1秒、关闭到5mm以内保持1秒。
输出目录`logs/a2_pi0/success_live_*/`保存逐步状态、动作、接触、视频和summary.json。
单次单种子只用于观察行为，不能作为总体成功率。

图形使用GLFW及Mesa软件渲染，GPU用于JAX推理；不可混用EGL离屏上下文和GLFW桌面窗口。
默认工具沙箱仍无GPU/桌面访问；本次经授权在沙箱外启动可以访问GPU和桌面。

### 2026-10-24首次真实闭环及首帧核查

seed42：ID3在18.53秒关闭窗口，未接触把手；ID6完整24秒超时，未抓握/拉开。
因此2000步模型的任务成功没有得到验证。两种图形后端下，实时初始化状态及三路RGB
与对应数据首帧完全相同。ready首预测夹持点误差55.90mm、warm17.51mm，与训练时
相同帧/种子56.54mm和17.29mm接近；问题在离线首帧上已经存在，不能只看聚合均值。
详细证据和后续对照建议：`results/live_start_audit_20261024/diagnosis.md`。

### 训练起点对照（已执行）

使用相同2000权重/seed42/执行前4步策略，train ID12 ready和train ID8 human各运行12秒。
两者均无把手接触、无抽屉位移；初始输入匹配数据。前4步实际夹持点相对同刻示教分别
偏离93.29mm和20.26mm。独立重放示教前4步可精确重现qpos。
因此当前模型在训练起点也未稳定执行，不能只把失败归因于验证起点陌生。
实际2000次更新仅37次采到各轨迹前15帧观测；起步采样不足是后续需受控检验的假设。
命令与完整结果见`results/train_start_control_20261024/README.md`。

### 早期采样配对续训（已完成）

从原2000权重分别执行uniform/early60各500步，其他训练设置相同；加密前75帧评估。
early60在验证首帧与原四帧总体位置指标上改善，但相同train ID12/8、seed42的四次12秒
闭环均无把手接触。ready前0～8帧仍是大误差区，未解决真实闭环。
新权重：`success13_uniform500_control_002/500`和`success13_early60_500_003/500`。
方案见`docs/drawer-early-sampling-control.md`，完整报告`results/early_sampling_control_20261024/README.md`。

### 深入原因核查（仅诊断，未改训练配置）

发现ready首帧对无关左夹爪状态过度敏感（0→0.001069rad令右侧预测变化87.91mm），
固定train检查帧左8维占总flow误差41.64%；所有状态/动作投影层在本项目都被冻结。
错误绝对位置首目标使右2/4/7执行器达到力矩上限。先做小样本起步拟合和逐项消融，
再决定接口解冻、无关维度屏蔽、残差动作及恢复数据方案。
完整分析与建议：`results/drawer_deep_diagnosis_20261024/README.md`。


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
