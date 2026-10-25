# 无损图像单轨迹重训

实验：`pi0_openarm_drawer_lora_lossless / overfit_roundtrip_lossless_001`。

本轮直接读取`data/drawer_trial/contact_roundtrip_rgb_001/episode.npz`中的三路
`images_ceiling`、`images_wrist_left`、`images_wrist_right`，保留原始HWC uint8 RGB。
不经过视频编码、解码或颜色空间转换。状态和动作从同一文件按相同帧索引读取，
转为与旧训练数据相同的float32。原始采集数据及旧实验均保留。

`drawer_lossless_dataset.py`提供单帧观测及未来16步动作，轨迹尾部重复末动作并提供
填充掩码。训练与推理共用`OpenArmInputs`、归一化和模型预处理。原有训练时图像增强
仍保留，本轮不同时调整增强、冻结规则或学习率；“无损”指原始观测来源，模型仍会
按现有约定缩放至224×224，训练时继续使用原配置的数据增强。

## 对齐检查

462帧状态/动作与旧Parquet的float32值逐元素一致。全部未来动作窗口与填充掩码
检查通过。原始float64专家控制在相同环境中执行，在0、30、149、191、281、401、461帧
检查三路渲染图与无损训练观测逐像素一致，关节状态float32值一致。

```bash
.venv/bin/python scripts/a2_pi0/drawer_lossless_dataset.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/a2_pi0/train_drawer_overfit.py \
  --lossless --prepare-only --exp-name overfit_roundtrip_lossless_001
```

## 训练和验证

与旧2000步实验保持相同的基础初始化、随机种子42、batch 1、16步预测、LoRA冻结规则、
学习率和采样顺序。每1000步保存检查点，在检查点中新增`input_protocol.json`记录图像
来源和原始episode SHA256。模型配置使用独立名称，统计沿用已核对相同的状态/动作统计。

```bash
.venv/bin/python scripts/a2_pi0/train_drawer_overfit.py \
  --lossless --steps 2000 --exp-name overfit_roundtrip_lossless_001
.venv/bin/python scripts/a2_pi0/evaluate_drawer_overfit.py \
  --checkpoint checkpoints/pi0_openarm_drawer_lora_lossless/overfit_roundtrip_lossless_001/2000
```

评估器从检查点协议自动选择无损训练帧进行离线拟合检查，闭环继续读取实际仿真RGB。
闭环执行前4步后重新观测，30 Hz，最多24秒，固定采样噪声42，与旧实验相同。
不注入中间专家状态，不执行专家动作，不启用把手绑定。仍为单训练轨迹验证，不代表泛化。

训练目录：`logs/a2_pi0/drawer_overfit_20261024-110824-532914`。

## 实测结果

2000步训练完成，新进程重载检查点、462帧无损离线评估、720步真实接触闭环和保存动作的
物理重放均已完成。结果文件：`results/a2_drawer_lossless_overfit_001.json`。
评估目录：`logs/a2_pi0/drawer_eval_20261024-112325-883140`。

| 指标 | 旧有损视频训练 | 本轮无损训练 |
|---|---:|---:|
| 固定帧flow损失，训练结束 | 0.03984 | 0.04303 |
| 右臂动作MAE，rad | 0.02408 | 0.02736 |
| 接近阶段右臂动作MAE，rad | 0.04640 | 0.05423 |
| 示教观测下，首动作目标接近阶段平均Z误差，mm | -6.73 | -6.68 |
| 初始离线/仿真整块动作最大差，rad | 0.09703 | 0 |
| 模型闭环最大抽屉开度，mm | 0 | 0 |
| 双侧手指抓握检测 | 无 | 无 |
| 开后关任务成功 | 否 | 否 |

本轮末帧夹持点相对把手XYZ误差为`[-175.03, -19.19, -43.48] mm`，没有到达把手。
右臂第5关节目标在501/720帧被控制限幅（按差值大于1e-9计），是该闭环偏离后的重要
异常信号；这里计的是请求目标超限，不等同于501次物理碰撞或触碰机械关节止挡。
原始模型动作在仿真中重新积分，720步最大qpos误差为0；GUI展示的是这些保存的模型
预测动作，非专家回放，也非再次在线推理。空格暂停、R重播，结束后保留末帧。

本次对照验证了无损训练和原始仿真输入的一致性，但相同2000步配方的拟合和任务效果
没有改善，不能认为图像压缩差异是此前失败的唯一原因，也不能据一次训练就认定无损
数据本身更差。现有LoRA配方的接近阶段拟合和闭环纠偏仍不足，后续应保留无损输入，
单独对照训练增强、状态/动作投影可训练性和接近/夹持帧权重，以真实动作和末端误差
作为单轨迹过拟合验收标准。本轮没有扩大训练或加入人工高度补偿。

```bash
.venv/bin/python scripts/a2_pi0/replay_drawer_model.py \
  logs/a2_pi0/drawer_eval_20261024-112325-883140/closed_loop.npz
```
