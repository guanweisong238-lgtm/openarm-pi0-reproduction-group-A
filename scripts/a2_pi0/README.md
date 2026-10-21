
# A2：π0 模型与训练

## 1. 任务目标

基于 OpenPI 官方开源代码和预训练模型，验证模型推理流程，选择并适配公开数据集，完成至少一次 π0 LoRA 微调。

## 2. 主要任务

### 阶段一：环境与模型

- [x] 检查 OpenPI 官方仓库及运行要求
- [x] 确认可用 GPU 型号与显存
- [x] 配置模型运行环境
- [x] 下载或获取预训练模型权重
- [x] 运行官方模型推理并记录结果（项目独立脚本，合成输入）

### 阶段二：公开数据集

- [ ] 确认选用的数据集名称和来源
- [ ] 记录数据集版本及使用许可
- [ ] 检查机器人类型、任务内容、观测和动作定义
- [ ] 验证数据能否被 OpenPI 读取
- [ ] 配合 A1 确认数据与 OpenArm 的兼容性

### 阶段三：模型微调

- [ ] 确认训练配置及数据处理方式
- [ ] 完成至少一次 LoRA 微调
- [ ] 保存训练日志和损失曲线
- [ ] 保存模型权重及对应配置
- [ ] 验证训练后的模型可以重新加载

## 3. 交付物

- 环境配置和运行说明
- 模型来源与版本记录
- 数据集来源及适配说明
- 训练配置
- 模型权重获取方式
- 训练日志与损失曲线
- 模型推理测试记录

## 4. 注意事项

1. 优先使用 OpenPI 官方代码和预训练模型。
2. 不直接假定公开数据集能够控制 OpenArm，需要先检查观测格式、动作空间和任务是否匹配。
3. 不将大型模型权重或完整数据集直接提交到 GitHub。
4. 记录实际使用的代码版本、数据版本和训练参数。
5. 如果遇到显存不足、依赖冲突或动作维度不匹配，应记录错误信息和处理过程。

## 5. 当前进度

- [x] 环境配置
- [x] 官方模型推理（合成输入验证）
- [ ] 数据集确认
- [ ] 数据读取验证
- [ ] LoRA 微调
- [ ] 权重复载验证

运行证据见 [2026-10-21 推理记录](../../results/a2-pi0-smoke-20261021.md)。

## 6. 独立环境与首次推理

以下命令均在小组仓库根目录执行，不使用其他 OpenPI 工作副本。

```bash
uv venv --python 3.11 .venv
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python --torch-backend=cu128 -r configs/a2-requirements.txt
uv pip check --python .venv/bin/python

# 先下载官方 pi0_aloha_sim 权重，再运行模型。
.venv/bin/python scripts/a2_pi0/infer_smoke.py --download-only
.venv/bin/python scripts/a2_pi0/infer_smoke.py
```

OpenPI 和客户端版本固定在 `configs/a2-requirements.txt` 中。权重位于
`weights/openpi/`，Hugging Face 与 JAX 缓存位于 `.cache/`，每次运行日志位于
`logs/a2_pi0/<时间戳>/`；这些目录均已被 Git 忽略。

完整推理检查使用随机图像、常量 14 维 ALOHA 状态和 `Transfer cube` 指令，执行两次推理。
验收条件是输出动作形状为 `(50, 14)` 且所有数值有限。
首次调用时间包含编译；第二次调用时间仅作单次观测，不作为正式延迟评估。
成功后保存 `actions.npy` 和 `summary.json`。这项检查不代表 OpenArm 适配完成或任务成功。

下载公开权重时，脚本显式使用 `token="anon"` 并禁用 gcsfs 的实验性 bucket 探测，
避免等待本地 Google 默认凭据。相邻的 `.lock` 文件是下载互斥锁，内容为空属于正常现象；
文件是否存在不代表权重已经下载完成。以脚本正常退出并输出 `Checkpoint downloaded` 为准。

项目使用 `checkpoint_download.py` 校验官方对象列表中的每个文件的大小和 MD5。
仅补下载缺失或不匹配的文件，通过临时文件校验后再替换，并将下载异常向上传播。
`infer_smoke.py --verify-only` 可仅检查完整性。完整推理前也会先校验权重。
该流程替代上游未读取后台下载异常、可能误报进度完成的目录下载实现。
