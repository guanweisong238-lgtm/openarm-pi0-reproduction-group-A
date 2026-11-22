# 任务 C 数据与模型脚本

在项目根目录使用 `.venv/bin/python` 运行。原始采集数据只读，派生输出使用独立目录。

- `prepare_real_episode.py`、`prepare_real_batch.py`：30 Hz 对齐；`check_real_alignment.py`：原始索引与图片哈希核验。
- `prepare_real_training_candidate.py`：完整 50 步窗口、按轨迹划分和训练集统计；已存在的候选目录不会覆盖。
- `audit_real_camera_batch.py`：按批次检查文件、时间戳与 JPEG 抽样。
- `review_real_episode.py`：四路图片，支持用 `--dataset` 指定新批次；`review_real_alignment.py`：真实图片与运动示意同步审核；`view_real_joint_prototype.py`：MuJoCo 运动示意（非物理任务重现）。
- `real_four_camera_config.py`、`real_four_camera_pi0.py`：四路模型；`real_four_camera_loader.py`：真实数据读取；`real_training_support.py`：权重加载与 LoRA 更新。
- `check_real_four_camera.py`、`check_real_four_camera_loader.py`、`smoke_real_four_camera.py`：模型输入和训练链路检查。
- `train_real_four_camera.py`：训练与固定验证；`verify_real_checkpoint.py`、`eval_real_four_camera.py`：重载及离线验证。

批量审核新数据的命令见[数据说明](../../data/README.md#2026-11-22-新采集批次)。上述对齐和训练脚本中的默认数据路径指向旧批次。

新批次必须显式指定数据路径和候选集，以下命令创建互不覆盖的新目录，按两种关抽屉方式分层（固定种子 42，按轨迹保留约 20% 验证）：

```bash
.venv/bin/python scripts/a2_pi0/prepare_real_batch.py --dataset data/real_20261122_camera_adjusted_new_block --output data/processed/real_camera_adjusted_alignment_v001
.venv/bin/python scripts/a2_pi0/prepare_real_training_candidate.py --source data/processed/real_camera_adjusted_alignment_v001 --output data/processed/real_camera_adjusted_candidate_v001 --stratify-at 35
```

候选集需要记录本次审核依据的 `review_decision.json`；不要把旧批次的审核结论复制成新批次的证据。之后运行：

```bash
.venv/bin/python scripts/a2_pi0/check_real_four_camera_loader.py --candidate data/processed/real_camera_adjusted_candidate_v001
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u scripts/a2_pi0/train_real_four_camera.py --candidate data/processed/real_camera_adjusted_candidate_v001 --steps 1000 --eval-every 100 --save-every 500 --name <新的实验名>
```

训练按候选集保存归一化统计，固定验证探针按操作方式分别报告损失。固定探针和训练损失均不是任务成功率。

示例命令

```bash
.venv/bin/python scripts/a2_pi0/check_real_four_camera_loader.py
.venv/bin/python scripts/a2_pi0/view_real_joint_prototype.py --headless
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u scripts/a2_pi0/train_real_four_camera.py --steps 1000 --eval-every 100 --save-every 500 --name <新的实验名>
```

训练脚本拒绝覆盖同名实验；仅保存 LoRA 适配器，不支持中断续训。该模型尚未完成真机闭环验收。

七步动作前缀与 14/20 时序的离线试验见 `real_prefix_pi0.py`、`real_action_window.py`、`train_real_prefix_pilot.py` 和 `eval_real_prefix_schedule.py`；这些均未连接机器人，需使用对应的 PrefixPi0 配置加载试验权重。
