# 11 月 28 日第 0 条示范的真机回放准备

回放范围为双臂各七个关节的实测角度。夹爪开合保留给后续单独验证。原始数据和 π0 工作台配置保持原样。

现场确认第 0 条采集时，左臂与底座没有接触。回放准备使用实测关节记录和驱动限位，真机空间以现场核对为准。

## 11 月 28 日真机首帧摆位结果

现场读回后，重新核对了 16 个起点角度。单次摆位按规划约用 14.54 秒，目标最高速度约 4.87°/秒。右臂到位误差最大 1.18°，左臂最大 1.25°，均小于 0.05 rad（约 2.86°）。试验只执行 `approach`。结束后独立确认 16 轴失能，两条 CAN 接口关闭。 [路线摘要](../../logs/openarm-demo0-approach-20261128.88f2hs/live_approach.json) · [摆位日志](../../logs/openarm-demo0-approach-20261128.88f2hs/approach.log) · [独立复查](../../logs/openarm-demo0-approach-20261128.88f2hs/after.json)

失能后手臂自然放下。右臂相对首帧的最大误差为 4.31°，左臂为 4.04°。现有 `short` 和 `full` 要求起点已在首帧 0.05 rad 以内，因此这次停机姿势下会拦截直接播放。下一次需先从新读回的姿势摆到首帧，再在同一次使能期间连续执行短段。该流程待补齐和验证。

## 已完成的离线检查

- 从两侧原始记录各读取 1,548 行，逐点核对了对齐后的 635 帧。左右顺序、角度和来源时间戳一致，14 个关节的全部原始角度都在当前驱动限位内。
- 已生成原示范前 1 秒的离线试跑候选。按 50 Hz 输出共 525 点，时长 10.48 秒，任一轴相对起点最大变化约 7.26°，最高角速度约 0.059 rad/s。[离线试跑数据](first_second_offline.npz)
- 按原始示范的 21.13 秒时间线、用对齐后相邻帧角度估算，速度峰值约 1.72 rad/s。当前轨迹检查上限为 0.20 rad/s 和 0.50 rad/s²，原速轨迹会被拦截。现有离线版将所有关节的整段时间统一拉长约 24 倍，主要受最急处的加速度约束。整段长 506.26 秒，最高角速度约 0.104 rad/s。后续可核对速度尖峰的原始时间戳，再尝试只放慢变化太快的区段，左右臂保持同步。 [整段离线数据](full_demo_offline.npz) · [逐项报告](report.json)

## 现场读回和分段回放

新入口 [`demo0_replay.py`](../../scripts/a2_pi0/deploy/dp_control/dp_pilot/demo0_replay.py) 使用官方控制驱动和工作台现有的限幅重力补偿。`check` 只核对离线文件。`read` 只请求双臂状态，不使能电机。它会核对 USB-CAN 和左右物理通道，要求两臂各收到八轴新鲜失能回包，再生成当前位置到第一帧的路线摘要。该摘要记录起点、终点、各轴角度范围和速度，实际空间仍由现场确认。

若两条接口仍为 DOWN，请由现场管理员确认接线、失能和总线独占后，仅配置 CAN 接口。

```bash
for iface in can0 can1; do
  sudo ip link set dev "$iface" type can bitrate 1000000 sample-point 0.75 \
    restart-ms 0 dbitrate 5000000 fd on dsample-point 0.75 dsjw 2 || break
  sudo ip link set dev "$iface" up || break
done
```

在仓库根目录用普通用户读回。每次使用新的文件名，便于保存各次现场结果。

```bash
pilot=scripts/a2_pi0/deploy/dp_control/dp_pilot/demo0_replay.py
plan=results/real_20261128_demo0_replay_preflight/live_approach.json
.cache/openarm-policy-driver/bin/python -B "$pilot" read --plan "$plan"
```

`run` 分为 `approach`（只到第一帧）、`short`（开头短段）和 `full`（整段）。每个阶段单独确认、执行和验证失能。收到新的现场运动授权后，先做 `approach`，确认路径净空、承托和实体急停可用。入口会在使能前复查姿势及电机模式，需在终端输入所提示的确认文字。去第一帧的目标速度上限是 0.09 rad/s，加速度上限是 0.10 rad/s²。实际姿势与已审核起点相差超过 0.01 rad，或计划超过 5 分钟，需重新读回。

任何一侧失去反馈、偏离目标、运行超时或出现 CAN 故障时，程序会尝试双臂失能并读回状态。下一次启动前重新读回，并按上面的起点要求核对。夹爪在各阶段保持该次启用时的角度。独立的实体停机仍需有人值守。

重新生成报告和离线试跑数据，在仓库根目录运行以下命令。

```bash
.venv/bin/python -B scripts/a2_pi0/deploy/dp_control/dp_pilot/prepare_demo0_replay.py \
  --out results/real_20261128_demo0_replay_preflight
```
