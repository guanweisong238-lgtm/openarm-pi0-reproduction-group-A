# 任务 C 策略上真机：接线、协议与联调顺序

更新：2026-11-22。本目录只包含"让模型输出真正驱动机械臂"所需的最小代码，
默认全部处于只算不发的状态。文中所有"已验证"仅指在本机离线运行过，
**没有任何一项在真机上验证过**。

## 1. 结论

模型留在这台机器（RTX 5060 Ti 16 GB + 12 GB π0 底座权重），机械臂电脑只加一个
薄节点。三件事必须同时成立：

1. 机械臂侧的 dora 图里，`dora-openarm` 节点的 `move_position` 输入改由
   `position-mux` 提供，而不是直接来自 VR leader；
2. 策略以 **30 Hz、50 步、16 维绝对关节目标（rad）** 的形式发布，与训练动作完全同义；
3. 人始终保留接管权：模式文件写 `leader`、任何 `arm_command`、或策略流超时，
   都会立刻把位置流交回人工 IK。

不要做的一件事：把训练脚本整体拷到机械臂电脑上跑。那边的 `.venv-dp` 只要能跑
MuJoCo 和 dora 即可；π0 一次前向在本机是 240 ms，在无 GPU 机器上是秒级，
33 ms 的控制周期根本无法维持。反过来只搬权重也不行——JAX/CUDA 版本、12 GB 底座权重、
编译缓存那套依赖没必要在那台机器上重装一遍。

## 2. 通路

```text
机械臂电脑（dora 图）                                本机（GPU）
────────────────────────────────────────────      ─────────────────────────
cameras/status_global   ┐
cameras/status_chest    │
cameras/status_wrist_l  ├─► policy_client_node ──TCP──► policy_server
cameras/status_wrist_r  │     30 Hz 出目标             四路 JPEG 解码
robot/state_left/right  ┘     ↓                        归一化 + 一次采样
                              policy/position_left/right ↓
leader/follower_position_* ─► position-mux ─► robot.move_position ─► CAN ─► 电机
```

- 相机输入取 `cameras/status_*`：四路采集服务在入缓冲前已把 global 和双腕旋转 180°、
  chest 保持原样（`rot180-v1` / `identity-v1`）。策略必须吃与训练同方向的图，
  所以**不要**接 `robot/camera_*` 或自己再转一次。
- `position-mux` 是新增的，和已有的 `arm-command-mux` 一个思路：两路输入、一路输出，
  人工随时可回。机器人节点的使能、限速、过跳变拒绝、Stop 逻辑一个都不改。
- 策略节点**不发送** `arm_command`。启动、对齐、B 键急停、断使能仍由现场人员操作。

## 3. 为什么必须分块异步

实测（`results/real_inference_latency_20261122/report.json`，10 步采样）：

| 阶段 | 中位 |
| --- | ---: |
| 预处理（4 路 JPEG 解码 + resize + 归一化） | 35 ms |
| 模型采样 | 239 ms |
| 合计 | 275 ms（5 步采样版 255 ms） |

控制周期 33.3 ms，一次推理 ≈ 8.4 个控制步。所以照搬"每步问一次模型"必然卡死，
必须按 50 步一块、异步重规划。客户端实际实现的时序（`--chunk-exec 20`）：

```text
第 0 步：请求第一块，动作到达前不发布
第 20 步：用当前观测发起第二块请求（旧计划还有 30 步 = 1.0 s 余量）
第 28 步左右：第二块到达，立刻按"绝对步号"接管：第 t 步执行 new[t - 新块起点]
             —— 因为新块起点就是第 20 步，所以从 new[8] 接着走，不跳、不插值
第 40 步：发起第三块
旧计划耗尽仍无新块：保持最后目标、记一次故障、重新请求，不猜、不外推
```

关键点是**按绝对步号对齐**，而不是"第几个动作"：新块的第 k 个动作对应哪一时刻由
它的起点步号决定，因此迟到的推理不会造成时间错位，最多是动作"旧了一点"。
`--chunk-exec 20` 时，执行中的动作最多比推理时刻晚 8 步（267 ms）。

带 `--prefix` 时额外要求新块保留 7 步已知前缀（`real_prefix_pi0.py`），
切换才真正连续；前缀模型已验证会原样保留（偏差 1.2e-7 rad），
但现有 prefix 检查点只有 100 步试探版，所以默认关闭，先用非 prefix 的硬切换跑通链路。

## 4. 协议

一个请求一个连接，TCP，长度前缀。上行：4 字节小端 JSON 头 + 每路 4 字节长度前缀的 JPEG。
下行：4 字节长度前缀的 JSON，动作是 base64 的 float32，形状 50×16。

请求头（客户端 → 服务端）：

```json
{"type":"obs","seq":12,"sent_ns":0,"state":[16 个 float，rad，右臂先],
 "state_ts_ns":0,"blob_count":4,
 "cameras":{"global":{"src_ts_ns":0,"seq":0},"chest":{...},"wrist_left":{...},"wrist_right":{...}},
 "known_prefix":[[7×16]，可选]}
```

响应（服务端 → 客户端）：

```json
{"ok":true,"seq":12,"shape":[50,16],"dtype":"float32","actions_b64":"...",
 "infer_ms":247.3,"recv_ns":0,"sent_ns":0,"action_min":-0.78,"action_max":1.9}
```

服务端返回的是**反归一化后的 rad**（`x * (std+1e-6) + mean`），客户端拿到即可下发。
失败时返回 `{"ok":false,"error":"..."}`，客户端记一次故障。

刻意不用 ZMQ / gRPC / HTTP：机械臂电脑那边不需要新增任何依赖
（只用 stdlib socket + 已有的 numpy / pyarrow / dora），也不需要在机器人网络里
再开一个常驻 broker。代价是单连接、无重连，这正好符合"故障就冻结、要人工重启"的取向。

## 5. 文件

| 文件 | 跑在哪 | 作用 |
| --- | --- | --- |
| `policy_server.py` | 本机 | 推理服务；含 `--self-test`、`--probe`、`--serve` |
| `policy_client_node.py` | 机械臂电脑 | dora 节点：取观测、出目标、夹爪与跳变限幅 |
| `position_mux_node.py` | 机械臂电脑 | leader / policy 位置流二选一，带超时与接近性联锁 |
| `dump_inputs.py` | 机械臂电脑 | 只读探针，打印各输入的真实载荷结构 |
| `dataflow_policy_patch.yaml` | 机械臂电脑 | 图接线片段（不是完整可跑图） |

本机自检已经通过，可复现：

```bash
cd /home/song03/scientific_research/openarm-pi0-reproduction-group-A
.venv/bin/python scripts/a2_pi0/deploy/policy_server.py --self-test   # 预处理与训练逐元素一致
.venv/bin/python scripts/a2_pi0/deploy/policy_server.py --probe       # 单样本开口环，打印 MAE 与时延
.venv/bin/python scripts/a2_pi0/deploy/policy_server.py --serve --bind 0.0.0.0 --port 5555
```

截止本文件写入时，实际跑过的证据：

| 检查 | 结果 |
| --- | --- |
| `--self-test` | 4 个验证样本的 state、四路 224×224 图像、prompt token 与训练 `__getitem__` **逐元素相等** |
| `--probe`（step_1000） | 端到端 259–261 ms，其中模型 243–245 ms；与录制动作 MAE 0.083–0.087 rad，最大 1.2 rad（夹爪二值差异） |
| `--probe --prefix`（prefix step_0100） | 258–260 ms，MAE 0.088–0.095 rad；前缀 7 步偏差 1.2e-7 rad |
| 协议与调度（本机 loopback） | 服务端 + 客户端 25 s：请求间隔稳定 20 步、rtt 250–263 ms、按 30 Hz 输出 739 条、零故障 |
| `position_mux_node`（假 dora 事件） | 差 0.3 rad 时拒绝切入、接近后切入、`arm_command` 立刻回 leader、回落后正确转发 leader |

`--self-test` 会逐个比较推理侧预处理与 `RealFourCameraDataset.__getitem__` 的
state、四路图像、prompt token，要求完全相等——这是"模型看到的东西和训练一致"的
唯一低成本证据，改任何预处理都必须先过它。

## 6. 联调顺序（每一阶段都有明确 go/no-go）

**阶段 0 · 服务（本机，无机械臂）**
`--probe` 能出数，MAE 与历史离线评估同量级；`--serve` 监听正常。
go：`infer_ms` 稳定在 250–300 ms，动作范围在 rad 量级（不是 0~1）。

**阶段 1 · 探针（机械臂电脑，机械臂不使能）**
`dump_inputs.py` 接上五路输入，确认 JPEG 在哪个字段、`state_*` 里 qpos 怎么取、
`leader/follower_position_*` 是什么载荷类型。这一步不改图、不发指令。
go：`policy_client_node --offline` 能解析出 4 路图 + 2 路 8 维状态；据实选定 `--encoder`。

**阶段 2 · 影子（不接管）**
图上加 policy 节点但模式保持 `leader`，`OPENARM_POLICY_PUBLISH` 不设，节点按 shadow 跑，
把 40 条以上预测写进 `policy.jsonl`。人工照常遥操走一遍完整任务。
go：单次 rtt 的 p95 < 400 ms；预测动作与当时人工目标的偏差随任务阶段收敛
（开局偏差大不算问题，中途发散就是问题）；无连续故障。

**阶段 3 · 空载（第一次真的动）**
夹具悬空或工作区清空，`--max-step-rad 0.10`（≈3 rad/s 上限），每块只执行 5 步就回
`leader`。第一次只让它走一个块。
go：无过跳变裁剪告警、无 CAN 报错、位置跟踪滞后可接受、按 B 能立刻停。

**阶段 4 · 单物体任务**
恢复正常 30 Hz、20 步/块，`--max-step-rad 0.25`。从训练同分布的初始位姿开始。
go：连续 10 次里至少能完成"接近 + 抓取"，失败要能说清是视觉、夹爪还是时序。

**阶段 5 · 完整任务与统计**
按 episode 记成功/失败、中断原因、是否需要人工接管。这时才谈成功率。
现有离线数字（验证 loss 0.0949）不是成功率，不能拿来做部署判据。

## 7. 安全协议

- **使能权**：只有现场人员用 UI/VR 长按 A 使能；策略节点无此能力，代码里也不发 `arm_command`。
- **急停**：VR 右手 B 与硬件空开仍是唯一急停手段。策略节点不能替代，也不假装能替代。
- **接管**：`echo leader > policy_mode.txt` 立即生效；任何 `arm_command` 也会强制回落。
- **联锁**：切入策略前要求策略与当前 leader 目标差 < 0.05 rad，否则拒绝并打印实际差值。
- **看门狗**：图像帧龄 > 150 ms 不发请求；策略流 > 300 ms 未更新自动回落 leader；
  连续 3 次故障（请求失败、形状异常、前缀不匹配）停止发布，机械臂停在最后目标。
  停在最后目标是安全的，因为驱动有限速和限位；它不会自己继续走。
- **限幅**：夹爪按数据核验文档的端点裁剪（左 [-0.1745, 0.7854]，右 [-0.7854, 0.1745] rad）；
  关节单步跳变超过 `--max-step-rad` 直接裁剪并告警。裁剪是最后一道网，不是主要手段。
- **不做的事**：不自动重试、不自动重连后继续、不在故障后自动补发、不插值补造缺失观测。

## 8. 只能现场确认的六件事

| 项 | 为什么必须确认 | 怎么确认 |
| --- | --- | --- |
| 四路相机到训练相机名的映射 | 接错一路等于给模型看错视角 | `dump_inputs.py` 看 `cameras/status_*`；对照采集时四路抽帧图 |
| 图像方向 | global/双腕采集时已转 180°，chest 未转 | 同上一张图对照，确认没有二次旋转 |
| `state_*` 的 qpos 顺序 | 训练是右臂 8 维 + 左臂 8 维 | 给右臂关节一个已知小动作，看 16 维里哪 8 个动 |
| `move_position` 载荷类型 | 类型不匹配会让机器人节点报错或读到空 | `dump_inputs.py` 看 `leader/follower_position_*`，据此选 `--encoder` |
| 执行频率 | 训练 30 Hz；驱动名义 250 Hz | 先 30 Hz 下发，观察跟踪与抖动；不要一上来插值到 250 Hz |
| 提示词 | 训练用完整任务文本，多说少说都是分布外 | 服务端已硬编码 `real_four_camera_loader.PROMPT`，勿改 |

## 9. 已知风险

- 训练只有 1000 步、不足 1 epoch 的 LoRA 探索性适配器，22 条轨迹同一批次采集。
  它可能只是"在验证集上 loss 低"，不代表能闭环完成长时序双臂任务。
- 四路相机仍有遮挡与强反光缺陷，腕部视角在抓取阶段最不可靠。
- 时延抖动比中位数更重要：本机离线是 275 ms，但经过网络、加上对面负载，p95 可能翻倍。
  阶段 2 必须实测 rtt 分布，而不是相信这张表。
- 20 步/块的切换是硬切换，没有平滑或时序集成。切换瞬间若有明显跳变，
  先降 `--max-step-rad`，再考虑 prefix 模型，不要用插值掩盖。
- 本目录代码没有在 `dora` 运行时里跑过（本机没有机械臂侧的 venv 与 upstream），
  `--encoder` 与 `dora.Node()` 的具体调用要以现场 `dump_inputs.py` 的结果为准。
