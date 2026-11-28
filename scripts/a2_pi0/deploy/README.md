# 任务 C 真机部署

## 2026-11-28 第 0 条示范回放

直接复现第 0 条示范的关节角度使用 [demo0_replay.py](dp_control/dp_pilot/demo0_replay.py)。`check` 检查离线轨迹，`read` 只读双臂状态并规划到首帧的路线。真机已验证 `run --stage approach`，双臂到位误差最大为右 1.18°、左 1.25°。结束后独立确认 16 轴失能，两个 CAN 接口关闭。短段、整段和摆位后直接接上播放的流程仍待真机验证。 [实测记录与操作说明](../../../results/real_20261128_demo0_replay_preflight/README.md)

离线短段长 10.48 秒，整段长 506.26 秒，只含双臂关节轨迹，夹爪保持本次启用时的角度。原始示范约 21 秒。当前回放上限为 0.20 rad/s 和 0.50 rad/s²，需要重新安排各段时间才能尝试更接近原速。 [轨迹检查](../../../results/real_20261128_demo0_replay_preflight/report.json)

## 2026-11-28 双臂工作台

在本仓库根目录运行离线检查。

```bash
scripts/a2_pi0/deploy/run_pi0.sh --check
```

该命令检查 Python 依赖、四路相机模块、示范起点和重力模型。真机工作台入口为 `scripts/a2_pi0/deploy/run_pi0.sh`，需要本机图形桌面和已准备好的两条 CAN FD 接口。旧 Dora 真机图、点动和位置复用入口已退役。Dora 模拟联调仍可使用 `dataflow_policy_mock.yaml`。

### 本机接线与启动条件

- 11 月 28 日读取系统设备信息，PEAK PCAN-USB Pro FD 的 USB 父路径为 `1-3.1:1.0`，`can0` 和 `can1` 均为 `DOWN / STOPPED`。沿用既有接线记录，`can0` 对应右臂，`can1` 对应左臂。该次检查未启用接口或收发电机帧，未确认双臂供电和在线状态。 [现场核验](../../../docs/on-site-verification.md#2026-11-28-电脑端连接检查)
- 现场若仍为该 USB 路径，普通启动时指定 `OPENARM_PI0_CAN_PARENT=1-3.1:1.0`。入口会检查接口为 UP、`ERROR-ACTIVE`、已开启 CAN FD，仲裁和数据速率分别为 1 和 5 Mbit/s，采样点均为 0.75，数据段 SJW 为 2，自动重启为 0。启动器只读系统接口信息。
- 本机枚举到四路 USB 相机，物理路径与 11 月 25 日记录一致。新工作台沿用相邻项目的四路服务，默认按其 bus 3 拓扑查相机。本机当前是 bus 1，且未找到工作台所需的 `/usr/local/lib/openarm-dp/uvc_pair_probe`。现场还需准备设备授权及适配实际机位的四路服务。工作台可以使用位于 `~/.cache/openarm-dp-field/camera.sock` 的兼容服务，须提供 `dp_snapshot`、四路机位身份及已应用的方向信息。

### 控制与停机

控制代码取自 `../openarm_diffusion_policy` 提交 `3b771160b4e0b5b7c8a3586689695a31d85954e8`，相邻仓库路径可用 `OPENARM_DP_SOURCE` 指定。逐臂控制和工作台逻辑在 [dp_control](dp_control)。模型使用本仓库的 `.venv`，逐臂进程默认使用 `.cache/openarm-policy-driver/bin/python`，需要官方驱动和 MuJoCo。本机该环境安装了 MuJoCo 3.8.1，也可用 `OPENARM_PI0_DRIVER_PYTHON` 指定等效环境。

1. 点①查询双臂八轴，检查两侧新鲜反馈均报告失能。
2. 确认工作区、承托和现场硬件停机后，点②确认使能并回官方位。工作台先检查四路观测和推理，双臂都确认使能后才放行回位。
3. 双臂回官方位后，现场核对整条运动路径，再点③去示范起点。
4. 双臂到位后，核对抽屉、方块和双臂，点④启动 π0。工作台只接收④之后的新观测所生成的计划。

双臂沿用原 pedestal PD，默认从首条位置指令开始施加 100% 模型重力前馈，J1–J7 的逐轴限幅为 6、6、2、6、1、1、1 N·m。`--no-gravity` 为零前馈对照。示范过渡和策略执行的关节目标限速为 0.30 rad/s，夹爪为 0.50 rad/s。工作台以 10 Hz 向逐臂进程发送从 π0 的 30 Hz、50×16 动作块取样得到的目标，两侧分别复核限位、反馈和有效期。策略阶段最多 45 秒。

红色停止键、Esc、窗口关闭或运行故障均请求失能并锁定会话。运动后复位需要收到两侧八轴失能确认；现场危险或软件无响应时使用实体停机。诊断记录保存在 `logs/openarm-pi0-trials/`，包含原观测时间、当前派发时间及逐臂反馈。

### 模型与验收状态

工作台默认使用 11 月 28 日批次续训第 2,000 步（累计 3,000 步）的检查点和同批次第 0 条示范首帧。新模型和起点的组合尚未通过真机闭环验收。新批次只教到放入方块并释放，不含推合抽屉。 [新批次数据](../../../data/README.md#2026-11-28-重训批次)

π0 工作台的控制迁移已通过离线和模拟测试。旧 Dora 回放中，旧实采快照的 π0 推理约 320 ms，并返回完整动作块。π0 工作台自身的实机回位和 π0 闭环尚待验收。以下为历史试验记录。

## 2026-11-25 双臂摆位与模型回放

现场确认 `1-4` 全局、`1-3.4` 胸前、`1-3.3` 左腕、`1-3.2` 右腕。摆位由官方 OpenArm 驱动执行限速目标，使用第 5 条示教首帧的关节位置和示教全开夹爪目标作为终点。目标速度上限为 0.08 rad/s，约 6 秒到终点附近，短时保持约 1 秒采样。全开时反馈为右 −45.15°、左 +44.99°，关节的最大到位误差为左 J4 的 4.44°，超过 0.05 rad 容差。四路画面与双臂反馈时间跨度 54.3 ms。采样后独立确认 16 轴失能、两条 CAN 接口 DOWN。模型随后读取保存的快照进行推理。 [摆位日志](../../../logs/openarm-demo-start.t6wgTJ/live.jsonl) · [同步观测](../../../logs/openarm-demo-start.t6wgTJ/capture.json) · [推理结果](../../../logs/openarm-demo-start.t6wgTJ/inference.json)

使用实拍快照可复现第二次规划越界。在仓库根目录运行，Dora 图的目标接收端为 mock。

```bash
export PATH="$PWD/.venv/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OPENARM_MOCK_CASE=model OPENARM_MOCK_SNAPSHOT_DIR="$PWD/logs/openarm-demo-start.t6wgTJ" \
  dora run scripts/a2_pi0/deploy/dataflow_policy_mock.yaml --stop-after 35s
```

- 首次推理约 320 ms，首个动作块从第 0 步发出，mock 两侧各收到连续 19 步。第二次推理的第 10 步右夹爪达到 +0.194 rad，首次超过 +0.174533 rad 上限，动作块被拒绝，后续目标停发。该动作块的右夹爪峰值为 +0.294 rad。 [Dora 回放](../../../logs/openarm-step-zero.KFkqOn/model.log)
- 固定画面和随机种子，只把右夹爪输入从 −0.788 rad 换到 −0.572 rad，越界步数从 0 增至 39。完整的第二次输入产生 40 步越界。保留初始右夹爪角、更新其余 15 维时为 0 步。换成第 5 条示教索引 11 的四路画面，保留第二次模拟反馈时仍有 42 步越界。对照支持右夹爪角变化是本次越界的主要触发因素。训练和验证候选窗口的右夹爪目标均在该上限内。 [输入对照](../../../logs/openarm-step-zero.KFkqOn/gripper-replan-check.json) · [样本清单](../../../data/processed/real_camera_adjusted_candidate_v001/manifest.json)
- 22 项离线回归通过。真实 Dora 进程的确定性回放验证了连续重规划、错失后续时间槽、初始目标跳变拒绝和推理超时停发。模型回放复用了固定图像，并即时回显目标作为反馈。真机闭环仍需验证。 [回放验收汇总](../../../logs/openarm-step-zero.KFkqOn/summary.json)

## 2026-11-25 右 J7 双编码器实测

左右 J7 固件均能用官方 `openarm_can` 分读 `p_m`（`0x50`，由电机侧换算到输出角）及 `xout`（`0x51`，输出轴编码器实测角）。标准位置反馈在本轮更接近 `p_m`。右腕失能时收到 90 组连续成对读数，约 30 Hz，每对回包间隔低于 0.3 ms，`xout` 读数峰峰值约 0.040°。右侧两路读数静态相差约 0.76°，左侧在同次快照的差值约 0.007°。这是两路读数的偏差，实际点动位移按每一路各自的基线计算。[双臂寄存器读回](../../../logs/openarm-policy-20261125/j7-encoders.fKIh73.log) · [右侧连续采样](../../../logs/openarm-policy-20261125/right-j7-continuous.hslOWk.log) · [左右只读对比](../../../logs/openarm-policy-20261125/j7-readonly-compare.4Oeraw.log)

官方 OpenArm 驱动负责使能、位置目标和失能，参数查询也由官方库发送。项目适配层负责目标限幅、超时、异常停止与记录，原始 CAN 另行旁听。受限入口以新鲜反馈为起点，只改变受试 J7 的目标，其他轴用其初始位置作为保持目标。右侧正向指令目标上限 0.00915 rad（约 0.524°），目标速度上限 0.005 rad/s（约 0.29°/s），整轮使能上限 3 秒。增益覆盖只作用于当次官方驱动实例中的 J7。四次试验每次下发 101 条 J7 目标，收到 102 组双编码器反馈，从使能到失能约 2.54–2.55 秒。其他关节在运行反馈中位置读数不变。结束后独立复查均确认右臂 8 轴失能，两条 CAN 接口恢复 DOWN。[旧受限入口源码](feedback_trial.py)

表中位移用使能后前 250 ms 与失能前末 250 ms 的各路角度中位数相减。目标幅度由原始 MIT 帧解码，扭矩是电机回报峰值。

| 右 J7 试验 | CAN 目标幅度 | 扭矩峰值 | 电机侧位移 | 输出轴位移 | 原始记录 |
| --- | ---: | ---: | ---: | ---: | --- |
| `Kp = 10` | +0.503° | 0.076 N·m | +0.017° | +0.017° | [日志](../../../logs/openarm-policy-20261125/right-j7-motion-encoders.h6vJSI.log) · [CAN](../../../logs/openarm-policy-20261125/right-j7-motion-encoders.h6vJSI.log.frames) |
| `Kp = 15` | +0.503° | 0.120 N·m | +0.033° | +0.030° | [日志](../../../logs/openarm-policy-20261125/right-j7-kp15.c61lhY.log) · [CAN](../../../logs/openarm-policy-20261125/right-j7-kp15.c61lhY.log.frames) |
| `Kp = 20` | +0.525° | 0.125 N·m | +0.191° | +0.047° | [日志](../../../logs/openarm-policy-20261125/right-j7-kp20.P1f5WC.log) · [CAN](../../../logs/openarm-policy-20261125/right-j7-kp20.P1f5WC.log.frames) |
| `Kp = 20` 原样复测 | +0.525° | 0.120 N·m | +0.138° | +0.051° | [日志](../../../logs/openarm-policy-20261125/right-j7-kp20-recovery.Sz5Uh1.log) · [CAN](../../../logs/openarm-policy-20261125/right-j7-kp20-recovery.Sz5Uh1.log.frames) |

用户手转电机时未感到异常卡滞。第一次 `Kp = 20` 的 J7 起始角比 `Kp = 15` 试验多约 7.2°，J5 也相差约 7.1°，跨增益对比含姿态差异。第二次 `Kp = 20` 的起始位置读数与第一次失能后的复查相同，再次测得电机侧位移大于输出轴位移。输出轴位移小，是否可用作精确跟踪仍待独立核对。历史 71 段记录筛出的 13 次近似起转片段，回报扭矩中位数约 0.15 N·m，姿态与本轮不同，当前姿态的起转阈值仍待测量。[历史实采数据](../../../data/README.md)

### 失能后的采样缺口

复测原始 CAN 在发出失能帧后约 0.64 ms 先收到 J7 状态帧，约 0.77 ms 收到 `p_m` 有效参数回复。两种消息共用 `0x17`，严格配对检查拒绝该组，即时 `xout` 查询随之未发出。该次 `p_m` 回包仍接近使能末段读数。约 1.002 秒后，8 轴失能确认和两路读数均有效。相对使能末段中位数，`p_m` 回退约 0.054°，`xout` 读数回退约 0.034°。缺少失能后即时成对数据，当前无法确定回退起点，或区分传动间隙、受力变形和读数波动。

### 夹爪模式读回与左臂状态

一次右 J8 临时切换由用户单独授权，在右臂失能时将运行时 `CTRL_MODE` 从 MIT（1）设为 POS_FORCE（4），读回确认通过。该次仅操作右 J8。后续点动在使能前通过官方库发出新查询，读取 J8 的具体模式值。固定版官方库的 `get_link_stats()` 返回实时对象，查询前保存计数值用于判断新回复。常规状态帧先到时，继续等实际填入模式寄存器的回复。左 J7 镜像点动入口已离线验证，尚未真机使能。按官方模型换算镜像姿态，上次两臂 J6 仍差约 22°，左右点动对照须先匹配姿态和负载。[右 J8 模式切换日志](../../../logs/openarm-policy-20261125/right-gripper-mode.n6orYM.log) · [现场与接口记录](../../../docs/on-site-verification.md)

回归检查共 18 项，包括左右接口、单轴增益覆盖、编码器缺失停机和失能后采样。运行 `.venv/bin/python -m unittest discover -s scripts/a2_pi0/deploy -p 'test_*_offline.py' -q` 可复查。失能后的回归用例覆盖正常采样，实际混帧路径仍待用本轮抓包复现。[最新交接事项](../../../docs/project-handoff.md#2026-11-25-实机部署交接) · [OpenArm V2 官方资料本地归档](../../../references/openarm-v2/index.html)

## 2026-11-24 右臂试验与联调记录

2026-11-24 执行右腕 J7 真机点动。官方驱动记录的最终目标相对初始位置为 +0.005 rad，107 组新鲜反馈中 J7 读数始终为 −0.017357 rad，未达到位移验收条件，3 秒观察窗口结束后停止。其他轴的最大反馈变化出现在 J6，约 0.00687 rad，夹爪位置读数无变化。官方 stop 和独立只读检查均确认右臂 8 轴失能，两条 CAN 接口恢复 DOWN。驱动运行期间记录 `write_no_buffer +3`，退出后的独立只读探针无总线故障增量；现有日志无法将这 3 次写失败归因到某个轴或解释 J7 的结果。未增加目标幅度、速度、力矩或重复运动。原始日志 `logs/openarm-policy-20261124/feedback-trial/motion-real.log`。

获准以原参数复测一次。57 组通过校验的反馈中，J7 最大变化约 +0.00038 rad（0.022°），仍未达到 +0.005 rad 目标的验收范围，夹爪角度读数无变化。保持目标的 J6 在下一组反馈触发 0.01 rad 偏差门槛，程序提前停机。运行期间再次出现 `write_no_buffer +3`，原因待查。独立只读检查确认右臂 8 轴失能，两条 CAN 接口恢复 DOWN。原始日志 `logs/openarm-policy-20261124/feedback-trial/motion-real-retry.log`。后续入口会记录被拒绝反馈的具体目标、实测值和时间戳。

离线复盘发现，复测最后一组通过校验的反馈中 J6 比保持目标高约 0.00916 rad，约 1.15 秒后的独立只读复查中比预检高约 0.3044 rad（17.4°）。旧日志无法区分过阈值后的变化是发生在失能前、过程中还是失能后。新版记录触发保护的整组位置与时间戳，先发送停止请求再输出这条日志。点动驱动保留官方失能顺序，记录进入停机、`disable_all` 返回及官方第一次读取的时间和位置，再记录逐轴核验新鲜度后的失能状态和位置。第一次官方读数未验证每轴应答，不视为新鲜反馈；`disable_all` 返回也不代表电机已经停稳。

新增的离线模拟验证了首条目标前发生 CAN 写入失败时不会发布运行反馈或发送位置目标，并走官方失能复查；模拟 J6 偏离时控制端先请求停止、不再发目标，再输出触发保护的读数。运行 `.venv/bin/python -m unittest discover -s scripts/a2_pi0/deploy -p 'test_*_offline.py' -v`，当前共 8 项通过。此次未访问硬件，未改动作幅度、速度或保护阈值；再次点动前仍需核对失能后承托。

与相邻项目对照，WebXR 手册记录右臂曾完成小幅跟随，使用相同的官方 pedestal 驱动节点，但由官方初始姿态接续并连续发送整臂目标。该流程的初始姿态与本机从新鲜反馈开始的点动不同。读取本仓库 10 段实采验证轨迹的右臂状态，J7 每段反馈跨度约 0.076–0.371 rad，J6 约 0.488–0.892 rad，证明历史采集中两轴均有反馈变化。原项目记录的 Dora 队列丢弃与本次 CAN 发送缓冲告警属于不同层级。本次两份点动日志均在首条位置目标前报告 `write_no_buffer +3`，真机仅反馈试验未报告此告警，现有记录无法定位具体写请求或判定 J7 受限原因。参考 `vr_teleoperation/docs/实机_WebXR_操作手册.md` 和 `vr_teleoperation/docs/PROJECT_STATUS.md`。

核对固定版本源码后，确认 `write_no_buffer` 对应本机 CAN 写入返回 `ENOBUFS` 或 `EAGAIN`，官方 Dora 的 `state` 消息省略了此计数。试验入口现直接读取官方驱动的发送计数，用 `trial_can_transmit` 区分使能前反馈、使能、首轮状态请求及启动确认，运行中继续检查反馈和位置发送。一旦发现写入失败，先拒绝继续发目标，通过已有退出流程复查 8 轴失能。使用固定版启停代码和模拟 CAN 验证了正常路径及六个阶段的写入失败。

离线用两份点动日志中驱动记录的 J7 目标，按固定版 [OpenArmCAN 编码公式](https://github.com/enactic/openarm_can/blob/f340d4b808fb177e1f297af54eb55fd51c6c7c10/src/openarm/damiao_motor/dm_motor_control.cpp#L136-L155) 和本次配置的零偏换算，首末位置编码均从 32722 变为 32735（共 14 个不同编码）。这排除了整个动作因软件编码精度不足而消失，但日志记录的是驱动目标，不能证明电机收到或执行。另增回归测试：模拟在使能阶段出现 3 次发送失败，节点不发布 started、无运行反馈或目标、进入失能复查。离线测试共 9 项通过。未改驱动参数或运行真机。

2026-11-24 获准执行分阶段日志下的右臂仅反馈真机试验。使能前反馈、使能、首轮状态请求和启动确认的 CAN 写入失败计数均为零。启动保护拒绝进入连续采样，Dora 收到 0 组运行反馈，且该数据流没有位置目标通道。启动保护同时检查电机状态、链路状态和位置变化，此次日志未保存具体触发项。试验入口已补充 `trial_startup_rejected` 日志，记录逐轴状态、链路状态和数值。退出后的独立只读检查确认右臂 8 轴全部失能、总线故障计数为零、试验前后位置读数一致，两条 CAN 接口均为 DOWN。单次反馈试验没有重现此前点动的发送告警。原始日志 `logs/openarm-policy-20261124/feedback-trial/transmit-feedback-real.K27Jw6.log`。

同日复测捕捉到启动保护的触发项。右臂 J1–J7 报告 `ENABLED`，第 8 轴夹爪报告 `DISABLED`，8 轴启动读数与使能前一致，链路正常，各阶段 CAN 写入失败计数为零。启动保护结束试验，无位置目标，连续反馈为 0 组。退出后的独立只读复查确认右臂 8 轴均失能，模式仍为 7 个 MIT 加 1 个 POS_FORCE，前后位置读数一致，双 CAN 接口恢复 DOWN。此前反馈试验曾收到夹爪 `ENABLED`。下一步需区分夹爪状态回报晚到和本次使能未生效，保留全轴使能要求。原始日志 `logs/openarm-policy-20261124/feedback-trial/startup-diagnostic-real.raRV3q.log`。

旧版仅反馈入口的夹爪启动诊断在 J1–J7 已使能、夹爪仍报告失能且链路和位置正常时，最多额外观察 60 ms、读取 3 组新鲜反馈，但无论观察结果如何都拒绝启动。旧版模拟日志 `logs/openarm-policy-20261124/feedback-trial/gripper-startup-offline.log`。

2026-11-24 获准执行一次真机夹爪启动诊断。首次反馈仍只有夹爪报告失能，约 11.09 ms 后的首组新鲜反馈中 8 轴均报告使能，期间未重发使能或发送位置目标。由此确认本次夹爪使能状态晚于首次检查出现，未测量实际状态切换的精确时刻。各阶段 CAN 写入失败计数为零。诊断按设计结束于拒绝启动，未进入 Dora 连续采样。退出后的独立只读检查确认 8 轴均失能、前后位置读数一致、总线故障计数为零，两条 CAN 接口恢复 DOWN。下一步先离线验证短时等待全部 8 轴确认使能的启动检查，再用仅反馈入口验证持续反馈。真机日志 `logs/openarm-policy-20261124/feedback-trial/gripper-startup-real.1hyKpy.log`。

右臂仅反馈入口最先接入短时等待。首次检查仅在 J1–J7 已使能、夹爪仍报告失能、位置正常、链路正常且总线计数与使能前一致时进入等待。最多等待 60 ms、检查 3 组逐轴有新增应答的反馈；只在窗口内确认 8 轴全部使能且其余检查仍正常时进入连续采样。等待时只请求反馈，不重发使能或下发位置目标。日志记录进入等待、逐组状态、确认耗时或拒绝原因。

从仓库根目录运行 `.venv/bin/python -m unittest discover -s scripts/a2_pi0/deploy -p 'test_*_offline.py' -v`，6 项测试通过（含故障子用例）。使用固定版官方节点和启停代码，替换 CAN 与 Dora 事件输入而不访问设备：立即确认和第 3 组才确认均发布了 10 组模拟运行反馈，并复查模拟失能；持续失能、缺少逐轴应答、反馈超时、位置偏差、链路掉线、总线计数变化、发送错误和关节状态异常均未发布运行反馈，并复查模拟失能。

2026-11-24 新版右臂仅反馈入口通过授权真机验收。首次反馈仅夹爪报告失能，11.12 ms 后首组追加反馈确认 8 轴全使能；随后收到 10 组运行反馈，首末组相隔约 224 ms。未重发使能或发送位置目标。各阶段记录的 CAN 故障计数均为零。停机与独立只读复查均确认 8 轴失能，双 CAN 接口最终均为 DOWN。运行反馈中 J5 相对初值最大变化约 +0.00038 rad，独立复查比预检高约 +0.00076 rad，其余轴的前后位置读数一致。此次结果不解释历史点动日志中的 `write_no_buffer` 或 J6 漂移。原始日志 `logs/openarm-policy-20261124/feedback-trial/gripper-confirmation-real.lZTvBt.log`。

仅反馈入口接入了同一套停机记录。对照最后一组 `trial_feedback`、控制端 `trial_stop_requested`、驱动 `trial_stop_timing` 和 `trial_disabled_confirmed` 中的时间和位置，可以比较停机前后的读数。驱动记录区分失能调用返回后第一次未核验新鲜度的读数，以及随后确认每轴有新增应答的失能反馈；前者不能当作失能确认。模拟验证了沿用原停机顺序、先调用失能再输出新增停机日志，两次读数保留各自时间及位置，且没有位置目标通道。离线回归共 9 项通过。

2026-11-24 现场值守下验证新版停机记录。右臂收满 10 组新鲜反馈（首末约 225 ms），无位置目标，启动与运行未记录 CAN 写入错误。末组使能态位置与预检读数一致（忽略浮点精度差）；失能调用返回约 1.002 秒后的新鲜读数中，J5 相对末组增加约 0.0191 rad，J6 减少约 0.00343 rad。随后的独立只读复查确认 8 轴均失能，J5 相对预检累计增加约 0.0221 rad（1.27°），J6 累计减少约 0.00420 rad（0.24°）。这不证明偏移具体发生在失能前、过程中还是失能后；第一次读数未核验新鲜度。可确认的是，两次已验明失能的读数之间 J5 仍变化约 0.00305 rad（0.17°）。CAN 故障计数为零，两接口均恢复 DOWN。未进入点动。原始日志 `logs/openarm-policy-20261124/feedback-trial/feedback-stop-timing-real.veUZRv.log`。

同日现场修正承托后，仅反馈复测一次。右臂起始姿态与上一轮不同；收到 10 组新鲜反馈（首末约 224 ms），无位置目标或 CAN 写入错误。8 轴使能期间读数与本次预检一致（忽略浮点精度差），官方停机复查和随后独立复查均确认 8 轴失能，8 轴位置均与本次预检完全一致。两接口均恢复 DOWN。结果仅适用于这次承托和姿态，不解释历史点动异常，也未验证手腕运动。原始日志 `logs/openarm-policy-20261124/feedback-trial/feedback-support-retry-real.VQTrRs.log`。

同日获准试转一次右 J7。点动入口要求首次启动反馈的 8 轴全使能，不沿用仅反馈入口的额外等待。本次首次反馈为 J1–J7 已使能、夹爪报告失能，位置与使能前一致，通信计数为零。启动保护拒绝放行，未发送位置目标，未再次尝试。官方与独立复查均确认 8 轴失能，位置与预检一致，两接口均为 DOWN。这次没有检验到 J7 的运动能力。原始日志 `logs/openarm-policy-20261124/feedback-trial/motion-supported-real.2QHT7H.log`。

上述试转之后，点动入口也接入了同一个夹爪等待窗口。仍需收到 8 轴新鲜使能反馈，并通过位置、通信检查才会启动；等待期间不发位置目标，原有角度、速度及动作保护不变。8 项离线测试通过，包含两种模式的正常放行、夹爪晚到及持续不使能、反馈缺失或超时、位置与链路异常等子用例。

新版点动获准在同一承托下尝试一次。使能前、使能及首次刷新阶段未报告发送错误；接下来启动确认阶段 `startup_feedback` 记录 `write_no_buffer +2`、`last_write_errno=105`（`ENOBUFS`）。尚未进入夹爪等待，未发送任何位置目标，未重试。官方与独立复查均确认 8 轴失能，位置与预检一致，两接口均为 DOWN。这次把写入失败定位到启动确认的刷新区间，不能据此确定具体失败帧或历史两次故障是否来自同一阶段。原始日志 `logs/openarm-policy-20261124/feedback-trial/motion-gripper-wait-real.Glx4eE.log`。

后续离线修改了共用启动流程：在官方首次刷新前清理已到达的消息并记录逐轴回复计数，等待 8 轴计数都增长，核对位置、链路和总线后才发启动确认查询。首轮等候最多 60 ms；第二轮发出后立即检查发送失败，再核对反馈。未调整 CAN 队列或运动参数，夹爪确认条件不变。11 项离线测试通过，包括首轮回报跨多次读取、缺轴、位置异常和第二轮发送失败。逐轴计数增长不是请求编号匹配，离线模拟也不能证明真机发送缓冲不再满。

2026-11-24 在现场值守下先验证新版共用启动：右臂仅反馈收满 10 组、未发动作目标、各阶段无 CAN 发送错误，独立复查确认 8 轴失能、位置与预检一致。之后试转一次右 J7，驱动记录 107 条目标，J7 目标从约 −0.14553 rad 到 −0.14053 rad（+0.005 rad），107 组新鲜反馈中 J7 位置读数始终为 −0.14553 rad，其余轴读数也无可分辨变化。达到 3 秒观察上限后停止，未满足运动验收；此次未重现 CAN 发送错误。驱动目标记录不等于电机确认执行。独立复查确认 8 轴失能，位置与预检完全一致，两条 CAN 接口均为 DOWN。没有自动重试或增加角度、速度和力度。日志：`logs/openarm-policy-20261124/feedback-trial/feedback-serial-start-real.u7zHih.log`、`logs/openarm-policy-20261124/feedback-trial/motion-serial-start-real.YH5ZOK.log`。

右腕 J7 点动入口 `dataflow_motion_trial.yaml` 已通过官方 Dora 节点加模拟电机的验证。首次动作只对右臂 J7 增加 0.005 rad（约 0.29°），目标速度不超过 0.005 rad/s（约 0.29°/s），整个观测窗口最多 3 秒。使用官方对齐检查，首条目标等于新鲜反馈位置，夹爪触发仅适用于原手柄流程，本次测试使用独立现场授权。其余关节与右夹爪发送初始保持目标，实际夹爪保持效果须观察。启动时官方驱动会同值重写右夹爪 POS_FORCE。验证结束或反馈丢失、非测试轴明显偏离、超时、越界时请求停止和失能，重新读取状态；意外移动、夹爪开合、承托移位或软件停止无响应时由现场值守人员操作双臂空开。左臂不使能。此入口的授权值是 `RIGHT_J7_POSITIVE_005_RAD`，仅供该具体点动试验，先前反馈试验的授权不适用于这一步。模拟故障与正常日志在 `logs/openarm-policy-20261124/mock-lifecycle/motion-*.log`。

2026-11-24 右臂官方 Dora 真机反馈试验通过。首次尝试在使能前因官方默认读窗口不足退出，独立复查确认 8 轴失能。将反馈等待改为最多 60 ms 且仍逐轴核对本次新增回复后复测成功。右臂 8 轴使能反馈完整，共读到 10 组，位置相对使能前无可分辨变化。官方 stop 后和独立只读复查均确认 8 轴失能，CAN 故障计数为零。两条 CAN 接口均恢复为 DOWN。没有位置目标、左臂使能或模型控制。日志见 `logs/openarm-policy-20261124/feedback-trial/retry-real.log`。授权限于本次反馈试验。

首次真机测试入口已准备为 `dataflow_feedback_trial.yaml`。它只连接右臂 `can0` 的官方 Dora 节点，保留官方使能和失能流程，跳过预设启动和停车轨迹。数据流没有位置目标通道，预定关节和夹爪位移均为 0。驱动在使能前核对 8 轴新鲜反馈和失能状态。使能后最多读取 10 组反馈，观察窗口上限 2 秒，之后请求失能并重新读取 8 轴状态。反馈过期、总线状态变化、任一轴相对于使能前或试验初始位置变化超过 0.02 rad（约 1.15°）时提前停止。此阈值用于触发停止，无法限制实际位移。现场出现意外移动、支撑位移或软件停止无响应时，由值守人员操作双臂空开并避开可能下落的区域。左臂和相机不参与这次试验。

此入口在实际 Dora 进程与模拟电机上通过正常结束和单轴反馈停止更新两项验证，结果在 `logs/openarm-policy-20261124/mock-lifecycle/feedback-trial-*.log`。右臂真机使能及失能结果见上文。运行真机前需确保原有 VR 和诊断工具退出，复查硬件空开值守及承托，再核查右臂 CAN 独占和新鲜只读反馈。

官方 V2 pedestal 驱动构造时会向右夹爪重发 POS_FORCE 模式设置，之前只读检查已确认它当时处于此模式。按用户的限制，首次真机授权须明确包括这次同值设置，以及右臂短时使能和失能。驱动卡死或电脑故障时不能依赖软件发送失能命令。真机图由 `.venv/bin/dora` 运行，主进程使用本机 `.venv`，官方节点使用 `.cache/openarm-policy-driver`。入口的 `OPENARM_FEEDBACK_TRIAL_AUTHORIZATION=RIGHT_FEEDBACK_ENABLE_DISABLE` 由执行者在取得现场授权后设置。

2026-11-24 用户确认机器为 OpenArm V2、零位已标定，实际接线沿用记录，物理 CAN1 / `can0` 接右臂，CAN2 / `can1` 接左臂。用户确认两臂有承托，硬件空开同时控制两臂，测试时有人值守。本次对话中空开处于通电状态，这不是下次连接的状态证明。承托细节、夹爪反馈映射与软件偏移的本机来源仍未核实。现场信息及其证据见 `docs/on-site-verification.md`。

下次连接先核对当前接线、承托、现场空开值守、现有控制进程和接口状态，并从新鲜反馈建立初态；不要沿用上次角度。已有本机只读授权，运动仍按用户要求在说明具体动作、范围、速度、持续时间和停止条件后取得现场明确授权。VR、策略与诊断程序不能争用同一 CAN 总线。

当前真机闭环的待验收事项见[项目交接](../../../docs/project-handoff.md#2026-11-25-实机部署交接)。先补齐右 J7 失能后的成对反馈，核对实际夹爪保持与左右实时机位映射，再接入实时观测和模型动作。

空启动和停止轨迹的官方驱动生命周期曾在模拟对象上验证。测试观察到使能、用模拟反馈初始化目标及失能，未调用轨迹。真机执行结果见本次验收表。

`small_motion_mock.py` 是独立的离线计算验证，模拟单关节 1 秒内变化 0.01 rad，并验证反馈维度、有限值、年龄、代次、范围及偏差拒绝。运行 `.venv/bin/python scripts/a2_pi0/deploy/small_motion_mock.py`。它使用合成反馈并立即回显目标，不进行实时等待，不接入 Dora 或官方驱动，也不验证真实限位、夹爪或停止效果，不能用于实机控制。

`dataflow_small_motion_mock.yaml` 将相同轨迹计算接到 Dora 定时器及模拟接收节点，使用合成初态、立即回显的反馈。2026-11-24 正常测试发送并接收 47 条目标，最短间隔约 20 ms；断联测试停止反馈后，以 `feedback_stale` 结束并停止发送。运行 `PATH="$PWD/.venv/bin:$PATH" SMALL_MOTION_CASE=normal dora run scripts/a2_pi0/deploy/dataflow_small_motion_mock.yaml --stop-after 3s`，将 `normal` 改为 `disconnect` 可复查断联。此模拟图不加载官方驱动，也未测试真实机械臂动力学、夹爪映射或实际停止。官方驱动的启停和异常路径还需另行验证。

`dataflow_small_motion_official_mock.yaml` 运行已安装的官方 Dora 节点，将其 `SingleArmDriver` 替换为内存模拟对象，不打开 CAN。它复用原项目的 `--side right --align-trigger gripper --config openarm_pedestal` 参数。2026-11-24 使用满足夹爪触发条件的合成数据，观察到 stopped、started、aligned 状态，并完成 45 条带代次目标和停止消息。运行 `PATH="$PWD/.venv/bin:$PATH" dora run scripts/a2_pi0/deploy/dataflow_small_motion_official_mock.yaml --stop-after 4s` 复查。该模拟不验证真实驱动的使能、夹爪、对齐或机械停止行为，不能用于实机控制。

用 `SMALL_MOTION_FAULT=feedback_delay` 注入一次 300 ms 的反馈延迟，采用原项目对齐参数的模拟图以 `feedback_stale` 退出并发送 stop，模拟驱动收到停止。该次控制端发送 11 条目标，模拟驱动处理了 7 条。控制端的发送与驱动实际处理存在时间差，测试不证明真机上排队命令会被清除，也不证明失联后机械臂停稳。

模拟控制端增加了启动后等待反馈的超时，以及运行中状态或代次改变时的拒绝处理。2026-11-24 模拟启动反馈延迟 2.5 秒时，控制端在 2 秒超时且发送目标数为零，驱动读取返回后才处理 stop。控制端现在最多同时等待一个反馈请求，避免读取阻塞时重复请求积压；这不限制动作消息积压。模拟状态变化的直接事件测试以 `status_or_epoch_changed` 拒绝，尚未做 Dora 端到端状态变化注入。模拟驱动异常退出时控制端因输入关闭而退出，官方节点异常路径未调用模拟 stop。真实受限入口仍未完成，真实夹爪反馈能否作为保持目标仍未确定。

官方模拟图的动作输入已设为单条队列。一次 300 ms 反馈延迟测试出现队列丢弃，控制端拒绝后模拟驱动没有再处理目标，但 stop 在约 225 ms 后才处理。该结果只覆盖一次模拟，不能推断所有调度情况或真机停稳。日志在 `logs/openarm-policy-20261124/mock-lifecycle/one-slot-delay-timing.log`。

官方模拟包装器现在对 Python 异常执行模拟 stop。故障注入验证了异常时的清理调用及控制端输入关闭退出；该包装器只使用内存模拟驱动，无法处理进程强杀、系统故障或读取永久阻塞，也未用于真机。另用官方 `SingleArmDriver.send_position` 和模拟电机检查左右夹爪输出，目标加对应软件偏移后发送，测试输入分别为 −0.4 和 +0.4 rad；使用当前示例配置得到同值。此测试不验证电机反馈准确性、夹爪机械运动或本机偏移来源。

`.venv/bin/python scripts/a2_pi0/deploy/test_official_gripper_offline.py` 可复查上述官方驱动夹爪换算。`SMALL_MOTION_FAULT=gripper_drift` 在官方模拟图中模拟夹爪反馈偏离 0.04 rad，控制端检测偏差后拒绝继续发送并请求停止。两项验证不访问 CAN，无法确认真机夹爪保持状态。

模拟中发现官方 Dora 节点即使驱动 `start()` 返回 false 仍会发布 started。模拟控制端增加八电机状态和总线检查，故障注入时在发送目标前以 `feedback_health` 拒绝；正常场景复测通过。模拟采用总线错误计数必须为零的保守条件，真机错误计数可能是累计值，需结合实际基线处理。本检查不证明使能、反馈或停止的真机行为。

`guarded_official_node.py` 在官方节点处理动作前拒绝缺少时间戳、超过有效期或代次格式错误的消息。启动失败触发异常，并在可执行 Python 清理时调用官方 stop。模拟控制器为动作附加 50 ms 有效期，测试了正常动作、过期动作拒绝及 Python 异常触发 stop。真机受限图也使用该适配层。时间有效期依赖本机墙钟，驱动永久阻塞、进程被强杀或断电时无法保证执行软件清理。

额外注入官方驱动启动失败时，包装器报告失败并调用模拟 stop，控制端因输入关闭退出，未发送动作。该验证只覆盖模拟驱动，真机启动配置、初始反馈和夹爪动作仍须现场验证。

### 左臂 J4、J5 跟随记录

已有授权运动流程使用 `guarded_official_node.py` 作为驱动入口时，可以给驱动进程设置 `OPENARM_TRACE_LEFT_TRACKING=1`。记录只在左臂成功启动后的运行段生效。每条记录中数组的顺序是 J4、J5，角度使用去除软件零偏后的关节坐标，单位为弧度。

- `left_tracking_command` 记录驱动入口目标、限位限速后的下发目标和实际调用下发的主机时间。拒绝的指令保留拒绝原因。CAN 写入数量和失败计数一并记录。
- `left_tracking_feedback` 只对读到新应答的轴填实测角度、速度和力矩。记录主机读取时间区间，并在下发写入计数正常时关联读到反馈时的最近目标。可用 `dispatch_ns` 对齐两类记录。该时间关系表示主机的观测顺序，反馈帧没有可用于确认某条命令已由电机收到的序号。若要独立确认线上指令及应答时间，需同时旁听左臂 CAN。
- `original_rad` 是进入官方驱动的目标。若当时处于对齐阶段，它是对齐逻辑产生的目标。评价原始示教指令时用对齐完成后的记录。

离线复查命令为 `.cache/openarm-policy-driver/bin/python -m unittest discover -s scripts/a2_pi0/deploy -p 'test_left_tracking_trace_offline.py'`。

## 2026-11-24 右手腕排查

2026-11-24 另做一次带旁听的右 J7 试转：驱动收到 107 组反馈，J7 位置读数不变，试验到时停止。独立复查 8 轴失能且位置与预检一致，两接口均为 DOWN。旁听程序在接口开启前绑定，日志为空，连已知存在的电机状态回复也未抓到。这不是线上没有动作指令的证据；再次用于点动前，应在接口就绪后启动旁听，并先用不使能的状态查询验证能捕获回复。试验日志 `logs/openarm-policy-20261124/feedback-trial/motion-passive-capture-real.3f7hIb.log`。

随后用不使能的状态查询验证旁听工具。正式旁听试转抓到 107 条发往 J7 的本机发送记录，期间 212 条 J7 使能态回复的位置编码不变；回报力矩随目标变化。本机发送回显不等于电机逐条确认。左右 J7 的关键设置只读对照一致，未发现地址、控制模式或驱动换算范围不符。证据见[交接表](../../../docs/project-handoff.md#验收到哪一步)。

旧采集数据中筛出 4 个近似对照：J7 从接近静止朝同方向转，其他关节几乎没动。起转前目标与实际角度差约 0.52°–1.11°，回报力矩最高约 0.061–0.178 N·m；整臂姿态不同，不能当作起转门槛。当天最后一次授权试转目标约 0.54°，回报力矩最高约 0.081 N·m，107 组反馈及 212 条原始使能态回复均未测到 J7 转动。试验超时退出，独立确认 8 轴失能、两接口为 DOWN。日志 `logs/openarm-policy-20261124/feedback-trial/motion-wrist-start-probe.crNuwK.log`。

现场报告无外部阻挡或异常阻力；用户感觉最后一次手腕似乎动了，但未有独立证据确认。下一步优先用已有录像对照读数，分清手腕相对前臂转动与整臂或夹爪晃动；无录像则由现场技术人员按厂家流程检查位置反馈和传动。未执行 90°、3° 或 5° 试转。不要靠加大角度、速度或力度验证读数是否可靠。

## 本机与模型

- 本仓库及其父目录未找到适用的 `AGENTS.md`。相邻 `openarm_diffusion_policy` 的项目与遥操作规则已查阅。
- Python 3.11.17、OpenPI 0.1.0（源码固定提交 `215abfb`）、JAX/JAXLIB 0.5.3、Flax 0.10.2、NumPy 1.26.4、Arrow 21.0.0。GPU 为 RTX 5060 Ti 16 GB。Dora SDK 和 CLI 1.0.1 已装入本项目 `.venv`。
- 本次选用累计 6500 步 LoRA（续训目录 `step_3000`）及本地 `pi0_base`。训练配置在同名 `results/real_camera_adjusted_lora_continue5000_20261122_v002/config.json`。模型加载会核对 LoRA 和归一化哈希。训练候选来自相机调整后的批次。
- 连接台架后，本机枚举到同一 PCAN-USB Pro FD 的两个 CAN 通道和四台 AK-Camera。只读探针确认两个通道都以 1 Mbps / 5 Mbps CAN FD 响应，各有 8 个电机。型号、接线和零位的用户确认见上文，夹爪反馈映射仍待核实。
- 相邻项目 `vr_teleoperation/local/robot_real_pedestal.yaml` 引用官方节点。其 `upstream/` 在本机缺失，真实启动器还引用采集电脑上的绝对路径。本机已从固定源码将 dora-openarm 1.3.0、openarm-driver 0.5.1、openarm-can 1.4.0 安装到 `.cache/openarm-policy-driver`，静态读取了安装包的 V2 pedestal 参考配置，并比对了 Arrow 格式和代次校验。 [采集根提交](https://github.com/enactic/dora-openarm-data-collection/tree/5be94fe9873b02d5123bedc96e07272ac0112c2d) · [官方 Dora 节点](https://github.com/enactic/dora-openarm/blob/1a6b29f41923fdc940959a60e24dfbe3467fbca1/src/dora_openarm/main.py) · [官方驱动及 V2 pedestal 配置](https://github.com/enactic/openarm_driver/tree/5c73d835b31875057695f1b8688e0bf9c0bf87b0/src/openarm_driver)
- 2026-11-24 恢复相邻项目固定版本 libuvc 和相机采集程序，离线 framing 测试通过。安装私有 TurboJPEG 2.1.2 文件并核对项目指定 SHA256，无损旋转 5 项离线测试通过。采集专用 Dora 1.1.0 SDK 和 CLI 安装至相邻项目 `deps/dora-capture-sdk/python`，二进制哈希均与项目 runtime-pins 一致。没有打开相机，当前 USB 物理端口与历史配置不同，仍须现场确认机位映射。

## 接口

| 信号 | 格式和语义 |
| --- | --- |
| 图像 `camera_{global,chest,wrist_left,wrist_right}` | Arrow UInt8 JPEG 字节。原始批次的 global 和双腕标记 `rot180-v1`，chest 标记 `identity-v1`。保留原方向转 RGB，对应训练槽位 `base_0_rgb`、`chest_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`，按 OpenPI 的 224×224 等比缩放填充处理。一次推理取一个四路观测，没有时序图像堆叠。 |
| 模型状态 | 16 维顺序为右 J1–J7、右夹爪、左 J1–J7、左夹爪。J1 到 J7 从肩到腕，位置单位 rad。状态归一化后补到 32 维。模型只使用 `qpos`、图像和固定任务文本。 |
| 模型动作 | 50×32 原始归一化输出，取前 16 列按训练统计 `x * (std + 1e-6) + mean` 还原为绝对关节目标，按状态顺序拆成双臂。直接使用关节目标，无需 IK。 |
| 夹爪 | 第 8 维为电机角。右侧开 −45°、闭 +10°，左侧开 +45°、闭 −10°，这是采集用目标端点。官方驱动对目标加关节零位偏置，反馈从电机角减偏置。开合方向和接触后的反馈需在本机验证。 |
| `move_position` | 每臂 1 行 Arrow struct `qpos: list<float32>`，8 个值，metadata 必须携带相应侧当前 `start_epoch`。状态未对齐、代次不一致或变化时，策略停发并锁定故障。 |
| `request_state` / `state` | 请求走独立节拍。固定驱动忽略请求载荷。反馈为 1 行 Arrow struct，含 `qpos/qvel/qtorque/tmos/trotor/motor_status/bus`，metadata 含 `observation_timestamp` 和 `start_epoch`。 |
| `command` / `status` | 官方节点从单元素 Arrow 字符串读取 `start` 或 `stop`，状态输出为 `stopped/started/aligned` 与当前代次。策略读取状态，启停控制归现场控制源。 |
| 时钟 | 相机 `timestamp_ns` 是主机接收时间。反馈 `observation_timestamp` 是驱动完成读取时间。同一主机上校验来源年龄，用单调时钟调度。采集训练使用因果对齐后的 30 Hz 网格。回放保留原始相机时刻，同时附加本次回放时刻。 |
| 频率 | 训练时域为 50 步 × 1/30 秒。mock 图每 20 步异步重规划，约 1.5 Hz，新的时间槽按 30 Hz 排列，调度器每 5 ms 检查，每槽最多发一次，允许 15 ms 调度延迟。热推理约 240 ms。录制状态和遥操目标约 250 Hz。官方节点按输入调用驱动，电机内部环路频率、真机发送和跟踪效果待验。 |

策略层验证整个动作块的形状、有限值、目标范围和相邻时间步速度。首次推理期间 mock 维持初始反馈，验证通过的动作块在首个可发送时刻从第 0 步开始。后续重规划期间继续发送上一动作块，切换时用新鲜双臂反馈和上一条已发送目标检查新动作块的第 0 步。每次发送都校验双臂代次，后续迟到的动作槽跳过，动作块有效期仍从原始观测时刻计算。超时、过期或拒绝会锁定本次运行。配置里的限位、0.10 rad 跟踪门槛和时间预算用于本机 mock 验证，真机参数须根据确认后的配置和授权测试重新确定。驱动正常跟随路径另做位置限位、速度修正与过跳变拒绝，不会将 30 Hz 目标插值为 250 Hz。

## 离线运行

在本仓库根目录执行。若需在同一 Python 环境重装 Dora，使用 `uv pip install --python .venv/bin/python dora-rs==1.0.1 dora-rs-cli==1.0.1`。

```bash
export PATH="$PWD/.venv/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
deploy=scripts/a2_pi0/deploy
run=real_camera_adjusted_lora_continue5000_20261122_v002

python "$deploy/policy_server.py" --run-dir "results/$run" --self-test
python "$deploy/policy_server.py" --run-dir "results/$run" \
  --ckpt "checkpoints/$run/step_3000" --probe --log /tmp/openarm-policy-probe.jsonl

OPENARM_MOCK_CASE=model OPENARM_MOCK_REPLAY_INDEX=773 \
  dora run "$deploy/dataflow_policy_mock.yaml" --stop-after 35s \
  > /tmp/openarm-dora-model.log 2>&1
OPENARM_MOCK_CASE=transport dora run "$deploy/dataflow_policy_mock.yaml" --stop-after 3s \
  > /tmp/openarm-dora-transport.log 2>&1
```

`model` 使用 checkpoint 和真实录制的四路 JPEG、初始反馈。mock 驱动即刻回显收到的关节目标。索引 773 选取 episode 5 尾段的一个验证观测，未指定索引时使用该验证轨迹首个观测。`transport` 使用明确标记的确定性测试动作，验证接收和反馈。可用 `pause`、`dimension`、`nan`、`range`、`jump`、`stale`、`timeout`、`epoch`、`status_stop` 和 `disconnect` 复查指定故障，使用 `--stop-after 3s` 或 Ctrl+C 结束 mock 图。

设置 `OPENARM_MOCK_SNAPSHOT_DIR` 可改用现场保存的初始快照，例如 `"$PWD/logs/openarm-demo-start.t6wgTJ"`。目录应包含方向已处理的四路同名 JPEG 和带 `qpos_rad`、`frame_ns` 的 `capture.json`。mock 重复发布这些静态画面并回显后续位置目标，因此该回放可以检验首个动作块和重规划保护。

探针 JSONL 和 Dora 日志包含观测时间、推理耗时、模型原始输出、反归一化目标、Dora 发布目标及反馈、拒绝原因。用 `seq`、`step` 和 `obs_timestamp_ns` 关联 `observation`、`inference`、`dispatch` 和 `mock_target`，检查 `reject`、`mock_watchdog`、`policy_exit` 和 `mock_exit` 定位中断。Dora 节点日志也存在部署目录的 `out/`。Dora 发布目标是节点输入层证据，真机驱动的限幅和 CAN 实际发送需要独立确认。

`camera_socket_node.py` 是后续实时图像入口。它通过 `OPENARM_CAMERA_SOCKET` 连接现有四相机服务，复用 `snapshot` 和 JPEG 校验，每 33 ms 请求已做方向处理的图像，四个 `camera_*` 输出接策略侧同名输入。按需用 `OPENARM_FOUR_CAMERA_DIR` 指定相邻采集项目的 `local/four_camera` 目录。旧桥的 `status_*` 只携带时间戳，`camera_*` 只在录制期间发出。

## 2026-11-24 基础联调验收

| 层级 | 结果 |
| --- | --- |
| 离线模型 | 4 个实采样本的预处理与训练逐元素相同。5 次同一实采观测探针加载了 6500 步 checkpoint，反归一化后均有有限的 50×16 输出，预热后端到端约 254–282 ms。 |
| Dora 传输 | 实际 Dora 运行验证了双臂载荷、对应代次、反馈回路、跳过错失时间槽、故障锁定和退出。四相机 Unix socket 转接也通过录制 JPEG 回放验证。 |
| 模型经 Dora 到 mock | 验证索引 773 收到 19 对目标并回传反馈，第二次重规划因目标与反馈差值超限而停发。索引 0 的第一次下发判定即触发门槛。 |
| 真机只读反馈 | `can0` 和 `can1` 各收到 8 个电机的新鲜位置、速度和力矩反馈。两边 ID 均为 1–8，回复 ID 为 0x11–0x18，7 个关节为 MIT，夹爪为 POS_FORCE。反馈显示电机均失能，总线无新增故障。两个接口已恢复为 DOWN。 |
| 官方 Dora 真机反馈 | 右臂短时使能读取 10 组反馈，结束后独立复查 8 轴失能。左臂仅通过 OpenArmCAN 只读反馈。 |
| 右 J7 点动 | 在真机测得电机侧小幅转动和输出轴较小的同步变化。两次 `Kp = 20` 均出现位移差，输出轴目标跟踪和失能后即时成对采样待验收。 |
| 任务闭环 | 待现场验证。 |

本机日志保存在 `logs/openarm-policy-20261124/` 和 `logs/openarm-policy-20261125/`，由 Git 忽略。

## 真机授权和停机边界

1. 获得真机只读授权并确认设备已连接后，先确认铭牌、左右线缆和独占总线，核对已安装的官方驱动版本及现场配置。使用经源码核对的参数读请求和状态刷新请求，只记录本次实际响应，逐侧验证八个电机反馈完整、时间新鲜、左右归属可追溯。采集机的 CAN 名称、USB 拓扑和配置中的零偏是参考。官方 `dora-openarm` 停止态会忽略 `request_state`，V2 pedestal 驱动构造时即会设置夹爪 POS_FORCE 模式。官方 `monitor` 和 `diagnose` 会使能，`discover` 会修改网络口状态及位速率。 [官方 OpenArmCAN 实现](https://github.com/enactic/openarm_can/tree/f340d4b808fb177e1f297af54eb55fd51c6c7c10)
   本机 `readonly_arm_probe.py` 由普通用户在已授权、通过管理员准备的单个 CAN FD 接口上运行。探针使用官方接口刷新一次反馈，用每轴的新增响应计数判断新鲜度，再读取电机 ID、硬件版本、固件版本、序列号及控制模式。记录的角度是电机原始反馈，须结合本机标定后用于策略。发现电机已使能时结束本次检查。
2. 运动需要下一次现场明确授权。测试前明确预计关节、范围、速度、持续时间、停止条件和现场急停方式，从双臂新鲜实际反馈建立初态，确认非测试轴的安全保持方式。先验收官方驱动与停止链路，再进行小范围命令验证及策略接入。真实图应只有一个目标源，策略运行时互斥停用 VR/遥操作与诊断。双臂分别使用已确认的接口和代次。
3. 固定驱动的 `start` 使能后执行配置启动轨迹。对齐可能发送增量运动，gripper 门控受实际目标影响。官方 `stop` 先发 stopped，随后执行默认停车轨迹再失能。退出默认走 stop，`--no-stop` 会移到启动位。无制动关节失能可能下落。策略停发与 mock 看门狗只验证软件行为，真机的停稳、承托与急停方式须现场确认。
