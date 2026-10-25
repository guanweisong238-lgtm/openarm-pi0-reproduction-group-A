# A2 模型与公开数据接口适配

本文件说明新增模型侧包装器的职责、配置、调用方式与验收边界。更新日期：2026-10-23。

## 1. 入口与范围

- `simulation/env.py`：原生执行器顺序的位置环境，默认20Hz、物理步长1ms，已有调用方式继续可用。
- `simulation/policy_env.py`：新增 `OpenArmPolicyEnv`，以右臂优先顺序、名义30Hz及三路图像对接A2。
- `configs/a2_interface.json`：频率、图像尺寸、指令及顶部相机初始位姿。
- `scripts/check_policy_interface.py`：真实模型上的数值、时间和离屏渲染验收。

官方模型子模块仍固定为 `a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c`，相机通过 `MjSpec` 在内存中处理：已有相机保留，缺失顶部相机时才添加备用相机；未修改上游MJCF或资产。

当前CELL场景的episode 0示教回放已完成，用户目视确认抓取并放入托盘。此结论只适用于该条示教，不代表模型自主控制成功或完整标定通过。

## 2. 独立环境与验收命令

在A1项目根目录执行：

```bash
git submodule update --init --recursive
uv sync --locked

# 数值、限幅和30Hz时钟，不需要图形上下文
uv run --locked python -m scripts.check_policy_interface --no-render

# 完整数值及三路图像验收；无窗口使用EGL
MUJOCO_GL=egl uv run --locked python -m scripts.check_policy_interface

# 已有原生接口的回归检查
uv run --locked python -m scripts.test_env
uv run --locked python -m scripts.test_control --headless
uv run --locked python -m scripts.validate_sample
```

依赖使用本项目锁文件中的 MuJoCo 3.16.0、NumPy 2.4.6，不把OpenPI/JAX或A2的NumPy约束装入A1。

产物写入Git忽略目录 `outputs/a2_interface/`：

- `ceiling.png`、`wrist_left.png`、`wrist_right.png`：本次渲染样张。
- `policy_observation.npz`：可在A2独立进程中检查的原始策略观测。
- `validation.json`：包含实际三路图像检查的完整验收记录。
- `numeric_validation.json`：单独的无渲染验收记录，不覆盖完整记录。

## 3. 观测和动作协议

```python
from simulation.policy_env import OpenArmPolicyEnv, PolicyInterfaceConfig

config = PolicyInterfaceConfig.from_file()
with OpenArmPolicyEnv(config, scene_path="openarm_mujoco/v2/cell/demo.xml") as env:
    observation = env.reset()
    # 这里只保持复位目标演示接口；模型输出须先由A2完成反归一化。
    action = observation["control.applied"].copy()
    next_observation = env.step(action)
```

### 状态与单步动作

`observation.state` 是实际测得的 `(16,) float32` 状态，顺序固定为：

```text
right_joint1..7, right_gripper, left_joint1..7, left_gripper
```

`step(action)` 接收已反归一化的 `(16,)` 位置目标，按仿真原生弧度解释；不接收模型内部32维向量或整块动作。A2模型当前输出 `(16, 16)` 动作块，由A3逐步调用，需结合新观测决定何时重推理，不代表必须一次执行完16步。

索引按执行器名称建立；pedestal原生顺序对应索引为 `[8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7]`；CELL为 `[9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8]`。CELL额外的 `lifter_ctrl` 不进入16维协议，每次reset后保持home目标0；这是位置目标保持，不是机械锁定。

此包装器已经完成双向重排，A3使用本接口时不得再次交换左右臂。若A3继续使用原生 `OpenArmEnv`，则仍需自行转换。

### 图像与任务指令

| 输出键 | MuJoCo相机 | 默认数组 |
| --- | --- | --- |
| `observation.images.ceiling` | `camera_ceiling` | `(600,960,3)` RGB uint8 |
| `observation.images.wrist_left` | `camera_wrist_left` | 同上 |
| `observation.images.wrist_right` | `camera_wrist_right` | 同上 |

三路图像在同一仿真状态渲染，不在相机之间推进物理。模型侧负责缩放至224×224，适配器不做图像归一化。`prompt` 默认是 `Pick up the cube.`。

`reset()`、`step()`、`get_observation()` 默认返回三路图像；传 `render=False` 只返回数值诊断，此结果不能直接用于A2推理。

### 时间与限幅反馈

| 字段 | 含义 |
| --- | --- |
| `timestamp` | 实际物理仿真时间，秒 |
| `nominal_timestamp` | 控制步数 / control_hz，秒 |
| `timing.error_seconds` | 实际时间减名义时间 |
| `timing.substeps` | 上一步实际执行的物理子步数，复位为0 |
| `timing.actual_dt_seconds` | 上一步实际推进的时间，复位为0 |
| `control.requested` | 模型顺序的未限幅目标 |
| `control.applied` | 模型顺序的实际下发目标，不能当作测得状态 |
| `control.clipped_mask` | 每一维是否发生限幅 |

物理步长保留1ms，30Hz采用绝对目标物理步编号调度33/34步。各周期实际间隔为33或34ms，名义30Hz，不声称每步严格等于1/30秒；单步时钟误差不超过0.5ms，300周期为10秒，无累计漂移。正式实时运行的墙钟时间和模型推理延迟仍需A3管理。

控制目标限制在执行器控制范围和关节几何范围的交集内。NaN、Inf或错误维度在推进仿真前被拒绝。公开示教中存在超限目标，回放时应统计限幅比例并核查采集端标定，不能把裁剪当成物理兼容性证明。

## 4. 顶部相机配置与待标定项

JSON配置字段：

| 字段 | 含义 |
| --- | --- |
| `control_hz` | 名义策略频率，当前30Hz |
| `width` / `height` | 三路渲染宽、高，当前960/600像素 |
| `prompt` | 当前任务指令 |
| `ceiling_position` | 世界坐标位置，单位米，初值 `[0.35,0,1.6]` |
| `ceiling_quaternion` | wxyz单位四元数，初值朝向世界-Z |
| `ceiling_fovy` | 垂直视场角，初值60度 |

以上顶部相机参数仅在场景缺少该相机时生效。CELL直接保留官方顶部及双腕相机，三者fovy均为66度。样张和单条回放通过，不等于相机或全部采集场景已完成标定。

仍待完成：

- 采集端单位、关节零点/方向及左右夹爪定义确认。
- 包装器reset仍恢复各场景home；CELL的home左夹爪已打开。当前联调另执行2秒平滑移动和5秒保持，定位episode 0首帧；该初始化不等于任意任务复位。
- 真实数据到仿真限位差异及限幅比例分析。
- CELL默认方块与托盘已支持episode 0示教抓放；其余轨迹初始条件、接触标定及自动成功判定仍待验证。
- 更多示教回放和实际策略的闭环评估。

## 5. 本机验收结果

2026-10-23，本机在独立A1环境中运行：

- 原生环境100步得到约5秒，原生T1–T6控制响应通过。
- 原有100帧样本回放通过，最大状态误差约 `1.24e-7 rad`。
- 新接口的动作/状态重排、非法输入拒绝、限幅反馈和复位检查通过。
- 300个名义30Hz周期得到 `9.999999999999897 s`，子步数为33/34，最大时钟误差约 `0.333334 ms`。
- EGL三路渲染均输出 `(600,960,3)` uint8，渲染前后物理时间不变。
- 将A1的NPZ观测快照交给A2独立环境，现有 `OpenArmInputs`、归一化、分词和补齐流程通过，得到状态 `(1,32)`、Token `(1,48)` 和三路 `(1,224,224,3)` 图像。

上述基础接口验收报告将标定和任务成功标记为未验收；后续CELL人工任务确认单独记录于第6节。

## 6. CELL示教回放验收及接口冻结记录

2026-10-23，场景 `openarm_mujoco/v2/cell/demo.xml`，数据集episode 0共263帧全部执行。用户目视确认“成功夹取并放入托盘”。记录为 `task_success_verified=true`，验证方式为人工观察；`calibration_verified=false`。

- 首帧最大误差：0.020106554 rad，未达到严格小于0.02 rad的检查条件。
- 平均绝对状态误差：0.004115203 rad；最大绝对误差：0.066250473 rad。
- 64帧右夹爪目标被限幅，其余维度无目标限幅。
- 原始报告：`outputs/a2_interface/cell_episode_000000_20261023-212812-678641/report.json`。

本轮联调固定使用本文16维协议、30Hz名义控制、960×600 RGB三路图像、右臂在前、绝对位置目标和升降轴home目标保持。模型检查点为10步试训练产物，模型推理与示教回放的任务结果分别记录。

`replay_episode.py --pose-only` 定位后会保存 `policy_observation.npz`，供A2独立环境推理。快照包含原始状态、prompt、三路图像、timestamp及模型顺序限位；不包含示教动作标签。首次联调使用重新构建的首帧快照，来源记录在快照旁的 `source.json`。
