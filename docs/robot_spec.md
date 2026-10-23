# OpenArm v2 机器人仿真接口技术规范


## 1. 基础物理与仿真参数

| 参数项 | 参数值 | 说明 |
| :--- | :--- | :--- |
| **场景入口 XML** | `openarm_mujoco/v2/pedestal/pedestal.xml` | 官方固定底座双臂标准场景 |
| **物理时间步长 (`physics_dt`)** | `0.001 s` ($1000\text{ Hz}$) | `model.opt.timestep`，物理引擎数值积分频率 |
| **策略控制周期 (`control_dt`)** | `0.05 s` ($20\text{ Hz}$) | `OpenArmEnv` 外部调用交互步频 |
| **子步迭代数 (`n_substeps`)** | `50` | 每次 `step()` 内部执行 `50` 次 `mj_step` |
| **广义位置维度 (`nq`)** | `18` | 14 个手臂关节 + 4 个手指关节 (含 2 被动) |
| **广义速度维度 (`nv`)** | `18` | 同上 |
| **执行器电机维度 (`nu`)** | `16` | 14 个手臂电机 + 2 个主动手指电机 |
| **板载相机数量 (`ncam`)** | `2` | 左右腕部嵌入式 RGB 相机 |
| **基准复位姿态** | 关键帧 `home` | 双肘弯曲约 90 度，双爪角度为 0，接近闭合 |

---

## 2. 16 维动作空间与执行器座次表 (Action Space)

外部策略（$\pi_0$ / Diffusion Policy）传入的动作数组 `action` **长度必须严格为 16**，单位全部为**弧度（$\text{rad}$）**，控制模式为**关节位置控制（Position Servo）**。

| 索引 | 执行器名称 (Actuator Name) | 对应关节 (Joint Name) | 控制范围 (ctrlrange, rad) | 物理含义 |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `left_joint1_ctrl` | `openarm_left_joint1` | `[-3.4907, 1.3963]` | 左臂基座回转 (Base) |
| **1** | `left_joint2_ctrl` | `openarm_left_joint2` | `[-3.3161, 0.1745]` | 左肩俯仰 (Shoulder) |
| **2** | `left_joint3_ctrl` | `openarm_left_joint3` | `[-1.5708, 1.5708]` | 左大臂旋转 (Roll) |
| **3** | `left_joint4_ctrl` | `openarm_left_joint4` | `[0.0000, 2.4435]` | 左肘弯曲 (Elbow) |
| **4** | `left_joint5_ctrl` | `openarm_left_joint5` | `[-1.5708, 1.5708]` | 左小臂旋转 (Wrist Roll) |
| **5** | `left_joint6_ctrl` | `openarm_left_joint6` | `[-0.7854, 0.7854]` | 左腕俯仰 (Wrist Pitch) |
| **6** | `left_joint7_ctrl` | `openarm_left_joint7` | `[-1.5708, 1.5708]` | 左腕侧摆 (Wrist Yaw) |
| **7** | `left_finger1_ctrl` | `openarm_left_finger_joint1` | `[-0.4000, 0.7854]` | **左夹爪主动手指** |
| **8** | `right_joint1_ctrl` | `openarm_right_joint1` | `[-1.3963, 3.4907]` | 右臂基座回转 (Base) |
| **9** | `right_joint2_ctrl` | `openarm_right_joint2` | `[-0.1745, 3.3161]` | 右肩俯仰 (Shoulder) |
| **10** | `right_joint3_ctrl` | `openarm_right_joint3` | `[-1.5708, 1.5708]` | 右大臂旋转 (Roll) |
| **11** | `right_joint4_ctrl` | `openarm_right_joint4` | `[0.0000, 2.4435]` | 右肘弯曲 (Elbow) |
| **12** | `right_joint5_ctrl` | `openarm_right_joint5` | `[-1.5708, 1.5708]` | 右小臂旋转 (Wrist Roll) |
| **13** | `right_joint6_ctrl` | `openarm_right_joint6` | `[-0.7854, 0.7854]` | 右腕俯仰 (Wrist Pitch) |
| **14** | `right_joint7_ctrl` | `openarm_right_joint7` | `[-1.5708, 1.5708]` | 右腕侧摆 (Wrist Yaw) |
| **15** | `right_finger1_ctrl` | `openarm_right_finger_joint1` | `[-0.7854, 0.4000]` | **右夹爪主动手指** |

---

## 3. 夹爪控制与物理特性约定 (Gripper Semantics)

机械爪通过旋转关节驱动（单位：$\text{rad}$），左右手呈**镜像对称**安装：

1. **左夹爪 (`left_finger1_ctrl`)**：
   - 几何范围：`[0.0, 0.7854]` $\text{rad}$ ($0^\circ \sim 45^\circ$)
   - 符号语义：**数值趋近于 `0.0` 时接近闭合；数值趋近于 `0.7854` (正数) 时张开**。
   - 被动手指：`openarm_left_finger_joint2` 无电机，通过等式约束联动跟随。
2. **右夹爪 (`right_finger1_ctrl`)**：
   - 几何范围：`[-0.7854, 0.0]` $\text{rad}$ ($-45^\circ \sim 0^\circ$)
   - 符号语义：**数值趋近于 `0.0` 时接近闭合；数值趋近于 `-0.7854` (负数) 时张开**。
   - 被动手指：`openarm_right_finger_joint2` 无电机，通过等式约束联动跟随。



---

## 4. 相机配置 (Camera Specifications)

OpenArm v2 左右腕部各有一台 RGB 相机。它们是两个不同机械臂上的视角，本工程未实现双目标定或深度重建。渲染使用 MJCF 中定义的相机位姿；输入尺寸以采集脚本的 320×240 为准：

| 相机键名 | XML 相机名 | 分辨率与通道 | 格式 | 视场角与视野 |
| :--- | :--- | :--- | :--- | :--- |
| **`left_wrist`** | `camera_wrist_left` | $320 \times 240 \times 3$ | RGB, uint8 | 左夹爪内嵌手腕视角，随机械臂运动，实际任务视野需与 B1 确认 |
| **`right_wrist`** | `camera_wrist_right` | $320 \times 240 \times 3$ | RGB, uint8 | 右夹爪内嵌手腕视角，随机械臂运动，实际任务视野需与 B1 确认 |

---

## 5. 观测空间字典协议 (`get_observation()`)

每次调用 `env.reset()` 或 `env.step(action)` 返回的观测字典包含：

```python
obs = {
    "timestamp": float,              # 当前物理仿真时间（秒），从 0.0 开始
    "joint_pos": np.ndarray,         # (16,) float64，16维执行器对应关节角度
    "joint_vel": np.ndarray,         # (16,) float64，16维执行器对应关节角速度
    "gripper_pos": np.ndarray,       # (2,) float64，[左夹爪角度, 右夹爪角度]
    "gripper_vel": np.ndarray,       # (2,) float64，[左夹爪角速度, 右夹爪角速度]
    "ctrl": np.ndarray,              # (16,) float64，当前执行器保持的目标位置
    "qpos_raw": np.ndarray,          # (18,) 包含被动关节的原始状态快照副本
    "qvel_raw": np.ndarray,          # (18,) 原始速度快照副本
}
```

---

## 6. 与 OpenPI / LeRobot 适配边界（W2 待联调）

- 本工程交付的是 16 维 MuJoCo 关节位置目标和原始状态，不包含已完成的 OpenPI／LeRobot 转换器。策略模型的填充维度、归一化、图像键名及动作块映射由 A2/A3 联调后确定，不能直接假定模型输出可原样传给环境。
- 当前 NPZ 中图像字段为 `images_left`、`images_right`；若后续转换为 `observation.images.wrist_left` 等键名，需要另外实现并验证转换。
- 每个样本记录执行前的 `(state_t, image_t, action_t)`，100 步后另存 `terminal_state` 和 `terminal_time=5.0`。因此状态转移可用 `states[1:]` 加末帧状态构造；当前没有保存末帧 RGB。
- `actions` 记录限幅后的实际控制目标，存储为 float32；与执行时 float64 比较使用数值容差，而非承诺位级相同。
- `joint_pos` 按执行器顺序提取，包含两个主动手指；`qpos_raw` 才包含被动手指。角度单位 rad，角速度单位 rad/s，时间单位 s。
- `get_observation()` 返回状态字典，RGB 由采样脚本单独渲染，并非环境字典中已有图像键。
