from pathlib import Path
import mujoco
import numpy as np
from simulation.env import OpenArmEnv

ROOT = Path(__file__).resolve().parents[1]
output_dir = ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 实例化环境 (控制周期 20Hz -> 0.05s)
env = OpenArmEnv(control_dt=0.05)
obs = env.reset()

# 2. 找到控制关节索引 (通过名称动态定位，避免硬编码)
left_elbow_idx = env.actuator_names.index("left_joint4_ctrl")
right_elbow_idx = env.actuator_names.index("right_joint4_ctrl")

# 3. 固化 Home 姿态作为基准动作 (防止增量累加漂移越界)
base_action = obs["ctrl"].copy()

timestamps = []
states = []          # 执行前观测状态 (16维)
actions = []         # 实际下发的动作 (16维)
imgs_left = []
imgs_right = []

print("🎬 开始修正版 5 秒轨迹录制 (共 100 步)...")

with mujoco.Renderer(env.model, width=320, height=240) as renderer:
    for step in range(100):
        # A. 因果律对齐：先记录 t 时刻的观测
        timestamps.append(float(obs["timestamp"]))
        states.append(obs["joint_pos"].copy())

        # 采集双腕图像 (同一瞬间，中间绝不 step)
        renderer.update_scene(env.data, camera="camera_wrist_left")
        imgs_left.append(renderer.render().copy())

        renderer.update_scene(env.data, camera="camera_wrist_right")
        imgs_right.append(renderer.render().copy())

        # B. 围绕固定基准位置进行平滑小幅正弦摆动 (振幅 0.08 rad)
        action = base_action.copy()
        oscillation = 0.08 * np.sin(step * 0.1)
        action[left_elbow_idx] += oscillation
        action[right_elbow_idx] += oscillation

        # C. 核心修复：先限幅检查，确保保存的 action 100% 等于实际执行的 action
        action_clipped = env.check_action(action)
        actions.append(action_clipped.copy())

        # D. 执行动作
        obs = env.step(action_clipped)

# 4. 提取终端最终观测 (5.0s 时刻)
terminal_state = obs["joint_pos"].copy()
terminal_time = float(obs["timestamp"])

# 5. 打包并写入完整元数据
npz_file = output_dir / "sample_episode.npz"

np.savez_compressed(
    npz_file,
    timestamps=np.array(timestamps, dtype=np.float32),      # (100,)
    states=np.array(states, dtype=np.float32),              # (100, 16)
    actions=np.array(actions, dtype=np.float32),            # (100, 16)
    images_left=np.array(imgs_left, dtype=np.uint8),        # (100, 240, 320, 3)
    images_right=np.array(imgs_right, dtype=np.uint8),      # (100, 240, 320, 3)
    terminal_state=np.array(terminal_state, dtype=np.float32), # (16,)
    terminal_time=np.array(terminal_time, dtype=np.float32),
    # 验收必需元数据
    scene_path="openarm_mujoco/v2/pedestal/pedestal.xml",
    model_commit="a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c",
    actuator_names=np.array(env.actuator_names),
    camera_names=np.array(["camera_wrist_left", "camera_wrist_right"]),
    units="rad",
    robot_version="v2",
    control_dt=0.05
)

print(f"🎉 成功生成修正版数据样本: {npz_file}")

# 6. 本地严苛自检断言
assert len(timestamps) == 100, "必须正好 100 帧"
assert np.allclose(np.diff(timestamps), 0.05, atol=1e-5), "时间间隔必须严格为 0.05s"
assert np.all(np.array(actions) >= env.model.actuator_ctrlrange[:, 0] - 1e-6), "动作不能下限越界"
assert np.all(np.array(actions) <= env.model.actuator_ctrlrange[:, 1] + 1e-6), "动作不能上限越界"
print("✅ 自检通过：无任何动作越界，元数据齐全，时序因果对齐！")
