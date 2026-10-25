# 本文件负责：将 A1 原生环境包装为 A2 当前模型所需的观测和位置动作接口。
# 对外使用右臂在前的16维顺序，提供顶部及双腕RGB、限幅反馈和30Hz名义时序。
# 保留原物理步长，以绝对物理步编号调度33/34步，避免控制周期误差持续累积。
# 优先保留场景原有相机；缺少顶部相机时才使用待标定的备用参数。
# CELL额外升降轴保持home控制目标，不进入模型的16维状态和动作。

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from simulation.env import DEFAULT_SCENE, OpenArmEnv, ROOT


# 一、显式协议：左右臂顺序和相机名称，不依赖模型内的偶然索引顺序。
POLICY_ACTUATORS = tuple(
    name
    for side in ("right", "left")
    for name in (
        *(f"{side}_joint{i}_ctrl" for i in range(1, 8)),
        f"{side}_finger1_ctrl",
    )
)
CAMERAS = {
    "observation.images.ceiling": "camera_ceiling",
    "observation.images.wrist_left": "camera_wrist_left",
    "observation.images.wrist_right": "camera_wrist_right",
}


@dataclass(frozen=True)
class PolicyInterfaceConfig:
    """保存名义控制频率、渲染尺寸、指令和待标定的顶部相机位姿。

    相机采用 MuJoCo 的 wxyz 四元数；单位姿态朝向本地 -Z，即竖直向下。
    默认位置仅用于接口验收，正式实验须对照采集视频确认外参与视场角。
    """

    control_hz: float = 30.0
    width: int = 960
    height: int = 600
    prompt: str = "Pick up the cube."
    ceiling_position: tuple = (0.35, 0.0, 1.6)
    ceiling_quaternion: tuple = (1.0, 0.0, 0.0, 0.0)
    ceiling_fovy: float = 60.0
    # 仅用于CELL视觉网格透明度对照；None保留官方外观，碰撞体不变。
    cell_visual_alpha: float | None = None

    @classmethod
    def from_file(cls, path=ROOT / "configs/a2_interface.json"):
        """从 JSON 加载配置；未知字段交给构造器报错，避免静默拼写错误。"""
        return cls(**json.loads(Path(path).read_text()))

    def validate(self):
        """检查尺寸、频率、指令和相机参数，非法值抛出 ValueError。"""
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz 必须为正有限数")
        for size in (self.width, self.height):
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError("渲染尺寸必须为正整数")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt 必须是非空字符串")
        if self.cell_visual_alpha is not None and (
            not np.isfinite(self.cell_visual_alpha) or not 0 <= self.cell_visual_alpha <= 1
        ):
            raise ValueError("cell_visual_alpha必须在[0,1]内")
        position = np.asarray(self.ceiling_position, dtype=float)
        quaternion = np.asarray(self.ceiling_quaternion, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("顶部相机位置必须为三维有限向量")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("顶部相机四元数必须为四维有限向量")
        if not np.isclose(np.linalg.norm(quaternion), 1.0):
            raise ValueError("顶部相机四元数必须为单位四元数")
        if not np.isfinite(self.ceiling_fovy) or not 0 < self.ceiling_fovy < 180:
            raise ValueError("顶部相机 fovy 必须在 (0, 180) 度内")


# 二、模型编辑：只在内存中添加顶部相机，不写回官方子模块。
def compile_policy_model(scene_path, config):
    """读取官方 MJCF，保留已有相机，仅在缺失时补充顶部相机。

    保留原有关节、执行器、物理步长与双腕相机；资源不存在时明确提示恢复子模块。
    """
    scene_path = Path(scene_path).resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"场景不存在：{scene_path}；请运行 git submodule update --init --recursive"
        )
    spec = mujoco.MjSpec.from_file(str(scene_path))
    camera = next((cam for cam in spec.cameras if cam.name == "camera_ceiling"), None)
    # 场景已有顶部相机时保留其外参和视场角。
    # 仅在场景缺少顶部相机时，使用配置中的备用参数添加相机。
    if camera is None:
        camera = spec.worldbody.add_camera(name="camera_ceiling")
        camera.pos = config.ceiling_position
        camera.quat = config.ceiling_quaternion
        camera.fovy = config.ceiling_fovy
    model = spec.compile()
    # 只改变非碰撞CELL外观网格的alpha；不调整质量、接触、关节或物理参数。
    if config.cell_visual_alpha is not None:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cell_vis")
        if gid < 0:
            raise ValueError("当前场景缺少CELL视觉网格cell_vis")
        if model.geom_contype[gid] != 0 or model.geom_conaffinity[gid] != 0:
            raise ValueError("只允许修改非碰撞视觉网格")
        model.geom_rgba[gid, 3] = config.cell_visual_alpha
    # 离屏缓冲区需覆盖请求尺寸，否则高于默认尺寸的 Renderer 会失败。
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)
    return model


class OpenArmPolicyEnv:
    """提供与当前 A2 策略直接匹配的字典协议，独立于 OpenPI/JAX 依赖。

    输入必须是已反归一化的单步16维位置目标；不接收补齐32维或整块动作。
    对外顺序为右臂七关节及夹爪，再接左臂七关节及夹爪。
    数值暂按仿真原生弧度解释，不自动翻转符号或学习采集端标定。
    """

    def __init__(self, config=None, scene_path=DEFAULT_SCENE):
        """编译三相机模型，按执行器名建立映射并准备惰性渲染器。"""
        self.config = config if config is not None else PolicyInterfaceConfig()
        self.config.validate()
        model = compile_policy_model(scene_path, self.config)
        self.env = OpenArmEnv(model=model)
        # 双臂协议按名称匹配；CELL只允许额外的升降轴执行器。
        actual_names = set(self.env.actuator_names)
        missing = set(POLICY_ACTUATORS) - actual_names
        unsupported = actual_names - set(POLICY_ACTUATORS) - {"lifter_ctrl"}
        if missing or unsupported:
            raise ValueError(
                f"场景执行器不匹配：缺少={sorted(missing)}，"
                f"不支持的额外执行器={sorted(unsupported)}"
            )
        self.policy_indices = np.array([
            self.env.actuator_names.index(name) for name in POLICY_ACTUATORS
        ])
        for name in CAMERAS.values():
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError(f"场景缺少相机：{name}")
        self.physics_dt = float(model.opt.timestep)
        if self.config.control_hz * self.physics_dt > 1:
            raise ValueError("控制频率不能高于物理积分频率")
        # 有些执行器允许超出几何限位的目标，桥接接口取两种范围的交集。
        # 限位使用原生执行器维度，包含CELL的升降轴。
        self.lower = np.full(model.nu, -np.inf)
        self.upper = np.full(model.nu, np.inf)
        for aid in range(model.nu):
            if model.actuator_ctrllimited[aid]:
                self.lower[aid], self.upper[aid] = model.actuator_ctrlrange[aid]
            jid = int(model.actuator_trnid[aid, 0])
            if model.jnt_limited[jid]:
                self.lower[aid] = max(self.lower[aid], model.jnt_range[jid, 0])
                self.upper[aid] = min(self.upper[aid], model.jnt_range[jid, 1])
        if np.any(self.lower > self.upper):
            raise ValueError("执行器与关节限位没有交集")
        self._renderer = None
        self.reset(render=False)

    def to_native(self, action):
        """映射16维双臂目标，额外升降轴使用reset保存的home目标。

        返回原生执行器维度的独立副本，不限幅、不换单位、不推进物理时间。
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("动作必须是数值有限的16维向量")
        native = self._held_native.copy()
        native[self.policy_indices] = action
        return native

    def to_policy(self, native):
        """从原生向量提取右臂优先的16维双臂值，排除额外升降轴。"""
        native = np.asarray(native, dtype=np.float64)
        expected = (self.env.model.nu,)
        if native.shape != expected or not np.isfinite(native).all():
            raise ValueError(f"原生向量必须是数值有限的{expected[0]}维向量")
        return native[self.policy_indices].copy()

    def reset(self, *, render=True):
        """恢复 home 和控制时钟；render=False 可跳过离屏图像用于数值验收。"""
        native = self.env.reset()
        # 每次reset重新保存home控制目标；to_native仅覆盖双臂对应位置。
        self._held_native = np.asarray(native["ctrl"], dtype=np.float64).copy()
        self._policy_step = 0
        self._physics_tick = 0
        self._last_substeps = 0
        self._requested = self.to_policy(native["ctrl"])
        self._applied = self._requested.copy()
        return self.get_observation(render=render)

    def step(self, action, *, render=True):
        """执行一个模型时间步，返回状态、可选三路图像和限幅反馈。

        物理步长仍为1ms时，30Hz周期交替推进33/34步；每个目标时间从
        绝对序号计算，误差不超过半个物理步且不累积。实际时间以 timestamp 为准。
        """
        native = self.to_native(action)
        applied = np.clip(native, self.lower, self.upper)
        if not np.isclose(self.env.data.time, self._physics_tick * self.physics_dt, atol=1e-9, rtol=0):
            raise RuntimeError("底层仿真被外部推进，请通过本接口 reset/step 管理时间")
        next_step = self._policy_step + 1
        target_tick = round(next_step / (self.config.control_hz * self.physics_dt))
        substeps = target_tick - self._physics_tick
        if substeps < 1:
            raise RuntimeError("当前控制周期不足一个物理积分步")
        self.env.data.ctrl[:] = applied
        for _ in range(substeps):
            mujoco.mj_step(self.env.model, self.env.data)
        mujoco.mj_forward(self.env.model, self.env.data)
        self._requested = self.to_policy(native)
        self._applied = self.to_policy(applied)
        self._policy_step = next_step
        self._physics_tick = target_tick
        self._last_substeps = substeps
        return self.get_observation(render=render)

    def get_observation(self, *, render=True):
        """返回当前实际状态和诊断字段，图像在同一时刻连续渲染且不推进物理。

        默认输出可直接交给 A2 OpenArmInputs；render=False 缺少图像，仅可用于数值检查。
        control.applied 是目标位置，不是实际测得的 observation.state。
        """
        raw = self.env.get_observation()
        nominal_time = self._policy_step / self.config.control_hz
        observation = {
            "observation.state": self.to_policy(raw["joint_pos"]).astype(np.float32),
            "prompt": self.config.prompt,
            "timestamp": raw["timestamp"],
            "nominal_timestamp": nominal_time,
            "timing.error_seconds": raw["timestamp"] - nominal_time,
            "timing.substeps": self._last_substeps,
            "timing.actual_dt_seconds": self._last_substeps * self.physics_dt,
            "control.requested": self._requested.copy(),
            "control.applied": self._applied.copy(),
            "control.clipped_mask": self._requested != self._applied,
        }
        if render:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.env.model, height=self.config.height, width=self.config.width
                )
            for key, camera in CAMERAS.items():
                self._renderer.update_scene(self.env.data, camera=camera)
                observation[key] = self._renderer.render().copy()
        return observation

    def close(self):
        """释放惰性创建的渲染资源，可以重复调用。"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        """进入上下文，保证异常退出时也释放渲染资源。"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文并释放资源，不吞掉调用方异常。"""
        self.close()
