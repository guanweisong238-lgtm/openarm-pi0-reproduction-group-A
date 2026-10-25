# 本文件负责：封装 OpenArm 原生执行器顺序的位置控制、状态读取与复位。
# 默认使用官方 pedestal 场景和 20 Hz 控制；模型侧协议由 policy_env.py 包装。
# 可注入已编译的 MuJoCo 模型，用于增加相机等场景编辑，不修改上游模型资产。

from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "openarm_mujoco/v2/pedestal/pedestal.xml"


class OpenArmEnv:
    """按原生执行器顺序提供关节位置目标接口，返回独立的状态快照。"""

    def __init__(self, scene_path=DEFAULT_SCENE, control_dt=0.05, *, model=None):
        """加载场景或接收已编译模型，验证控制周期和关节传动约定。

        model 非空时优先使用注入模型；默认调用方式和物理步长保持原定义。
        control_dt 必须为物理步长的整数倍，否则抛出 ValueError。
        """
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive and finite")
        ratio = self.control_dt / self.model.opt.timestep
        self.n_substeps = round(ratio)
        if self.n_substeps < 1 or not np.isclose(ratio, self.n_substeps):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        self.home_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if self.home_id < 0:
            raise ValueError("Missing home keyframe")
        joint_ids = []
        for aid in range(self.model.nu):
            if self.model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError("Only joint transmissions are supported")
            jid = int(self.model.actuator_trnid[aid, 0])
            if jid < 0 or int(self.model.jnt_type[jid]) not in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
            ):
                raise ValueError("Only hinge/slide joints are supported")
            if not np.allclose(self.model.actuator_gear[aid], [1, 0, 0, 0, 0, 0]):
                raise ValueError("Expected unit actuator gear")
            joint_ids.append(jid)
        self.qpos_idx = self.model.jnt_qposadr[joint_ids]
        self.qvel_idx = self.model.jnt_dofadr[joint_ids]
        self.actuator_names = [
            self.model.actuator(i).name for i in range(self.model.nu)
        ]
        self.gripper_idx = np.array([
            self.actuator_names.index("left_finger1_ctrl"),
            self.actuator_names.index("right_finger1_ctrl"),
        ])
        self.reset()

    def check_action(self, action):
        """检查原生顺序动作的维度和有限值，按执行器控制范围限幅并返回副本。"""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.model.nu,):
            raise ValueError(f"Expected action shape {(self.model.nu,)}")
        if not np.isfinite(action).all():
            raise ValueError("Action contains NaN or Inf")
        action = action.copy()
        mask = self.model.actuator_ctrllimited.astype(bool)
        action[mask] = np.clip(
            action[mask], self.model.actuator_ctrlrange[mask, 0],
            self.model.actuator_ctrlrange[mask, 1]
        )
        return action

    def reset(self):
        """恢复 home 关键帧，保持对应关节目标并返回初始状态。"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        # For the verified unit-gear position actuators, hold the home pose.
        self.data.ctrl[:] = self.check_action(self.data.qpos[self.qpos_idx])
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, action):
        """下发原生顺序的位置目标，推进一个固定控制周期后返回状态。"""
        self.data.ctrl[:] = self.check_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_observation(self):
        """返回时间、实际关节状态和控制目标的副本，不包含 RGB 图像。"""
        return {
            "timestamp": float(self.data.time),
            "qpos_raw": self.data.qpos.copy(),
            "qvel_raw": self.data.qvel.copy(),
            "joint_pos": self.data.qpos[self.qpos_idx].copy(),
            "joint_vel": self.data.qvel[self.qvel_idx].copy(),
            "gripper_pos": self.data.qpos[self.qpos_idx[self.gripper_idx]].copy(),
            "gripper_vel": self.data.qvel[self.qvel_idx[self.gripper_idx]].copy(),
            "ctrl": self.data.ctrl.copy(),
        }
