from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "openarm_mujoco/v2/pedestal/pedestal.xml"


class OpenArmEnv:
    def __init__(self, scene_path=DEFAULT_SCENE, control_dt=0.05):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
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
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        # For the verified unit-gear position actuators, hold the home pose.
        self.data.ctrl[:] = self.check_action(self.data.qpos[self.qpos_idx])
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, action):
        self.data.ctrl[:] = self.check_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_observation(self):
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
