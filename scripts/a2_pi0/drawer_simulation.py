"""与抽屉采集一致的真实接触环境；只在reset加载示教起始物理状态。

模型观测仅含实际关节位置、当前三路图像和任务文本。抽屉位置只供外部评分，
不注入动作、未来状态、专家阶段或时钟。step沿用30Hz和臂关节重力补偿。
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
import hashlib
import json
from pathlib import Path
import subprocess
import mujoco
import numpy as np
from collect_drawer_trial import ROOT, UPSTREAM_COMMIT


class DrawerSimulation:
    def __init__(self, episode_root=ROOT / "data/drawer_trial/contact_roundtrip_rgb_001"):
        self.episode_root = Path(episode_root)
        self.provenance = json.loads((self.episode_root / "report.json").read_text())
        source = ROOT / ".cache/openarm-control"
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if commit != UPSTREAM_COMMIT or mujoco.__version__ != self.provenance["mujoco_version"]:
            raise ValueError("仿真代码或MuJoCo版本与采集不一致")
        scene = source / self.provenance["source_scene"]
        if hashlib.sha256(scene.read_bytes()).hexdigest() != self.provenance["scene_sha256"]:
            raise ValueError("场景哈希与采集不一致")
        spec = mujoco.MjSpec.from_file(str(scene))
        overrides = self.provenance.get("scene_overrides", {})
        next(b for b in spec.bodies if b.name == "cabinet_drawer").pos[0] += overrides.get("drawer_offset_x_m", 0)
        next(b for b in spec.bodies if b.name == "cabinet_drawer").pos[1] += overrides.get("drawer_offset_y_m", 0)
        if overrides.get("drawer_handle_radius_m") is not None:
            next(g for g in spec.geoms if g.name == "drawer_handle").size[0] = overrides["drawer_handle_radius_m"]
        spec.worldbody.add_camera(name="camera_ceiling")
        self.model = model = spec.compile()
        model.vis.quality.shadowsize = self.provenance["shadow_map_size"]
        model.vis.global_.offwidth = max(320, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(240, model.vis.global_.offheight)
        for name, pose in self.provenance["camera_local_poses"].items():
            cid = model.camera(name).id
            model.cam_pos[cid], model.cam_quat[cid], model.cam_fovy[cid] = pose["position"], pose["quaternion_wxyz"], pose["fovy_degrees"]
        with np.load(self.episode_root / "episode.npz", allow_pickle=False) as episode:
            self.initial_state = episode["initial_integration_state"].copy()
            self.initial_eq = episode["initial_eq_data"].copy()
            names = episode["actuator_names"].tolist()
        self.actuator_ids = np.array([model.actuator(name).id for name in names])
        self.qpos_ids = model.jnt_qposadr[model.actuator_trnid[self.actuator_ids, 0]]
        self.velocity_ids = model.jnt_dofadr[model.actuator_trnid[self.actuator_ids, 0]]
        arm_joints = [model.joint(f"openarm_{side}_joint{i}").id for side in ("right", "left") for i in range(1, 8)]
        self.arm_dofs = model.jnt_dofadr[arm_joints]
        self.weld_mask = model.eq_type == mujoco.mjtEq.mjEQ_WELD
        self.drawer_qpos = model.joint("drawer_slide").qposadr[0]
        low, high = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
        for aid in range(model.nu):
            jid = model.actuator_trnid[aid, 0]
            if model.jnt_limited[jid]:
                low[aid], high[aid] = max(low[aid], model.jnt_range[jid, 0]), min(high[aid], model.jnt_range[jid, 1])
        self.low, self.high = low[self.actuator_ids], high[self.actuator_ids]
        self.data = mujoco.MjData(model)
        self.renderer = None
        self.reset()

    def reset(self):
        self.model.eq_data[:] = self.initial_eq
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, self.initial_state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        self.steps = int(self.provenance.get('initial_control_step_offset', 0))
        if np.any(self.data.eq_active[self.weld_mask]):
            raise RuntimeError("初始状态包含活动抓取绑定")

    def observation(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        result = {"observation.state": self.data.qpos[self.qpos_ids].astype(np.float32),
                  "observation.velocity": self.data.qvel[self.velocity_ids].astype(np.float32),
                  "prompt": self.provenance["prompt"]}
        for key, camera in self.provenance["camera_mapping"].items():
            self.renderer.update_scene(self.data, camera=camera)
            result[f"observation.images.{key}"] = self.renderer.render().copy()
        return result

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("动作必须是有限的16维位置目标")
        if np.any(self.data.eq_active[self.weld_mask]):
            raise RuntimeError("闭环中出现了活动抓取绑定")
        applied = np.clip(action, self.low, self.high)
        self.data.ctrl[self.actuator_ids] = applied
        dt = self.model.opt.timestep
        ticks = round((self.steps + 1) / 30 / dt) - round(self.steps / 30 / dt)
        for _ in range(ticks):
            mujoco.mj_forward(self.model, self.data)
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.arm_dofs] = self.data.qfrc_bias[self.arm_dofs]
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("仿真状态非有限")
        return applied

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


if __name__ == "__main__":
    environment = DrawerSimulation()
    maximum = 0.0
    with np.load(environment.episode_root / "episode.npz", allow_pickle=False) as episode:
        for index, action in enumerate(episode["action"]):
            maximum = max(maximum, float(np.max(np.abs(environment.data.qpos-episode["qpos"][index]))))
            environment.step(action)
        maximum = max(maximum, float(np.max(np.abs(environment.data.qpos-episode["terminal_qpos"]))))
    if maximum >= 1e-6:
        raise RuntimeError(f"环境动作回放与示教不同：{maximum}")
    print(f"PASS：闭环环境动作回放误差={maximum}")
