"""A1 抽屉任务仿真环境与真实接触控制器（位于 openarm-a1/simulation/）。

包含：
1. DrawerSimulation：与抽屉采集一致的 30Hz 真实接触环境，通过 drawer_scene_generator
   恢复外形预设、材质配色、尺寸缩放与 X/Y/Z 位姿。
2. finger_handle_contacts：内/外指片与抽屉把手组件的真实接触统计。
3. ContactDrawerController：禁用抓取 weld 的真实接触专家控制器。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import mujoco
import numpy as np

from simulation.drawer_scene_generator import compile_drawer_scene, resolve_drawer_variation

A1_ROOT = Path(__file__).resolve().parents[1]
A2_ROOT = A1_ROOT.parent / "openarm-pi0-reproduction-group-A"
UPSTREAM_COMMIT = "14ea59faad6706b9add5d7f8d386e43239a6360f"


def resolve_upstream_source(source: Path | None = None) -> Path:
    """定位 openarm-control 上游资产目录（优先本工程 .cache，其次同级 A2 工程 .cache）。"""
    if source is not None and Path(source).exists():
        return Path(source).resolve()
    for candidate in (A1_ROOT / ".cache/openarm-control", A2_ROOT / ".cache/openarm-control"):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("未找到 .cache/openarm-control 上游资产目录")


def finger_handle_contacts(model: mujoco.MjModel, data: mujoco.MjData, *, include_stems: bool = False):
    """统计右臂内/外指片与抽屉把手横杆（及可选连接杆）的接触点数与法向合力。"""
    handles = {model.geom("drawer_handle").id}
    if include_stems:
        handles.update(model.geom(name).id for name in ("drawer_handle_stem1", "drawer_handle_stem2"))
    fingers = [
        {model.geom(f"finger_{side}_right_collision_{i:02d}").id for i in range(3)}
        for side in ("inner", "outer")
    ]
    counts, forces = np.zeros(2, dtype=int), np.zeros(2)
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 in handles:
            other = contact.geom2
        elif contact.geom2 in handles:
            other = contact.geom1
        else:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        for side, ids in enumerate(fingers):
            if other in ids:
                counts[side] += 1
                forces[side] += max(0.0, wrench[0])
    return counts, forces


class DrawerSimulation:
    """30Hz 抽屉闭环仿真环境，支持从 episode 目录或直接传入变体配置初始化。"""

    def __init__(self, episode_root: str | Path, *, source: Path | None = None):
        self.episode_root = Path(episode_root)
        self.provenance = json.loads((self.episode_root / "report.json").read_text())
        source_dir = resolve_upstream_source(source)
        commit = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
        if commit != UPSTREAM_COMMIT or mujoco.__version__ != self.provenance["mujoco_version"]:
            raise ValueError(f"仿真源码或 MuJoCo 版本不一致: {mujoco.__version__} vs {self.provenance['mujoco_version']}")
        scene = source_dir / self.provenance["source_scene"]
        if hashlib.sha256(scene.read_bytes()).hexdigest() != self.provenance["scene_sha256"]:
            raise ValueError("场景文件哈希与采集记录不一致")

        overrides = self.provenance.get("scene_overrides", {})
        self.variation = resolve_drawer_variation(overrides_dict=overrides)
        self.model = model = compile_drawer_scene(
            str(scene),
            self.variation,
            camera_local_poses=self.provenance.get("camera_local_poses"),
            shadow_size=self.provenance.get("shadow_map_size", 1024),
        )
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
            jid = int(model.actuator_trnid[aid, 0])
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
        self.steps = int(self.provenance.get("initial_control_step_offset", 0))
        if np.any(self.data.eq_active[self.weld_mask]):
            raise RuntimeError("初始状态包含活动抓取绑定")

    def observation(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        result = {
            "observation.state": self.data.qpos[self.qpos_ids].astype(np.float32),
            "observation.velocity": self.data.qvel[self.velocity_ids].astype(np.float32),
            "prompt": self.provenance["prompt"],
        }
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
