"""Pedestal gravity feedforward on the official driver's checked arm packets."""
import time

import mujoco as mj
import numpy as np

from dp_pilot.dp_paths import MODEL
from dp_pilot.gravity_profile import CAP_NM, RAMP_S
TIMEOUT = .20
# Nominal joint-space experiment ceilings, not measured load or certified limits.
CAP = np.array(CAP_NM)


class Gravity:
    def __init__(self, side):
        if side not in ("left", "right"):
            raise ValueError("unknown arm")
        m = self.model = mj.MjModel.from_xml_path(str(MODEL))
        self.data = mj.MjData(m)
        joints = [m.joint(f"openarm_{side}_joint{i}") for i in range(1, 8)]
        self.pos = [j.qposadr.item() for j in joints]
        self.dof = [j.dofadr.item() for j in joints]
        self.fingers = [m.joint(f"openarm_{side}_finger_joint{i}").qposadr.item() for i in (1, 2)]

    def torque(self, q):
        q = np.asarray(q)
        if q.shape != (8,) or not np.isfinite(q).all():
            raise ValueError("invalid measured pose")
        self.data.qpos[self.pos] = q[:7]
        self.data.qpos[self.fingers] = q[7]
        mj.mj_forward(self.model, self.data)
        tau = self.data.qfrc_bias[self.dof].copy()
        if not np.isfinite(tau).all():
            raise ValueError("nonfinite model gravity")
        return tau


class Ramp:
    def __init__(self, *, immediate=False):
        self.immediate = immediate
        self.last, self.scale = None, 0.

    def step(self, now, maximum, *, gap_timeout=TIMEOUT):
        if self.last is not None:
            dt = now - self.last
            if not 0 <= dt < gap_timeout:
                raise RuntimeError("teleop command stream interrupted")
            if not self.immediate:
                self.scale = min(maximum, self.scale + min(dt, .04) / RAMP_S)
        if self.immediate:
            self.scale = maximum
        self.last = now
        return self.scale


class Proxy:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class ArmPackets(Proxy):
    """Only modify the torque field of the checked, already constructed arm command."""
    def mit_control_all(self, params):
        if len(params) != 7 or any(p.tau != 0 for p in params):
            raise RuntimeError("unexpected MIT layout or pre-existing feedforward")
        if self.pending is not None:
            sample_ns, expected, effort = self.pending
            if not 0 <= time.time_ns() - sample_ns < 50_000_000:
                raise RuntimeError("stale joint feedback at dispatch")
            if not 0 <= time.time_ns() - expected < int(TIMEOUT*1e9):
                raise RuntimeError("stale VR target at dispatch")
            for p, tau in zip(params, effort, strict=True):
                p.tau = float(tau)
        return self.wrapped.mit_control_all(params)


class Bus(Proxy):
    def __init__(self, wrapped):
        super().__init__(wrapped)
        self.arm = ArmPackets(wrapped.get_arm())
        self.arm.pending = None

    def get_arm(self):
        return self.arm
