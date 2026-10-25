"""以真实接触尝试夹持抽屉把手，禁止使用抓取 weld。

采用靠近指尖的实际夹持点，IK 优先保证位置；只有左右指片均接触把手才拉。
执行与记录复用 collect_drawer_trial 的30Hz循环，本模块不修改场景和物理参数。
"""

import mujoco
import numpy as np
from scipy.optimize import least_squares

from openarm_control.articulated import ArticulatedController
from openarm_control.grasp import front_orientation
from openarm_control.kinematics import orientation_error


def finger_handle_contacts(model, data, *, include_stems=False):
    """内/外指片接触；默认保留历史横杆口径，人工示教可选择整个把手组件。"""
    handles = {model.geom("drawer_handle").id}
    if include_stems:
        handles.update(model.geom(name).id for name in ("drawer_handle_stem1", "drawer_handle_stem2"))
    fingers = [{model.geom(f"finger_{side}_right_collision_{i:02d}").id for i in range(3)}
               for side in ("inner", "outer")]
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


class ContactDrawerController(ArticulatedController):
    contact_pitch = 15.0
    contact_open = -0.25
    pull_distance = 0.09
    free_pitch = False
    return_drawer = False
    approach_offset = np.array([-0.10, 0.0, 0.055])

    def __init__(self, model, data):
        super().__init__(model, data)
        self.right.king.tool_offset = np.array([0.0, 0.0, -0.16])
        self.grasp_diagnostics = []

    def _weld(self, *args):
        raise RuntimeError("真实接触控制器禁止启用辅助绑定")

    def _solve_contact_pose(self, target, q_seed, orientation, branch_limit=1.2, pitch_free=False):
        """在独立IK状态上求解，优先保证真实夹持点位置，拒绝厘米级偏差。"""
        king = self.right.king
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        low, high = king.jnt_low + 1e-5, king.jnt_high - 1e-5
        offset = king.tool_offset
        base = self.model.geom("openarm_body_link0_collision").id
        elbow = self.model.geom("link4_right_collision_00").id

        def residual(q):
            scratch.qpos[king.qpos_indices] = q
            mujoco.mj_kinematics(self.model, scratch)
            rotation = scratch.site_xmat[king.site_id].reshape(3, 3)
            point = scratch.site_xpos[king.site_id] + rotation @ offset
            clearance = mujoco.mj_geomDistance(self.model, scratch, base, elbow, 0.1, None)
            rotation_error = orientation_error(rotation, orientation)
            if pitch_free:
                # 把手沿世界Y轴，允许腕部绕杆轴调整俯仰，仍约束另外两轴。
                rotation_error[1] *= 0.03
            return np.r_[100 * (point - target), rotation_error,
                         100 * max(0.0, 0.006 - clearance), 0.005 * (q - q_seed)]

        solution = least_squares(residual, np.clip(q_seed, low, high), bounds=(low, high),
                                 max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        error = np.linalg.norm(residual(solution.x)[:3]) / 100
        if error > 0.002 or np.max(np.abs(solution.x - q_seed)) > branch_limit:
            raise RuntimeError(f"夹持点IK不满足要求：位置误差{error:.6f}m")
        return solution.x

    def _diagnose(self, stage):
        mujoco.mj_forward(self.model, self.data)
        counts, forces = finger_handle_contacts(self.model, self.data)
        point, _ = self.right.king.forward_kinematics()
        handle = self._geom_pos("drawer_handle")
        value = {"stage": stage, "time": float(self.data.time),
                 "tool_error_m": float(np.linalg.norm(point - handle)),
                 "finger_contact_counts": counts.tolist(), "finger_normal_forces": forces.tolist(),
                 "drawer_open_m": -float(self.data.qpos[self.model.joint("drawer_slide").qposadr[0]])}
        self.grasp_diagnostics.append(value)
        print(value, flush=True)
        return counts, forces

    def open_drawer(self, distance=None, viewer=None, dt_realtime=False):
        distance = self.pull_distance if distance is None else distance
        active, other = self.right, self.left
        self._park(other)
        handle = self._geom_pos("drawer_handle")
        orientation = front_orientation(np.radians(self.contact_pitch))
        q = self.data.qpos[active.king.qpos_indices].copy()
        # 先在把手前方、上方到位，再沿短笛卡尔路径接近。
        pre = handle + np.asarray(self.approach_offset)
        # 先选择能到达把手的分支，再向后求接近位，避免从ready选中镜像肘分支。
        grasp_seed = (np.array([-1.1, 3.3, -0.8, 1.8, -1.5, 0.7, 0.3])
                      if self.contact_pitch > 40 else np.array([0.1, 0.0, 0.8, 1.8, 1.3, 0.6, -0.2]))
        q = self._solve_contact_pose(handle, grasp_seed, orientation, branch_limit=np.inf)
        q = self._solve_contact_pose(pre, q, orientation, branch_limit=np.inf)
        self._drive(active, other, [q], 2.0, grip=self.contact_open)
        self._drive(active, other, [q], 0.5, grip=self.contact_open)
        path = []
        for fraction in np.linspace(0, 1, 15)[1:]:
            q = self._solve_contact_pose(pre + fraction * (handle - pre), q, orientation)
            path.append(q.copy())
        self._drive(active, other, path, 2.0, grip=self.contact_open)
        self._drive(active, other, [q], 0.5, grip=self.contact_open)
        self._diagnose("aligned_open")
        # 渐闭合，避免大幅目标跳变。
        holding_grip = active.arm.gripper_closed
        for grip in np.linspace(self.contact_open, active.arm.gripper_closed, 13)[1:]:
            q = self._solve_contact_pose(self._geom_pos("drawer_handle"), q, orientation)
            self._drive(active, other, [q], 0.1, grip=float(grip))
            counts, forces = finger_handle_contacts(self.model, self.data)
            if np.all(counts > 0) and np.all(forces > 1.0):
                holding_grip = float(grip)
                break
        self._drive(active, other, [q], 0.5, grip=holding_grip)
        counts, forces = self._diagnose("closed")
        if not np.all(counts > 0) or not np.all(forces > 0.01):
            print("双侧手指尚未夹持把手，停止拉动", flush=True)
            return False
        point, actual_orientation = active.king.forward_kinematics()
        path = []
        for delta in np.linspace(0, distance, 31)[1:]:
            try:
                q = self._solve_contact_pose(point - [delta, 0, 0], q, actual_orientation, pitch_free=self.free_pitch)
            except RuntimeError as error:
                self.grasp_diagnostics.append({"stage": "planning_stop", "requested_delta": float(delta), "reason": str(error)})
                break
            path.append(q.copy())
        if not path:
            return False
        self._drive(active, other, path, 3.0, grip=holding_grip)
        self._drive(active, other, [q], 1.0, grip=holding_grip)
        self._diagnose("pulled")
        opened = -float(self.data.qpos[self.model.joint("drawer_slide").qposadr[0]]) >= 0.08
        if self.return_drawer:
            # 保持夹持与原重力补偿，沿已通过IK的同一分支反向推回。
            return_path = list(reversed(path[:-1])) + [
                self._solve_contact_pose(point, path[0], actual_orientation)]
            self._drive(active, other, return_path, 3.0, grip=holding_grip)
            self._drive(active, other, [return_path[-1]], 1.0, grip=holding_grip)
            self._diagnose("returned")
            return opened and abs(float(self.data.qpos[self.model.joint("drawer_slide").qposadr[0]])) <= 0.005
        return opened
