"""以真实接触尝试夹持抽屉把手及桌面物体（方块、橡皮、胶棒、空塑料瓶），禁止使用抓取 weld（A1/A2 共享控制器）。

采用靠近指尖的实际夹持点，IK 优先保证位置与姿态；只有左右指片均产生真实接触力才拉动抽屉或提举物体。
执行与记录复用 collect_drawer_trial 的 30Hz 循环，本模块不修改物理参数。
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

from openarm_control.articulated import ArticulatedController
from openarm_control.grasp import front_orientation, topdown_orientation
from openarm_control.kinematics import orientation_error
from simulation.drawer_scene_generator import OBJECT_CATALOG, PREOPENED_RIGHT_READY_QPOS
from simulation.drawer_simulation import (
    finger_handle_contacts,
    finger_object_contacts,
    is_object_inside_drawer,
)


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
        """在独立 IK 状态上求解，优先保证真实夹持点位置，拒绝厘米级偏差。"""
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
                rotation_error[1] *= 0.03
            return np.r_[
                100 * (point - target),
                rotation_error,
                100 * max(0.0, 0.006 - clearance),
                0.005 * (q - q_seed),
            ]

        solution = least_squares(
            residual,
            np.clip(q_seed, low, high),
            bounds=(low, high),
            max_nfev=250,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        error = np.linalg.norm(residual(solution.x)[:3]) / 100
        if error > 0.002 or np.max(np.abs(solution.x - q_seed)) > branch_limit:
            raise RuntimeError(f"夹持点IK不满足要求：位置误差{error:.6f}m")
        return solution.x

    def _solve_topdown_pose(self, target, q_seed, orientation, branch_limit=1.2):
        """求解保持竖直朝下姿态的真实接触夹持点 IK，用于桌面物体抓取与空心抽屉投放。"""
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
            rot_err = orientation_error(rotation, orientation)
            return np.r_[
                100 * (point - target),
                8 * rot_err,
                100 * max(0.0, 0.006 - clearance),
                0.005 * (q - q_seed),
            ]

        solution = least_squares(
            residual,
            np.clip(q_seed, low, high),
            bounds=(low, high),
            max_nfev=250,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        error = np.linalg.norm(residual(solution.x)[:3]) / 100
        if error > 0.0025 or np.max(np.abs(solution.x - q_seed)) > branch_limit:
            raise RuntimeError(f"俯视夹持点IK不满足要求：位置误差{error:.6f}m")
        return solution.x

    def _diagnose(self, stage):
        mujoco.mj_forward(self.model, self.data)
        counts, forces = finger_handle_contacts(self.model, self.data)
        point, _ = self.right.king.forward_kinematics()
        handle = self._geom_pos("drawer_handle")
        value = {
            "stage": stage,
            "time": float(self.data.time),
            "tool_error_m": float(np.linalg.norm(point - handle)),
            "finger_contact_counts": counts.tolist(),
            "finger_normal_forces": forces.tolist(),
            "drawer_open_m": -float(self.data.qpos[self.model.joint("drawer_slide").qposadr[0]]),
        }
        self.grasp_diagnostics.append(value)
        print(value, flush=True)
        return counts, forces

    def _diagnose_object(self, stage: str, obj_key: str, drawer_body: str = "drawer"):
        mujoco.mj_forward(self.model, self.data)
        spec = OBJECT_CATALOG[obj_key]
        counts, forces = finger_object_contacts(self.model, self.data, spec["collision_geom"])
        inside, rel = is_object_inside_drawer(self.model, self.data, spec["body_name"], drawer_body)
        slide_name = "drawer_2_slide" if drawer_body == "drawer_2" else "drawer_slide"
        value = {
            "stage": stage,
            "target_object": obj_key,
            "time": float(self.data.time),
            "object_contact_counts": counts.tolist(),
            "object_normal_forces": forces.tolist(),
            "object_inside_drawer": inside,
            "object_rel_to_drawer_m": [round(float(x), 4) for x in rel],
            "drawer_open_m": -float(self.data.qpos[self.model.joint(slide_name).qposadr[0]]),
        }
        self.grasp_diagnostics.append(value)
        print(value, flush=True)
        return counts, forces, inside

    def open_drawer(self, distance=None, viewer=None, dt_realtime=False):
        distance = self.pull_distance if distance is None else distance
        active, other = self.right, self.left
        self._park(other)
        handle = self._geom_pos("drawer_handle")
        orientation = front_orientation(np.radians(self.contact_pitch))
        q = self.data.qpos[active.king.qpos_indices].copy()
        pre = handle + np.asarray(self.approach_offset)
        grasp_seed = (
            np.array([-1.1, 3.3, -0.8, 1.8, -1.5, 0.7, 0.3])
            if self.contact_pitch > 40
            else np.array([0.1, 0.0, 0.8, 1.8, 1.3, 0.6, -0.2])
        )
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
                q = self._solve_contact_pose(
                    point - [delta, 0, 0], q, actual_orientation, pitch_free=self.free_pitch
                )
            except RuntimeError as error:
                self.grasp_diagnostics.append(
                    {"stage": "planning_stop", "requested_delta": float(delta), "reason": str(error)}
                )
                break
            path.append(q.copy())
        if not path:
            return False
        self._drive(active, other, path, 3.0, grip=holding_grip)
        self._drive(active, other, [q], 1.0, grip=holding_grip)
        self._diagnose("pulled")
        opened = -float(self.data.qpos[self.model.joint("drawer_slide").qposadr[0]]) >= 0.08
        if self.return_drawer:
            return_path = list(reversed(path[:-1])) + [
                self._solve_contact_pose(point, path[0], actual_orientation)
            ]
            self._drive(active, other, return_path, 3.0, grip=holding_grip)
            self._drive(active, other, [return_path[-1]], 1.0, grip=holding_grip)
            self._diagnose("returned")
            return opened and abs(float(self.data.qpos[self.model.joint("drawer_slide").qposadr[0]])) <= 0.005
        return opened

    def put_object_in_drawer(
        self,
        target_object: str,
        *,
        target_drawer: str = "primary",
        viewer=None,
        dt_realtime: bool = False,
    ) -> bool:
        """真实接触抓取指定桌面物体（cube / eraser / glue_stick / plastic_bottle）并放入空心抽屉槽内。

        支持两种起始模式：
        1. 抽屉起始已打开（initial_drawer_open_m >= 0.06）：直接俯视抓取目标物体并放入空心抽屉；
        2. 抽屉起始关闭（initial_drawer_open_m < 0.06）：先真实接触拉开抽屉，松开把手并抬升至安全高度，
           再俯视抓取目标物体放入抽屉。
        """
        if target_object not in OBJECT_CATALOG:
            raise ValueError(f"未知目标物体: {target_object}，可选: {list(OBJECT_CATALOG)}")
        active, other = self.right, self.left
        self._park(other)

        drawer_body = "drawer_2" if target_drawer == "secondary" else "drawer"
        slide_joint = "drawer_2_slide" if target_drawer == "secondary" else "drawer_slide"
        current_open = -float(self.data.qpos[self.model.joint(slide_joint).qposadr[0]])
        q_raised = np.asarray(PREOPENED_RIGHT_READY_QPOS, dtype=float)

        # 若抽屉起始处于关闭状态，先执行真实接触开抽屉并安全退离把手
        if current_open < 0.06:
            saved_return = self.return_drawer
            self.return_drawer = False
            opened = self.open_drawer(
                distance=max(self.pull_distance, 0.10), viewer=viewer, dt_realtime=dt_realtime
            )
            self.return_drawer = saved_return
            if not opened:
                return False
            q_cur = self.data.qpos[active.king.qpos_indices].copy()
            self._drive(active, other, [q_cur], 0.5, grip=-0.28)
            pt_now, ori_now = active.king.forward_kinematics()
            back_path = []
            q_step = q_cur.copy()
            for frac in np.linspace(0, 1, 7)[1:]:
                q_step = self._solve_contact_pose(
                    pt_now + frac * np.array([-0.055, 0.0, 0.0]),
                    q_step,
                    ori_now,
                    branch_limit=0.6,
                    pitch_free=True,
                )
                back_path.append(q_step.copy())
            self._drive(active, other, back_path, 0.8, grip=-0.28)
            q_high = self._solve_contact_pose(
                pt_now + np.array([-0.055, 0.08, 0.085]),
                q_step,
                ori_now,
                branch_limit=0.8,
                pitch_free=True,
            )
            self._drive(active, other, [q_high], 0.8, grip=-0.28)
            self._drive(active, other, [q_raised], 1.0, grip=-0.26)

        spec = OBJECT_CATALOG[target_object]
        obj_body = spec["body_name"]
        obj_geom = spec["collision_geom"]
        bid = self.model.body(obj_body).id
        obj_pos = self.data.xpos[bid].copy()

        obj_mat = self.data.xmat[bid].reshape(3, 3)
        obj_yaw = float(np.arctan2(obj_mat[1, 0], obj_mat[0, 0]))
        grip_yaw = obj_yaw + np.pi if obj_yaw <= 0.0 else obj_yaw
        R_top = topdown_orientation(grip_yaw)

        grasp_z = 0.413
        pick_pt = np.array([obj_pos[0], obj_pos[1], grasp_z])
        hover_pt = np.array([obj_pos[0], obj_pos[1], 0.56])
        drawer_pos = self.data.xpos[self.model.body(drawer_body).id].copy()
        place_x_off = -0.015 if drawer_pos[0] > 0.36 else 0.008
        place_hover = np.array([drawer_pos[0] + place_x_off, drawer_pos[1], 0.56])
        place_drop = np.array([drawer_pos[0] + place_x_off, drawer_pos[1], 0.532])

        q_seed = self.data.qpos[active.king.qpos_indices].copy()
        if np.max(np.abs(q_seed - q_raised)) > 0.6:
            q_seed = q_raised.copy()
        q_hover = self._solve_topdown_pose(hover_pt, q_seed, R_top, branch_limit=1.0)
        open_grip = -0.26
        self._drive(active, other, [q_hover], 1.2, grip=open_grip)
        self._drive(active, other, [q_hover], 0.3, grip=open_grip)

        desc_path = []
        q = q_hover.copy()
        for frac in np.linspace(0, 1, 11)[1:]:
            q = self._solve_topdown_pose(hover_pt + frac * (pick_pt - hover_pt), q, R_top, branch_limit=0.8)
            desc_path.append(q.copy())
        self._drive(active, other, desc_path, 1.2, grip=open_grip)
        self._drive(active, other, [q], 0.4, grip=open_grip)
        self._diagnose_object("object_aligned", target_object, drawer_body)

        holding_grip = -0.08
        for grip in np.linspace(open_grip, active.arm.gripper_closed, 17)[1:]:
            self._drive(active, other, [q], 0.1, grip=float(grip))
            counts, forces = finger_object_contacts(self.model, self.data, obj_geom)
            if np.all(counts > 0) and np.all(forces > 6.0):
                holding_grip = float(grip)
                break
        self._drive(active, other, [q], 0.3, grip=holding_grip)
        counts, forces, _ = self._diagnose_object("object_grasped", target_object, drawer_body)
        if not np.all(counts > 0) or not np.all(forces > 0.5):
            print(f"双侧手指未能夹稳物体 {target_object}，停止提举", flush=True)
            return False

        lift_path = []
        for frac in np.linspace(0, 1, 13)[1:]:
            q = self._solve_topdown_pose(pick_pt + frac * (hover_pt - pick_pt), q, R_top, branch_limit=0.8)
            lift_path.append(q.copy())
        self._drive(active, other, lift_path, 1.4, grip=holding_grip)
        self._diagnose_object("object_lifted", target_object, drawer_body)

        trans_path = []
        for frac in np.linspace(0, 1, 16)[1:]:
            q = self._solve_topdown_pose(hover_pt + frac * (place_hover - hover_pt), q, R_top, branch_limit=0.8)
            trans_path.append(q.copy())
        self._drive(active, other, trans_path, 1.6, grip=holding_grip)

        drop_path = []
        for frac in np.linspace(0, 1, 11)[1:]:
            q = self._solve_topdown_pose(place_hover + frac * (place_drop - place_hover), q, R_top, branch_limit=0.8)
            drop_path.append(q.copy())
        self._drive(active, other, drop_path, 1.0, grip=holding_grip)
        self._drive(active, other, [q], 0.3, grip=holding_grip)

        # 松开手指，让物体自由落入空心抽屉底板
        self._drive(active, other, [q], 0.5, grip=open_grip)

        ret_path = []
        for frac in np.linspace(0, 1, 9)[1:]:
            q = self._solve_topdown_pose(place_drop + frac * (place_hover - place_drop), q, R_top, branch_limit=0.8)
            ret_path.append(q.copy())
        self._drive(active, other, ret_path, 0.8, grip=open_grip)
        self._drive(active, other, [ret_path[-1]], 0.5, grip=open_grip)

        _, _, inside = self._diagnose_object("object_placed_in_drawer", target_object, drawer_body)
        return bool(inside)
