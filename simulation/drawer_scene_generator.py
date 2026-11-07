"""统一抽屉与桌面物体场景参数化生成器（由 A1 在 simulation/ 下维护，供 A1/A2 共用）。

核心特性：
1. 100% 向后兼容：当不启用空心抽屉改造、桌面物体与头部双目相机时，默认参数与上游
   articulated_scene.xml 保持逐字节一致，完全兼容历史 v003/016/020 数据与模型。
2. 真实“空心抽屉槽”（hollow_drawer=True）：将原有的实心块 drawer_box 升级为由底板 + 左右侧壁 +
   后壁 + 前面板围成的开口空心抽屉盒（总质量仍保持 0.20 kg），支持将不同物体真正放入抽屉内部。
3. 5 种抽屉外形家族（standard / wide_low / narrow_tall / deep_compact / large_modern）
   × 5 种材质配色方案，并支持“单抽屉（single）”与“桌面双抽屉（dual_side_by_side）”布局。
4. 4 种可抓取桌面物体（OBJECT_CATALOG）：
   - cube（红色积木方块）
   - eraser（蓝白纸套美术橡皮）
   - glue_stick（橙盖固体胶棒）
   - plastic_bottle（半透明蓝盖空塑料瓶）
   支持自由组合摆放位置、指定目标放入物体与自动生成自然语言指令（Prompt）。
5. 机器人头部 Stereolabs ZED Mini 双目深度相机（enable_head_stereo=True）：
   按 ZED Mini 真机规格（124.5×30.5×26.5 mm 外壳、63 mm 双目基线）在机器人头部建立可视化相机模型
   及 camera_head_left / camera_head_right 双路光学相机。
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import mujoco
import numpy as np

SCHEMA_VERSION = 2
GENERATOR_VERSION = "2.0.0"

TABLE_TOP_WORLD_Z = 0.40
BASE_CABINET_POS = (0.40, -0.28, 0.46)
BASE_WALL_HALF_THICKNESS = 0.005

SHAPE_PRESETS: dict[str, dict[str, Any]] = {
    "standard": {
        "description": "标准基准款（与上游原始抽屉外廓100%一致）",
        "width_scale": 1.0,
        "depth_scale": 1.0,
        "height_scale": 1.0,
        "handle_span_scale": 1.0,
        "handle_standoff_scale": 1.0,
        "handle_radius_m": 0.007,
        "panel_style": "inset",
    },
    "wide_low": {
        "description": "宽矮款（加宽25%、压低12%，配加长把手与上下外框饰条）",
        "width_scale": 1.25,
        "depth_scale": 1.0,
        "height_scale": 0.88,
        "handle_span_scale": 1.25,
        "handle_standoff_scale": 1.0,
        "handle_radius_m": 0.007,
        "panel_style": "framed",
    },
    "narrow_tall": {
        "description": "窄高款（收窄15%、加高18%，配紧凑把手与中心浮雕面板）",
        "width_scale": 0.85,
        "depth_scale": 1.0,
        "height_scale": 1.18,
        "handle_span_scale": 0.85,
        "handle_standoff_scale": 1.0,
        "handle_radius_m": 0.0075,
        "panel_style": "raised_center",
    },
    "deep_compact": {
        "description": "深匣紧凑款（加深15%、收窄10%、略降高，配8.5mm粗把手与双横条饰面）",
        "width_scale": 0.90,
        "depth_scale": 1.15,
        "height_scale": 0.94,
        "handle_span_scale": 0.92,
        "handle_standoff_scale": 1.0,
        "handle_radius_m": 0.0085,
        "panel_style": "twin_rail",
    },
    "large_modern": {
        "description": "大号宽体款（加宽18%、加深10%、加高12%，配8mm把手与顶檐帽条）",
        "width_scale": 1.18,
        "depth_scale": 1.10,
        "height_scale": 1.12,
        "handle_span_scale": 1.15,
        "handle_standoff_scale": 1.0,
        "handle_radius_m": 0.008,
        "panel_style": "bevel_top",
    },
}

APPEARANCE_PRESETS: dict[str, dict[str, Any]] = {
    "classic_wood_blue": {
        "description": "经典暖木色柜体 + 蓝灰抽屉面板 + 深灰把手（默认）",
        "color_label_en": "blue",
        "color_label_zh": "蓝灰色",
        "cabinet_rgba": [0.55, 0.45, 0.32, 1.0],
        "drawer_box_rgba": [0.30, 0.45, 0.62, 1.0],
        "drawer_front_rgba": [0.22, 0.36, 0.52, 1.0],
        "handle_rgba": [0.20, 0.20, 0.22, 1.0],
        "trim_rgba": [0.16, 0.28, 0.42, 1.0],
    },
    "light_oak_white": {
        "description": "浅橡木色柜体 + 象牙白面板 + 银灰金属把手",
        "color_label_en": "white",
        "color_label_zh": "白色浅橡木",
        "cabinet_rgba": [0.76, 0.64, 0.48, 1.0],
        "drawer_box_rgba": [0.86, 0.86, 0.82, 1.0],
        "drawer_front_rgba": [0.92, 0.91, 0.88, 1.0],
        "handle_rgba": [0.52, 0.55, 0.60, 1.0],
        "trim_rgba": [0.68, 0.56, 0.40, 1.0],
    },
    "walnut_slate": {
        "description": "深胡桃木柜体 + 岩灰面板 + 哑光黑把手",
        "color_label_en": "dark slate",
        "color_label_zh": "深胡桃木岩灰",
        "cabinet_rgba": [0.32, 0.22, 0.15, 1.0],
        "drawer_box_rgba": [0.35, 0.37, 0.40, 1.0],
        "drawer_front_rgba": [0.26, 0.28, 0.32, 1.0],
        "handle_rgba": [0.12, 0.12, 0.14, 1.0],
        "trim_rgba": [0.46, 0.33, 0.22, 1.0],
    },
    "industrial_steel": {
        "description": "工业冷灰钢柜 + 暖橙面板 + 炭黑把手",
        "color_label_en": "orange",
        "color_label_zh": "橙色工业钢",
        "cabinet_rgba": [0.42, 0.45, 0.48, 1.0],
        "drawer_box_rgba": [0.52, 0.55, 0.58, 1.0],
        "drawer_front_rgba": [0.82, 0.50, 0.18, 1.0],
        "handle_rgba": [0.15, 0.15, 0.16, 1.0],
        "trim_rgba": [0.26, 0.29, 0.32, 1.0],
    },
    "sage_green": {
        "description": "浅灰白柜体 + 鼠尾草绿面板 + 黄铜色把手",
        "color_label_en": "green",
        "color_label_zh": "鼠尾草绿",
        "cabinet_rgba": [0.78, 0.76, 0.72, 1.0],
        "drawer_box_rgba": [0.48, 0.58, 0.50, 1.0],
        "drawer_front_rgba": [0.36, 0.49, 0.41, 1.0],
        "handle_rgba": [0.72, 0.58, 0.26, 1.0],
        "trim_rgba": [0.25, 0.36, 0.29, 1.0],
    },
}

# 4 种桌面可操作物体规格表（方块、橡皮、胶棒、空塑料瓶）
OBJECT_CATALOG: dict[str, dict[str, Any]] = {
    "cube": {
        "body_name": "obj_cube",
        "collision_geom": "obj_cube_geom",
        "label_en": "red cube",
        "label_zh": "方块",
        "half_size_xyz_m": [0.017, 0.017, 0.017],
        "table_rest_z_m": 0.417,
        "default_xy_m": [0.32, -0.09],
        "mass_kg": 0.045,
        "rgba": [0.88, 0.20, 0.18, 1.0],
        "grasp_yaw_deg": -60.0,
        "close_grip_target": -0.02,
    },
    "eraser": {
        "body_name": "obj_eraser",
        "collision_geom": "obj_eraser_geom",
        "label_en": "eraser",
        "label_zh": "橡皮",
        "half_size_xyz_m": [0.023, 0.015, 0.014],
        "table_rest_z_m": 0.414,
        "default_xy_m": [0.32, -0.02],
        "mass_kg": 0.038,
        "rgba": [0.95, 0.95, 0.92, 1.0],
        "sleeve_rgba": [0.18, 0.42, 0.78, 1.0],
        "grasp_yaw_deg": -60.0,
        "close_grip_target": -0.01,
    },
    "glue_stick": {
        "body_name": "obj_glue_stick",
        "collision_geom": "obj_glue_stick_geom",
        "label_en": "glue stick",
        "label_zh": "胶棒",
        # 水平横放（长 54mm，直径 28mm），防滚动且易抓取、易落入抽屉槽
        "half_size_xyz_m": [0.027, 0.014, 0.014],
        "table_rest_z_m": 0.414,
        "default_xy_m": [0.25, -0.09],
        "mass_kg": 0.036,
        "rgba": [0.94, 0.94, 0.96, 1.0],
        "cap_rgba": [0.95, 0.45, 0.10, 1.0],
        "label_band_rgba": [0.20, 0.65, 0.38, 1.0],
        "grasp_yaw_deg": -60.0,
        "close_grip_target": -0.01,
    },
    "plastic_bottle": {
        "body_name": "obj_plastic_bottle",
        "collision_geom": "obj_plastic_bottle_geom",
        "label_en": "empty plastic bottle",
        "label_zh": "空塑料瓶",
        # 紧凑空塑料瓶（总长 60mm，瓶身宽 28mm），横放防滚并适配抽屉内腔
        "half_size_xyz_m": [0.030, 0.014, 0.014],
        "table_rest_z_m": 0.414,
        "default_xy_m": [0.25, -0.02],
        "mass_kg": 0.028,
        "rgba": [0.70, 0.88, 0.98, 0.72],
        "cap_rgba": [0.12, 0.35, 0.85, 1.0],
        "label_band_rgba": [0.22, 0.72, 0.42, 0.95],
        "grasp_yaw_deg": -60.0,
        "close_grip_target": -0.01,
    },
}

# 当抽屉初始已预拉开（initial_drawer_open_m > 0.02）时，右臂采用抬高准备姿态以避开拉出的抽屉面板
PREOPENED_RIGHT_READY_QPOS: list[float] = [
    -0.704239,
    2.777109,
    0.617503,
    1.813381,
    0.942334,
    0.237361,
    0.993102,
]

# 相机映射预设（支持向后兼容的三相机与 ZED Mini 头部双目配置）
CAMERA_PRESETS: dict[str, dict[str, str]] = {
    "legacy_3cam": {
        "ceiling": "camera_ceiling",
        "wrist_left": "camera_wrist_left",
        "wrist_right": "camera_wrist_right",
    },
    "zed_4cam": {
        "head_left": "camera_head_left",
        "head_right": "camera_head_right",
        "wrist_left": "camera_wrist_left",
        "wrist_right": "camera_wrist_right",
    },
    "all_5cam": {
        "head_left": "camera_head_left",
        "head_right": "camera_head_right",
        "ceiling": "camera_ceiling",
        "wrist_left": "camera_wrist_left",
        "wrist_right": "camera_wrist_right",
    },
}

# Stereolabs ZED Mini 双目深度相机默认规格（63mm 双目基线）
DEFAULT_ZED_MINI_SPEC: dict[str, Any] = {
    "model_name": "Stereolabs ZED Mini",
    "mount_pos_world_m": [0.04, 0.0, 0.82],
    "lookat_world_m": [0.35, -0.08, 0.44],
    "baseline_m": 0.063,
    "fovy_degrees": 54.0,
    "housing_half_size_xyz_m": [0.01325, 0.06225, 0.01525],
}


def build_task_prompt(
    *,
    target_object: str | None,
    drawer_layout: str = "single",
    target_drawer: str = "primary",
    primary_appearance: str = "classic_wood_blue",
    secondary_appearance: str = "light_oak_white",
    initial_drawer_open_m: float = 0.0,
    return_drawer: bool = False,
) -> str:
    """根据目标物体、抽屉布局与起始状态生成自然语言任务指令（Prompt）。"""
    if not target_object:
        return "Open the drawer, then close it." if return_drawer else "Open the drawer."
    obj_info = OBJECT_CATALOG[target_object]
    obj_name = obj_info["label_en"]
    if drawer_layout == "dual_side_by_side":
        app_key = secondary_appearance if target_drawer == "secondary" else primary_appearance
        color_label = APPEARANCE_PRESETS[app_key]["color_label_en"]
        drawer_desc = f"the {color_label} drawer"
    else:
        drawer_desc = "the drawer"

    if initial_drawer_open_m >= 0.06:
        if return_drawer:
            return f"Put the {obj_name} into {drawer_desc}, then close it."
        return f"Put the {obj_name} into {drawer_desc}."
    if return_drawer:
        return f"Open {drawer_desc}, put the {obj_name} into it, and close {drawer_desc}."
    return f"Open {drawer_desc} and put the {obj_name} into it."


def _compute_camera_pose_looking_at(
    cam_pos: list[float] | np.ndarray,
    target_pos: list[float] | np.ndarray,
) -> tuple[list[float], list[float]]:
    """计算从 cam_pos 指向 target_pos 且水平轴平行于世界 XY 平面的相机位姿 (pos, quat_wxyz)。"""
    pos = np.asarray(cam_pos, dtype=float)
    target = np.asarray(target_pos, dtype=float)
    # MuJoCo 相机沿着局部 -Z 轴观察，+X 向右，+Y 向上
    forward = target - pos
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(forward, world_up)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([0.0, -1.0, 0.0])
    else:
        x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x_axis, y_axis, z_axis]).ravel())
    return [round(float(v), 6) for v in pos], [round(float(v), 6) for v in quat]


def resolve_drawer_variation(
    *,
    variant_preset: str = "standard",
    appearance_preset: str = "classic_wood_blue",
    drawer_offset_x_m: float = 0.0,
    drawer_offset_y_m: float = 0.0,
    drawer_offset_z_m: float = 0.0,
    width_scale: float | None = None,
    depth_scale: float | None = None,
    height_scale: float | None = None,
    handle_span_scale: float | None = None,
    handle_standoff_scale: float | None = None,
    handle_radius_m: float | None = None,
    panel_style: str | None = None,
    hollow_drawer: bool = False,
    initial_drawer_open_m: float = 0.0,
    drawer_layout: str = "single",
    secondary_variant_preset: str = "wide_low",
    secondary_appearance_preset: str = "light_oak_white",
    secondary_pos_xyz_m: list[float] | None = None,
    secondary_initial_open_m: float = 0.0,
    objects: list[str] | list[dict[str, Any]] | None = None,
    target_object: str | None = None,
    target_drawer: str = "primary",
    remove_unused_fixtures: bool | None = None,
    enable_head_stereo: bool = False,
    head_camera_mount_pos_m: list[float] | None = None,
    head_camera_lookat_m: list[float] | None = None,
    head_camera_baseline_m: float = 0.063,
    head_camera_fovy_deg: float = 54.0,
    overrides_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """解析并校验抽屉、桌面物体与头部双目相机参数，返回包含派生几何与确定性哈希的完整配置字典。"""
    if overrides_dict is not None:
        dv = (
            overrides_dict["drawer_variation"]
            if "drawer_variation" in overrides_dict and isinstance(overrides_dict["drawer_variation"], dict)
            else overrides_dict
        )
        sec_sub = dv.get("secondary_drawer") if isinstance(dv.get("secondary_drawer"), dict) else {}
        head_sub = dv.get("head_stereo_camera") if isinstance(dv.get("head_stereo_camera"), dict) else {}
        explicit_hr = dv.get("explicit_handle_radius")
        restored_hr = (
            None
            if explicit_hr is False
            else dv.get(
                "handle_radius_m",
                dv.get("drawer_handle_radius_m", dv.get("handle_radius", overrides_dict.get("drawer_handle_radius_m"))),
            )
        )
        return resolve_drawer_variation(
            variant_preset=dv.get("variant_preset", "standard"),
            appearance_preset=dv.get("appearance_preset", "classic_wood_blue"),
            drawer_offset_x_m=float(
                dv.get(
                    "drawer_offset_x_m",
                    dv.get("drawer_offset_x", dv.get("drawer_x", overrides_dict.get("drawer_offset_x_m", 0.0))),
                )
            ),
            drawer_offset_y_m=float(
                dv.get(
                    "drawer_offset_y_m",
                    dv.get("drawer_offset_y", dv.get("drawer_y", overrides_dict.get("drawer_offset_y_m", 0.0))),
                )
            ),
            drawer_offset_z_m=float(
                dv.get(
                    "drawer_offset_z_m",
                    dv.get("drawer_offset_z", dv.get("drawer_z", overrides_dict.get("drawer_offset_z_m", 0.0))),
                )
            ),
            width_scale=dv.get("width_scale"),
            depth_scale=dv.get("depth_scale"),
            height_scale=dv.get("height_scale"),
            handle_span_scale=dv.get("handle_span_scale"),
            handle_standoff_scale=dv.get("handle_standoff_scale"),
            handle_radius_m=restored_hr,
            panel_style=dv.get("panel_style"),
            hollow_drawer=bool(dv.get("hollow_drawer", False)),
            initial_drawer_open_m=float(dv.get("initial_drawer_open_m", 0.0)),
            drawer_layout=str(dv.get("drawer_layout", "single")),
            secondary_variant_preset=str(
                dv.get("secondary_variant_preset", sec_sub.get("variant_preset", "wide_low"))
            ),
            secondary_appearance_preset=str(
                dv.get("secondary_appearance_preset", sec_sub.get("appearance_preset", "light_oak_white"))
            ),
            secondary_pos_xyz_m=dv.get("secondary_pos_xyz_m", sec_sub.get("cabinet_pos_xyz_m")),
            secondary_initial_open_m=float(
                dv.get("secondary_initial_open_m", sec_sub.get("initial_open_m", 0.0))
            ),
            objects=dv.get("objects"),
            target_object=dv.get("target_object"),
            target_drawer=str(dv.get("target_drawer", "primary")),
            remove_unused_fixtures=dv.get("remove_unused_fixtures"),
            enable_head_stereo=bool(dv.get("enable_head_stereo", head_sub.get("enabled", False))),
            head_camera_mount_pos_m=dv.get("head_camera_mount_pos_m", head_sub.get("mount_pos_world_m")),
            head_camera_lookat_m=dv.get("head_camera_lookat_m", head_sub.get("lookat_world_m")),
            head_camera_baseline_m=float(dv.get("head_camera_baseline_m", head_sub.get("baseline_m", 0.063))),
            head_camera_fovy_deg=float(dv.get("head_camera_fovy_deg", head_sub.get("fovy_degrees", 54.0))),
        )

    if variant_preset not in SHAPE_PRESETS:
        raise ValueError(f"未知抽屉外形预设: {variant_preset}，可选: {list(SHAPE_PRESETS)}")
    if appearance_preset not in APPEARANCE_PRESETS:
        raise ValueError(f"未知抽屉外观预设: {appearance_preset}，可选: {list(APPEARANCE_PRESETS)}")
    if drawer_layout not in ("single", "dual_side_by_side"):
        raise ValueError(f"未知抽屉布局: {drawer_layout}，可选: ('single', 'dual_side_by_side')")

    # 解析桌面物体列表
    resolved_objects: list[dict[str, Any]] = []
    if objects:
        for idx, item in enumerate(objects):
            if isinstance(item, str):
                obj_key = item
                custom_xy = None
                custom_yaw = None
            elif isinstance(item, dict):
                obj_key = item["type"]
                custom_xy = item.get("xy_m")
                if custom_xy is None and "initial_pos_xyz_m" in item:
                    custom_xy = item["initial_pos_xyz_m"][:2]
                custom_yaw = item.get("yaw_deg")
            else:
                raise ValueError(f"非法的物体配置项: {item}")
            if obj_key not in OBJECT_CATALOG:
                raise ValueError(f"未知物体类型: {obj_key}，可选: {list(OBJECT_CATALOG)}")
            spec_entry = copy.deepcopy(OBJECT_CATALOG[obj_key])
            xy = list(spec_entry["default_xy_m"] if custom_xy is None else custom_xy)
            yaw_deg = float(spec_entry["grasp_yaw_deg"] if custom_yaw is None else custom_yaw)
            half_yaw = np.radians(yaw_deg) * 0.5
            quat = [round(float(np.cos(half_yaw)), 6), 0.0, 0.0, round(float(np.sin(half_yaw)), 6)]
            resolved_objects.append(
                {
                    "type": obj_key,
                    "body_name": spec_entry["body_name"],
                    "collision_geom": spec_entry["collision_geom"],
                    "label_en": spec_entry["label_en"],
                    "label_zh": spec_entry["label_zh"],
                    "half_size_xyz_m": spec_entry["half_size_xyz_m"],
                    "initial_pos_xyz_m": [round(float(xy[0]), 6), round(float(xy[1]), 6), spec_entry["table_rest_z_m"]],
                    "initial_quat_wxyz": quat,
                    "yaw_deg": yaw_deg,
                    "mass_kg": spec_entry["mass_kg"],
                    "close_grip_target": spec_entry["close_grip_target"],
                }
            )

    if target_object is not None:
        if target_object not in OBJECT_CATALOG:
            raise ValueError(f"未知目标物体: {target_object}，可选: {list(OBJECT_CATALOG)}")
        if not any(o["type"] == target_object for o in resolved_objects):
            spec_entry = copy.deepcopy(OBJECT_CATALOG[target_object])
            xy = list(spec_entry["default_xy_m"])
            yaw_deg = float(spec_entry["grasp_yaw_deg"])
            half_yaw = np.radians(yaw_deg) * 0.5
            quat = [round(float(np.cos(half_yaw)), 6), 0.0, 0.0, round(float(np.sin(half_yaw)), 6)]
            resolved_objects.insert(
                0,
                {
                    "type": target_object,
                    "body_name": spec_entry["body_name"],
                    "collision_geom": spec_entry["collision_geom"],
                    "label_en": spec_entry["label_en"],
                    "label_zh": spec_entry["label_zh"],
                    "half_size_xyz_m": spec_entry["half_size_xyz_m"],
                    "initial_pos_xyz_m": [round(float(xy[0]), 6), round(float(xy[1]), 6), spec_entry["table_rest_z_m"]],
                    "initial_quat_wxyz": quat,
                    "yaw_deg": yaw_deg,
                    "mass_kg": spec_entry["mass_kg"],
                    "close_grip_target": spec_entry["close_grip_target"],
                },
            )

    # 当存在桌面物体或双抽屉时，默认启用空心抽屉并移除无关阀门/柜门
    use_hollow = bool(hollow_drawer or resolved_objects or drawer_layout != "single")
    use_clean_table = (
        bool(resolved_objects or drawer_layout != "single")
        if remove_unused_fixtures is None
        else bool(remove_unused_fixtures)
    )

    preset = SHAPE_PRESETS[variant_preset]
    appearance = APPEARANCE_PRESETS[appearance_preset]

    sy = float(preset["width_scale"] if width_scale is None else width_scale)
    sx = float(preset["depth_scale"] if depth_scale is None else depth_scale)
    sz = float(preset["height_scale"] if height_scale is None else height_scale)
    h_span_scale = float(preset["handle_span_scale"] if handle_span_scale is None else handle_span_scale)
    h_standoff_scale = float(
        preset["handle_standoff_scale"] if handle_standoff_scale is None else handle_standoff_scale
    )
    explicit_handle_radius = handle_radius_m is not None or variant_preset != "standard"
    h_radius = float(preset["handle_radius_m"] if handle_radius_m is None else handle_radius_m)
    resolved_panel_style = str(preset["panel_style"] if panel_style is None else panel_style)

    dx = float(drawer_offset_x_m)
    dy = float(drawer_offset_y_m)
    dz = float(drawer_offset_z_m)
    init_open = float(initial_drawer_open_m)

    if not (0.75 <= sy <= 1.35):
        raise ValueError(f"width_scale ({sy}) 超出安全范围 [0.75, 1.35]")
    if not (0.80 <= sx <= 1.30):
        raise ValueError(f"depth_scale ({sx}) 超出安全范围 [0.80, 1.30]")
    if not (0.75 <= sz <= 1.30):
        raise ValueError(f"height_scale ({sz}) 超出安全范围 [0.75, 1.30]")
    if not (0.005 <= h_radius <= 0.020):
        raise ValueError(f"handle_radius_m ({h_radius}) 超出安全范围 [0.005, 0.020]")
    if not (0.75 <= h_span_scale <= 1.40):
        raise ValueError(f"handle_span_scale ({h_span_scale}) 超出安全范围 [0.75, 1.40]")
    if not (0.80 <= h_standoff_scale <= 1.25):
        raise ValueError(f"handle_standoff_scale ({h_standoff_scale}) 超出安全范围 [0.80, 1.25]")
    if not (-0.10 <= dx <= 0.14):
        raise ValueError(f"drawer_offset_x_m ({dx}) 超出安全范围 [-0.10, 0.14]")
    if not (-0.06 <= dy <= 0.06):
        raise ValueError(f"drawer_offset_y_m ({dy}) 超出安全范围 [-0.06, 0.06]")
    if not (-0.04 <= dz <= 0.06):
        raise ValueError(f"drawer_offset_z_m ({dz}) 超出安全范围 [-0.04, 0.06]")
    if not (0.0 <= init_open <= 0.10):
        raise ValueError(f"initial_drawer_open_m ({init_open}) 超出安全范围 [0.0, 0.10]")

    base_full_height_m = 0.06 + dz
    if base_full_height_m < 0.010:
        raise ValueError(f"柜体高度偏移 dz={dz} 过低，导致底座高度 {base_full_height_m:.3f}m < 0.010m")

    t = BASE_WALL_HALF_THICKNESS
    cab_half_w = 0.05 * sy
    cab_half_d = 0.05 * sx
    cab_half_h = 0.05 * sz

    box_half_w = 0.038 * sy
    box_half_d = 0.038 * sx
    box_half_h = 0.030 * sz
    front_half_w = 0.040 * sy

    clearance_y_box = (cab_half_w - t) - box_half_w
    clearance_y_front = (cab_half_w - t) - front_half_w
    clearance_z_bottom = (0.045 * sz - box_half_h) - (2 * t)
    clearance_z_top = (2 * cab_half_h) - (0.045 * sz + box_half_h)
    if min(clearance_y_box, clearance_y_front, clearance_z_bottom, clearance_z_top) < 0.002:
        raise ValueError("抽屉盒或前面板与柜体壁板滑动间隙不足 2mm")

    cabinet_world_x = BASE_CABINET_POS[0] + dx
    cabinet_world_y = BASE_CABINET_POS[1] + dy
    cabinet_world_z = BASE_CABINET_POS[2] + dz
    cabinet_max_y = cabinet_world_y + cab_half_w + t
    if not use_clean_table and cabinet_max_y > -0.14:
        raise ValueError(f"柜体左壁外沿 y={cabinet_max_y:.3f}m 过于靠近阀门组件")

    handle_standoff_m = 0.040 * h_standoff_scale
    handle_local_x = -0.052 - handle_standoff_m
    handle_half_span_m = 0.020 * h_span_scale
    drawer_local_z = 0.045 * sz
    handle_world_closed = [
        round(cabinet_world_x + handle_local_x, 6),
        round(cabinet_world_y, 6),
        round(cabinet_world_z + drawer_local_z, 6),
    ]

    # 第二抽屉（dual_side_by_side 模式）解析
    secondary_drawer_cfg: dict[str, Any] | None = None
    if drawer_layout == "dual_side_by_side":
        sec_preset = SHAPE_PRESETS[secondary_variant_preset]
        sec_app = APPEARANCE_PRESETS[secondary_appearance_preset]
        sec_pos = [0.50, 0.22, 0.46] if secondary_pos_xyz_m is None else [float(v) for v in secondary_pos_xyz_m]
        secondary_drawer_cfg = {
            "variant_preset": secondary_variant_preset,
            "appearance_preset": secondary_appearance_preset,
            "cabinet_pos_xyz_m": [round(v, 6) for v in sec_pos],
            "width_scale": float(sec_preset["width_scale"]),
            "depth_scale": float(sec_preset["depth_scale"]),
            "height_scale": float(sec_preset["height_scale"]),
            "handle_span_scale": float(sec_preset["handle_span_scale"]),
            "handle_standoff_scale": float(sec_preset["handle_standoff_scale"]),
            "handle_radius_m": float(sec_preset["handle_radius_m"]),
            "panel_style": str(sec_preset["panel_style"]),
            "initial_open_m": round(float(secondary_initial_open_m), 6),
            "appearance": copy.deepcopy(sec_app),
        }

    # ZED Mini 头部双目相机位姿计算
    mount_pos = (
        list(DEFAULT_ZED_MINI_SPEC["mount_pos_world_m"])
        if head_camera_mount_pos_m is None
        else [float(v) for v in head_camera_mount_pos_m]
    )
    lookat_pos = (
        list(DEFAULT_ZED_MINI_SPEC["lookat_world_m"])
        if head_camera_lookat_m is None
        else [float(v) for v in head_camera_lookat_m]
    )
    _, head_quat = _compute_camera_pose_looking_at(mount_pos, lookat_pos)
    half_baseline = 0.5 * float(head_camera_baseline_m)
    head_stereo_cfg = {
        "enabled": bool(enable_head_stereo),
        "model_name": DEFAULT_ZED_MINI_SPEC["model_name"],
        "mount_pos_world_m": [round(v, 6) for v in mount_pos],
        "lookat_world_m": [round(v, 6) for v in lookat_pos],
        "orientation_quat_wxyz": head_quat,
        "baseline_m": round(float(head_camera_baseline_m), 6),
        "fovy_degrees": round(float(head_camera_fovy_deg), 6),
        "left_camera_name": "camera_head_left",
        "right_camera_name": "camera_head_right",
        "left_camera_world_pos_m": [round(mount_pos[0], 6), round(mount_pos[1] + half_baseline, 6), round(mount_pos[2], 6)],
        "right_camera_world_pos_m": [round(mount_pos[0], 6), round(mount_pos[1] - half_baseline, 6), round(mount_pos[2], 6)],
    }

    is_baseline_geometry = (
        variant_preset == "standard"
        and appearance_preset == "classic_wood_blue"
        and abs(dz) < 1e-12
        and abs(sy - 1.0) < 1e-12
        and abs(sx - 1.0) < 1e-12
        and abs(sz - 1.0) < 1e-12
        and abs(h_span_scale - 1.0) < 1e-12
        and abs(h_standoff_scale - 1.0) < 1e-12
        and resolved_panel_style == "inset"
        and not use_hollow
        and abs(init_open) < 1e-12
        and drawer_layout == "single"
        and not resolved_objects
        and not use_clean_table
        and not enable_head_stereo
    )

    core_params: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "variant_preset": variant_preset,
        "appearance_preset": appearance_preset,
        "panel_style": resolved_panel_style,
        "drawer_offset_x_m": dx,
        "drawer_offset_y_m": dy,
        "drawer_offset_z_m": dz,
        "width_scale": sy,
        "depth_scale": sx,
        "height_scale": sz,
        "handle_span_scale": h_span_scale,
        "handle_standoff_scale": h_standoff_scale,
        "handle_radius_m": h_radius,
        "explicit_handle_radius": explicit_handle_radius,
        "hollow_drawer": use_hollow,
        "initial_drawer_open_m": init_open,
        "drawer_layout": drawer_layout,
        "secondary_drawer": secondary_drawer_cfg,
        "objects": resolved_objects,
        "target_object": target_object,
        "target_drawer": target_drawer,
        "remove_unused_fixtures": use_clean_table,
        "head_stereo_camera": head_stereo_cfg,
        "is_baseline_geometry": is_baseline_geometry,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(core_params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    wall_t = 0.003
    floor_h = 0.004
    return {
        **core_params,
        "config_sha256": config_sha256,
        "appearance": copy.deepcopy(appearance),
        "derived_geometry": {
            "cabinet_world_pos_m": [round(cabinet_world_x, 6), round(cabinet_world_y, 6), round(cabinet_world_z, 6)],
            "handle_world_pos_closed_m": handle_world_closed,
            "base_bottom_world_z_m": round(cabinet_world_z - base_full_height_m, 6),
            "base_full_height_m": round(base_full_height_m, 6),
            "cabinet_outer_size_xyz_m": [
                round(2 * (cab_half_d + t), 6),
                round(2 * (cab_half_w + t), 6),
                round(2 * (cab_half_h + t), 6),
            ],
            "drawer_box_outer_size_xyz_m": [
                round(2 * box_half_d, 6),
                round(2 * box_half_w, 6),
                round(2 * box_half_h, 6),
            ],
            "drawer_cavity_inner_size_xyz_m": [
                round(2 * box_half_d - wall_t, 6),
                round(2 * (box_half_w - wall_t), 6),
                round(2 * box_half_h - 2 * floor_h, 6),
            ],
            "handle_bar_length_m": round(2 * handle_half_span_m, 6),
            "handle_standoff_m": round(handle_standoff_m, 6),
            "slide_stroke_m": 0.10,
        },
    }


def build_scene_overrides_dict(variation: dict[str, Any]) -> dict[str, Any]:
    """生成写入 report.json['scene_overrides'] 的字典，同时保留旧字段名以确保向后兼容。"""
    return {
        "drawer_handle_radius_m": (
            variation["handle_radius_m"] if variation.get("explicit_handle_radius") else None
        ),
        "drawer_offset_y_m": variation["drawer_offset_y_m"],
        "drawer_offset_x_m": variation["drawer_offset_x_m"],
        "drawer_offset_z_m": variation["drawer_offset_z_m"],
        "drawer_variation": variation,
    }


def _add_secondary_drawer_to_spec(spec: mujoco.MjSpec, sec_cfg: dict[str, Any]) -> None:
    """在 worldbody 中动态添加第二个独立空心抽屉柜（cabinet_drawer_2 / drawer_2）。"""
    sy = sec_cfg["width_scale"]
    sx = sec_cfg["depth_scale"]
    sz = sec_cfg["height_scale"]
    pos = sec_cfg["cabinet_pos_xyz_m"]
    dz = pos[2] - BASE_CABINET_POS[2]
    t = BASE_WALL_HALF_THICKNESS
    extra_half_d = 0.05 * (sx - 1.0)
    cab_half_d = 0.05 * sx
    cab_half_w = 0.05 * sy
    cab_half_h = 0.05 * sz
    base_half_z = 0.03 + 0.5 * dz
    app = sec_cfg["appearance"]

    cab2 = spec.worldbody.add_body(name="cabinet_drawer_2", pos=pos)
    for name, size, gpos in (
        ("drawer_2_base", [cab_half_d, cab_half_w + t, base_half_z], [extra_half_d, 0.0, -base_half_z]),
        ("drawer_2_bottom", [cab_half_d, cab_half_w + t, t], [extra_half_d, 0.0, t]),
        ("drawer_2_top", [cab_half_d, cab_half_w + t, t], [extra_half_d, 0.0, 2.0 * cab_half_h + t]),
        ("drawer_2_back", [t, cab_half_w + t, cab_half_h], [0.05 + 2.0 * extra_half_d, 0.0, cab_half_h + t]),
        ("drawer_2_sideL", [cab_half_d + t, t, cab_half_h], [extra_half_d, cab_half_w, cab_half_h + t]),
        ("drawer_2_sideR", [cab_half_d + t, t, cab_half_h], [extra_half_d, -cab_half_w, cab_half_h + t]),
    ):
        g = cab2.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=gpos,
            rgba=app["cabinet_rgba"],
        )
        g.contype = 1
        g.conaffinity = 1

    drawer2 = cab2.add_body(name="drawer_2", pos=[0.0, 0.0, 0.045 * sz])
    j2 = drawer2.add_joint(
        name="drawer_2_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1.0, 0.0, 0.0],
        range=[-0.10, 0.0],
        damping=1.0,
    )
    j2.frictionloss = 0.05

    box_extra_half_d = 0.038 * (sx - 1.0)
    box_hx = 0.038 * sx
    box_hy = 0.038 * sy
    box_hz = 0.030 * sz
    floor_hz = 0.004
    wall_t = 0.003
    wall_hz = box_hz - floor_hz

    g_floor = drawer2.add_geom(
        name="drawer_2_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[box_hx, box_hy, floor_hz],
        pos=[box_extra_half_d, 0.0, -box_hz + floor_hz],
        rgba=app["drawer_box_rgba"],
        mass=0.08,
    )
    g_floor.contype = 1
    g_floor.conaffinity = 1

    for wname, wsize, wpos in (
        ("drawer_2_wall_left", [box_hx, wall_t, wall_hz], [box_extra_half_d, box_hy - wall_t, floor_hz]),
        ("drawer_2_wall_right", [box_hx, wall_t, wall_hz], [box_extra_half_d, -box_hy + wall_t, floor_hz]),
        (
            "drawer_2_wall_back",
            [wall_t, box_hy - 2 * wall_t, wall_hz],
            [box_extra_half_d + box_hx - wall_t, 0.0, floor_hz],
        ),
    ):
        gw = drawer2.add_geom(
            name=wname,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=wsize,
            pos=wpos,
            rgba=app["drawer_box_rgba"],
            mass=0.04,
        )
        gw.contype = 1
        gw.conaffinity = 1

    g_front = drawer2.add_geom(
        name="drawer_2_front",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.006, 0.040 * sy, 0.028 * sz],
        pos=[-0.046, 0.0, 0.0],
        rgba=app["drawer_front_rgba"],
        mass=0.05,
    )
    g_front.contype = 1
    g_front.conaffinity = 1

    front_outer_x = -0.052
    standoff = 0.040 * sec_cfg["handle_standoff_scale"]
    handle_x = front_outer_x - standoff
    half_span = 0.020 * sec_cfg["handle_span_scale"]
    stem_y = 0.90 * half_span

    g_handle = drawer2.add_geom(
        name="drawer_2_handle",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[handle_x, -half_span, 0.0, handle_x, half_span, 0.0],
        size=[sec_cfg["handle_radius_m"], 0.0, 0.0],
        rgba=app["handle_rgba"],
        mass=0.02,
    )
    g_handle.contype = 1
    g_handle.conaffinity = 1

    for idx, sy_sign in enumerate((-1.0, 1.0), start=1):
        g_stem = drawer2.add_geom(
            name=f"drawer_2_handle_stem{idx}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=[front_outer_x, sy_sign * stem_y, 0.0, handle_x, sy_sign * stem_y, 0.0],
            size=[0.004, 0.0, 0.0],
            rgba=app["handle_rgba"],
            mass=0.005,
        )
        g_stem.contype = 1
        g_stem.conaffinity = 1


def _add_tabletop_objects_to_spec(spec: mujoco.MjSpec, objects: list[dict[str, Any]]) -> None:
    """在桌面上生成自由刚体物体（方块、橡皮、胶棒、空塑料瓶），每个物体带 <freejoint/>。"""
    for obj in objects:
        otype = obj["type"]
        cat = OBJECT_CATALOG[otype]
        body = spec.worldbody.add_body(
            name=obj["body_name"],
            pos=obj["initial_pos_xyz_m"],
            quat=obj["initial_quat_wxyz"],
        )
        body.add_freejoint(name=f"{obj['body_name']}_joint")
        hx, hy, hz = obj["half_size_xyz_m"]

        # 主碰撞与夹持几何体：采用高摩擦系数与柔和接触刚度，确保真实接触夹持稳定不打滑
        col = body.add_geom(
            name=obj["collision_geom"],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hx, hy, hz],
            pos=[0.0, 0.0, 0.0],
            rgba=cat["rgba"],
            mass=obj["mass_kg"],
        )
        col.contype = 1
        col.conaffinity = 1
        col.condim = 6
        col.friction[:] = [2.5, 0.15, 0.05]
        col.solref[:] = [0.004, 1.0]

        # 各物体的精细视觉部件（contype=0, conaffinity=0, mass=0，不干扰碰撞包络）
        if otype == "cube":
            top_inlay = body.add_geom(
                name="obj_cube_top_inlay",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[hx * 0.65, hy * 0.65, 0.001],
                pos=[0.0, 0.0, hz + 0.0005],
                rgba=[0.98, 0.82, 0.25, 1.0],
                mass=0.0,
            )
            top_inlay.contype = 0
            top_inlay.conaffinity = 0
        elif otype == "eraser":
            sleeve = body.add_geom(
                name="obj_eraser_sleeve",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[hx * 0.62, hy + 0.0006, hz + 0.0006],
                pos=[-hx * 0.12, 0.0, 0.0],
                rgba=cat["sleeve_rgba"],
                mass=0.0,
            )
            sleeve.contype = 0
            sleeve.conaffinity = 0
        elif otype == "glue_stick":
            col.rgba[:] = [1.0, 1.0, 1.0, 0.0]
            barrel = body.add_geom(
                name="obj_glue_barrel_vis",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[-hx, 0.0, 0.0, hx * 0.38, 0.0, 0.0],
                size=[hy, 0.0, 0.0],
                rgba=cat["rgba"],
                mass=0.0,
            )
            barrel.contype = 0
            barrel.conaffinity = 0
            cap = body.add_geom(
                name="obj_glue_cap_vis",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[hx * 0.38, 0.0, 0.0, hx, 0.0, 0.0],
                size=[hy * 1.03, 0.0, 0.0],
                rgba=cat["cap_rgba"],
                mass=0.0,
            )
            cap.contype = 0
            cap.conaffinity = 0
            band = body.add_geom(
                name="obj_glue_label_vis",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[-hx * 0.45, 0.0, 0.0, hx * 0.20, 0.0, 0.0],
                size=[hy * 1.02, 0.0, 0.0],
                rgba=cat["label_band_rgba"],
                mass=0.0,
            )
            band.contype = 0
            band.conaffinity = 0
        elif otype == "plastic_bottle":
            col.rgba[:] = [1.0, 1.0, 1.0, 0.0]
            bottle_body = body.add_geom(
                name="obj_bottle_body_vis",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[-hx, 0.0, 0.0, hx * 0.40, 0.0, 0.0],
                size=[hy, 0.0, 0.0],
                rgba=cat["rgba"],
                mass=0.0,
            )
            bottle_body.contype = 0
            bottle_body.conaffinity = 0
            bottle_label = body.add_geom(
                name="obj_bottle_label_vis",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[-hx * 0.45, 0.0, 0.0, hx * 0.15, 0.0, 0.0],
                size=[hy * 1.02, 0.0, 0.0],
                rgba=cat["label_band_rgba"],
                mass=0.0,
            )
            bottle_label.contype = 0
            bottle_label.conaffinity = 0
            bottle_neck = body.add_geom(
                name="obj_bottle_neck_vis",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[hx * 0.40, 0.0, 0.0, hx * 0.72, 0.0, 0.0],
                size=[hy * 0.58, 0.0, 0.0],
                rgba=cat["rgba"],
                mass=0.0,
            )
            bottle_neck.contype = 0
            bottle_neck.conaffinity = 0
            bottle_cap = body.add_geom(
                name="obj_bottle_cap_vis",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[hx * 0.72, 0.0, 0.0, hx, 0.0, 0.0],
                size=[hy * 0.62, 0.0, 0.0],
                rgba=cat["cap_rgba"],
                mass=0.0,
            )
            bottle_cap.contype = 0
            bottle_cap.conaffinity = 0


def _add_zed_mini_head_cameras_to_spec(spec: mujoco.MjSpec, head_cfg: dict[str, Any]) -> None:
    """在机器人头部安装 Stereolabs ZED Mini 双目深度相机（含 3D 外壳模型与左右双路光学相机）。"""
    mount_pos = head_cfg["mount_pos_world_m"]
    quat = head_cfg["orientation_quat_wxyz"]
    fovy = head_cfg["fovy_degrees"]
    half_b = 0.5 * head_cfg["baseline_m"]

    # 添加左右双目相机（在世界坐标系下保持严格水平 63mm 双目基线与相同光轴姿态）
    cam_left = spec.worldbody.add_camera(name=head_cfg["left_camera_name"])
    cam_left.pos[:] = head_cfg["left_camera_world_pos_m"]
    cam_left.quat[:] = quat
    cam_left.fovy = fovy

    cam_right = spec.worldbody.add_camera(name=head_cfg["right_camera_name"])
    cam_right.pos[:] = head_cfg["right_camera_world_pos_m"]
    cam_right.quat[:] = quat
    cam_right.fovy = fovy

    # 如果启用了头部双目模型，在机器人颈部上方添加 ZED Mini 支架与双目相机横梁实体（纯视觉，不参与碰撞）
    if not head_cfg.get("enabled", False):
        return

    stalk = spec.worldbody.add_geom(
        name="zed_mini_mount_stalk",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0.0, 0.0, 0.698, mount_pos[0] - 0.024, mount_pos[1], mount_pos[2]],
        size=[0.009, 0.0, 0.0],
        rgba=[0.18, 0.19, 0.22, 1.0],
        mass=0.0,
    )
    stalk.contype = 0
    stalk.conaffinity = 0

    head_body = spec.worldbody.add_body(
        name="zed_mini_head",
        pos=[mount_pos[0] - 0.022, mount_pos[1], mount_pos[2]],
    )
    hx, hy, hz = DEFAULT_ZED_MINI_SPEC["housing_half_size_xyz_m"]
    housing = head_body.add_geom(
        name="zed_mini_housing",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, hz],
        pos=[0.0, 0.0, 0.0],
        rgba=[0.16, 0.17, 0.19, 1.0],
        mass=0.0,
    )
    housing.contype = 0
    housing.conaffinity = 0

    vis_plate = head_body.add_geom(
        name="zed_mini_front_visor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.002, hy * 0.94, hz * 0.82],
        pos=[hx + 0.001, 0.0, 0.0],
        rgba=[0.07, 0.07, 0.09, 1.0],
        mass=0.0,
    )
    vis_plate.contype = 0
    vis_plate.conaffinity = 0

    for side_name, y_off in (("left", half_b), ("right", -half_b)):
        bezel = head_body.add_geom(
            name=f"zed_mini_lens_bezel_{side_name}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=[hx, y_off, 0.0, hx + 0.004, y_off, 0.0],
            size=[0.0095, 0.0, 0.0],
            rgba=[0.28, 0.30, 0.34, 1.0],
            mass=0.0,
        )
        bezel.contype = 0
        bezel.conaffinity = 0
        pupil = head_body.add_geom(
            name=f"zed_mini_lens_glass_{side_name}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=[hx + 0.003, y_off, 0.0, hx + 0.005, y_off, 0.0],
            size=[0.0065, 0.0, 0.0],
            rgba=[0.06, 0.18, 0.35, 1.0],
            mass=0.0,
        )
        pupil.contype = 0
        pupil.conaffinity = 0


def apply_drawer_variation_to_spec(spec: mujoco.MjSpec, variation: dict[str, Any]) -> None:
    """将解析后的抽屉、桌面物体与头部双目参数应用到 MjSpec（原地修改内存中的 spec，不改动磁盘 XML）。"""
    cabinet = next(b for b in spec.bodies if b.name == "cabinet_drawer")
    cabinet.pos[0] += variation["drawer_offset_x_m"]
    cabinet.pos[1] += variation["drawer_offset_y_m"]

    if variation.get("is_baseline_geometry", False):
        if variation.get("explicit_handle_radius"):
            next(g for g in spec.geoms if g.name == "drawer_handle").size[0] = variation["handle_radius_m"]
        return

    cabinet.pos[2] += variation["drawer_offset_z_m"]

    sy = variation["width_scale"]
    sx = variation["depth_scale"]
    sz = variation["height_scale"]
    dz = variation["drawer_offset_z_m"]
    t = BASE_WALL_HALF_THICKNESS

    extra_half_d = 0.05 * (sx - 1.0)
    cab_half_d = 0.05 * sx
    cab_half_w = 0.05 * sy
    cab_half_h = 0.05 * sz
    base_half_z = 0.03 + 0.5 * dz

    geoms = {g.name: g for g in spec.geoms if g.name}

    # 如果移除了无关的阀门和柜门，将其移至地下并关闭碰撞与可见度，同时适度拓宽桌面以容纳双抽屉与多物体
    if variation.get("remove_unused_fixtures", False):
        for body_name in ("valve_base", "cabinet_door"):
            b = next((body for body in spec.bodies if body.name == body_name), None)
            if b is not None:
                b.pos[:] = [0.0, 0.0, -2.0]
        for geom_name in (
            "valve_pipe",
            "valve_hub",
            "valve_lever",
            "valve_grip",
            "door_post",
            "door_panel",
            "door_handle",
        ):
            if geom_name in geoms:
                geoms[geom_name].contype = 0
                geoms[geom_name].conaffinity = 0
                geoms[geom_name].rgba[:] = [0.0, 0.0, 0.0, 0.0]
        table_body = next((body for body in spec.bodies if body.name == "table"), None)
        if table_body is not None and "table_top" in geoms:
            table_body.pos[:] = [0.33, 0.0, 0.36]
            geoms["table_top"].size[:] = [0.27, 0.42, 0.04]

    g_base = geoms["drawer_base"]
    g_base.size[:] = [cab_half_d, cab_half_w + t, base_half_z]
    g_base.pos[:] = [extra_half_d, 0.0, -base_half_z]

    g_bottom = geoms["drawer_bottom"]
    g_bottom.size[:] = [cab_half_d, cab_half_w + t, t]
    g_bottom.pos[:] = [extra_half_d, 0.0, t]

    g_top = geoms["drawer_top"]
    g_top.size[:] = [cab_half_d, cab_half_w + t, t]
    g_top.pos[:] = [extra_half_d, 0.0, 2.0 * cab_half_h + t]

    g_back = geoms["drawer_back"]
    g_back.size[:] = [t, cab_half_w + t, cab_half_h]
    g_back.pos[:] = [0.05 + 2.0 * extra_half_d, 0.0, cab_half_h + t]

    g_side_l = geoms["drawer_sideL"]
    g_side_l.size[:] = [cab_half_d + t, t, cab_half_h]
    g_side_l.pos[:] = [extra_half_d, cab_half_w, cab_half_h + t]

    g_side_r = geoms["drawer_sideR"]
    g_side_r.size[:] = [cab_half_d + t, t, cab_half_h]
    g_side_r.pos[:] = [extra_half_d, -cab_half_w, cab_half_h + t]

    drawer_body = next(b for b in spec.bodies if b.name == "drawer")
    drawer_body.pos[2] = 0.045 * sz

    box_extra_half_d = 0.038 * (sx - 1.0)
    box_hx = 0.038 * sx
    box_hy = 0.038 * sy
    box_hz = 0.030 * sz
    g_box = geoms["drawer_box"]

    app = variation["appearance"]
    if variation.get("hollow_drawer", False):
        # 真实“空心抽屉槽”：原 drawer_box 变为底板（厚 8mm），并新增左、右、后三面盒壁（壁厚 6mm）
        floor_hz = 0.004
        wall_t = 0.003
        wall_hz = box_hz - floor_hz
        g_box.size[:] = [box_hx, box_hy, floor_hz]
        g_box.pos[:] = [box_extra_half_d, 0.0, -box_hz + floor_hz]
        g_box.mass = 0.08
        g_box.friction[:] = [0.8, 0.02, 0.01]

        for wname, wsize, wpos in (
            ("drawer_wall_left", [box_hx, wall_t, wall_hz], [box_extra_half_d, box_hy - wall_t, floor_hz]),
            ("drawer_wall_right", [box_hx, wall_t, wall_hz], [box_extra_half_d, -box_hy + wall_t, floor_hz]),
            (
                "drawer_wall_back",
                [wall_t, box_hy - 2 * wall_t, wall_hz],
                [box_extra_half_d + box_hx - wall_t, 0.0, floor_hz],
            ),
        ):
            gw = drawer_body.add_geom(
                name=wname,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=wsize,
                pos=wpos,
                rgba=app["drawer_box_rgba"],
                mass=0.04,
            )
            gw.contype = 1
            gw.conaffinity = 1
            gw.friction[:] = [0.6, 0.01, 0.01]
    else:
        g_box.size[:] = [box_hx, box_hy, box_hz]
        g_box.pos[:] = [box_extra_half_d, 0.0, 0.0]

    g_front = geoms["drawer_front"]
    g_front.size[:] = [0.006, 0.040 * sy, 0.028 * sz]
    g_front.pos[:] = [-0.046, 0.0, 0.0]

    front_outer_x = -0.052
    standoff = 0.040 * variation["handle_standoff_scale"]
    handle_x = front_outer_x - standoff
    half_span = 0.020 * variation["handle_span_scale"]
    stem_y = 0.90 * half_span

    g_handle = geoms["drawer_handle"]
    g_handle.fromto[:] = [handle_x, -half_span, 0.0, handle_x, half_span, 0.0]
    g_handle.size[0] = variation["handle_radius_m"]

    g_stem1 = geoms["drawer_handle_stem1"]
    g_stem1.fromto[:] = [front_outer_x, -stem_y, 0.0, handle_x, -stem_y, 0.0]

    g_stem2 = geoms["drawer_handle_stem2"]
    g_stem2.fromto[:] = [front_outer_x, stem_y, 0.0, handle_x, stem_y, 0.0]

    for cab_geom in (g_base, g_bottom, g_top, g_back, g_side_l, g_side_r):
        cab_geom.rgba[:] = app["cabinet_rgba"]
    g_box.rgba[:] = app["drawer_box_rgba"]
    g_front.rgba[:] = app["drawer_front_rgba"]
    for h_geom in (g_handle, g_stem1, g_stem2):
        h_geom.rgba[:] = app["handle_rgba"]

    panel_style = variation.get("panel_style", "inset")
    trim_rgba = app["trim_rgba"]
    if panel_style == "framed":
        for idx, z_sign in enumerate((-1.0, 1.0), start=1):
            rail = drawer_body.add_geom(
                name=f"drawer_trim_rail_{idx}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.0015, 0.038 * sy, 0.003 * sz],
                pos=[-0.0525, 0.0, z_sign * 0.022 * sz],
                rgba=trim_rgba,
                mass=0.0,
            )
            rail.contype = 0
            rail.conaffinity = 0
    elif panel_style == "raised_center":
        plaque = drawer_body.add_geom(
            name="drawer_trim_center",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.0015, 0.030 * sy, 0.018 * sz],
            pos=[-0.0525, 0.0, 0.0],
            rgba=trim_rgba,
            mass=0.0,
        )
        plaque.contype = 0
        plaque.conaffinity = 0
    elif panel_style == "twin_rail":
        for idx, z_sign in enumerate((-1.0, 1.0), start=1):
            strip = drawer_body.add_geom(
                name=f"drawer_trim_strip_{idx}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.0015, 0.034 * sy, 0.0025 * sz],
                pos=[-0.0525, 0.0, z_sign * 0.014 * sz],
                rgba=trim_rgba,
                mass=0.0,
            )
            strip.contype = 0
            strip.conaffinity = 0
    elif panel_style == "bevel_top":
        crown = cabinet.add_geom(
            name="cabinet_crown_trim",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[cab_half_d + 0.004, cab_half_w + t + 0.004, 0.004],
            pos=[extra_half_d, 0.0, 2.0 * cab_half_h + 2.0 * t + 0.004],
            rgba=trim_rgba,
            mass=0.0,
        )
        crown.contype = 0
        crown.conaffinity = 0

    # 添加可选第二抽屉、桌面可抓取物体及头部 ZED Mini 双目相机
    extra_qpos: list[float] = []
    if variation.get("drawer_layout") == "dual_side_by_side" and variation.get("secondary_drawer"):
        sec_cfg = variation["secondary_drawer"]
        _add_secondary_drawer_to_spec(spec, sec_cfg)
        extra_qpos.append(-float(sec_cfg.get("initial_open_m", 0.0)))

    objects = variation.get("objects") or []
    if objects:
        _add_tabletop_objects_to_spec(spec, objects)
        for obj in objects:
            extra_qpos.extend(obj["initial_pos_xyz_m"])
            extra_qpos.extend(obj["initial_quat_wxyz"])

    head_cfg = variation.get("head_stereo_camera")
    if head_cfg and head_cfg.get("enabled", False):
        _add_zed_mini_head_cameras_to_spec(spec, head_cfg)

    # 同步更新 ready keyframe 的 qpos（含初始抽屉开启位移、第二抽屉位移及全部自由物体的 7D 位姿）
    if spec.keys:
        ready_key = next((k for k in spec.keys if k.name == "ready"), spec.keys[0])
        base_qpos = list(ready_key.qpos)
        init_open_val = float(variation.get("initial_drawer_open_m", 0.0))
        if len(base_qpos) >= 19:
            base_qpos[18] = -init_open_val
            if init_open_val > 0.02:
                base_qpos[9:16] = PREOPENED_RIGHT_READY_QPOS
        if extra_qpos:
            ready_key.qpos = base_qpos + extra_qpos
        else:
            ready_key.qpos = base_qpos


def configure_wrist_side_cameras(model: mujoco.MjModel) -> None:
    """将左右腕部相机固定在腕部侧向局部坐标系，避开掌部遮挡，不依赖抽屉真值。"""
    position = np.array([0.07, 0.10, -0.04])
    target = np.array([0.0, 0.0, -0.14])
    z_axis = position - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.column_stack([x_axis, y_axis, z_axis]).ravel())
    for name in ("camera_wrist_left", "camera_wrist_right"):
        cid = model.camera(name).id
        model.cam_pos[cid], model.cam_quat[cid], model.cam_fovy[cid] = position, quaternion, 85


def compile_drawer_scene(
    scene_path: str,
    variation: dict[str, Any],
    *,
    wrist_view: str = "source",
    camera_local_poses: dict[str, dict[str, Any]] | None = None,
    shadow_size: int = 1024,
) -> mujoco.MjModel:
    """从上游场景文件与变体配置统一编译 MuJoCo 模型。"""
    spec = mujoco.MjSpec.from_file(str(scene_path))
    apply_drawer_variation_to_spec(spec, variation)
    top = next(cam for cam in spec.cameras if cam.name == "tablecam")
    camera = spec.worldbody.add_camera(name="camera_ceiling")
    camera.pos, camera.quat, camera.fovy = top.pos, top.quat, top.fovy
    model = spec.compile()
    if variation.get("objects"):
        for arm_side in ("right", "left"):
            for finger_side in ("inner", "outer"):
                for idx in range(3):
                    gname = f"finger_{finger_side}_{arm_side}_collision_{idx:02d}"
                    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                    if gid >= 0:
                        model.geom_condim[gid] = 6
                        model.geom_friction[gid] = [2.5, 0.15, 0.05]
        for hname in ("drawer_handle", "drawer_handle_stem1", "drawer_handle_stem2"):
            hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, hname)
            if hid >= 0:
                model.geom_priority[hid] = 2
                model.geom_condim[hid] = 3
                model.geom_friction[hid] = [1.0, 0.01, 0.01]
                model.geom_solref[hid] = [0.005, 1.0]
    if camera_local_poses is not None:
        for name, pose in camera_local_poses.items():
            try:
                cid = model.camera(name).id
            except KeyError:
                continue
            model.cam_pos[cid] = pose["position"]
            model.cam_quat[cid] = pose["quaternion_wxyz"]
            model.cam_fovy[cid] = pose["fovy_degrees"]
    elif wrist_view == "side":
        configure_wrist_side_cameras(model)
    model.vis.quality.shadowsize = shadow_size
    model.vis.global_.offwidth = max(320, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(240, model.vis.global_.offheight)
    return model
