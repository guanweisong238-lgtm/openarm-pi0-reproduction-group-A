"""统一抽屉场景参数化配置与生成模块（由 A1 在 simulation/ 下维护，供 A1/A2 共用）。

核心特性：
1. 默认参数（variant_preset="standard", appearance_preset="classic_wood_blue", 零偏移、1.0缩放）
   与上游 articulated_scene.xml 保持 100% 数值与几何一致，完全兼容历史示教与评测。
2. 支持 5 种抽屉外形家族（standard / wide_low / narrow_tall / deep_compact / large_modern）
   与 5 种材质配色方案（classic_wood_blue / light_oak_white / walnut_slate / industrial_steel / sage_green）。
3. 支持柜体与抽屉宽（Y）、深（X）、高（Z）独立缩放，自动联动更新底板、顶板、背板、左右侧板、
   抽屉盒、前面板、把手横杆与连接杆，并自动校验滑动间隙与夹爪抓握空间。
4. 支持柜体三维位置与高度可变（drawer_offset_x / y / z），改变高度 Z 时自动伸缩底座 drawer_base，
   保证底座下表面始终紧贴桌面（world z = 0.40m），绝不悬空或穿模。
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import mujoco
import numpy as np

SCHEMA_VERSION = 1
GENERATOR_VERSION = "1.0.0"

TABLE_TOP_WORLD_Z = 0.40
BASE_CABINET_POS = (0.40, -0.28, 0.46)
BASE_WALL_HALF_THICKNESS = 0.005

SHAPE_PRESETS: dict[str, dict[str, Any]] = {
    "standard": {
        "description": "标准基准款（与上游原始抽屉几何100%一致）",
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
        "cabinet_rgba": [0.55, 0.45, 0.32, 1.0],
        "drawer_box_rgba": [0.30, 0.45, 0.62, 1.0],
        "drawer_front_rgba": [0.22, 0.36, 0.52, 1.0],
        "handle_rgba": [0.20, 0.20, 0.22, 1.0],
        "trim_rgba": [0.16, 0.28, 0.42, 1.0],
    },
    "light_oak_white": {
        "description": "浅橡木色柜体 + 象牙白面板 + 银灰金属把手",
        "cabinet_rgba": [0.76, 0.64, 0.48, 1.0],
        "drawer_box_rgba": [0.86, 0.86, 0.82, 1.0],
        "drawer_front_rgba": [0.92, 0.91, 0.88, 1.0],
        "handle_rgba": [0.52, 0.55, 0.60, 1.0],
        "trim_rgba": [0.68, 0.56, 0.40, 1.0],
    },
    "walnut_slate": {
        "description": "深胡桃木柜体 + 岩灰面板 + 哑光黑把手",
        "cabinet_rgba": [0.32, 0.22, 0.15, 1.0],
        "drawer_box_rgba": [0.35, 0.37, 0.40, 1.0],
        "drawer_front_rgba": [0.26, 0.28, 0.32, 1.0],
        "handle_rgba": [0.12, 0.12, 0.14, 1.0],
        "trim_rgba": [0.46, 0.33, 0.22, 1.0],
    },
    "industrial_steel": {
        "description": "工业冷灰钢柜 + 暖橙面板 + 炭黑把手",
        "cabinet_rgba": [0.42, 0.45, 0.48, 1.0],
        "drawer_box_rgba": [0.52, 0.55, 0.58, 1.0],
        "drawer_front_rgba": [0.82, 0.50, 0.18, 1.0],
        "handle_rgba": [0.15, 0.15, 0.16, 1.0],
        "trim_rgba": [0.26, 0.29, 0.32, 1.0],
    },
    "sage_green": {
        "description": "浅灰白柜体 + 鼠尾草绿面板 + 黄铜色把手",
        "cabinet_rgba": [0.78, 0.76, 0.72, 1.0],
        "drawer_box_rgba": [0.48, 0.58, 0.50, 1.0],
        "drawer_front_rgba": [0.36, 0.49, 0.41, 1.0],
        "handle_rgba": [0.72, 0.58, 0.26, 1.0],
        "trim_rgba": [0.25, 0.36, 0.29, 1.0],
    },
}


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
    overrides_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """解析并校验抽屉场景变体参数，返回包含派生几何与确定性哈希的完整配置字典。"""
    if overrides_dict is not None:
        if "drawer_variation" in overrides_dict and isinstance(overrides_dict["drawer_variation"], dict):
            dv = overrides_dict["drawer_variation"]
            return resolve_drawer_variation(
                variant_preset=dv.get("variant_preset", "standard"),
                appearance_preset=dv.get("appearance_preset", "classic_wood_blue"),
                drawer_offset_x_m=float(dv.get("drawer_offset_x_m", overrides_dict.get("drawer_offset_x_m", 0.0))),
                drawer_offset_y_m=float(dv.get("drawer_offset_y_m", overrides_dict.get("drawer_offset_y_m", 0.0))),
                drawer_offset_z_m=float(dv.get("drawer_offset_z_m", overrides_dict.get("drawer_offset_z_m", 0.0))),
                width_scale=dv.get("width_scale"),
                depth_scale=dv.get("depth_scale"),
                height_scale=dv.get("height_scale"),
                handle_span_scale=dv.get("handle_span_scale"),
                handle_standoff_scale=dv.get("handle_standoff_scale"),
                handle_radius_m=dv.get("handle_radius_m", overrides_dict.get("drawer_handle_radius_m")),
                panel_style=dv.get("panel_style"),
            )
        return resolve_drawer_variation(
            variant_preset=overrides_dict.get("variant_preset", "standard"),
            appearance_preset=overrides_dict.get("appearance_preset", "classic_wood_blue"),
            drawer_offset_x_m=float(
                overrides_dict.get(
                    "drawer_offset_x_m",
                    overrides_dict.get("drawer_offset_x", overrides_dict.get("drawer_x", 0.0)),
                )
            ),
            drawer_offset_y_m=float(
                overrides_dict.get(
                    "drawer_offset_y_m",
                    overrides_dict.get("drawer_offset_y", overrides_dict.get("drawer_y", 0.0)),
                )
            ),
            drawer_offset_z_m=float(
                overrides_dict.get(
                    "drawer_offset_z_m",
                    overrides_dict.get("drawer_offset_z", overrides_dict.get("drawer_z", 0.0)),
                )
            ),
            width_scale=overrides_dict.get("width_scale"),
            depth_scale=overrides_dict.get("depth_scale"),
            height_scale=overrides_dict.get("height_scale"),
            handle_span_scale=overrides_dict.get("handle_span_scale"),
            handle_standoff_scale=overrides_dict.get("handle_standoff_scale"),
            handle_radius_m=overrides_dict.get(
                "drawer_handle_radius_m",
                overrides_dict.get("handle_radius_m", overrides_dict.get("handle_radius")),
            ),
            panel_style=overrides_dict.get("panel_style"),
        )

    if variant_preset not in SHAPE_PRESETS:
        raise ValueError(f"未知抽屉外形预设: {variant_preset}，可选: {list(SHAPE_PRESETS)}")
    if appearance_preset not in APPEARANCE_PRESETS:
        raise ValueError(f"未知抽屉外观预设: {appearance_preset}，可选: {list(APPEARANCE_PRESETS)}")

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
    if not (-0.05 <= dy <= 0.05):
        raise ValueError(f"drawer_offset_y_m ({dy}) 超出安全范围 [-0.05, 0.05]")
    if not (-0.04 <= dz <= 0.06):
        raise ValueError(f"drawer_offset_z_m ({dz}) 超出安全范围 [-0.04, 0.06]")

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
    front_half_h = 0.028 * sz

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
    if cabinet_max_y > -0.14:
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
    )

    core_params = {
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
        "is_baseline_geometry": is_baseline_geometry,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(core_params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

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


def apply_drawer_variation_to_spec(spec: mujoco.MjSpec, variation: dict[str, Any]) -> None:
    """将解析后的抽屉变体参数应用到 MjSpec（原地修改内存中的 spec，不改动磁盘 XML）。"""
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
    g_box = geoms["drawer_box"]
    g_box.size[:] = [0.038 * sx, 0.038 * sy, 0.030 * sz]
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

    app = variation["appearance"]
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
    if camera_local_poses is not None:
        for name, pose in camera_local_poses.items():
            cid = model.camera(name).id
            model.cam_pos[cid] = pose["position"]
            model.cam_quat[cid] = pose["quaternion_wxyz"]
            model.cam_fovy[cid] = pose["fovy_degrees"]
    elif wrist_view == "side":
        configure_wrist_side_cameras(model)
    model.vis.quality.shadowsize = shadow_size
    model.vis.global_.offwidth = max(320, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(240, model.vis.global_.offheight)
    return model
