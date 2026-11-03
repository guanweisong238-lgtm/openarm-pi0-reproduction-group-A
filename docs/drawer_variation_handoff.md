# 抽屉多形态与多高度参数化仿真交接说明（A1 -> A2）

- **更新日期**：2026-11-03
- **交付分支**：`a1-simulation`
- **适用任务**：面向 $\pi_0$ 抽屉开启/往返模型泛化能力提升的多外形、多尺寸、多位置/高度数据扩采与闭环评估

---

## 1. 交付模块与职责边界

针对 A2 提出的多样化抽屉采集需求（增加抽屉外形与配色、改变柜体/抽屉/把手尺寸、支持三维位置与高度可变），A1 已将全部场景参数化生成、真实接触专家采集、回放与验证接口整理至以下位置：

| 路径 | 职责说明 |
| :--- | :--- |
| `simulation/drawer_scene_generator.py` | **核心场景生成器**：定义 5 种外形预设（`SHAPE_PRESETS`）、5 种材质配色（`APPEARANCE_PRESETS`）、宽/深/高联动缩放、X/Y/Z 三维偏移（含底座防悬空自动伸缩）、滑动间隙校验与确定性哈希 `config_sha256` |
| `simulation/drawer_simulation.py` | **30Hz 闭环仿真环境**：提供 `DrawerSimulation` 与 `finger_handle_contacts`，从 `report.json["scene_overrides"]` 自动重建变体场景 |
| `scripts/a1_simulation/contact_drawer_controller.py` | **真实接触专家**：`ContactDrawerController`，全程禁用 `weld` 绑定，自动读取变体把手中心并求解无碰撞 IK |
| `scripts/a1_simulation/collect_drawer_trial.py` | **变体示教采集入口**：支持命令行传入预设、尺寸缩放比例、三维位姿偏移及初始关节扰动 |
| `scripts/a1_simulation/replay_drawer_trial.py` | **独立回放校验**：支持 `--headless` 零误差断言（`max_qpos_error < 1e-6`）与实时 3D 窗口循环观看 |
| `scripts/a1_simulation/verify_drawer_variants.py` | **三级可控验证入口**：支持 `--mode scene`（几何与渲染）、`--mode expert`（接触专家与回放）、`--mode policy`（模型闭环推理） |
| `configs/drawer_variants_preset.json` | **代表性变体清单**：预置 `v01`–`v05` 五款不同外形、配色、尺寸与高度的抽屉采集配置 |

> **向后兼容性保证**：当不传任何新增变体参数（即默认 `variant_preset="standard"`, `appearance_preset="classic_wood_blue"`, `drawer_offset_z=0.0`, 各缩放系数为 `1.0`）时，`drawer_scene_generator.py` 编译出的 MuJoCo 模型与改造前逐字节 100% 一致，历史 13 条 `v003` 数据及评估集无需重采即可直接兼容。

---

## 2. 抽屉外形、配色与几何可调参数规格

### 2.1 五种抽屉外形家族（`--variant-preset`）

| 预设名称 | 宽度缩放 (`Y`) | 深度缩放 (`X`) | 高度缩放 (`Z`) | 把手长度缩放 | 把手半径 | 面板装饰结构 (`panel_style`) | 柜体外部尺寸 `[深X, 宽Y, 高Z]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `standard` | `1.00` | `1.00` | `1.00` | `1.00` | `7.0 mm` | `inset`（原版内嵌） | `110 × 110 × 110 mm` |
| `wide_low` | `1.25` | `1.00` | `0.88` | `1.25` | `7.0 mm` | `framed`（上下外框饰条） | `110 × 135 × 98 mm` |
| `narrow_tall` | `0.85` | `1.00` | `1.18` | `0.85` | `7.5 mm` | `raised_center`（中心浮雕板） | `110 × 95 × 128 mm` |
| `deep_compact` | `0.90` | `1.15` | `0.94` | `0.92` | `8.5 mm` | `twin_rail`（双横条饰面） | `125 × 100 × 104 mm` |
| `large_modern` | `1.18` | `1.10` | `1.12` | `1.15` | `8.0 mm` | `bevel_top`（柜顶帽檐饰条） | `120 × 128 × 122 mm` |

### 2.2 五种材质配色预设（`--appearance-preset`）

- `classic_wood_blue`：经典暖木色柜体 + 蓝灰面板 + 深灰把手（默认）
- `light_oak_white`：浅橡木色柜体 + 象牙白面板 + 银灰金属把手
- `walnut_slate`：深胡桃木色柜体 + 深岩灰面板 + 哑光黑把手
- `industrial_steel`：工业冷灰钢柜 + 暖橙面板 + 炭黑把手
- `sage_green`：浅灰白柜体 + 鼠尾草绿面板 + 黄铜色把手

### 2.3 三维位置、高度与尺寸连续调节参数

| 命令行参数 | 默认值 | 安全合法范围 | 几何联动与物理保护机制 |
| :--- | :---: | :---: | :--- |
| `--drawer-offset-x` | `0.0` | `[-0.10, +0.14] m` | 柜体沿世界 X 轴平移（接触专家推荐在 `[0.098, 0.115] m` 区间采样） |
| `--drawer-offset-y` | `0.0` | `[-0.05, +0.05] m` | 柜体沿世界 Y 轴横移，自动校验与左侧阀门间距 |
| `--drawer-offset-z` | `0.0` | `[-0.04, +0.06] m` | 柜体沿世界 Z 轴高度偏移；**底座 `drawer_base` 自动按 $H_{\text{base}} = 0.06 + dz$ 伸缩**，底面恒贴合桌面 `z = 0.40 m` |
| `--width-scale` | 随预设 | `[0.75, 1.35]` | 联动缩放柜体、抽屉盒与前面板宽度，保持滑动间隙 $\ge 4\text{ mm}$ |
| `--depth-scale` | 随预设 | `[0.80, 1.30]` | 锚定柜体前开口平面（局部 $x = -0.05\text{ m}$）不变，向柜后 $+X$ 延伸深度 |
| `--height-scale` | 随预设 | `[0.75, 1.30]` | 联动缩放柜体、抽屉盒与前面板高度及抽屉中心高度 |
| `--handle-radius` | 随预设 | `[0.005, 0.020] m` | 把手圆柱横杆半径 |
| `--handle-span-scale` | 随预设 | `[0.75, 1.40]` | 把手横杆长度与两侧连接杆（`stem1/2`）间距同步缩放 |
| `--handle-standoff-scale` | 随预设 | `[0.80, 1.25]` | 把手横杆离前面板外表面距离缩放 |

---

## 3. A2 对接与快速使用命令

### 3.1 一键运行三级验收（场景 / 接触专家 / 模型权重闭环）

在 `openarm-a1` 根目录下：

```bash
# 1. 验证 5 款抽屉几何无穿模、底座贴合桌面，并生成三路相机拼图至 outputs/drawer_variants_preview/
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode scene

# 2. 验证 5 款抽屉真实接触专家拉开 >=80mm 且独立回放误差为 0.0
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode expert

# 3. 使用已就位的模型检查点运行闭环推理验证（自动输出闭环录像至 outputs/drawer_variants_verify/）
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode policy --seconds 18
```

### 3.2 采集带三路 RGB 的多样化抽屉示教

```bash
uv run --locked python -m scripts.a1_simulation.collect_drawer_trial \
  --mode contact --expert contact \
  --variant-preset wide_low \
  --appearance-preset light_oak_white \
  --drawer-offset-x 0.10 --drawer-offset-y 0.004 --drawer-offset-z 0.008 \
  --pull-distance 0.10 --return-drawer --wrist-view side \
  --initial-right-jitter 0.02 --seed 202 \
  --output outputs/custom_drawer_demo/wide_low_oak_001
```

### 3.3 独立回放校验（无头或 3D 窗口）

```bash
# 无头校验（断言 max_qpos_error < 1e-6）
uv run --locked python -m scripts.a1_simulation.replay_drawer_trial \
  --episode outputs/custom_drawer_demo/wide_low_oak_001/episode.npz --headless

# 弹出 3D 交互窗口观看回放
uv run --locked python -m scripts.a1_simulation.replay_drawer_trial \
  --episode outputs/custom_drawer_demo/wide_low_oak_001/episode.npz
```
