# 抽屉与桌面物体多形态参数化仿真 + ZED Mini 双目相机交接说明（A1 -> A2）

- **更新日期**：2026-11-07
- **交付分支**：`a1-simulation`（Schema v2 / Generator v2.0.0）
- **适用任务**：
  1. 面向 $\pi_0$ 模型泛化能力提升的**多外形、多配色、多尺寸、多高度/位置抽屉**示教扩采与闭环评估；
  2. 面向最终展示目标的**“根据自然语言指令将不同物体（方块、橡皮、胶棒、空塑料瓶）放入不同空心抽屉”**仿真采集与回放；
  3. **机器人头部 Stereolabs ZED Mini 双目深度相机（63 mm 基线）**视觉观测输出。

---

## 1. 交付模块与职责边界

A1 已将全部场景参数化生成、空心抽屉槽与双抽屉布局、4 种桌面可抓取物体、ZED Mini 头部双目相机、真实接触专家采集、回放与四级验证接口整理至以下位置：

| 路径 | 职责说明 |
| :--- | :--- |
| `simulation/drawer_scene_generator.py` | **核心场景生成器（v2.0.0）**：定义 5 种抽屉外形预设（`SHAPE_PRESETS`）、5 种材质配色（`APPEARANCE_PRESETS`）、**真实空心抽屉槽（`hollow_drawer`）**、**单/双抽屉布局（`drawer_layout`）**、**4 种桌面物体（`OBJECT_CATALOG`）**、**头部 ZED Mini 双目相机（`DEFAULT_ZED_MINI_SPEC` / `CAMERA_PRESETS`）**及自然语言指令自动生成器（`build_task_prompt`） |
| `simulation/drawer_simulation.py` | **30Hz 闭环仿真环境**：提供 `DrawerSimulation`、`finger_handle_contacts`、`finger_object_contacts` 与 `is_object_inside_drawer`（抽屉内腔三维落入判定），从 `report.json["scene_overrides"]` 自动重建完整变体场景 |
| `scripts/a1_simulation/contact_drawer_controller.py` | **真实接触专家**：`ContactDrawerController`，全程禁用 `weld` 绑定，支持真实接触拉/推抽屉（`open_drawer`）以及抓取桌面目标物体放入空心抽屉槽（`put_object_in_drawer`，兼容**抽屉预打开**与**先开抽屉再抓物体放入**两种流程） |
| `scripts/a1_simulation/collect_drawer_trial.py` | **变体示教采集入口**：支持命令行传入抽屉变体、空心抽屉、双抽屉、桌面物体组合、目标放入物体、头部双目相机预设及语言指令 `prompt` |
| `scripts/a1_simulation/replay_drawer_trial.py` | **独立回放校验**：支持 `--headless` 零误差断言（`max_qpos_error < 1e-6`）、物体落入抽屉内腔校验（`target_object_inside_drawer`）与实时 3D 窗口循环观看 |
| `scripts/a1_simulation/verify_drawer_variants.py` | **四级可控验证入口**：支持 `--mode scene`（五款抽屉几何与渲染）、`--mode expert`（接触专家开抽屉与回放）、`--mode objects_zed`（ZED Mini 双目 + 四种物体放入不同抽屉全流程验证）、`--mode policy`（模型闭环推理） |
| `configs/drawer_variants_preset.json` | **五款抽屉变体清单**：预置 `v01`–`v05` 五款不同外形、配色、尺寸与高度的抽屉采集配置 |
| `configs/drawer_objects_zed_preset.json` | **四物体入抽屉 + ZED Mini 双目配置清单**：预置 `obj01`–`obj05` 覆盖方块、橡皮、胶棒、空塑料瓶、不同空心抽屉、双抽屉并排以及“预开抽屉”与“先拉开抽屉再抓放物体”的完整验证配置 |

> **100% 向后兼容性保证**：当不启用空心抽屉、桌面物体与头部双目相机（即使用默认参数）时，`drawer_scene_generator.py` 编译出的 MuJoCo 模型与改造前逐字节 100% 一致，历史 `v003` 数据与 `016/2000` 检查点完全兼容。

---

## 2. 抽屉结构、桌面物体与头部双目相机规格

### 2.1 真实“空心抽屉槽”（`--hollow-drawer`）与单/双抽屉布局（`--drawer-layout`）

1. **真实空心抽屉槽（`hollow_drawer=True`）**：
   - 将原实心块 `drawer_box` 改造为由 **`8 mm` 厚底板 + `6 mm` 厚左右侧壁（`drawer_wall_left/right`）+ 后壁（`drawer_wall_back`）+ 前面板（`drawer_front`）** 围成的顶部开口空心容器；
   - 5 种抽屉外形家族（`standard` / `wide_low` / `narrow_tall` / `deep_compact` / `large_modern`）均支持空心化，标准款内腔净空尺寸为 `146 × 140 × 52 mm`，总质量保持 `0.20 kg` 不变。
2. **单抽屉 / 双抽屉并排布局（`--drawer-layout`）**：
   - `single`（默认）：主抽屉位于右臂工作区（默认 $y = -0.28\text{ m}$）；
   - `dual_side_by_side`：桌面上并排放置两个不同款式/颜色的空心抽屉（主抽屉 `cabinet_drawer` 与第二抽屉 `cabinet_drawer_2`，可通过 `--secondary-variant-preset` 与 `--secondary-appearance-preset` 独立指定外形与颜色），支撑“把某物体放进指定颜色/样式的抽屉”语言指令任务。
3. **抽屉起始状态可配（`--initial-drawer-open`）**：
   - `--initial-drawer-open 0.09`：抽屉起始处于拉开 `90 mm` 状态（右臂 `ready` 初态自动抬升至安全高度 `[0.26, -0.10, 0.58]` 避免碰撞前面板），机器人直接根据指令抓取目标物体投入抽屉槽；
   - `--initial-drawer-open 0.0`：抽屉起始关闭，机器人先以真实接触拉开抽屉 $\ge 85\text{ mm}$、平滑松开退离把手，再抓取目标物体放入抽屉。

### 2.2 四种桌面可抓取物体（`OBJECT_CATALOG` / `--objects` / `--target-object`）

每个物体均为带 `<freejoint/>` 的 6-DOF 自由刚体，包含高摩擦接触包络（`condim=6`, `friction=[2.5, 0.15, 0.05]`）与多部件精细视觉造型：

| 物体标识 (`type`) | 中文名称 | 英文标签 (`label_en`) | 碰撞半尺寸 `[hx, hy, hz]` | 质量 | 视觉结构与配色特征 | 默认桌面初始位置 `[x, y, z]` |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| `cube` | **方块** | `red cube` | `17 × 17 × 17 mm` | `45 g` | 红色正方体积木 + 顶部明黄嵌块 | `[0.32, -0.09, 0.417] m` |
| `eraser` | **橡皮** | `eraser` | `23 × 15 × 14 mm` | `38 g` | 象牙白橡皮主体 + 经典天蓝纸套护封 | `[0.32, -0.02, 0.414] m` |
| `glue_stick` | **胶棒** | `glue stick` | `27 × 14 × 14 mm` | `36 g` | 白色圆柱管身 + 鲜橙色笔帽 + 蓝标腰封 | `[0.25, -0.09, 0.414] m` |
| `plastic_bottle` | **空塑料瓶** | `empty plastic bottle` | `30 × 14 × 14 mm` | `28 g` | 半透明水蓝瓶身 + 缩颈瓶颈 + 深蓝瓶盖 + 白色标签环 | `[0.25, -0.02, 0.414] m` |

- **自然语言指令自动生成（`build_task_prompt`）**：
  - 单抽屉预开：`Put the red cube into the drawer.` / `Put the empty plastic bottle into the drawer.`
  - 双抽屉指代：`Put the red cube into the blue drawer.`
  - 先开抽屉再放物体：`Open the blue drawer and put the red cube into it.`
  - 也支持通过 `--prompt "..."` 传入任意自定义中/英文指令写入 `report.json["prompt"]`。

### 2.3 机器人头部 Stereolabs ZED Mini 双目深度相机（`--enable-head-stereo` / `--camera-preset`）

参照 Stereolabs ZED Mini 真机规格在机器人头部（颈部支架上方）建立 3D 相机实体与双路光学镜头：
- **外观模型**：深空灰圆角机身横梁（`124.5 × 30.5 × 26.5 mm`）、黑色前视镜面面板、左右双镜头金属外圈与深蓝光学镀膜镜片（纯视觉几何 `contype=0, conaffinity=0`，不干扰机械臂碰撞）；
- **光学参数**：
  - 左目相机名：`camera_head_left`，世界坐标 `[0.04, +0.0315, 0.82] m`
  - 右目相机名：`camera_head_right`，世界坐标 `[0.04, -0.0315, 0.82] m`
  - **双目水平基线（Stereo Baseline）**：严格 **`63.0 mm`（`0.063 m`）**
  - **垂直视场角（`fovy`）**：`54.0°`（俯视光轴指向桌面与抽屉工作区中心 `[0.35, -0.08, 0.44] m`）
- **采集相机组合预设（`--camera-preset`）**：
  - `legacy_3cam`（默认兼容）：`ceiling` + `wrist_left` + `wrist_right`
  - `zed_4cam`（推荐多模态采集）：`head_left` + `head_right` + `wrist_left` + `wrist_right`
  - `all_5cam`（全量五路）：`ceiling` + `head_left` + `head_right` + `wrist_left` + `wrist_right`

---

## 3. A2 对接与快速使用命令

### 3.1 一键运行四级验收（场景 / 抽屉专家 / 四物体入抽屉+ZED双目 / 模型闭环）

在 `openarm-a1` 根目录下：

```bash
# 1. 验证 5 款抽屉几何无穿模、底座贴合桌面，生成对比图至 outputs/drawer_variants_preview/all_variants_comparison.png
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode scene

# 2. 验证 5 款抽屉真实接触专家拉开 >=80mm 且独立回放误差为 0.0
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode expert

# 3. 验证 ZED Mini 双目相机（63mm基线）与 4 种物体（方块/橡皮/胶棒/空塑料瓶）放入不同空心抽屉 + 先开抽屉再放物体
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode objects_zed

# 4. 使用已就位的模型检查点运行闭环推理验证
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode policy --seconds 18
```

### 3.2 采集“语言指令抓取指定物体放入空心抽屉 + ZED Mini 双目”示教

```bash
# 示例 A：4 物体同台（预开抽屉），抓取空塑料瓶放入鼠尾草绿大号空心抽屉，录制 ZED 双目 + 双腕四路 RGB
uv run --locked python -m scripts.a1_simulation.collect_drawer_trial \
  --mode contact --expert contact \
  --variant-preset large_modern --appearance-preset sage_green \
  --hollow-drawer --initial-drawer-open 0.09 \
  --objects cube eraser glue_stick plastic_bottle \
  --target-object plastic_bottle \
  --enable-head-stereo --camera-preset zed_4cam \
  --output outputs/custom_object_demo/bottle_into_sage_001

# 示例 B：双抽屉并排 + 抽屉起始关闭，先拉开蓝灰主抽屉，再抓取红色方块放入抽屉内
uv run --locked python -m scripts.a1_simulation.collect_drawer_trial \
  --mode contact --expert contact \
  --variant-preset standard --appearance-preset classic_wood_blue \
  --drawer-offset-x 0.10 --pull-distance 0.10 \
  --hollow-drawer --initial-drawer-open 0.0 \
  --drawer-layout dual_side_by_side \
  --objects cube eraser glue_stick plastic_bottle \
  --target-object cube \
  --enable-head-stereo --camera-preset all_5cam \
  --output outputs/custom_object_demo/open_and_put_cube_001
```

### 3.3 独立回放校验（无头或 3D 窗口）

```bash
# 无头校验（断言 max_qpos_error == 0.0 且 target_object_inside_drawer == True）
uv run --locked python -m scripts.a1_simulation.replay_drawer_trial \
  --episode outputs/drawer_variants_verify/objects_zed_trials/obj01_cube_into_standard_blue/episode.npz --headless

# 弹出 3D 交互窗口观看抓取放入抽屉全过程
uv run --locked python -m scripts.a1_simulation.replay_drawer_trial \
  --episode outputs/drawer_variants_verify/objects_zed_trials/obj01_cube_into_standard_blue/episode.npz
```
