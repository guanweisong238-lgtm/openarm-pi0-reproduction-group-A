# OpenArm v2 A1 仿真环境（WSL + uv）

## 抽屉与桌面物体多形态仿真 + ZED Mini 头部双目相机（2026-11-07，供 A2 泛化与语言指令入抽屉交接）

针对 $\pi_0$ 模型泛化能力提升及最终展示目标（**根据自然语言指令将不同物体放进不同抽屉 + 机器人头部双路深度相机**），统一参数化场景生成器 `simulation/drawer_scene_generator.py`（v2.0.0）与 30Hz 真实接触闭环环境 `simulation/drawer_simulation.py` 支持：
- **5 种抽屉外形预设**（`standard` / `wide_low` / `narrow_tall` / `deep_compact` / `large_modern`）× **5 种材质配色预设**（`classic_wood_blue` / `light_oak_white` / `walnut_slate` / `industrial_steel` / `sage_green`）；
- **真实“空心抽屉槽”（`--hollow-drawer`）与单/双抽屉布局（`--drawer-layout single|dual_side_by_side`）**：将实心块升级为底板+左右侧壁+后壁+前面板围成的真实开口容器，支持单抽屉或双抽屉并排摆放；
- **4 种桌面可抓取物体（`--objects` / `--target-object`）**：**方块（`cube`）、橡皮（`eraser`）、胶棒（`glue_stick`）、空塑料瓶（`plastic_bottle`）**，支持真实接触（零 `weld`）抓取放入抽屉（兼容**抽屉预打开**与**先拉开抽屉再抓物体放入**两种流程）及自然语言指令自动生成；
- **机器人头部 Stereolabs ZED Mini 双目深度相机（`--enable-head-stereo` / `--camera-preset`）**：按 ZED Mini 规格构建头部 3D 双目相机模型与 `camera_head_left` / `camera_head_right` 双路光学镜头（严格 **63.0 mm 水平双目基线**，支持 `legacy_3cam` / `zed_4cam` / `all_5cam` 预设）；
- **柜体三维位置与高度可变**（`--drawer-offset-x/y/z`，调节高度 Z 时底座 `drawer_base` 自动伸缩，确保底面恒贴合桌面 `z = 0.40 m`，不悬空、不穿模）；
- **100% 向后兼容**：默认参数下编译出的模型与改造前逐字节一致，完整参数规格与交接说明见 [抽屉与桌面物体参数化仿真交接文档](docs/drawer_variation_handoff.md)。

```bash
# ① 场景几何与三路相机渲染验证（导出五款抽屉对比图至 outputs/drawer_variants_preview/）
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode scene

# ② 5 款抽屉变体真实接触专家（无 weld）拉取与零误差独立回放验证
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode expert

# ③ 头部 ZED Mini 双目相机（63mm基线）+ 4 种物体（方块/橡皮/胶棒/空塑料瓶）放入不同空心抽屉全流程验证
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode objects_zed

# ④ 使用已就位的模型检查点运行闭环推理验证（需同机配置好检查点与 GPU 环境）
uv run --locked python -m scripts.a1_simulation.verify_drawer_variants --mode policy --seconds 18
```

---

## A2 模型接口适配（2026-10-23）

新增 `OpenArmPolicyEnv`，提供右臂优先的16维状态/动作、顶部和双腕RGB、名义30Hz步进及限幅反馈。
配置、字段、运行方式和验收范围见 [A2模型接口说明](docs/a2_model_interface.md)。
原生 `OpenArmEnv` 的20Hz入口继续可用；调用新包装器时左右臂顺序已经转换，不要在A3重复交换。

```bash
uv run --locked python -m scripts.check_policy_interface --no-render
MUJOCO_GL=egl uv run --locked python -m scripts.check_policy_interface
```

顶部相机为可配置的初始位姿，仍需与公开数据对齐；当前没有完成抓方块场景或成功率评估。

---

## 一、 系统与依赖环境规格

* **操作系统**：Windows 11 + WSL2 (Ubuntu 22.04 LTS)
* **包管理器**：`uv` (Astral)
* **Python 版本**：3.11.x (通过 `.python-version` 锁定)
* **核心依赖库**：
  * `mujoco == 3.16.0`
  * `numpy == 2.4.6`
  * `pillow >= 12.3.0`
  * `openarm_mujoco` (Git Submodule, commit: `a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c`)

---

## 二、 Quick Start

所有命令均在 **WSL2 Ubuntu 终端**中执行：

### 0. 首次获取与系统准备

在 WSL 中准备 Linux 版 uv（参阅 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)），并安装 Git 与图形运行库：

```bash
sudo apt update
sudo apt install -y git curl libgl1 libegl1 libglfw3
mkdir -p ~/projects
cd ~/projects
git clone --branch a1-simulation --recurse-submodules https://github.com/guanweisong238-lgtm/openarm-pi0-reproduction-group-A.git openarm-a1
```

已有工程目录时跳过克隆，勿覆盖原目录。相机使用当前已验证的 WSLg/GLFW 后端；不要在未安装对应运行库时强制设置 OSMesa。Python 与依赖由 uv 在各自机器上恢复，虚拟环境不提交。

### 1. 环境同步与依赖恢复
拉取仓库后，进入工程根目录，直接使用锁文件一键同步虚拟环境：
```bash
cd ~/projects/openarm-a1

# 首次克隆使用 --branch a1-simulation --recurse-submodules
# 已克隆的仓库先恢复固定版本的模型子模块
git submodule update --init --recursive

# 检查/同步环境
uv sync --locked
```

### 2. 核心功能验证与交互演示
依次运行以下脚本，验证无窗口环境接口、相机渲染与控制响应：

```bash
# ① 验证环境 Step/Reset 基础接口 (5秒/100步无头测试)
uv run --locked python -m scripts.test_env

# ② 验证左右手腕嵌入式相机拍照 (输出至 outputs/)
uv run --locked python -m scripts.capture_camera

# ③ 无窗口数值校验：左右臂、双臂、双夹爪响应、回位和复位
uv run --locked python -m scripts.test_control --headless

# 可选交互式演示：仅在所有响应检查通过后报告成功
uv run --locked python -m scripts.test_control

# ④ 生成 5 秒状态／动作／双相机样本
uv run --locked python -m scripts.record_sample

# ⑤ 校验样本字段、限幅、时间对齐及完整回放
uv run --locked python -m scripts.validate_sample
```

### 3. 打开官方交互式查看器 (Viewer)
```bash
uv run --locked openarm-mujoco-launch openarm_mujoco/v2/pedestal/pedestal.xml --keyframe home
```

---

## 三、 工程目录结构

```text
openarm-a1/
├── pyproject.toml                  # 项目依赖与 uv workspace 配置
├── uv.lock                         # 精确依赖锁文件（还需固定模型源码和系统依赖）
├── .python-version                 # Python 3.11 版本定义
├── .gitignore                      # Git 忽略配置
├── openarm_mujoco/                 # OpenArm 官方模型源码与资产 (Submodule)
├── configs/                        # 环境与多变体采样配置
│   ├── a2_interface.json           # A2 30Hz 策略接口与相机参数配置
│   ├── a2_dataset_render.json      # 数据集重渲染相机配置
│   ├── drawer_variants_preset.json # 5 套代表性抽屉外形/配色/高度变体配置清单
│   └── drawer_objects_zed_preset.json # 4 种桌面物体入空心抽屉 + ZED Mini 头部双目配置清单
├── simulation/                     # 仿真环境核心实现
│   ├── __init__.py
│   ├── env.py                      # OpenArmEnv 标准 20Hz 环境封装类
│   ├── policy_env.py               # OpenArmPolicyEnv 右臂优先 30Hz 策略包装环境
│   ├── drawer_scene_generator.py   # 抽屉多外形、多配色、尺寸缩放与三维位姿统一生成器
│   └── drawer_simulation.py        # DrawerSimulation 30Hz 抽屉真实接触闭环环境
├── scripts/                        # 验证、采集与回放脚本
│   ├── __init__.py
│   ├── inspect_robot.py            # 机器人模型结构探测工具
│   ├── test_control.py             # T1-T6 数值检查及可选交互演示
│   ├── test_env.py                 # 接口单元测试脚本
│   ├── capture_camera.py           # 双腕相机截图工具
│   ├── record_sample.py            # 5秒时序对齐示教采样工具
│   ├── validate_sample.py          # 样本元数据、时序与回放校验
│   ├── check_policy_interface.py   # A2 策略包装接口数值与渲染验收脚本
│   └── a1_simulation/              # 抽屉泛化采集、回放与模型闭环脚本
│       ├── contact_drawer_controller.py # 禁用 weld 的真实双指接触专家控制器
│       ├── collect_drawer_trial.py      # 参数化抽屉变体示教采集入口
│       ├── replay_drawer_trial.py       # 抽屉变体示教无头校验与 3D 窗口回放
│       ├── verify_drawer_variants.py    # 抽屉变体三层递进式一键可控验证脚本
│       ├── replay_episode.py            # LeRobot parquet 公开示教重渲染与回放
│       ├── resident_policy_client.py    # 常驻策略进程通信客户端
│       ├── run_model_closed_loop.py     # CELL 场景模型闭环推理运行脚本
│       └── view_model_chunk.py          # 单块模型动作预测对比查看脚本
├── docs/                           # 架构与交接规格文档
│   ├── drawer_variation_handoff.md # 抽屉多形态与多高度参数化仿真交接文档 (A1->A2)
│   ├── a2_model_interface.md       # A2 模型与公开数据接口适配说明
│   ├── robot_spec.md               # 16维关节/夹爪/相机详尽规格说明书 (必读!)
│   ├── environment.txt             # 实际运行环境软硬件配置快照
│   └── openarm-commit.txt          # 锁定的 OpenArm 模型 Commit 哈希
├── outputs/                        # 基础验收产物目录（大型衍生验证产物已 Git 忽略）
│   ├── model_info.txt              # 原始模型探测报告
│   ├── camera_wrist_left.png       # 左腕相机样张 (240x320 RGB)
│   ├── camera_wrist_right.png      # 右腕相机样张 (240x320 RGB)
│   └── sample_episode.npz          # 5秒 100 步示教数据样条
└── README.md                       # 复现步骤和接口入口
```

---

## 四、 API 使用方式 (`OpenArmEnv`)


```python
from simulation.env import OpenArmEnv

# 实例化环境 (控制周期默认为 20Hz / 0.05s，内置 50 次物理 sub-stepping)
env = OpenArmEnv(control_dt=0.05)

# 1. 复位
obs = env.reset()

# 2. 发送 16 维动作 (左右臂各 7 轴 + 左右各 1 夹爪)
action = obs["ctrl"].copy()
action[3] += 0.1  # 左肘微弯 0.1 弧度

# 3. 单步推进
obs = env.step(action)

# 4. 获取观测
# 包含: obs["joint_pos"] (16维), obs["gripper_pos"] (2维), obs["timestamp"]
```

更详细的各电机角度范围、夹爪正负号约定与相机参数，请参阅 [`docs/robot_spec.md`](docs/robot_spec.md)。

---

## 五、 团队协作与分支规范

* 本工程为 A 组团队仓库：[openarm-pi0-reproduction-group-A](https://github.com/guanweisong238-lgtm/openarm-pi0-reproduction-group-A)
* A1 交付分支为：**`a1-simulation`**


## 六、验收说明

基础验收包括 v2 模型加载、双臂与夹爪控制、状态接口、双腕相机，以及具有终端状态和元数据的 5 秒采样。`step(action)` 返回一个观测字典，不是 Gymnasium 四元组。`get_observation()` 返回状态数据，RGB 图像由采集脚本渲染。

控制脚本在 headless 和 GUI 模式均校验实际响应、回位和复位；测试期间关闭窗口会返回中断状态，不报告全部通过。抽屉多形态与多高度参数化仿真已通过场景几何、真实接触专家回放及模型闭环三级验证，详见 [`docs/drawer_variation_handoff.md`](docs/drawer_variation_handoff.md)。
