# OpenArm v2 A1 仿真环境（WSL + uv）

## A2 模型接口适配（2026-10-23）

新增 `OpenArmPolicyEnv`，提供右臂优先的16维状态/动作、顶部和双腕RGB、名义30Hz步进及限幅反馈。
配置、字段、运行方式和验收范围见 [A2模型接口说明](docs/a2_model_interface.md)。
原生 `OpenArmEnv` 的20Hz入口继续可用；调用新包装器时左右臂顺序已经转换，不要在A3重复交换。

```bash
uv run --locked python -m scripts.check_policy_interface --no-render
MUJOCO_GL=egl uv run --locked python -m scripts.check_policy_interface
```

顶部相机为可配置的初始位姿，仍需与公开数据对齐；当前没有完成抓方块场景或成功率评估。

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
├── pyproject.toml              # 项目依赖与 uv workspace 配置
├── uv.lock                     # 精确依赖锁文件（还需固定模型源码和系统依赖）
├── .python-version             # Python 3.11 版本定义
├── .gitignore                  # Git 忽略配置
├── openarm_mujoco/             # OpenArm 官方模型源码与资产 (Submodule)
├── simulation/                 # 仿真环境核心实现
│   ├── __init__.py
│   └── env.py                  # OpenArmEnv 标准环境封装类 (reset, step, get_observation)
├── scripts/                    # 验证与测试脚本
│   ├── __init__.py
│   ├── inspect_robot.py        # 机器人模型结构探测工具
│   ├── test_control.py         # T1-T6 数值检查及可选交互演示
│   ├── test_env.py             # 接口单元测试脚本
│   ├── capture_camera.py       # 双腕相机截图工具
│   ├── record_sample.py        # 5秒时序对齐示教采样工具
│   └── validate_sample.py      # 样本元数据、时序与回放校验
├── docs/                       # 架构与规格文档
│   ├── robot_spec.md           # 16维关节/夹爪/相机详尽规格说明书 (必读!)
│   ├── environment.txt         # 实际运行环境软硬件配置快照
│   └── openarm-commit.txt      # 锁定的 OpenArm 模型 Commit 哈希
├── outputs/                    # 产物输出目录
│   ├── model_info.txt          # 原始模型探测报告
│   ├── camera_wrist_left.png   # 左腕相机样张 (240x320 RGB)
│   ├── camera_wrist_right.png  # 右腕相机样张 (240x320 RGB)
│   └── sample_episode.npz      # 5秒 100 步示教数据样条
└── README.md                  # 复现步骤和接口入口
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



控制脚本在 headless 和 GUI 模式均校验实际响应、回位和复位；测试期间关闭窗口会返回中断状态，不报告全部通过。正式 W2 示教任务、OpenPI 数据适配及团队接口确认仍需另行完成。
