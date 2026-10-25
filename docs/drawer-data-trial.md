# OpenArm 拉抽屉数据试采

日期：2026-10-24。目标是检查公开控制代码能否生成当前 π0 输入协议可读取的示教。

## 来源与运行环境

- 源码：[Manas-arumalla/openarm-control](https://github.com/Manas-arumalla/openarm-control)。
- 固定提交：`14ea59faad6706b9add5d7f8d386e43239a6360f`，Apache-2.0。
- 本地源码：`.cache/openarm-control/`；不改动或复制上游文件到项目代码。
- 实测运行时：A2 `.venv`，MuJoCo 3.8.1。
- 原生拉抽屉演示实测报告 `ok=True`，拉出约 82 mm。源码中的 `imitation.collect`
  只支持 reach/insert，不能直接用它导出 drawer。

## 试采方法

项目入口：`scripts/a2_pi0/collect_drawer_trial.py`。

复用上游 `ArticulatedController.open_drawer` 的 IK 和路径规划，独立覆盖执行循环：

- 30 Hz 下发并保持位置目标，2 ms 物理积分，按绝对控制时刻交替推进 16/17 子步。
- 每一帧记录执行前的实际状态及同一时刻的 RGB，标签为随后真正执行的限幅位置目标。
- 16 维顺序为右臂 7 关节、右夹爪、左臂 7 关节、左夹爪，单位为弧度。
- 顶部视角沿用上游 tablecam 并别名为 camera_ceiling。
  `--wrist-view source` 保留上游双腕相机，但实测大部分视野为地面或遮挡物；
  `--wrist-view side` 在腕部固定局部位置 `[0.07, 0.10, -0.04]`，
  朝向局部点 `[0, 0, -0.14]`，视场角85°。右腕预览可看到抽屉及把手。
  这是仿真中的虚拟安装位姿，未声称与真实相机标定一致，也不逐帧追踪物体真值。
- 降低阴影图到1024以加速离屏渲染，不改变物理参数。相机局部位姿保存到报告。
- RGB 为 320×240。LeRobot 的名义时间戳为 `index/30`，NPZ 额外保留真实物理时间，
  两者误差不超过 1 ms。未来动作尾部由读取器在本轨迹末帧填充。
- 保留上游 14 个机械臂关节的重力补偿，每个物理子步重算。
- 初始化沿用 ready，左臂由上游 `_park` 在开始记录前直接置零；这不是从 CELL home 出发的轨迹。
- `assisted` 保留上游抓取绑定；`contact` 禁用绑定。绑定事件不进入模型输入或16维动作。

独立回放只载入起始物理状态，此后执行保存的动作、绑定事件和重力补偿，不注入中间状态。
因此回放验证证明相同环境机制下的可复现性，不能证明16维动作在无绑定场景也能成功。

## 结果与适用范围

| 模式 | 帧数 | 时长 | 最远拉出 | 结束拉出 | ≥8 cm 保持1秒 | 独立回放最大 qpos 误差 |
|---|---:|---:|---:|---:|---|---:|
| assisted，30 Hz | 447 | 14.9 s | 102.6 mm | 68.8 mm | 未通过 | 0 |
| contact，30 Hz | 402 | 13.4 s | 2.74 mm | 约0 mm | 未通过 | 0 |

最远位移略超过100 mm名义限位，是 MuJoCo 软约束的实际测量值。以上试采均不满足此前
提出的“8 cm 并保持1秒”标准。不能仅凭上游返回 `True` 将轨迹标成成功。

这批数据用于数据接口与辅助抓取实验的诊断，不作为当前 CELL 场景已验收的成功训练集。
它与当前场景的基座、桌面、物体、相机不同；后续需要把控制与采集流程适配到选定训练场景，
修正真实接触抓取及退出动作，再采集多个初始条件并按轨迹划分训练/验证集。

首版三路RGB共447帧已全部通过 LeRobot v2.1 读取及现有 `OpenArmInputs` 检查，
每一帧的16维状态、16步未来动作及末端填充均与原始记录比对通过。
侧向腕部视角版的447帧也全部通过相同检查，独立回放误差仍为0，
诊断视频447帧、30 fps检查通过。这证明数据格式兼容，不证明任务成功。
侧向腕部视角的数据另存，不覆盖首版。

## 复现命令

在本项目根目录执行。脚本拒绝覆盖已有输出，重复试验需换新目录。

```bash
git clone https://github.com/Manas-arumalla/openarm-control.git .cache/openarm-control
git -C .cache/openarm-control checkout 14ea59faad6706b9add5d7f8d386e43239a6360f
uv pip install --python .venv/bin/python --target .cache/drawer-deps --no-deps -r configs/drawer-trial-extras.txt

# 原生演示检查，不写数据
PYTHONPATH=.cache/drawer-deps:.cache/openarm-control .venv/bin/python -m openarm_control.demos.demo_articulated --task drawer --headless

# 两种模式分别保存数值轨迹并独立回放
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode assisted --no-images --output data/drawer_trial/assisted_numeric_001
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode contact --no-images --output data/drawer_trial/contact_numeric_001

# 三路RGB、诊断视频、LeRobot v2.1写入与全部帧读取检查
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode assisted --export-lerobot --output data/drawer_trial/assisted_rgb_001

# 改进腕部视野，单独保存
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode assisted --wrist-view side --export-lerobot --output data/drawer_trial/assisted_rgb_side_001
```

`episode.npz` 保留原始图像、状态、动作、物理诊断及回放数据；`report.json` 记录模式、
来源提交、场景哈希和实测指标；`lerobot_v21/` 用于格式检查。单轨迹只标记 `train: 0:1`，
不伪造验证集，也不替换现有抓方块配置、归一化统计或检查点。

汇总实测证据：`results/a2_drawer_data_trial_20261024.json`。
三视角视频：`data/drawer_trial/assisted_rgb_side_001/preview.mp4`。

## 在仿真窗口中执行示教

```bash
# 先无窗口独立检查，再打开实时循环回放窗口
.venv/bin/python scripts/a2_pi0/replay_drawer_trial.py --headless
.venv/bin/python scripts/a2_pi0/replay_drawer_trial.py
```

默认窗口现已切换到下述无绑定往返轨迹（462帧、15.4秒）。窗口显示上游 pedestal 场景，
执行已保存的动作，不运行专家重新规划，也不注入中间关节状态。
末帧停留2秒后复位再播放。空格暂停，R重新开始，
关闭窗口退出。每轮完成后的独立执行误差及抽屉位移写入 `logs/a2_pi0/drawer_replay_*/report.json`。
旧版辅助绑定轨迹需显式传入 `--episode data/drawer_trial/assisted_rgb_side_001/episode.npz`。

## 用户指出“粘着拉动”后的纠正

此前辅助绑定轨迹没有真实夹住把手，不作为抓取成功证据。已停止该循环回放。

新控制器 `contact_drawer_controller.py` 使用原版14 mm直径把手，保持原碰撞体、摩擦、
滑轨和夹爪限位；最终试验没有加粗把手。试验中尝试的24 mm把手均另存且未被选用。
修正包括：将实际夹持点设为腕部局部 `[0,0,-0.16]`，IK优先保证夹持点位置，
加入基座与肘部6 mm避碰余量；接近时半开夹爪，缓慢闭合，并在双侧法向力均超过1 N时
停止继续挤压。原先一直命令闭合到0会将细把手挤向指尖边缘，导致滑脱。

`contact_force_stop_001` 实测：

- 无活动抓取weld，无施加在抽屉上的外力，只补偿14个机械臂关节的重力。
- 张开到位时夹持点误差0.261 mm，夹持后0.447 mm。
- 共339帧、11.3秒，结束拉出35.3 mm；独立回放误差0。
- 从夹持完成到拉动、保持结束的150帧全部检测到双侧把手接触力。
- 最后1秒两侧最小法向力分别为4.68 N和14.78 N。
- 更长的规划在36.7 mm目标处因位置误差2.72 mm超过2 mm门限而停止，不能宣称达到8 cm目标。

复现真实接触试采：

```bash
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode contact --expert contact --pull-distance 0.05 --no-images --output data/drawer_trial/contact_force_stop_001
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode contact --expert contact --pull-distance 0.05 --wrist-view side --export-lerobot --output data/drawer_trial/contact_force_stop_rgb_001
.venv/bin/python scripts/a2_pi0/replay_drawer_trial.py --episode data/drawer_trial/contact_force_stop_001/episode.npz
```

本次通过的是“原把手的无绑定双侧接触短距离夹持与拉动”诊断，尚未通过8 cm完整任务。

## 扩大行程及推回抽屉

原位置的抽屉在向基座拉动时受姿态、关节限位与肘部避碰限制。试验将柜体沿世界X轴
向前移动100 mm（柜体中心X由0.40变为0.50 m），原14 mm把手、摩擦、夹爪、滑轨行程
和机器人模型不变。该布局变化显式保存在 `scene_overrides.drawer_offset_x_m` 中，回放时恢复。

- `contact_reachable_002`：342帧、11.4秒，峰值93.1 mm，结束86.3 mm，
  最后1秒始终超过80 mm，无抓取weld，独立回放误差0。
- `contact_roundtrip_001`：462帧、15.4秒；夹持、拉出、保持、保持夹持沿同一关节分支推回、
  关闭保持。峰值93.1 mm，结束关闭偏差0.478 mm；打开≥80 mm保持1秒和
  关闭误差≤5 mm保持1秒均通过，独立回放误差0。
- 往返轨迹的拉动、推回及保持270帧中234帧检测到双侧接触；并非每一帧都维持双侧接触。
  全部帧的抓取weld均关闭，抽屉没有施加外力。末端完成任务时仍保持夹持目标，没有安排松手退开。

```bash
# 长行程单向拉出：数据和视频均另存
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode contact --expert contact --drawer-offset-x 0.10 --pull-distance 0.10 --no-images --output data/drawer_trial/contact_reachable_002
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode contact --expert contact --drawer-offset-x 0.10 --pull-distance 0.10 --wrist-view side --export-lerobot --output data/drawer_trial/contact_reachable_rgb_001

# 拉出后推回：用完整往返指令保存图像、动作和未来动作序列
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode contact --expert contact --drawer-offset-x 0.10 --pull-distance 0.10 --return-drawer --no-images --output data/drawer_trial/contact_roundtrip_001
.venv/bin/python scripts/a2_pi0/collect_drawer_trial.py --mode contact --expert contact --drawer-offset-x 0.10 --pull-distance 0.10 --return-drawer --wrist-view side --export-lerobot --output data/drawer_trial/contact_roundtrip_rgb_001

# 默认打开完整往返示教
.venv/bin/python scripts/a2_pi0/replay_drawer_trial.py
```

往返任务指令为 `Open the drawer, then close it.`，成功指标是轨迹中间连续1秒打开8 cm，
最后连续1秒关闭误差小于等于5 mm。`success_8cm_held_1s` 保留“最终打开”的旧语义，
往返应读取 `opened_8cm_held_1s_during_episode` 和 `success_open_then_close`。
单条成功示教尚无独立验证集；多次循环回放是同一轨迹的执行核验，不是泛化成功率。

`contact_roundtrip_rgb_001/lerobot_v21` 已导出并逐帧验证全部462帧，任务文本、16维状态、
16步未来动作及轨迹末端填充均通过。RGB版物理状态和动作与数值版逐元素一致；
三视角视频462帧、30 fps检查通过。汇总证据见 `results/a2_drawer_contact_cycle_20261024.json`。
