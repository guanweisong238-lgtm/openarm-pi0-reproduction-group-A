# Week 1 本地验证记录

验证日期：2026-10-23（WSL Ubuntu 22.04）。模型为 OpenArm v2 pedestal，源码 commit `a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c`。

## 已通过

- 在独立副本中运行 `uv sync --locked`，从同一锁文件建立新的 Python 3.11.17 环境。
- 模型结构：nq=18、nv=18、nu=16、相机数 2，物理步长 0.001 s。
- `python -m scripts.test_env`：100 个 0.05 s 控制周期得到 5 s 仿真，状态有限。
- `python -m scripts.test_control --headless`：左右臂、双臂同时运动、左右夹爪实际响应、回位、复位均通过。手臂目标变化 0.15 rad 时实际响应约 0.15012 rad；夹爪目标 ±0.35 rad 时响应约 ±0.34984 rad。
- 独立检查错误动作形状、NaN、Inf 拒绝及控制范围限幅；复位与观测副本有效。
- 在独立环境中运行左右腕相机采集与 100 步采样，并运行可提交的 `scripts.validate_sample`。
- 当前工程样本回放最大状态误差约 `1.24e-7 rad`，末帧误差约 `5.73e-8 rad`；时间、形状、元数据、动作限幅及两个 RGB 流条目数通过。
- 提前关闭 viewer 时中断检查，不打印全部通过；README Python API 示例可运行。

## 范围

交互 viewer 的窗口操作本次没有完整重跑；动作与中断逻辑通过无窗口检查。相机使用 WSLg/GLFW。录像不在本轮范围内。正式示教任务、OpenPI/LeRobot 转换、团队接口确认和另一台机器的独立复现属于后续工作，不能由本地数值检查代替。

复核命令与输入字段见 README、`scripts/test_control.py`、`scripts/validate_sample.py`。开发机的详细过程日志保存在未提交的 `codex/`，本记录可独立阅读。
