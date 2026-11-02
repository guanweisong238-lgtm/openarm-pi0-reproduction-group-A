# A组每周进度记录

## 项目信息

- 项目：OpenArm + MuJoCo + π0 / OpenPI
- 执行周期：2026-10-20 至 2026-11-19
- 组长：A组组长
- 更新频率：每周更新，遇到重要阻塞及时记录
- 最近更新：2026-11-02（A2 视角，按实际运行证据填写）

> 本表只勾有证据的项。证据位置写在该项后面；未完成项注明原因与下一步。

## W1：环境跑通（10月20日—10月26日）

### A1：仿真与数据适配
- [x] OpenArm + MuJoCo 环境启动成功（`openarm-a1` 仓库，`simulation/env.py`、`simulation/policy_env.py`）
- [x] 验证关节控制（`scripts/test_control.py` 双臂、双夹爪响应与回位检查）
- [x] 获取相机图像、关节状态和夹爪状态（顶部 + 双腕 320×240 RGB，16 维位置，见 `openarm-a1/docs/a2_model_interface.md`）
- [x] 整理环境启动说明（`openarm-a1/README.md`、`docs/robot_spec.md`）

### A2：π0 模型与训练
- [x] 配置 OpenPI 运行环境（项目 `.venv`，JAX 0.5.3，`configs/a2-requirements.txt`）
- [x] 确认 GPU 型号与可用显存（RTX 5060 Ti 16 GB，见 `scripts/a2_pi0/README.md`）
- [x] 下载并加载预训练模型（`pi0_base` 33 个文件通过大小与 MD5 校验）
- [x] 运行官方推理示例（LoRA 10 步更新 + 新进程重载推理）
- [x] 初步确认公开数据集及其来源（`openarm_pickup` 30 条轨迹已下载并转 LeRobot v2.1；**结论：不采用**，原因见 W2 第一条）

### A3：闭环执行
- [ ] 验证仿真动作执行接口 — 由 A2 在 drawer 任务上完成（见 W3），A3 目录仍为模板
- [ ] 验证执行动作后能够获取新观测 — 同上
- [ ] 明确模型与仿真的连接方案 — `docs/interfaces.md` 已冻结 CELL 接口，drawer 任务接口见评估协议

### W1 验收
- [x] 三个模块均有运行证据 — A1/A2 有；**A3 无**（目录只有模板 README）
- [x] 保存环境配置和启动命令
- [x] 完成至少30秒仿真录像 — `results/drawer_pull_success_20261025/pi0_endpoint_original_start.mp4`

## W2：数据适配（10月27日—11月2日）

- [x] 确认实际使用的公开数据集及版本 — **结论：公开 `openarm_pickup` 不适用**（相机槽位、任务与 OpenArm 抽屉场景不匹配），改用自采数据 `data/processed/openarm_drawer_success_v003`（13 条，10 train / 3 validation）
- [x] 检查机器人类型、任务内容、观测和动作定义（`docs/interfaces.md`、`scripts/a2_pi0/README.md`）
- [x] 完成数据格式适配与校验（13 条示教 + 1200 条反馈监督查询 `data/drawer_feedback_queries/v001`）
- [x] 验证数据可以被 OpenPI 读取（016 训练实际跑通，2000 步）
- [x] 验证示教动作或数据样本能够正确回放（`DrawerSimulation` 自检 PASS，重放误差 < 1e-6；批量评估每次运行都复核）
- [ ] 与B组确认双方采用的共同数据及评测条件 — **不在本月范围**：B 组独立推进，A 组数据格式与协议已可对外，但未与 B 组对齐

## W3：训练与闭环（11月3日—11月9日）

- [x] 完成至少一次 π0 LoRA 微调（015 → 016；LoRA + 三 IO 投影共 50,087,968 参数）
- [x] 保存训练配置、权重和训练日志（`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000`，日志 `logs/a2_pi0/drawer_feedback_20261025-031627-720249/`）
- [x] 验证模型权重可以重新加载（批量评估每次子进程都执行 `LOADING_POLICY` 并校验协议哈希）
- [x] 完成模型预测、动作执行和下一帧观测的闭环（ID12 成功拉开 101.82 mm；ID2 在 10-25 记录中成功 90.78 mm）
- [x] 保存自主执行视频及失败案例（`results/drawer_pull_success_20261025/`；失败证据 `physical_scores.json` 与 `logs/a2_pi0/*`）

## W4：联合验收（11月10日—11月19日）

- [x] 固定最终代码、模型和测试配置（`docs/drawer-eval-protocol.md` 冻结判据与运行条件；`deployment_config.json` 固定部署配置）
- [x] 完成至少30次仿真测试 — 020/2000同一权重完成固定33条隔离评估，33/33通过，原独立评分函数重放33/33通过；见 `results/drawer_eval_020_20261102/`
- [x] 统计任务成功率 — train 10/10、validation 3/3、eval_only 20/20，已产出 `metrics.json` / `metrics.md`
- [x] 统计任务完成时间 — 中位数4.73秒仿真，最慢ID112为15.57秒；推理期暂停，不等于实时
- [x] 统计推理延迟 — 合并1241次实际推理，墙钟p50=138.40ms、p95=139.51ms（RTX 5060 Ti）
- [x] 整理失败案例及原因 — 016全量13/33、017全量0/33以及018/019筛查失败均保留；020/2000本轮33条无失败，修复过程见 `results/drawer_repair_20261101/`
- [x] 完成演示视频和复现报告 — `results/drawer_eval_020_20261102/README.md`；原生ID12实测通过，35帧录像完整解码

## 评估工具链（2026-11-01 新增）

| 工具 | 用途 | 状态 |
| --- | --- | --- |
| `docs/drawer-eval-protocol.md` | 冻结成功判据、运行条件、指标与失败分类 | 已冻结（含 2 条变更记录） |
| `docs/eval-runbook.md` | 操作手册：冒烟、全量、续跑、出表、常见错误 | 已就绪 |
| `scripts/a2_pi0/run_drawer_eval_batch.py` | 无窗口批量闭环（单进程多 trial） | 已验证 |
| `scripts/a2_pi0/run_drawer_eval_isolated.py` | 逐 trial 独立进程（消除批次依赖） | 冒烟通过 |
| `scripts/a2_pi0/build_drawer_eval_starts.py` | 生成 eval-only 扰动起点 | 已跑，产出 20 条 |
| `scripts/a2_pi0/check_drawer_selfcheck.py` | 自检判定（同实现重复一致） | 已跑，ID2 repeat 2 PASS |
| `scripts/a2_pi0/summarize_drawer_eval.py` | 分组出表（train/validation/eval_only） | 已验证格式 |
| `scripts/a2_pi0/compare_render_backends.py` | 两渲染后端像素对比 | 已验证 |

## 每次评估必须记录的口径

1. **渲染后端**：`MUJOCO_GL=glfw`。换成 `egl` 会让成败判定翻转（实测 ID12 成功→失败、ID6 失败→成功），跨批次比较前必须核对 `run_config.json` 的 `gl_backend`。
2. **不要用「复现历史单次数字」做校准标准**：批量无窗口实现与交互式脚本首步动作必然差 6e-4 量级，接触任务下该差异足以翻转结果。校准标准是「同实现重复运行一致」。
3. **结果不得依赖批次构成**：必须用 `run_drawer_eval_isolated.py`，逐 trial 进程隔离。
4. **仿真时间 ≠ 实时性能**：推理期间仿真暂停，延迟另按墙钟 p50/p95 记录。

## 每周会议记录

### 本周完成（10月27日—11月2日）

- 自采抽屉数据定版：13 条成功示教（10 train / 3 validation）+ 1200 条反馈监督查询。
- π0 一步确定性改造版 016 在 ID12 闭环成功（101.82 mm），ID2 在历史记录中成功，ID6 未通过。
- 建成完整评估体系：协议、批量脚本、隔离驱动、自检判定、汇总出表、运行手册。
- 建 20 条 eval-only 扰动起点（柜体 X/Y 偏移、把手粗细、初始姿态），合计 33 条可测起点，全部通过初态加载。
- 评估工具校准过程中修复三个会让数据失效的问题：渲染后端、预热依赖批次、跨 trial 状态污染。

### 当前阻塞

- 原GPU访问阻塞已通过授权宿主执行解决，训练和33条隔离评估均已完成。目标已达成；推荐020/2000并使用checkpoint原始权重精度。
- A3 目录仍为空模板：闭环执行、批量测试、失败记录代码层面未开始（当前由 A2 以 drawer 任务替代承担）。

### 原下周计划（11月3日—11月9日，现已完成本轮验收）

1. 跑满 33 次隔离评估，产出 `metrics.json` / `metrics.md`（分组成功率、峰值开度、首次成功步、仿真耗时、推理延迟、失败模式分布）。
2. 按失败模式分布决定补哪类数据：`single_finger` / `grasp_slip` 优先补夹持与恢复监督；`hold_fail` 优先补保持段。
3. 在补数据基础上重训 017，超参与 016 保持一致（2000 步、lr 1e-5、warmup 100），用**同一批 33 个起点**复测才可比。
4. 写 `results/drawer_eval_<日期>/README.md`，明确"不能宣称什么"。

### 当时记录的协助事项（GPU阻塞已解除）

- 一台可长时间占用 GPU 的机器（约 40 分钟/轮，后续重训 017 另需数小时）。
- A3 侧确认是否接手批量测试与视频记录，避免与 A2 的评估脚本重复建设。

## 实验记录要求

每项已完成任务都应尽量附上代码提交记录、运行日志、测试结果或视频等证据。未完成的任务应注明具体原因和下一步处理方案。

注意：本进度表是组内管理模板。任务完成状态应以实际运行和测试证据为准，不应仅根据代码已经编写来判定完成。

## 2026-11-02 自主修复验收

017训练后全量0/33，未因离线loss下降而判定成功。修复训练/部署KV-cache路径与权重精度加载，补充独立随机场景反馈，并加入旧成功策略保留标签及位置/夹爪加权损失后，020/2000完成固定33条全部通过。实际训练父起点仅原train10个，验证/评估状态未进梯度或专家标签；数据和完整物理证据已审计。

本轮6000步课程保留2000/4000/6000步，第一份2000步候选通过全量后停止后续筛查。此结果是固定开发基准上的33/33，不宣称新独立测试集100%泛化。原多步flow、关闭往返和真机仍未验收。正式证据与运行命令见 `results/drawer_eval_020_20261102/README.md`。
