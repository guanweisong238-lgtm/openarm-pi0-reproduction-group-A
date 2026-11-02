# 抽屉评估运行手册

面向要亲手跑评估的人（A2/A3或组长）。当前推荐020/2000，固定33条与独立重放均已通过；结果见[正式报告](../results/drawer_eval_020_20261102/README.md)。成功判据沿用[原协议](drawer-eval-protocol.md)，新候选精度和运行口径见[修复协议](drawer-repair-eval-protocol.md)。

## 0. 前提检查

| 项 | 要求 | 如何确认 |
| --- | --- | --- |
| GPU | 必须可被 JAX 使用 | `.venv/bin/python -c "import jax; print(jax.default_backend())"` 输出 `gpu` |
| 图形环境 | `glfw` 后端需要 `DISPLAY` | `echo $DISPLAY` 非空；纯 ssh 无 X 的会话要在桌面终端跑 |
| 工作目录 | 项目根目录 | `cd ~/scientific_research/openarm-pi0-reproduction-group-A` |
| 检查点 | `feedback_endpoint_020_randomized_rehearsal/2000` | 运行配置记录协议SHA256；本模型为 `5cf3c5a364fcb9d4978058f7d3e046144bc1db90c3d771dd35481b97feef4cf4` |

**不要**设 `DRAWER_EVAL_GL=egl`。渲染后端是固定实验条件，换后端会改变成败判定（见协议第 9 节）。

### 0.1 无 GPU 沙箱/容器里能做什么（2026-11-02 实测）

在设备节点被隔离的环境（`/dev/nvidia*` 不存在、JAX 回落到 `cpu`）里：

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 跑正式评估 | **不可以** | 软件渲染的像素与宿主机 GPU 不同，成败会翻转，不能作为评估结果 |
| 训练 | **不可以** | `jax.default_backend()` 为 `cpu`，2000 步课程不现实 |
| 只读诊断（几何、遮挡、图像结构） | 可以 | 结论只依赖运动学和相机模型，与渲染后端无关 |
| `glfw` 渲染本身 | 可以，需两个环境变量 | 见下 |

`glfw` 在该环境下的默认失败与修复：

```text
GLFWError: (65543) 'GLX: Failed to create context: BadValue'
→ gladLoadGL error
```

原因不是"没有 X server"（`echo $DISPLAY` 非空、`xdpyinfo` 能连上），而是客户端被指向了 **NVIDIA 的 GLX 库**，而 `/dev/nvidia*` 不在容器内。改用 Mesa 的 GLX 库即可：

```bash
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
glxinfo -B   # 应显示 Vendor: Mesa, Device: llvmpipe, Accelerated: no
```

在此设置下 `MUJOCO_GL=glfw` 与 `MUJOCO_GL=egl` 渲染出的三路观测图像**逐像素完全相同**（2026-11-02 实测，240×320×3，max diff = 0）。这说明沙箱里两条后端路径等价，但**都等价于 llvmpipe 软件渲染，不等于宿主机 GPU 渲染**。因此沙箱内可以检查"模型看到了什么结构"，不能据此判断成败。

`mknod` 补设备节点不可行（无 `CAP_MKNOD`，`Operation not permitted`），该限制只能在宿主侧放开。

## 1. 跑之前：冒烟测试（约 1.5 分钟）

```bash
.venv/bin/python scripts/a2_pi0/run_drawer_eval_isolated.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000 \
  --weight-precision checkpoint \
  --episode-ids 100 101 \
  --output logs/a2_pi0/drawer_eval_isolated_smoke_$(date +%Y%m%d)
```

看到 `ISOLATED_DONE trials=2 success=?` 即表示链路通。若报 `批量评估要求 JAX GPU`，说明环境里 JAX 看不到显卡，先解决环境再往下。

## 2. 正式跑：33 次全量（约 25–60 分钟）

```bash
.venv/bin/python scripts/a2_pi0/run_drawer_eval_isolated.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000 \
  --weight-precision checkpoint \
  --extra-manifest data/processed/openarm_drawer_eval_v001/manifest.json \
  --output logs/a2_pi0/drawer_eval_020_reproduce_001
```

只跑真正的泛化数字（eval_only 20 条，约 15–40 分钟）：

```bash
.venv/bin/python scripts/a2_pi0/run_drawer_eval_isolated.py \
  --checkpoint checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000 \
  --weight-precision checkpoint \
  --episode-ids 100 101 102 103 104 105 106 107 108 109 \
                110 111 112 113 114 115 116 117 118 119 \
  --output logs/a2_pi0/drawer_eval_020_evalonly_reproduce_001
```

**为什么用 `isolated` 而不是 `run_drawer_eval_batch.py` 直接跑**：后者在单进程内连续跑多个 trial，实测会出现跨 trial 状态污染（同一 ID2 在不同批次构成下分别为 20.20 mm 与 -0.000000 mm）。`isolated` 逐 trial 起子进程，代价是每个 trial 多约 20 秒加载。

## 3. 中途断了怎么办

每个 trial 的 `summary.json` 与 `closed_loop.npz` 都在 `<输出目录>/trial_XXX_episode_YYY/` 下，`index.json` 是逐条追加写的。查询已完成的 trial：

```bash
.venv/bin/python -c "import json;print([r['episode_id'] for r in json.load(open('logs/a2_pi0/<输出目录>/index.json'))])"
```

续跑时把未完成的起点用 `--episode-ids` 列出来，输出到**同一目录**是安全的（`--append` 会复用并追加 index）。

## 4. 出结果表

```bash
.venv/bin/python scripts/a2_pi0/summarize_drawer_eval.py \
  --run logs/a2_pi0/drawer_eval_020_reproduce_001
```

产出 `metrics.json`（机器可读）与 `metrics.md`（可直接贴进报告），含：

- **分组成功率**：`train` / `validation` / `eval_only` 三组分开，其中 `eval_only` 是唯一可称"泛化"的数字
- 峰值开度、首次成功步、成功时仿真耗时、推理延迟 p50/p95
- 失败模式分布（7 类，定义见协议第 7 节）
- 逐条明细表

## 5. 结果怎么读（三条硬规矩）

1. **`train` 组成功率不能当成绩报**：那 10 条起点模型训练时见过。
2. **仿真耗时 ≠ 实时性能**：推理期间仿真暂停，表中另列推理墙钟延迟（p50/p95），后者才反映部署可行性。
3. **不要与 10-25 的 live 数字并列比较**：那是交互式脚本的单次观察，与本表的批量实现不是同一参照系（协议第 9 节）。

## 6. 常见错误与处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `批量评估要求 JAX GPU` | 当前环境看不到 GPU | 换到有显卡的终端；检查 `/dev/nvidia*` 是否存在 |
| `FileExistsError` | 输出目录已存在且未加 `--append` | 换一个新目录名 |
| `检查点资产统计与协议版本不一致` | 归一化统计与检查点不匹配 | 不要改统计文件，回到正确的检查点 |
| `libEGL warning` | 走了 EGL 路径 | 确认没有设 `DRAWER_EVAL_GL=egl` |
| `GLX: Failed to create context: BadValue` / `gladLoadGL error` | 客户端用了 NVIDIA 的 GLX 库，但容器内没有 `/dev/nvidia*` | 设 `__GLX_VENDOR_LIBRARY_NAME=mesa` 与 `LIBGL_ALWAYS_SOFTWARE=1`，见 0.1 节；这只是让渲染能跑，不能用于评估 |
| 所有起点都 `no_contact` | 渲染或夹爪链路异常 | 先跑协议第 5 节的独立重放复核，看 `evidence_valid` |

## 7. 跑完之后

1. 把 `metrics.md` 贴进 `results/drawer_eval_<日期>/README.md`，并写清"不能宣称什么"（见协议第 12 节边界）。
2. 按失败模式分布决定下一轮补哪类数据：`grasp_slip` / `single_finger` 优先补夹持与恢复监督；`hold_fail` 优先补保持段。
3. 当前020/2000已经33/33通过，无需为本轮验收继续训练。将来更换模型、数据或精度须新建运行目录并完整记录条件，不覆盖本次结果。
