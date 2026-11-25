#!/usr/bin/env bash
# 把 Task C 累计 6,500 步模型打包成单个自包含压缩包。
# 用法： bash scripts/package_model_bundle.sh [输出目录，默认 <root>/exports]
# 可选环境变量：INCLUDE_BASE=0 只打适配器，不打 12 GB 基座参数。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/exports}"
NAME="pi0_taskC_step6500_20261123"
LORA_RUN="real_camera_adjusted_lora_continue5000_20261122_v002"
LORA_STEP="step_3000"
BASE_PARAMS="weights/openpi/openpi-assets/checkpoints/pi0_base/params"
TRAIN_DATA="real_camera_adjusted_candidate_v001"   # 本轮训练实际使用的数据候选
DEFAULT_DATA="real_training_candidate_v001"        # real_four_camera_loader.DEFAULT
INCLUDE_BASE="${INCLUDE_BASE:-1}"

command -v zstd >/dev/null || { echo "缺少 zstd，请先安装" >&2; exit 1; }
mkdir -p "$OUT_DIR"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pi0bundle.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
S="$WORK/$NAME"
mkdir -p "$S"/{weights/openpi,checkpoints/$LORA_RUN/lora,results/$LORA_RUN,data/processed,scripts,third_party,env,tools}
mkdir -p "$S/data/processed/$TRAIN_DATA" "$S/data/processed/$DEFAULT_DATA"

echo "[1/6] 建立骨架与符号链接"
mkdir -p "$S/weights/openpi/openpi-assets/checkpoints/pi0_base"
if [ "$INCLUDE_BASE" = "1" ]; then
  ln -s "$ROOT/$BASE_PARAMS" "$S/$BASE_PARAMS"
fi
ln -s "$ROOT/weights/openpi/big_vision" "$S/weights/openpi/big_vision"
cp -a "$ROOT/checkpoints/$LORA_RUN/$LORA_STEP" "$S/checkpoints/$LORA_RUN/lora/$LORA_STEP"

echo "[2/6] 复制运行元数据与代码"
for f in config.json norm_stats.json manifest.json summary.json review_decision.json validation.jsonl run.log metrics.jsonl; do
  [ -f "$ROOT/results/$LORA_RUN/$f" ] && cp -a "$ROOT/results/$LORA_RUN/$f" "$S/results/$LORA_RUN/$f"
done
cp -a "$ROOT/results/$LORA_RUN/source_snapshot" "$S/results/$LORA_RUN/source_snapshot"
for d in "$TRAIN_DATA" "$DEFAULT_DATA"; do
  for f in norm_stats.json manifest.json review_decision.json README.md; do
    [ -f "$ROOT/data/processed/$d/$f" ] && cp -a "$ROOT/data/processed/$d/$f" "$S/data/processed/$d/$f"
  done
done
tar -C "$ROOT/scripts" -cf - --exclude='__pycache__' --exclude='*.pyc' a2_pi0 | tar -C "$S/scripts" -xf -
tar -C "$ROOT/.venv/lib/python3.11/site-packages" -cf - --exclude='__pycache__' --exclude='*.pyc' openpi | tar -C "$S/third_party" -xf -

echo "[3/6] 写环境清单"
"$ROOT/.venv/bin/python" - "$S/env/versions.txt" <<'PY'
import importlib.metadata as md, platform, sys
pkgs = ["openpi", "jax", "jaxlib", "flax", "numpy", "orbax-checkpoint", "optax", "torch", "transformers", "pillow"]
lines = [f"python {platform.python_version()}", f"platform {platform.platform()}"]
for n in pkgs:
    try:
        lines.append(f"{n} {md.version(n)}")
    except Exception:
        lines.append(f"{n} not-installed")
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
PY

echo "[4/6] 写说明文档与工具"
cat > "$S/tools/verify_bundle.py" <<'PY'
#!/usr/bin/env python3
"""核对包内一致性：检查点哈希、训练统计量一致性、LoRA 键名与形状。不加载 JAX。"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

m = json.loads((root / "manifest.json").read_text())
ck = root / m["checkpoint_dir_in_bundle"]
meta = json.loads((ck / "checkpoint.json").read_text())
train_stats = root / m["training_dataset"] / "norm_stats.json"
default_stats = root / "data/processed" / m["loader_default_dataset"] / "norm_stats.json"
problems, warnings = [], []

if sha(ck / "lora_params.npz") != meta["lora_sha256"]:
    problems.append("lora_params.npz 与 checkpoint.json 记录的 sha256 不一致")
if sha(ck / "norm_stats.json") != meta["normalization_sha256"]:
    problems.append("norm_stats.json 与 checkpoint.json 记录的 sha256 不一致")
if sha(train_stats) != sha(ck / "norm_stats.json"):
    problems.append("训练数据候选的 norm_stats 与检查点内副本不一致")
if sha(default_stats) != sha(ck / "norm_stats.json"):
    warnings.append("real_four_camera_loader.DEFAULT 指向的旧数据集统计量与本次训练不同，"
                    "直接跑 policy_server.py 会被其哈希校验拒绝；请用 tools/serve_step6500.sh 启动")

with np.load(ck / "lora_params.npz", allow_pickle=False) as f:
    keys = sorted(f.files)
    params = sum(int(np.prod(f[k].shape)) for k in keys)
    dtypes = sorted({str(f[k].dtype) for k in keys})
print(f"累计步数: {m['cumulative_steps']}  适配器: {m['checkpoint_dir_in_bundle']}")
print(f"LoRA 张量: {len(keys)} 个，参数 {params:,}，dtype {dtypes}")
print(f"基座参数: {'已包含' if m['base_weights_included'] else '未包含'}"
      f"（{m.get('weights_bytes', 0)/2**30:.1f} GiB，{m['file_count']} 个文件）")
for w in warnings:
    print("注意:", w)
if problems:
    print("发现问题：")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("一致性检查通过（未做 JAX 重载与数值复算）")
PY

cat > "$S/tools/serve_step6500.sh" <<EOF
#!/usr/bin/env bash
# 启动 6,500 步模型的推理服务；把 loader 默认数据集指向本轮训练实际使用的候选，
# 否则 policy_server 内的统计量哈希校验会拒绝加载。参数原样透传给 policy_server.py。
set -euo pipefail
cd "\$(dirname "\${BASH_SOURCE[0]}")/.."
CKPT="checkpoints/$LORA_RUN/lora/$LORA_STEP"
RUN="results/$LORA_RUN"
EXTRA=()
case " \$* " in *" --ckpt "*) ;; *) EXTRA+=(--ckpt "\$CKPT");; esac
case " \$* " in *" --run-dir "*) ;; *) EXTRA+=(--run-dir "\$RUN");; esac
exec "\${PYTHON:-python}" - "\${EXTRA[@]}" "\$@" <<'PY'
import pathlib, runpy, sys
root = pathlib.Path.cwd()
sys.path.insert(0, str(root / "scripts/a2_pi0"))
import real_four_camera_loader as loader
loader.DEFAULT = root / "data/processed/$TRAIN_DATA"
sys.argv = ["policy_server.py", *sys.argv[1:]]
runpy.run_path(str(root / "scripts/a2_pi0/deploy/policy_server.py"), run_name="__main__")
PY
EOF
chmod +x "$S/tools/serve_step6500.sh"

LORA_SHA="$(python3 -c "import json;print(json.load(open('$S/checkpoints/$LORA_RUN/lora/$LORA_STEP/checkpoint.json'))['lora_sha256'])")"
cat > "$S/README.md" <<EOF
# Task C 累计 6,500 步模型包

自包含离线包：pi0_base 基座参数 + 累计 6,500 步的四相机 LoRA 适配器 + 归一化统计 + 训练/推理代码 + 运行元数据。
解压目录即项目根，脚本用 \`__file__\` 推算路径，无需改代码。

## 版本

| 项 | 值 |
| --- | --- |
| 项目 | openarm-pi0-reproduction-group-A / Task C（四相机，拉抽屉+抓方块入屉） |
| 基座 | pi0_base，\`weights/openpi/openpi-assets/checkpoints/pi0_base/params\` |
| 适配器 | \`$LORA_RUN/$LORA_STEP\`，本轮追加 3,000 步 → 累计 6,500 步 |
| 选择依据 | 固定 20 个验证窗口 flow loss 最低（0.057397） |
| 适配器 SHA-256 | \`$LORA_SHA\` |
| 训练数据候选 | \`data/processed/$TRAIN_DATA\`（本轮 run 实际使用） |
| 打包时间 | $(date '+%Y-%m-%d %H:%M %Z') |

## 目录

\`\`\`
$NAME/
├── weights/openpi/openpi-assets/checkpoints/pi0_base/params/   # 基座参数（约 12 GB）
├── weights/openpi/big_vision/paligemma_tokenizer.model         # 分词器
├── checkpoints/$LORA_RUN/lora/$LORA_STEP/                      # LoRA + norm_stats + checkpoint.json
├── results/$LORA_RUN/                                          # config.json / 训练日志 / 验证损失 / 源快照
├── data/processed/$TRAIN_DATA/                                 # 训练用数据集的元数据（3 个文件）
├── data/processed/$DEFAULT_DATA/                               # 旧数据集元数据，仅供对照
├── scripts/a2_pi0/                                             # 训练与推理代码（含 deploy/）
├── third_party/openpi/                                         # 本机安装的 openpi 0.1.0 源码
├── env/versions.txt                                            # 环境与依赖版本
├── manifest.json / FILES.json / SHA256SUMS.txt
└── tools/verify_bundle.py, tools/serve_step6500.sh
\`\`\`

## 校验

\`\`\`bash
cd $NAME
sha256sum -c SHA256SUMS.txt | tail -3      # 需读取 12 GB，约 1 分钟
python3 tools/verify_bundle.py .           # 复核 LoRA 哈希、键名/形状、统计量一致性
\`\`\`

## 加载

依赖与打包机器一致：Python 3.11、jax/jaxlib 0.5.3、flax 0.10.2、numpy 1.26.4、orbax-checkpoint 0.11.13（见 \`env/versions.txt\`）。

方式一，起推理服务（推荐用包内启动脚本）：

\`\`\`bash
bash tools/serve_step6500.sh --self-test
bash tools/serve_step6500.sh --serve --bind 0.0.0.0 --port 5555
\`\`\`

**注意**：\`policy_server.py\` 会把统计量与 \`real_four_camera_loader.DEFAULT\`（旧数据集 \`$DEFAULT_DATA\`）比对，而本轮训练用的是 \`$TRAIN_DATA\`，两者 \`norm_stats.json\` 数值不同（68 个数值中 64 个不同）。因此直接用 \`policy_server.py\` 启动会在哈希校验处被拒；\`tools/serve_step6500.sh\` 已把 \`loader.DEFAULT\` 指向训练用候选。自定义启动时需做同样处理。
协议见 \`scripts/a2_pi0/deploy/README.md\`。

方式二，在自己的代码里加载适配器（与 \`verify_real_checkpoint.py\` 同序）：

\`\`\`python
import dataclasses, sys, jax, numpy as np, flax.nnx as nnx
from flax import traverse_util
sys.path.insert(0, "scripts/a2_pi0")
from real_training_support import get_train_config, load_parameters
from real_four_camera_config import FourCameraPi0Config

ck = "checkpoints/$LORA_RUN/lora/$LORA_STEP"
cfg = dataclasses.replace(get_train_config(),
    model=FourCameraPi0Config(paligemma_variant="gemma_2b_lora",
                              action_expert_variant="gemma_300m_lora",
                              action_horizon=50), seed=42)
params = load_parameters(cfg)                      # 先装基座
abstract = nnx.eval_shape(cfg.model.create, jax.random.key(cfg.seed))
graph = nnx.graphdef(abstract)
model = nnx.merge(graph, params)
state = nnx.state(model).filter(cfg.trainable_filter)
expected = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
with np.load(f"{ck}/lora_params.npz", allow_pickle=False) as f:
    assert set(f.files) == set(expected)
    for k, v in expected.items():
        assert f[k].shape == v.shape and f[k].dtype == v.dtype
    flat = {k: f[k] for k in f.files}
state.replace_by_pure_dict(traverse_util.unflatten_dict(flat, sep="/"))
nnx.update(model, state)                           # 得到完整可推理模型
\`\`\`

## 已知限制

- 只含 LoRA 适配器与基座，**没有优化器状态，不能续训**（checkpoint.json 已注明）。
- 基座 \`pi0.zip\`（11 GB 原始下载包）未纳入；参数目录完整，加载不需要它。
- 完整数据集（约 21 万张 jpeg）未纳入，只放 manifest/norm_stats/review_decision 元数据；跑回放评测需另配 \`data/processed/$TRAIN_DATA\` 的完整内容。
- \`assets/\`（MuJoCo 仿真场景）不属于模型，未纳入。
- 该检查点只验证过录制观测驱动的目标回放（轨迹 5、38），方块接触与闭环性能尚未测试，交接处仍有速度尖峰。
EOF

echo "[5/6] 生成 SHA256SUMS 与清单（读取 12 GB，需要一会儿）"
cd "$S"
find -L . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
"$ROOT/.venv/bin/python" - "$S" "$NAME" "$LORA_RUN" "$LORA_STEP" "$INCLUDE_BASE" "$TRAIN_DATA" "$DEFAULT_DATA" <<'PY'
import json, os, pathlib, sys, time
stage, name, run, step, include_base, train_data, default_data = sys.argv[1:8]
root = pathlib.Path(stage)

files, weights_bytes = [], 0
for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
    dirnames.sort()
    for fn in sorted(filenames):
        p = pathlib.Path(dirpath) / fn
        rel = p.relative_to(root).as_posix()
        if rel in ("SHA256SUMS.txt", "manifest.json", "FILES.json"):
            continue
        size = p.stat().st_size
        if rel.startswith("weights/openpi/"):
            weights_bytes += size
        files.append({"path": rel, "bytes": size})

ck = json.loads((root / "checkpoints" / run / "lora" / step / "checkpoint.json").read_text())
run_cfg = json.loads((root / "results" / run / "config.json").read_text())
manifest = {
    "bundle": name,
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_project": "openarm-pi0-reproduction-group-A",
    "cumulative_steps": 6500,
    "this_round_steps": 3000,
    "run": run,
    "checkpoint_dir_in_bundle": f"checkpoints/{run}/lora/{step}",
    "lora_sha256": ck["lora_sha256"],
    "normalization_sha256": ck["normalization_sha256"],
    "training_dataset": f"data/processed/{train_data}",
    "loader_default_dataset": default_data,
    "training_dataset_declared": run_cfg.get("candidate"),
    "base_checkpoint_declared": ck["base_checkpoint"],
    "base_weights_included": include_base == "1",
    "weights_bytes": weights_bytes,
    "file_count": len(files),
    "total_bytes": sum(f["bytes"] for f in files),
    "notes": ck["note"],
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(root / "FILES.json").write_text(json.dumps(files, indent=1) + "\n")
print(f"files={manifest['file_count']}  total={manifest['total_bytes']/2**30:.2f} GiB  weights={weights_bytes/2**30:.2f} GiB")
PY

echo "[6/6] 压缩"
ARCHIVE="$OUT_DIR/$NAME.tar.zst"
tar -C "$WORK" -h -cf - "$NAME" | zstd -T0 -3 -q -f -o "$ARCHIVE"
(cd "$OUT_DIR" && sha256sum "$NAME.tar.zst") > "$OUT_DIR/$NAME.tar.zst.sha256"
cat "$OUT_DIR/$NAME.tar.zst.sha256"
du -h "$ARCHIVE"
echo "完成: $ARCHIVE"
