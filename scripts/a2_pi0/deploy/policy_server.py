#!/usr/bin/env python3
"""任务 C 策略推理服务：本机 GPU 出 50x16 绝对关节目标（rad）。

本文件只做两件事：把训练用的预处理原样复现，然后对外提供观测->动作的服务。
它不接触机械臂、不打开 CAN、不知道对面是谁；真机安全边界全部在机械臂侧节点。
协议见 deploy/README.md。

用法：
  .venv/bin/python scripts/a2_pi0/deploy/policy_server.py --self-test
  .venv/bin/python scripts/a2_pi0/deploy/policy_server.py --probe
  .venv/bin/python scripts/a2_pi0/deploy/policy_server.py --serve --bind 0.0.0.0 --port 5555
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # scripts/a2_pi0/deploy
SCRIPTS = HERE.parent                            # scripts/a2_pi0
ROOT = HERE.parents[2]                           # 项目根
sys.path.insert(0, str(SCRIPTS))

# 必须在 openpi 初始化前设置缓存与权重路径，复用训练侧的同一套设置。
from real_training_support import get_train_config, load_parameters  # noqa: E402
from real_four_camera_config import CAMERA_MAPPING, FourCameraInputs, FourCameraPi0Config  # noqa: E402
from real_four_camera_loader import DEFAULT, PROMPT, RealFourCameraDataset  # noqa: E402
from openpi import transforms  # noqa: E402
from openpi.models import model as model_lib  # noqa: E402
from openpi.models.tokenizer import PaligemmaTokenizer  # noqa: E402
from openpi.shared import normalize  # noqa: E402

CAM_ORDER = ("global", "chest", "wrist_left", "wrist_right")
DEFAULT_CKPT = ROOT / "checkpoints/real_pi0_four_camera_lora_001/step_1000"
IMAGE_SIZE = 224
MODEL_ACTION_DIM = 32

# 数据核验文档给出的夹爪端点（rad），只用于报告越界，不在服务端裁剪。
GRIPPER_LIMITS = {
    "right": (-0.785398, 0.174533),
    "left": (-0.174533, 0.785398),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_jpeg(blob: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(blob)) as im:
        return np.asarray(im.convert("RGB"))


class Preprocessor:
    """与 RealFourCameraDataset.__getitem__ 完全同序的推理侧预处理。"""

    def __init__(self, stats_root: Path = DEFAULT):
        self.stats = normalize.load(str(stats_root))
        self.adapter = FourCameraInputs()
        self.resize = transforms.ResizeImages(IMAGE_SIZE, IMAGE_SIZE)
        self.pad = transforms.PadStatesAndActions(MODEL_ACTION_DIM)
        # 推理时没有 actions，用非 strict；state 仍按训练统计归一化，数值与训练一致。
        self.norm = transforms.Normalize(self.stats, strict=False)
        self.tokenizer = PaligemmaTokenizer(48)
        tokens = self.tokenizer._tokenizer.encode(PROMPT, add_bos=True) + self.tokenizer._tokenizer.encode("\n")
        if len(tokens) > 48:
            raise ValueError(f"prompt would be truncated: {len(tokens)} tokens")
        self.prompt_tokens = np.asarray(self.tokenizer.tokenize(PROMPT)[0], dtype=np.int32)
        self.prompt_mask = np.asarray(self.tokenizer.tokenize(PROMPT)[1], dtype=bool)
        self.action_mean = np.asarray(self.stats["actions"].mean, dtype=np.float32)
        self.action_std = np.asarray(self.stats["actions"].std, dtype=np.float32)

    def __call__(self, images: dict[str, np.ndarray], state16, prompt: str = PROMPT) -> dict:
        sample = {
            "images": {k: images[k] for k in CAM_ORDER},
            "state": np.asarray(state16, dtype=np.float32),
            "prompt": prompt,
        }
        data = self.adapter(sample)
        data = self.norm(data)
        data["state"] = np.asarray(data["state"], dtype=np.float32)
        data = self.resize(data)
        data = self.pad(data)
        data["image"] = {k: np.asarray(v, dtype=np.uint8) for k, v in data["image"].items()}
        data["tokenized_prompt"] = self.prompt_tokens.copy()
        data["tokenized_prompt_mask"] = self.prompt_mask.copy()
        data.pop("prompt", None)
        return data

    def unnormalize(self, actions: np.ndarray) -> np.ndarray:
        return actions * (self.action_std[: actions.shape[-1]] + 1e-6) + self.action_mean[: actions.shape[-1]]


class PolicyService:
    def __init__(self, ckpt: Path = DEFAULT_CKPT, run_dir: Path | None = None, num_steps: int = 10,
                 prefix: bool = False, horizon: int = 50):
        import jax
        import flax.nnx as nnx
        from flax import traverse_util

        self.jax, self.nnx, self.traverse_util = jax, nnx, traverse_util
        run_dir = run_dir or (ROOT / "results/real_pi0_four_camera_lora_001")
        run = json.loads((run_dir / "config.json").read_text())
        recorded = next((run[k] for k in ("norm_stats_sha256", "norm_sha256", "stats_sha256") if k in run), None)
        self.ckpt = ckpt
        self.candidate = Path(run.get("candidate", DEFAULT))
        self.candidate = ROOT / self.candidate
        self.stats_root = ckpt if (ckpt / "norm_stats.json").is_file() else self.candidate
        dataset_sha = sha256(self.stats_root / "norm_stats.json")
        meta = json.loads((ckpt / "checkpoint.json").read_text())
        if recorded != dataset_sha or meta.get("normalization_sha256", dataset_sha) != dataset_sha:
            raise ValueError("归一化统计与训练配置或检查点不一致")
        if not prefix:
            if run.get("action_semantics") != "absolute IK target, right arm then left arm, rad":
                raise ValueError("缺少受支持的绝对关节动作语义")
            if run.get("action_horizon") != horizon or run.get("training_frequency_hz") != 30:
                raise ValueError("训练动作窗口或时间间隔不匹配")

        model_class = FourCameraPi0Config
        if prefix:
            from real_prefix_pi0 import PrefixPi0Config

            model_class = PrefixPi0Config
        cfg = dataclasses.replace(
            get_train_config(),
            model=model_class(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
                              action_horizon=horizon),
            seed=int(run.get("seed", 0)),
        )
        self.cfg = cfg
        params = load_parameters(cfg)
        abstract = nnx.eval_shape(cfg.model.create, jax.random.key(cfg.seed))
        self.graph = nnx.graphdef(abstract)
        model = nnx.merge(self.graph, params)
        self._load_adapter(model, ckpt)
        self.params = nnx.state(model)

        self.pre = Preprocessor(self.stats_root)
        self.num_steps = num_steps
        self.prefix = prefix
        self._lock = threading.Lock()

        def sample(params, obs, known, key):
            m = nnx.merge(self.graph, params)
            if self.prefix:
                return m.sample_actions(key, obs, known_prefix=known, num_steps=self.num_steps)
            return m.sample_actions(key, obs, num_steps=self.num_steps)

        self._sample = jax.jit(sample)

    def _load_adapter(self, model, ckpt: Path) -> None:
        meta = json.loads((ckpt / "checkpoint.json").read_text())
        if sha256(ckpt / "lora_params.npz") != meta["lora_sha256"]:
            raise ValueError(f"LoRA 适配器哈希不匹配: {ckpt}")
        state = self.nnx.state(model).filter(self.cfg.trainable_filter)
        expected = self.traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
        with np.load(ckpt / "lora_params.npz", allow_pickle=False) as f:
            if set(f.files) != set(expected):
                raise ValueError("适配器键与模型不匹配")
            flat = {k: f[k] for k in f.files}
        state.replace_by_pure_dict(self.traverse_util.unflatten_dict(flat, sep="/"))
        self.nnx.update(model, state)

    def warmup(self) -> float:
        started = time.perf_counter()
        dummy = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        obs = self._make_obs(self.pre({k: dummy for k in CAM_ORDER}, np.zeros(16, np.float32)))
        out = self._sample(self.params, obs, np.zeros((1, 7, MODEL_ACTION_DIM), np.float32), self.jax.random.key(0))
        self.jax.block_until_ready(out)
        return time.perf_counter() - started

    def _make_obs(self, data: dict):
        batch = self.jax.tree.map(lambda x: np.stack([x]), data)
        batch.pop("actions", None)
        return model_lib.Observation.from_dict(batch)

    def infer(self, images: dict[str, np.ndarray], state16: np.ndarray, *, known_prefix=None, seed=None,
              include_raw=False):
        state16 = np.asarray(state16, dtype=np.float32)
        if state16.shape != (16,) or not np.isfinite(state16).all():
            raise ValueError("state 必须是有限的 16 维反馈")
        data = self.pre(images, state16)
        obs = self._make_obs(data)
        key = self.jax.random.key(int(seed if seed is not None else np.random.randint(0, 2**31)))
        prefix = np.zeros((1, 7, MODEL_ACTION_DIM), np.float32)
        if known_prefix is not None:
            known = np.asarray(known_prefix, dtype=np.float32)
            if known.shape != (7, 16):
                raise ValueError("known_prefix 必须是 (7,16)")
            prefix[0, :, :16] = self.pre.norm({"actions": known})["actions"][:7]
        with self._lock:
            t0 = time.perf_counter()
            out = self._sample(self.params, obs, prefix, key)
            self.jax.block_until_ready(out)
            infer_ms = (time.perf_counter() - t0) * 1000
        raw = np.asarray(out, dtype=np.float32)
        if raw.shape != (1, self.cfg.model.action_horizon, MODEL_ACTION_DIM) or not np.isfinite(raw).all():
            raise ValueError("模型输出维度或数值异常")
        actions = self.pre.unnormalize(raw[0, :, :16])
        if not np.isfinite(actions).all():
            raise ValueError("反归一化产生非有限动作")
        if include_raw:
            return actions, infer_ms, raw[0]
        return actions, infer_ms


# ---------------------------------------------------------------- 传输层
def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def recv_message(conn: socket.socket) -> tuple[dict, list[bytes]]:
    (head_len,) = struct.unpack(">I", recv_exact(conn, 4))
    head = json.loads(recv_exact(conn, head_len))
    blobs = []
    for _ in range(head.pop("blob_count", 0)):
        (n,) = struct.unpack(">I", recv_exact(conn, 4))
        blobs.append(recv_exact(conn, n))
    return head, blobs


def send_message(conn: socket.socket, head: dict) -> None:
    payload = json.dumps(head, separators=(",", ":")).encode()
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def handle(conn: socket.socket, service: PolicyService) -> None:
    while True:
        head, blobs = recv_message(conn)
        recv_ns = time.time_ns()
        try:
            if head.get("type") != "obs":
                raise ValueError(f"未知消息类型 {head.get('type')}")
            if len(blobs) != len(CAM_ORDER):
                raise ValueError(f"需要 {len(CAM_ORDER)} 路 JPEG，收到 {len(blobs)}")
            state = np.asarray(head["state"], dtype=np.float32)
            if state.shape != (16,) or not np.isfinite(state).all():
                raise ValueError("state 必须是有限的 16 维")
            images = {name: decode_jpeg(blob) for name, blob in zip(CAM_ORDER, blobs)}
            actions, infer_ms = service.infer(images, state, known_prefix=head.get("known_prefix"),
                                              seed=head.get("seed"))
            send_message(conn, {
                "ok": True, "seq": head.get("seq"), "shape": list(actions.shape), "dtype": "float32",
                "actions_b64": base64.b64encode(actions.astype("<f4").tobytes()).decode(),
                "infer_ms": round(infer_ms, 1), "recv_ns": recv_ns, "sent_ns": time.time_ns(),
                "action_min": float(actions.min()), "action_max": float(actions.max()),
            })
            print(f"[server] seq={head.get('seq')} infer={infer_ms:.0f}ms "
                  f"range=[{actions.min():.3f},{actions.max():.3f}]", flush=True)
        except Exception as exc:  # 单次失败不杀服务，交回客户端决定是否停
            try:
                send_message(conn, {"ok": False, "seq": head.get("seq"), "error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                return
            print(f"[server] ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def serve(service: PolicyService, bind: str, port: int) -> None:
    service.warmup()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, port))
    srv.listen(1)
    print(f"[server] listening on {bind}:{port}", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[server] client {addr}", flush=True)
        try:
            handle(conn, service)
        except (ConnectionError, OSError) as exc:
            print(f"[server] client gone: {exc}", flush=True)
        finally:
            conn.close()


# ---------------------------------------------------------------- 自检
def self_test(candidate: Path = DEFAULT, stats_root: Path | None = None) -> None:
    """证明推理侧预处理与训练侧 __getitem__ 逐元素一致，不需要加载模型。"""
    pre = Preprocessor(stats_root or candidate)
    ds = RealFourCameraDataset("validation", root=candidate, cache_episodes=4)

    for index in (0, 7, len(ds) // 2, len(ds) - 1):
        raw = ds.get_raw(index)
        mine = pre(raw["images"], raw["state"])
        ref = ds[index]
        if not np.array_equal(mine["state"], ref["state"]):
            raise AssertionError(f"index {index}: state 预处理与训练不一致")
        if not np.array_equal(mine["tokenized_prompt"], ref["tokenized_prompt"]):
            raise AssertionError(f"index {index}: prompt token 不一致")
        for key in mine["image"]:
            if not np.array_equal(mine["image"][key], ref["image"][key]):
                raise AssertionError(f"index {index}: 图像 {key} 预处理不一致")
        print(f"[self-test] index {index} 通过 state={mine['state'].shape} "
              f"image={mine['image']['chest_0_rgb'].shape} 张量数={len(mine['image'])}")
    print("[self-test] 预处理与训练管线逐元素一致")


def probe(service: PolicyService, log: Path | None = None) -> None:
    ds = RealFourCameraDataset("validation", root=service.candidate, cache_episodes=1)
    raw = ds.get_raw(0)
    print("[probe] warm-up 编译中", flush=True)
    print(f"[probe] 编译耗时 {service.warmup():.1f}s", flush=True)
    rows = []
    for i in range(5):
        t0 = time.perf_counter()
        actions, infer_ms, normalized = service.infer(raw["images"], raw["state"], seed=1000 + i,
                                                       include_raw=True)
        total = (time.perf_counter() - t0) * 1000
        ref = np.asarray(raw["actions"], dtype=np.float32)
        err = np.abs(actions - ref)
        rows.append({"total_ms": total, "infer_ms": infer_ms,
                     "mae_vs_recorded": float(err.mean()), "max_vs_recorded": float(err.max())})
        if log is not None:
            record = dict(rows[-1], obs_timestamp_ns=raw["timestamp_ns"], episode=raw["episode_id"],
                          seed=1000 + i, feedback_rad=raw["state"].tolist(),
                          model_normalized=normalized.tolist(), target_right_rad=actions[:, :8].tolist(),
                          target_left_rad=actions[:, 8:].tolist(), sent_target=None)
            with log.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"[probe] {i+1}/5 total={total:.0f}ms infer={infer_ms:.0f}ms "
              f"MAE={err.mean():.3f}rad max={err.max():.3f}rad", flush=True)
    print(json.dumps({"checkpoint": str(service.ckpt), "rows": rows,
                      "note": "单样本开口环对比；不是闭环成功率，也不代表真机行为"}, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--run-dir", type=Path, default=ROOT / "results/real_pi0_four_camera_lora_001")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--prefix", action="store_true", help="使用前缀条件模型（需配套 prefix 适配器）")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--log", type=Path, help="保存离线探针的模型原始输出、目标和反馈")
    args = ap.parse_args()

    if args.self_test:
        run = json.loads((args.run_dir / "config.json").read_text())
        self_test(ROOT / Path(run.get("candidate", DEFAULT)))
        return
    service = PolicyService(args.ckpt, args.run_dir, args.num_steps, args.prefix)
    if args.probe:
        probe(service, args.log)
    elif args.serve:
        serve(service, args.bind, args.port)
    else:
        ap.error("需要 --self-test / --probe / --serve 之一")


if __name__ == "__main__":
    main()
