#!/usr/bin/env python3
"""机械臂侧 dora 节点：把网络策略输出变成 position_left/position_right 目标流。

三种模式，默认最保守：
  observe  只统计输入是否到齐，不发请求、不发指令
  shadow   正常请求策略并记录预测动作，但绝不发布到机械臂（影子对比用）
  publish  真正发布关节目标，需要 --publish 且 OPENARM_POLICY_PUBLISH=1

它不发送 Arm Start / Arm Stop，不打开 CAN；使能、急停、断使能仍由 UI 与 VR B 键负责。
动作语义与训练一致：30 Hz、50 步、16 维绝对 IK 目标 [右 J1..J7, 右夹爪, 左 J1..J7, 左夹爪]（rad）。

时序（与 scripts/a2_pi0/real_action_window.py 的参考实现同构）：
  计划 start 时刻记为 s，执行到 s+14 时用当前观测发起下一次推理，
  新计划在到达后立刻接管：第 t 步执行 new[t - s_new]。
  带 --prefix 时新计划必须保留 7 步已知前缀，否则判故障、不切换。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

CAM_ORDER = ("global", "chest", "wrist_left", "wrist_right")
GRIPPER_LIMITS = {"right": (-0.785398, 0.174533), "left": (-0.174533, 0.785398)}
PREFIX = 7


# ------------------------------------------------------------------ 取值适配
def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, bytearray)):
        return np.frombuffer(value, np.uint8)
    if hasattr(value, "to_numpy"):
        try:
            return np.asarray(value.to_numpy(zero_copy_only=False))
        except (TypeError, ValueError):
            return np.asarray(value.to_pylist())
    if hasattr(value, "as_py"):
        return np.asarray(value.as_py())
    return np.asarray(value)


def extract_jpeg(value):
    """从相机事件取出 JPEG 字节；失败时报出实际结构，便于现场改适配。"""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        for key in ("bytes", "data", "jpeg", "image_data", "payload", "image", "value"):
            if key in value:
                return extract_jpeg(value[key])
        raise ValueError(f"字典里找不到图像字节，键={sorted(value)}")
    if hasattr(value, "as_py") and not isinstance(value, np.ndarray):
        py = value.as_py()
        if isinstance(py, dict):
            return extract_jpeg(py)
        if isinstance(py, (bytes, bytearray)):
            return bytes(py)
    arr = np.asarray(_to_numpy(value))
    if arr.dtype == np.uint8 and arr.ndim == 1:
        blob = arr.tobytes()
        if blob[:2] != b"\xff\xd8":
            raise ValueError("收到 uint8 一维数组但不是 JPEG（缺少 FFD8 头）")
        return blob
    raise ValueError(f"不支持的相机载荷 type={type(value).__name__} shape={arr.shape} "
                     f"dtype={arr.dtype}；若为原始 HWC 图像需在客户端编码，见 README")


def extract_qpos(value):
    if isinstance(value, dict):
        for key in ("qpos", "position", "positions", "joint_positions"):
            if key in value:
                return extract_qpos(value[key])
        raise ValueError(f"状态字典里没有 qpos，键={sorted(value)}")
    if hasattr(value, "as_py") and not isinstance(value, np.ndarray):
        py = value.as_py()
        if isinstance(py, dict):
            return extract_qpos(py)
    arr = np.asarray(_to_numpy(value), dtype=np.float64).reshape(-1)
    if arr.size != 8:
        raise ValueError(f"期望每臂 8 维 qpos，收到 {arr.size} 维")
    return arr


# ------------------------------------------------------------------ 网络
def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("服务端关闭连接")
        buf += chunk
    return bytes(buf)


def request_actions(host, port, timeout, frames, state16, seq, state_ts_ns, known_prefix=None):
    head = {
        "type": "obs", "seq": seq, "sent_ns": time.time_ns(),
        "state": [float(x) for x in state16], "state_ts_ns": int(state_ts_ns),
        "blob_count": len(CAM_ORDER),
        "cameras": {n: {"src_ts_ns": int(frames[n]["src_ts_ns"]), "seq": int(frames[n]["seq"])} for n in CAM_ORDER},
    }
    if known_prefix is not None:
        head["known_prefix"] = np.asarray(known_prefix, np.float32).tolist()
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        payload = json.dumps(head, separators=(",", ":")).encode()
        conn.sendall(struct.pack(">I", len(payload)) + payload)
        for name in CAM_ORDER:
            blob = frames[name]["blob"]
            conn.sendall(struct.pack(">I", len(blob)) + blob)
        (n,) = struct.unpack(">I", _recv_exact(conn, 4))
        resp = json.loads(_recv_exact(conn, n))
    if not resp.get("ok"):
        raise RuntimeError(f"服务端拒绝: {resp.get('error')}")
    actions = np.frombuffer(base64.b64decode(resp["actions_b64"]), dtype="<f4")
    return actions.reshape(resp["shape"]).astype(np.float32), resp


# ------------------------------------------------------------------ 主逻辑
class Controller:
    def __init__(self, args, sender=None):
        self.args = args
        self.sender = sender
        self.frames = {}
        self.qpos = {}
        self.state_ts_ns = 0
        self.step = 0
        self.plans = []            # [{'actions': (K,16), 'start': int}]
        self.pending = False
        self.seq = 0
        self.last_action = None
        self.faults = 0
        self.stopped = None
        self.late_warned = False
        self._exhausted = False
        self.last_request_step = -10**9
        self._last_fault_ns = 0
        self.log = open(args.log, "a", buffering=1) if args.log else None
        self.publish = bool(args.publish and os.environ.get("OPENARM_POLICY_PUBLISH") == "1")

    # -- 输入 ----------------------------------------------------------
    def on_event(self, ident, value, metadata):
        if ident == "tick":
            self.on_tick()
            return
        if ident.startswith("camera_"):
            name = ident[len("camera_"):]
            if name not in CAM_ORDER:
                return
            meta = metadata or {}
            try:
                blob = extract_jpeg(value)
            except ValueError as exc:
                self.fault(f"相机 {name} 载荷无法解析: {exc}")
                return
            src = meta.get("source_timestamp_ns") or meta.get("timestamp_ns") or 0
            self.frames[name] = {"blob": blob, "recv_ns": time.monotonic_ns(),
                                 "src_ts_ns": int(src), "seq": int(meta.get("sequence", 0))}
            return
        if ident in ("state_left", "state_right"):
            try:
                self.qpos[ident.split("_")[1]] = extract_qpos(value)
                self.state_ts_ns = time.monotonic_ns()
            except ValueError as exc:
                self.fault(f"{ident} 无法解析: {exc}")

    def ready(self):
        return all(n in self.frames for n in CAM_ORDER) and all(k in self.qpos for k in ("left", "right"))

    # -- 故障 ----------------------------------------------------------
    def fault(self, message):
        self.faults += 1
        now = time.monotonic_ns()
        if now - self._last_fault_ns > 500e6:
            self._last_fault_ns = now
            print(f"[policy] FAULT({self.faults}) {message}", file=sys.stderr, flush=True)
        if self.publish and self.faults >= self.args.fault_limit:
            self.stopped = message
            print(f"[policy] 故障达上限，停止发布并保持最后目标：{message}", file=sys.stderr, flush=True)

    # -- 计划管理 ------------------------------------------------------
    def active_plan(self):
        usable = [p for p in self.plans if p["start"] <= self.step]
        if not usable:
            return None
        plan = max(usable, key=lambda p: p["start"])
        self.plans = [p for p in self.plans if p["start"] >= plan["start"]]
        return plan

    def fire_request(self, plan, rel):
        # 先记时刻：即使本次因未就绪/失鲜被拒，也要等到下个重规划周期再试，避免 30 Hz 猛打
        self.last_request_step = self.step
        if self.pending or self.stopped or not self.ready():
            return
        newest = max(self.frames[n]["recv_ns"] for n in CAM_ORDER)
        age_ms = (time.monotonic_ns() - newest) / 1e6
        if age_ms > self.args.stale_ms:
            self.fault(f"图像帧龄 {age_ms:.0f}ms 超过 {self.args.stale_ms}ms")
            return
        prefix = None
        if self.args.prefix:
            if plan is None or rel + PREFIX > plan["actions"].shape[0]:
                self.fault("旧计划剩余不足以构造 7 步前缀")
                return
            prefix = plan["actions"][rel:rel + PREFIX]
        self.pending = True
        self.seq += 1
        frames = {n: dict(self.frames[n]) for n in CAM_ORDER}
        state16 = np.concatenate([self.qpos["right"], self.qpos["left"]]).astype(np.float32)
        threading.Thread(target=self._worker, args=(frames, state16, self.seq, prefix, self.step),
                         daemon=True).start()

    def _worker(self, frames, state16, seq, prefix, request_step):
        started = time.time_ns()
        try:
            actions, resp = request_actions(self.args.server_host, self.args.server_port, self.args.timeout,
                                            frames, state16, seq, self.state_ts_ns, prefix)
            if actions.shape != (50, 16):
                raise RuntimeError(f"动作形状异常 {actions.shape}")
            if prefix is not None and not np.allclose(actions[:PREFIX], prefix, rtol=0, atol=1e-6):
                raise RuntimeError("新计划未保留 7 步已知前缀，拒绝切换")
            self.plans.append({"actions": actions, "start": request_step})
            self._exhausted = False
            self.late_warned = False
            rtt = (time.time_ns() - started) / 1e6
            print(f"[policy] seq={seq} 计划到达 start={request_step} rtt={rtt:.0f}ms "
                  f"infer={resp.get('infer_ms')}ms", flush=True)
            self.faults = 0
        except Exception as exc:
            self.fault(f"请求失败 seq={seq}: {type(exc).__name__}: {exc}")
        finally:
            self.pending = False

    # -- 执行 ----------------------------------------------------------
    def clamp(self, action):
        out = np.array(action, dtype=np.float32, copy=True)
        for idx, side in ((7, "right"), (15, "left")):
            low, high = GRIPPER_LIMITS[side]
            out[idx] = float(np.clip(out[idx], low, high))
        if self.last_action is not None:
            delta = out[:15] - self.last_action[:15]
            bad = np.abs(delta) > self.args.max_step_rad
            if bad.any():
                print(f"[policy] 单步跳变裁剪 关节={np.flatnonzero(bad).tolist()} "
                      f"max={float(np.abs(delta).max()):.3f}rad", file=sys.stderr, flush=True)
                out[:15] = np.where(bad, self.last_action[:15] + np.sign(delta) * self.args.max_step_rad, out[:15])
        return out

    def emit(self, action, note=""):
        action = self.clamp(action)
        if not np.isfinite(action).all():
            self.fault("动作含非有限值，拒绝发布")
            return
        if self.publish and self.sender is not None:
            self.sender.send_output("position_right", action[:8])
            self.sender.send_output("position_left", action[8:])
        self.last_action = action
        if self.log:
            self.log.write(json.dumps({"t_ns": time.monotonic_ns(), "step": self.step,
                                       "mode": "publish" if self.publish else "shadow",
                                       "action": action.tolist(), "note": note}) + "\n")

    def on_tick(self):
        if self.stopped or not self.ready():
            return
        plan = self.active_plan()
        if plan is None:
            if not self.pending:
                self.fire_request(None, 0)
            return
        rel = self.step - plan["start"]
        if rel >= plan["actions"].shape[0]:
            # 计划耗尽：保持最后目标，并重新请求；只报一次，避免刷屏
            if not self._exhausted:
                self._exhausted = True
                self.fault("计划已耗尽且没有可用新计划，保持最后目标并重新请求")
            if not self.pending:
                self.fire_request(None, 0)
            return
        self.emit(plan["actions"][rel])
        self.step += 1
        if self.step - self.last_request_step >= self.args.chunk_exec:
            self.fire_request(plan, self.step - plan["start"])


class Sender:
    """把 8 维目标按现场确认的 dora 载荷类型发出。"""

    def __init__(self, node, encoder):
        self.node = node
        self.encoder = encoder

    def send_output(self, ident, values):
        values = np.asarray(values, dtype=np.float32)
        if self.encoder == "pa_array":
            import pyarrow as pa

            self.node.send_output(ident, pa.array(values))
        elif self.encoder == "pa_list":
            import pyarrow as pa

            self.node.send_output(ident, pa.array(values.tolist()))
        elif self.encoder == "pa_struct":
            import pyarrow as pa

            self.node.send_output(ident, pa.StructArray.from_arrays([pa.array(values)], names=["qpos"]))
        elif self.encoder == "pa_table":
            import pyarrow as pa

            self.node.send_output(ident, pa.table({"qpos": pa.array(values)}))
        else:
            raise ValueError(f"未知 encoder {self.encoder}")


def run_dora(args):
    import dora

    node = dora.Node()
    ctrl = Controller(args, Sender(node, args.encoder))
    mode = "publish" if ctrl.publish else ("shadow(未授权降级)" if args.publish else "shadow")
    print(f"[policy] mode={mode} server={args.server_host}:{args.server_port} hz={args.hz}", flush=True)
    ready_reported = False
    for event in node:
        if event["type"] != "INPUT":
            continue
        ctrl.on_event(event["id"], event["value"], event.get("metadata"))
        if ctrl.ready() and not ready_reported:
            ready_reported = True
            print("[policy] 四路相机与双臂状态已到齐，开始按 30 Hz 出计划", flush=True)


def run_offline(args):
    """不接 dora，用假观测验证调度/网络/服务连通性。"""
    ctrl = Controller(args)
    period = 1.0 / args.hz
    if args.offline_fake:
        dummy = np.zeros((120, 160, 3), np.uint8)
        from PIL import Image
        import io

        buf = io.BytesIO()
        Image.fromarray(dummy).save(buf, format="JPEG")
        for name in CAM_ORDER:
            ctrl.frames[name] = {"blob": buf.getvalue(), "recv_ns": time.monotonic_ns(),
                                 "src_ts_ns": time.time_ns(), "seq": 0}
        ctrl.qpos.update({"right": np.zeros(8), "left": np.zeros(8)})
        print("[policy] offline-fake：使用全零假观测，仅验证协议与节奏", flush=True)
    deadline = time.monotonic()
    while True:
        if args.offline_fake:
            # 模拟 30 fps 相机持续到帧，否则帧龄检查会一直失败
            now_ns = time.monotonic_ns()
            for name in CAM_ORDER:
                ctrl.frames[name].update({"recv_ns": now_ns, "src_ts_ns": time.time_ns()})
            ctrl.state_ts_ns = now_ns
        ctrl.on_tick()
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-host", default="192.168.1.10")
    ap.add_argument("--server-port", type=int, default=5555)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--hz", type=float, default=30.0, help="发布/执行频率，必须与训练采样频率一致")
    ap.add_argument("--chunk-exec", type=int, default=20,
                    help="每执行多少步重规划一次（必须小于 50 且留出推理时间余量）")
    ap.add_argument("--prefix", action="store_true", help="7 步前缀条件，要求服务端同为 prefix 模型")
    ap.add_argument("--stale-ms", type=float, default=150.0)
    ap.add_argument("--fault-limit", type=int, default=3)
    ap.add_argument("--max-step-rad", type=float, default=0.25)
    ap.add_argument("--encoder", choices=("pa_array", "pa_list", "pa_struct", "pa_table"), default="pa_array")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--offline-fake", action="store_true")
    args = ap.parse_args()
    if args.publish and os.environ.get("OPENARM_POLICY_PUBLISH") != "1":
        print("[policy] 未设置 OPENARM_POLICY_PUBLISH=1，本次降级为 shadow 模式", file=sys.stderr, flush=True)
    if args.offline:
        run_offline(args)
    else:
        run_dora(args)


if __name__ == "__main__":
    main()
