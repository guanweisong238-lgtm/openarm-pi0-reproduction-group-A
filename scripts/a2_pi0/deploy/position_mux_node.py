#!/usr/bin/env python3
"""位置指令复用器：在人工 leader（IK 目标）和策略目标之间切换。

插在 leader 与 dora-openarm 之间，接线见 deploy/dataflow_policy_patch.yaml：

  leader/follower_position_*  ->  leader_*
  policy/position_*           ->  policy_*
  position-mux/position_*     ->  robot/position_*（move_position）

要点：
  * 默认永远转发 leader，人工照旧可以随时接管；
  * 只有收到 mode="policy" 且两路目标足够接近、策略流新鲜时才切到策略；
  * 策略流超过 stale_ms 没有更新、或收到 mode="leader"、或收到任何 arm_command，
    立即无条件回到 leader；已回落后需要新的显式 mode="policy" 才能再进；
  * 转发时保留 leader 的载荷类型（可用 --coerce 用 leader 的 schema 重编码策略值），
    避免两侧 dora 类型不一致把机器人节点弄崩。
"""
from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("旧真机入口已停用，请运行 scripts/a2_pi0/deploy/run_pi0.sh")

import argparse
import time

import numpy as np

SIDES = ("left", "right")


def as_array(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(-1)
    if hasattr(value, "to_numpy"):
        try:
            return np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return np.asarray(value.to_pylist(), dtype=np.float32).reshape(-1)
    if hasattr(value, "as_py"):
        return np.asarray(value.as_py(), dtype=np.float32).reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def coerce(template, values):
    """用 leader 载荷的 schema 重新编码 8 维目标。"""
    import pyarrow as pa

    if template is None or not isinstance(template, (pa.Array, pa.ChunkedArray, pa.StructArray)):
        return pa.array(np.asarray(values, dtype=np.float32))
    if isinstance(template, pa.StructArray):
        fields = [pa.array(np.asarray(values[i:i + 1], dtype=np.float32)) if template.type[i] == pa.float32()
                  else pa.array(np.asarray(values[i:i + 1], dtype=np.float64)) for i in range(template.type.num_fields)]
        return pa.StructArray.from_arrays(fields, names=template.type.names)
    dtype = template.type if pa.types.is_floating(template.type) else pa.float32()
    return pa.array(np.asarray(values, dtype=np.float32), type=dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stale-ms", type=float, default=300.0)
    ap.add_argument("--switch-tolerance", type=float, default=0.05, help="切入策略前允许的最大目标差（rad）")
    ap.add_argument("--coerce", action="store_true", help="按 leader 的 schema 重编码策略值")
    ap.add_argument("--mode-file", default=None,
                    help="可选的模式开关文件：内容为 leader 或 policy，每收到 tick 时读取")
    args = ap.parse_args()

    import dora

    node = dora.Node()
    leader = {s: None for s in SIDES}
    leader_ts = {s: 0.0 for s in SIDES}
    policy = {s: None for s in SIDES}
    policy_ts = {s: 0.0 for s in SIDES}
    mode = "leader"
    blocked = False

    def forward(side, value, template, from_policy=False):
        if args.coerce and from_policy:
            node.send_output(f"position_{side}", coerce(template, as_array(value)))
        else:
            node.send_output(f"position_{side}", value)

    for event in node:
        if event["type"] != "INPUT":
            continue
        ident, value = event["id"], event["value"]
        now = time.monotonic()

        if ident == "tick":
            if args.mode_file:
                try:
                    want = open(args.mode_file, encoding="utf-8").read().strip().lower()
                except OSError:
                    want = ""
                if want and want != mode:
                    ident, value = "mode", want
                else:
                    continue
            else:
                continue

        if ident == "mode":
            want = str(value.as_py() if hasattr(value, "as_py") else value).strip().lower()
            if want == "policy":
                if any(policy[s] is None for s in SIDES) or any(leader[s] is None for s in SIDES) or any(
                        now - policy_ts[s] > args.stale_ms / 1000 for s in SIDES):
                    print("[mux] 策略或 leader 流未就绪/已超时，拒绝切入", flush=True)
                    blocked = True
                    continue
                gap = max(float(np.abs(as_array(policy[s]) - as_array(leader[s])).max()) for s in SIDES)
                if gap > args.switch_tolerance:
                    print(f"[mux] 策略与当前目标相差 {gap:.3f} rad > {args.switch_tolerance}，拒绝切入", flush=True)
                    blocked = True
                    continue
                mode, blocked = "policy", False
                print("[mux] 已切换到策略", flush=True)
            else:
                mode, blocked = "leader", False
                print("[mux] 已回到人工 leader", flush=True)
            continue

        if ident == "arm_command":
            if mode != "leader":
                print("[mux] 收到 arm_command，强制回到 leader", flush=True)
            mode, blocked = "leader", False
            continue

        side = ident.split("_")[-1]
        if side not in SIDES:
            continue
        if ident.startswith("leader_"):
            leader[side], leader_ts[side] = value, now
            if mode == "leader":
                forward(side, value, value)
        elif ident.startswith("policy_"):
            policy[side], policy_ts[side] = value, now
            if mode == "policy":
                forward(side, value, leader[side], from_policy=True)
        else:
            continue

        if mode == "policy" and now - min(policy_ts.values()) > args.stale_ms / 1000:
            print("[mux] 策略流超时，自动回到 leader", flush=True)
            mode = "leader"
            if leader[side] is not None:
                forward(side, leader[side], leader[side])
