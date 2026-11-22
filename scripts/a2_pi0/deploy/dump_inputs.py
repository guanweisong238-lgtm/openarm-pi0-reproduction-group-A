#!/usr/bin/env python3
"""现场探针：把每个输入的 dora 载荷类型、形状、metadata 打出来。

用于接入前确认三件只能用现场数据回答的事：
  1. cameras/status_* 里 JPEG 字节到底放在哪个字段；
  2. robot/state_* 的 qpos 结构；
  3. leader/follower_position_* 的载荷类型（决定 policy_client 的 --encoder）。
只读，不发任何输出，不使能机械臂。

用法（机械臂电脑）：
  dora run -d deploy/probe_dataflow.yaml     # 或在现有图里加一个 probe 节点
"""
from __future__ import annotations

import sys
import time

import numpy as np


def describe(value):
    dtype = getattr(value, "dtype", None)
    if dtype is None and hasattr(value, "type"):
        dtype = value.type
    shape = getattr(value, "shape", None)
    extra = ""
    if isinstance(value, (bytes, bytearray)):
        extra = f" bytes={len(value)} head={bytes(value[:2]).hex()}"
    elif isinstance(value, dict):
        extra = f" keys={sorted(value)}"
    elif shape is None and hasattr(value, "to_numpy"):
        try:
            arr = np.asarray(value.to_numpy(zero_copy_only=False))
            shape, dtype, extra = arr.shape, arr.dtype, f" sample={arr.reshape(-1)[:4]}"
        except Exception as exc:  # noqa: BLE001
            extra = f" to_numpy 失败: {exc}"
    return f"type={type(value).__name__} dtype={dtype} shape={shape}{extra}"


def main() -> None:
    import dora

    node = dora.Node()
    seen = {}
    for event in node:
        if event["type"] != "INPUT":
            continue
        ident = event["id"]
        now = time.monotonic()
        last = seen.get(ident, 0.0)
        if now - last < 1.0:      # 每个输入每秒最多打一行
            continue
        seen[ident] = now
        meta = event.get("metadata") or {}
        print(f"[probe] {ident}: {describe(event['value'])} metadata_keys={sorted(meta)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
