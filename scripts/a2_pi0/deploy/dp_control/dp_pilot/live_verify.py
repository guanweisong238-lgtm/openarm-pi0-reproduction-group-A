"""PC-only field verification: live observations, shadow inference, per-arm stop."""
import json
import fcntl
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import SimpleQueue

from dp_paths import HERE, SDK_PY
SIDES = ("left", "right")


class LiveSession:
    def __init__(self):
        self.locked = threading.Event()
        self.samples = {s: deque(maxlen=32) for s in SIDES}
        self.guard = threading.Lock()
        self.workers = {}
        self.events = SimpleQueue()
        self.opened_at = None
        self.stop_reason = None
        self.predictions_shown = 0

    def connect(self):
        if self.locked.is_set() or self.workers:
            raise RuntimeError("session already started or locked")
        directory = Path.home()/".cache/openarm-dp-field"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.ownership = (directory/"verify.lock").open("a")
        try:
            fcntl.flock(self.ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.ownership.close()
            raise RuntimeError("已有真机验证面板占用，禁止重复连接") from exc
        self.opened_at = time.monotonic()
        try:
            for side in SIDES:
                p = subprocess.Popen([str(SDK_PY), "-B", str(HERE/"live_arm_probe.py"),
                                      "--field-session", side], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
                self.workers[side] = p
                threading.Thread(target=self.read, args=(side,p), daemon=True).start()
        except Exception as exc:
            self.stop("启动失败："+str(exc))
            raise

    def stop(self, reason="操作员请求停止"):
        if self.stop_reason is None:
            self.stop_reason = reason
        self.locked.set()
        for p in tuple(self.workers.values()):
            # EOF is the stop request, not a signal killing the CAN owner.
            if not p.stdin.closed:
                try:
                    p.stdin.close()
                except OSError:
                    pass

    def locked_status(self):
        return (f"会话已锁定：{self.stop_reason}。已显示实时预测 {self.predictions_shown} 次。"
                "停稳未确认；有危险请使用实体急停。")

    def read(self, side, process):
        ended = False
        try:
            for line in process.stdout:
                msg = json.loads(line)
                if msg["kind"] == "feedback":
                    with self.guard:
                        self.samples[side].append((msg["stamp"], msg["q"]))
                else:
                    self.events.put((side, msg))
                    ended = msg["kind"] in ("disabled", "unconfirmed")
        except (ValueError, KeyError, OSError) as exc:
            self.events.put((side, {"kind":"fault", "reason":str(exc)}))
        finally:
            process.wait()
            if not ended:
                self.events.put((side, {"kind":"unconfirmed", "reason":"feedback process exited"}))

    def at(self, anchor):
        with self.guard:
            arms = {s: next((v for v in reversed(self.samples[s]) if v[0] <= anchor), None)
                    for s in SIDES}
        if any(v is None or anchor-v[0] >= 100_000_000 for v in arms.values()):
            raise ValueError("missing/stale arm feedback")
        return arms

    def check_watchdog(self):
        if self.opened_at is None or self.locked.is_set() or time.monotonic()-self.opened_at < 2:
            return
        now = time.time_ns()
        with self.guard:
            for s, samples in self.samples.items():
                if not samples or not 0 <= now-samples[-1][0] < 250_000_000:
                    self.stop(f"{s} 反馈中断（看门狗）")
                    self.events.put((s, {"kind":"fault", "reason":"反馈中断，已请求双臂失能"}))
                    return
