"""Per-arm supervised motion owner; all position writes use the official driver."""
import argparse
from collections import deque
import hashlib
import json
import socket
import signal
import time

import numpy as np

from live_arm_probe import await_enable, before_refresh, fresh, open_probe
from task_reference import DEMO_ID, DEMO_FRAME, RELEASE, start_row
from gravity_profile import DEFAULT_PERCENT

LEASE = .50
TRIAL_SECONDS = 45
JOINT_RATE = .30
GRIP_RATE = .50
POLICY_RESPONSE_S = .18
LEFT_J5_PREDICTION_BOUND = .50


def task_start_pose(side, limits):
    row = start_row()
    if row.shape != (16,) or hashlib.sha256(row.tobytes()).hexdigest() != RELEASE.pose_sha256:
        raise ValueError(f"episode {DEMO_ID} 第 {DEMO_FRAME} 帧反馈与核对记录不符")
    q = row[{"left":slice(0,8), "right":slice(8,16)}[side]].astype(float)
    if not np.all((limits[:,0] <= q) & (q <= limits[:,1])):
        raise ValueError(f"episode {DEMO_ID} 起点超出关节限位")
    return q


def pose_error(q, pose):
    return f"目标={pose.round(3).tolist()} 实测={q.round(3).tolist()} 偏差(实测-目标, rad)={(q-pose).round(3).tolist()}"


def check_pose_hold(now, since, q, pose):
    if now-since > 30:
        raise RuntimeError("到位待命达到 30 秒上限")
    if np.any(np.abs(q-pose) > .10):
        raise RuntimeError("待命时偏离目标：" + pose_error(q,pose))


class ArrivalWindow:
    """Only extend settling for axes that are demonstrably closing their errors."""
    def __init__(self, start, nominal):
        self.start, self.nominal = start, nominal
        self.history = deque()

    def check(self, now, q, pose):
        err = np.abs(q-pose)
        self.history.append((now,err))
        while len(self.history)>1 and self.history[1][0] <= now-.5:
            self.history.popleft()
        elapsed = now-self.start
        if elapsed <= self.nominal: return
        since, prior = self.history[0]
        outside = err > .08
        # Preserve the full in-tolerance dwell even when the last axis enters late.
        if (elapsed > self.nominal+2. or now-since < .49 or
                np.any((prior-err)[outside] < .005)):
            raise RuntimeError("到位超时或超差轴未持续趋近：" + pose_error(q,pose))


def accept_target(value, q, limits, stamp, now, *, side=None):
    x = np.asarray(value, dtype=float)
    q = np.asarray(q, dtype=float)
    if type(stamp) != int or not 0 <= now-stamp < 100_000_000:
        raise ValueError(f"expired action: age_ms={(now-stamp)/1e6:.1f}" if type(stamp) is int
                         else "invalid action timestamp")
    if x.shape != (8,) or q.shape != (8,) or not np.isfinite([x,q]).all():
        raise ValueError("invalid target/feedback")
    for i in np.flatnonzero((x < limits[:,0]) | (x > limits[:,1])):
        raise ValueError(f"outside official joint limits: J{i+1} 目标={x[i]:.4f} rad，"
                         f"限位={limits[i].tolist()}")
    bound = np.array([.35]*7+[1.05])
    if side == "left":
        bound[4] = LEFT_J5_PREDICTION_BOUND
    for i in np.flatnonzero(np.abs(x-q) > bound):
        raise ValueError(f"predicted first target too far from feedback: J{i+1} "
                         f"目标={x[i]:.4f} 实测={q[i]:.4f} 差={x[i]-q[i]:.4f} "
                         f"允许绝对差={bound[i]:.2f} rad")
    return x


def policy_step(last, target, dt):
    # Ease toward the current validated target; never coast toward an older one.
    dt = min(max(dt, 0.), .04)
    bound = np.array([JOINT_RATE]*7+[GRIP_RATE])*dt
    return last + np.clip(-np.expm1(-dt/POLICY_RESPONSE_S)*(target-last), -bound, bound)


class Control:
    def __init__(self, channel):
        self.sock = channel
        self.sock.setblocking(False)
        self.last_ping = time.monotonic()
        self.kind = "preflight"
        self.command = None
        self.pending = None
        self.last_anchor = 0
        self.peer_enabled = False

    def emit(self, kind, **fields):
        self.sock.send(json.dumps(dict(kind=kind, **fields),allow_nan=False).encode())

    def check(self):
        while True:
            try:
                data = self.sock.recv(8192)
            except BlockingIOError:
                break
            if not data:
                raise RuntimeError("PC disconnected")
            m = json.loads(data)
            if m["kind"] == "stop":
                raise RuntimeError("PC stop")
            if m["kind"] == "ping":
                stamp = m["mono"]
                if not 0 <= time.monotonic()-stamp <= LEASE:
                    raise RuntimeError("stale PC heartbeat")
                self.last_ping = stamp
            elif m["kind"] == "home" and self.kind == "preflight" and m["confirm"] == "START":
                self.command = "home"
            elif m["kind"] == "release_home" and self.kind == "await_peer":
                self.peer_enabled = True
            elif m["kind"] == "task_start" and self.kind == "home_ready":
                self.command = "task_start"
            elif m["kind"] == "target" and self.kind in ("ready","running"):
                if m["stamp"] <= self.last_anchor:
                    raise RuntimeError("out-of-order policy target")
                self.last_anchor = m["stamp"]
                self.pending = m
            else:
                raise RuntimeError("unexpected motion message")
        if time.monotonic()-self.last_ping > LEASE:
            raise RuntimeError("PC heartbeat timeout")


def run(side, channel, gravity_percent=DEFAULT_PERCENT):
    from openarm_driver import Config, SingleArmDriver
    io = Control(channel)
    label = side+"_arm"
    driver = None
    reason = "unknown"
    trace = deque(maxlen=192)
    stage = "连接前检查"
    def report(kind, **fields):
        try:
            io.emit(kind, **fields)
        except OSError:
            pass

    class MotionConfig(Config):
        def get_stop_config(self):
            return {"moves": []}  # Never execute a parking trajectory on a fault.

        def get_gripper_posforce_limits(self):
            speed, torque = super().get_gripper_posforce_limits()
            return np.array([min(speed, GRIP_RATE), torque])

    class CheckedDriver(SingleArmDriver):
        def set_latest_state(self, *args, **kwargs):
            super().set_latest_state(*args, **kwargs)
            if getattr(self, "_mark_start_refresh", False):
                # Official start(): receive after enable, refresh_all, receive again.
                self._start_pending = before_refresh(groups)
                self._mark_start_refresh = False

        def _on_start(self):
            nonlocal stage
            stage = "启动/等待八轴使能回包"
            groups = [(self.openarm.get_arm(),7),(self.openarm.get_gripper(),1)]
            def read():
                return fresh(self.openarm,self.joint_offsets,groups,1)
            def read_enable():
                if self._start_pending is None:
                    return read()
                pending, self._start_pending = self._start_pending, None
                return fresh(self.openarm,self.joint_offsets,groups,1,pending=pending)
            if self._mark_start_refresh or self._start_pending is None:
                raise RuntimeError("未取得官方启动刷新前的八轴计数")
            await_enable(read_enable, io.check)
            stage = "已使能/等待双臂回位许可"
            io.kind = "await_peer"
            io.emit("enabled")
            deadline = time.monotonic() + 1.
            while not io.peer_enabled:
                io.check()
                io.emit("feedback", **read())
                if time.monotonic() >= deadline:
                    raise TimeoutError("双臂回位许可超时")
                time.sleep(.005)
            io.check()
            self.last_command = np.asarray(read()["q"], dtype=float)
            self.last_command_time_s = time.monotonic()
            io.kind = "homing"
            stage = "官方回预设位"
            return super()._on_start()

        def send_position(self, position):
            io.check()  # Also interrupts the official homing interpolation between ticks.
            groups = [(self.openarm.get_arm(),7),(self.openarm.get_gripper(),1)]
            sample = fresh(self.openarm,self.joint_offsets,groups,1)
            self.observed = np.asarray(sample["q"])
            if io.kind != "running":
                io.emit("feedback", **sample)
            io.check()
            if io.kind == "running" and not 0 <= time.time_ns()-latest < 200_000_000:
                raise RuntimeError("action expired before dispatch")
            if io.kind=="running" and side=="left" and abs(position[4]-self.observed[4]) > .35:
                raise RuntimeError("左 J5 下发前指令与新反馈偏差超过 0.35 rad")
            scale = ramp.step(time.monotonic(),gravity_percent/100)
            tau = np.clip(scale*gravity.torque(sample["q"]), -CAP, CAP)
            self.openarm.arm.pending = sample["stamp"], latest if io.kind=="running" else sample["stamp"], tau
            try:
                sent = super().send_position(position)
            finally:
                self.openarm.arm.pending = None
            if io.kind != "running":
                return sent
            if sent:
                io.emit("feedback", **sample, anchor_ns=latest,
                        dispatch_ns=self.last_command_dispatch_timestamp_ns,
                        command=self.last_command.tolist(), scale=scale, tau_ff_nm=tau.tolist())
                if side=="left":
                    trace.append((latest, sample["stamp"], self.last_command_dispatch_timestamp_ns,
                                  float(target[4]), float(self.last_command[4]), float(self.observed[4])))
            return sent

    try:
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from vr_gravity import Bus, CAP, Gravity, Ramp
        gravity, ramp = Gravity(side), Ramp(immediate=True)
        cfg = MotionConfig("openarm_pedestal")
        limits = cfg.get_joint_limits(label)
        demo_pose = task_start_pose(side, limits)
        pose = np.asarray(cfg.get_start_config()["moves"][-1]["position"][label])
        bus, offsets = open_probe(side)
        # Read-only identity/state gate before opening the motion-mode driver.
        initial = fresh(bus, offsets)
        io.last_ping = time.monotonic()
        io.emit("checked", limits=limits.tolist(), **initial)
        while io.command != "home":
            io.check()
            io.emit("feedback", **fresh(bus, offsets))
            time.sleep(.015)
        del bus
        io.kind = "homing"
        stage = "初始化运动驱动，检查未使能"
        driver = CheckedDriver(label,cfg)
        driver.openarm = Bus(driver.openarm)
        groups = [(driver.openarm.get_arm(),7),(driver.openarm.get_gripper(),1)]
        fresh(driver.openarm,driver.joint_offsets,groups)
        io.check()
        stage = "官方启动/回预设位"
        driver._mark_start_refresh = True
        driver._start_pending = None
        if not driver.start():
            raise RuntimeError("official homing rejected")
        stage = "回位后等待反馈到位"
        target = pose
        home_end = time.monotonic()
        arrival = ArrivalWindow(home_end,2.)
        tick = home_end
        goal_since = None
        ready_since = None
        latest = 0
        trial_started = None
        while True:
            io.check()
            sample = fresh(driver.openarm,driver.joint_offsets,groups,1)
            q = np.array(sample["q"])
            now = time.monotonic()
            if io.kind == "home_ready" and io.command == "task_start":
                check_pose_hold(now, ready_since, q, pose)
                target = pose = demo_pose
                io.kind, stage = "approach", f"限速移动到 episode {DEMO_ID} 起点"
                home_end, goal_since = now, None
                arrival = ArrivalWindow(now,6.)
            if io.kind in ("homing", "approach"):
                official = io.kind == "homing"
                arrival.check(now,q,pose)
                ramp_done = official or np.allclose(driver.last_command, pose, atol=.001, rtol=0)
                if ramp_done and np.all(np.abs(q-pose) <= .08):
                    if goal_since is None: goal_since = now
                    if now-goal_since >= .20:
                        io.kind = "home_ready" if official else "ready"
                        stage = "官方启动位待命" if official else "示范起点待命"
                        ready_since = now
                        io.emit(io.kind)
                else:
                    goal_since = None
            if io.kind in ("ready", "home_ready"):
                check_pose_hold(now, ready_since, q, pose)
            if io.pending is not None:
                m, io.pending = io.pending, None
                target = accept_target(m["q"],q,limits,m["stamp"],time.time_ns(),
                                       side=side)
                latest = m["stamp"]
                if trial_started is None: trial_started = now
                io.kind = "running"
                stage = "执行策略"
            if io.kind == "running":
                if not 0 <= time.time_ns()-latest < 200_000_000:
                    raise RuntimeError("执行中预测已过期")
                if now-trial_started >= TRIAL_SECONDS:
                    raise RuntimeError("达到单次策略试验时长上限")
            if io.kind in ("running", "approach") and np.any(
                    np.abs(driver.last_command-q) > [.15]*7+[.55]):
                raise RuntimeError("跟踪偏差过大：" + pose_error(q, driver.last_command))
            # Keep both the pilot rates and the official driver's own checks active.
            dt = min(max(now-tick,0),.04)
            tick = now
            if io.kind == "running":
                cmd = policy_step(driver.last_command, target, dt)
            else:
                bound = np.array([JOINT_RATE]*7+[GRIP_RATE])*dt
                cmd = driver.last_command + np.clip(target-driver.last_command,-bound,bound)
            if not driver.send_position(cmd):
                raise RuntimeError("official driver rejected target")
            time.sleep(.014)
    except BaseException as exc:
        reason = f"{stage}: {str(exc) or type(exc).__name__}"
    finally:
        report("fault", reason=reason)
        if driver is not None:
            try:
                driver.stop()
                fresh(driver.openarm,driver.joint_offsets,
                      [(driver.openarm.get_arm(),7),(driver.openarm.get_gripper(),1)])
                report("disabled")
            except Exception as exc:
                report("unconfirmed", reason=str(exc))
        else:
            report("unconfirmed", reason="no motion driver opened; physical state not confirmed")
        # Only transmit diagnostics after the stop attempt; keep each packet below 8 KiB.
        if trace:
            rows = list(trace)
            for i in range(0,len(rows),24):
                report("left_j5_trace", columns=["anchor_ns","feedback_ns","dispatch_ns",
                       "prediction_rad","driver_command_rad","feedback_rad"], rows=rows[i:i+24])
        channel.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fd",type=int,required=True)
    p.add_argument("--side",choices=("left","right"),required=True)
    p.add_argument("--gravity-percent",type=int,choices=(0,100),default=DEFAULT_PERCENT)
    args = p.parse_args()
    def interrupted(*_):
        raise InterruptedError("control process interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    run(args.side,socket.socket(fileno=args.fd),args.gravity_percent)
