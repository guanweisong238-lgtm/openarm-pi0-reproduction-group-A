"""Read a live approach plan or run one explicitly selected episode-0 phase."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import openarm_can as can
from openarm_driver import Config, SingleArmDriver

from dp_paths import ROOT
from live_arm_probe import await_enable, fresh, open_probe


SIDES = ("right", "left")
INTERFACES = ("can0", "can1")
AXES = [f"{side}_J{i}" for side in SIDES for i in range(1, 8)]
ARTIFACTS = ROOT / "results/real_20261128_demo0_replay_preflight"
FILES = {"short": "first_second_offline.npz", "full": "full_demo_offline.npz"}
DT = .02
START_DRIFT = .01
AT_START = .05
TRACK_LIMIT = .10
GRIP_LIMIT = .02
LATE_LIMIT = .015
PLAN_AGE_NS = 300_000_000_000


def arm_limits(config):
    return np.vstack([config.get_joint_limits(f"{side}_arm")[:7] for side in SIDES])


def check_interfaces(config):
    rows = json.loads(subprocess.check_output(["ip", "-j", "-details", "link", "show", "type", "can"]))
    interfaces = {row['ifname']: row for row in rows}
    expected_parent = os.environ.get("OPENARM_PI0_CAN_PARENT", "1-3.1:1.0")
    for channel, (side, name) in enumerate(zip(SIDES, INTERFACES, strict=True)):
        row = interfaces.get(name, {})
        if config.get_can_interface(f"{side}_arm") != name or (row.get("parentbus"), row.get("parentdev")) != ("usb", expected_parent):
            raise RuntimeError(f"{side}: {name} 未匹配本机 USB-CAN 配置")
        net = Path("/sys/class/net") / name
        if int((net / "dev_id").read_text(), 0) != channel:
            raise RuntimeError(f"{name} 的物理通道与 {side} 臂不符")
        usb = (net / "device").resolve().parent
        if ((usb / "idVendor").read_text().strip(), (usb / "idProduct").read_text().strip()) != ("0c72", "0011"):
            raise RuntimeError(f"{name} 不是已核对的 PEAK 双路适配器")
        if "UP" not in row["flags"]:
            raise RuntimeError(f"{name} 为 DOWN，需要现场先配置两条 CAN FD 接口")
        cfg = row.get("linkinfo", {}).get("info_data", {})
        bit, data = cfg.get("bittiming", {}), cfg.get("data_bittiming", {})
        if (cfg.get("ctrlmode") != ["FD"] or cfg.get("state") != "ERROR-ACTIVE" or cfg.get("restart_ms") != 0
                or (bit.get("bitrate"), bit.get("sample_point")) != (1_000_000, .75)
                or (data.get("bitrate"), data.get("sample_point"), data.get("sjw")) != (5_000_000, .75, 2)):
            raise RuntimeError(f"{name} 的 CAN FD 速率、采样点或状态不符")


@contextmanager
def exclusive():
    directory = Path.home() / ".cache/openarm-dp-field"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "verify.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def check_path(q, limits, speed, accel):
    if q.ndim != 2 or q.shape[1] != 14 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("轨迹格式不是至少两帧有限的双臂七轴角度")
    if np.any((q < limits[:, 0]) | (q > limits[:, 1])):
        raise ValueError("轨迹超出驱动关节角度限位")
    rates = np.diff(q, axis=0) / DT
    acceleration = np.diff(np.vstack((np.zeros((1, 14)), rates, np.zeros((1, 14)))), axis=0) / DT
    peak_rate = float(np.max(abs(rates)))
    peak_accel = float(np.max(abs(acceleration)))
    if peak_rate > speed or peak_accel > accel:
        raise ValueError(f"轨迹超限：速度 {peak_rate:.4f} rad/s，加速度 {peak_accel:.4f} rad/s²")
    return dict(duration_s=(len(q)-1)*DT, peak_speed_rad_s=peak_rate, peak_accel_rad_s2=peak_accel,
                min_rad=q.min(0).tolist(), max_rad=q.max(0).tolist())


def approach_path(start, target, limits):
    delta = target-start
    travel = float(np.max(abs(delta)))
    duration = max(2., 1.875*travel/.085, math.sqrt((10/math.sqrt(3))*travel/.095))
    steps = math.ceil(duration/DT)
    u = np.linspace(0., 1., steps+1)
    blend = u**3*(10 + u*(-15 + 6*u))
    path = start + blend[:, None]*delta
    path[-1] = target
    return path, check_path(path, limits, .09, .10)


def load_replay(stage, limits):
    path = ARTIFACTS / FILES[stage]
    data = path.read_bytes()
    with np.load(BytesIO(data), allow_pickle=False) as file:
        q, ts = file["q_rad"], file["time_s"]
        if (file["axes"].tolist() != AXES or ts.shape != (len(q),)
                or not np.allclose(ts, np.arange(len(q))*DT, rtol=0, atol=1e-8)):
            raise ValueError(f"{path.name} 的轴顺序或时间不符")
    summary = check_path(q, limits, .20, .50)
    return q, {**summary, "sha256": hashlib.sha256(data).hexdigest()}


def pair(samples):
    stamps = [sample["stamp"] for sample in samples]
    now = time.time_ns()
    if max(stamps)-min(stamps) > 50_000_000 or not all(0 <= now-stamp <= 100_000_000 for stamp in stamps):
        raise RuntimeError("双臂反馈不同步或已经过期")
    q = np.asarray([sample["q"] for sample in samples])
    if q.shape != (2, 8) or not np.isfinite(q).all():
        raise ValueError("双臂角度反馈格式异常")
    return q


def read_pose():
    sockets = [open_probe(side) for side in SIDES]
    return pair([fresh(bus, offsets) for bus, offsets in sockets])


def make_plan(q, limits):
    short, short_info = load_replay("short", limits)
    route, approach = approach_path(q[:, :7].ravel(), short[0], limits)
    checkpoints = np.linspace(0, len(route)-1, 5, dtype=int)
    approach["checkpoints"] = [{"elapsed_s": float(i*DT), "q_rad": route[i].tolist()} for i in checkpoints]
    return dict(episode=0, read_ns=time.time_ns(), axes=AXES, disabled_q_rad=q.tolist(),
                first_frame_rad=short[0].tolist(), first_frame_source_sha256=short_info["sha256"],
                approach=approach, field_clearance_verified=False)


def validate_plan(plan, q, limits):
    if (plan.get("episode") != 0 or plan.get("axes") != AXES or type(plan.get("read_ns")) is not int
            or not 0 <= time.time_ns()-plan["read_ns"] <= PLAN_AGE_NS):
        raise ValueError("需要最近 5 分钟内的第 0 条现场读回计划")
    reference = np.asarray(plan["disabled_q_rad"], dtype=float)
    if reference.shape != (2, 8) or not np.isfinite(reference).all() or np.max(abs(reference-q)) > START_DRIFT:
        raise ValueError("当前角度与审核计划相差超过 0.01 rad，请重新读回和核对路线")
    short, info = load_replay("short", limits)
    if info["sha256"] != plan.get("first_frame_source_sha256") or not np.array_equal(short[0], plan["first_frame_rad"]):
        raise ValueError("试跑轨迹已改变，请重新读回和核对路线")
    return reference, short[0]


class ReplayDriver(SingleArmDriver):
    def groups(self):
        return [(self.openarm.get_arm(), 7), (self.openarm.get_gripper(), 1)]

    def read(self, status):
        self.openarm.recv_all(0)
        return fresh(self.openarm, self.joint_offsets, self.groups(), status)

    def verify_modes(self):
        register = can.MotorVariable.CTRL_MODE.value
        collections = self.groups()
        self.openarm.recv_all(0)
        before = [[dev.get_link_stats(i).responses for i in range(n)] for dev, n in collections]
        self.openarm.set_callback_mode_all(can.CallbackMode.PARAM)
        try:
            for dev, _ in collections:
                dev.query_param_all(register)
            deadline = time.monotonic() + .06
            while time.monotonic() < deadline:
                self.openarm.recv_all(1000)
                if not self.openarm.is_bus_healthy():
                    break
                if all(dev.get_link_stats(i).responses > counts[i]
                       for (dev, n), counts in zip(collections, before, strict=True) for i in range(n)):
                    modes = [motor.get_param(register) for dev, _ in collections for motor in dev.get_motors()]
                    if modes == [can.ControlMode.MIT.value]*7 + [can.ControlMode.POS_FORCE.value]:
                        return
                    if all(mode != -1 for mode in modes):
                        break
        finally:
            self.openarm.recv_all(0)
            self.openarm.set_callback_mode_all(can.CallbackMode.STATE)
        raise RuntimeError(f"{self.arm_side}: MIT／POS_FORCE 模式未读回确认")

    def _on_start(self):
        # The official start sequence enables motors. Home motion is a separate phase.
        sample = await_enable(lambda: self.read(1), lambda: None)
        self.last_command = np.asarray(sample["q"])
        return True


def track(q, target, grippers):
    if np.max(abs(q[:, :7].ravel()-target)) > TRACK_LIMIT or np.max(abs(q[:, 7]-grippers)) > GRIP_LIMIT:
        raise RuntimeError("双臂关节跟踪或夹爪保持偏差过大")


def play(drivers, path, hold, phase, *, gravity=None):
    prior = path[0]
    settled_since = None
    ticks = len(path) + math.ceil(3/DT)
    start_time = time.monotonic()
    for index in range(ticks):
        deadline = start_time + index*DT
        time.sleep(max(0., deadline-time.monotonic()))
        q = pair([driver.read(1) for driver in drivers])
        track(q, prior, hold)
        if index >= len(path) and np.max(abs(q[:, :7].ravel()-path[-1])) <= AT_START:
            settled_since = time.monotonic() if settled_since is None else settled_since
            if time.monotonic()-settled_since >= .20:
                return q
        else:
            settled_since = None
        target = path[min(index, len(path)-1)]
        for i, driver in enumerate(drivers):
            if time.monotonic()-deadline > LATE_LIMIT:
                raise TimeoutError(f"{phase}: 控制节拍超时，停止发送目标")
            position = np.r_[target[i*7:(i+1)*7], hold[i]]
            if gravity is not None:
                model, cap = gravity[i]
                stamp = time.time_ns()
                driver.openarm.arm.pending = stamp, stamp, np.clip(model.torque(q[i]), -cap, cap)
            try:
                if not driver.send_position(position) or not np.allclose(driver.last_command, position, rtol=0, atol=1e-6):
                    raise RuntimeError(f"{driver.arm_side}: 驱动拒绝或裁剪目标")
            finally:
                if gravity is not None:
                    driver.openarm.arm.pending = None
            if not driver.openarm.is_bus_healthy():
                raise RuntimeError(f"{driver.arm_side}: CAN 故障")
        prior = target
    raise TimeoutError(f"{phase}: 双臂未稳定到位")


def stop_both(drivers):
    for driver in drivers:
        try:
            driver.openarm.disable_all()
        except Exception as exc:
            print(f"{driver.arm_side} 请求失能失败：{exc}", file=sys.stderr)
    unconfirmed = []
    for driver in drivers:
        try:
            driver.read(0)
        except Exception as exc:
            unconfirmed.append(driver.arm_side)
            print(f"{driver.arm_side} 失能未确认：{exc}", file=sys.stderr)
        else:
            print(f"{driver.arm_side} 八轴报告失能", flush=True)
    if unconfirmed:
        raise RuntimeError("未确认双臂失能，请使用现场硬件停机")


def execute(stage, path, reference, config):
    if not config.get_gripper_posforce():
        raise RuntimeError("需要 pedestal 七轴 MIT 和夹爪 POS_FORCE 配置")
    gripper_limits = np.asarray([config.get_joint_limits(f"{side}_arm")[7] for side in SIDES])
    # Use the workbench's existing capped gravity torque. No model contact checks.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vr_gravity import Bus, CAP, Gravity

    gravity = [(Gravity(side), CAP) for side in SIDES]
    drivers = []
    armed = False
    try:
        for side in SIDES:
            driver = ReplayDriver(f"{side}_arm", config)
            driver.openarm = Bus(driver.openarm)
            drivers.append(driver)
            driver.verify_modes()
        q = pair([driver.read(0) for driver in drivers])
        if np.max(abs(q-reference)) > START_DRIFT:
            raise RuntimeError("连接驱动后，关节角已偏离审核计划")
        if np.any((q[:, 7] < gripper_limits[:, 0]) | (q[:, 7] > gripper_limits[:, 1])):
            raise RuntimeError("当前夹爪角度超出保持范围")
        armed = True
        for driver in drivers:
            if not driver.start():
                raise RuntimeError(f"{driver.arm_side}: 使能检查失败")
        q = pair([driver.read(1) for driver in drivers])
        if np.max(abs(q-reference)) > START_DRIFT:
            raise RuntimeError("使能后关节角偏离审核起点")
        if np.any((q[:, 7] < gripper_limits[:, 0]) | (q[:, 7] > gripper_limits[:, 1])):
            raise RuntimeError("使能后夹爪角度超出保持范围")
        first = path[-1] if stage == "approach" else path[0]
        entry, _ = approach_path(q[:, :7].ravel(), first, arm_limits(config))
        # Start driving from the confirmed enabled posture, with unchanged J8 targets.
        final = play(drivers, entry, q[:, 7], "approach", gravity=gravity)
        if stage != "approach":
            final = play(drivers, path, q[:, 7], stage, gravity=gravity)
        print(json.dumps({"phase": stage, "final_q_rad": final.tolist()}, allow_nan=False), flush=True)
    finally:
        if armed:
            blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
            old = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
            try:
                stop_both(drivers)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="离线检查两份回放文件")
    read = commands.add_parser("read", help="只读取双臂角度，生成去第一帧的计划")
    read.add_argument("--plan", type=Path, required=True)
    run = commands.add_parser("run", help="现场确认后执行一个阶段")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--stage", choices=("approach", "short", "full"), required=True)
    args = parser.parse_args()
    config = Config("openarm_pedestal")
    limits = arm_limits(config)
    if args.command == "check":
        print(json.dumps({stage: load_replay(stage, limits)[1] for stage in FILES}, indent=2))
        return
    with exclusive():
        check_interfaces(config)
        if args.command == "read":
            plan = make_plan(read_pose(), limits)
            with args.plan.open("x") as out:
                json.dump(plan, out, indent=2, allow_nan=False)
            print(f"双臂八轴均报告失能。路线摘要已保存到 {args.plan}，现场核对路径净空。")
            return
        if not sys.stdin.isatty():
            raise RuntimeError("运动需从现场交互终端启动")
        plan = json.loads(args.plan.read_text())
        preview_reference, first = validate_plan(plan, read_pose(), limits)
        if args.stage == "approach":
            path, summary = approach_path(preview_reference[:, :7].ravel(), first, limits)
        else:
            path, summary = load_replay(args.stage, limits)
            if not np.array_equal(path[0], first) or np.max(abs(preview_reference[:, :7].ravel()-first)) > AT_START:
                raise RuntimeError("仅从第 0 条第一帧附近开始播放，请先到位")
        print(json.dumps({"stage": args.stage, "summary": summary}, indent=2), flush=True)
        token = f"RUN {args.stage.upper()}"
        if input(f"现场已核对路线、承托和实体急停，输入 {token} 才使能：") != token:
            raise RuntimeError("未确认使能")
        reference, _ = validate_plan(plan, read_pose(), limits)
        execute(args.stage, path, reference, config)


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise InterruptedError(f"控制进程收到信号 {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(str(exc) or type(exc).__name__) from exc
