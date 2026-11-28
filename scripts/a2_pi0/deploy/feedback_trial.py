#!/usr/bin/env python3
"""Run a right-arm feedback trial or a separately authorized J7 trial."""
if __name__ == "__main__":
    raise SystemExit("旧真机入口已停用，请运行 scripts/a2_pi0/deploy/run_pi0.sh")

import json
import math
import os
import sys
import time

import dora
import pyarrow as pa

TEST_AXIS = 6  # Wrist J7.
TEST_TRAVEL = 0.005
START_TEST_TRAVEL = 0.0095
START_TEST_MEASURED = 0.002
ENCODER_TRIAL_TRAVEL = 0.00915  # Up to 24 MIT position counts, about 0.525 degrees.
ENCODER_TRIAL_WINDOW_S = 2.85  # Leave time to dispatch disable within 3 s.
ENCODER_READ_TIMEOUT_NS = 10_000_000
ENCODER_TRIAL_AUTHORIZATION = "RIGHT_J7_DUAL_ENCODER_POSITIVE_054_DEG"
ENCODER_KP15_AUTHORIZATION = "RIGHT_J7_DUAL_ENCODER_POSITIVE_054_DEG_KP15"
ENCODER_KP20_AUTHORIZATION = "RIGHT_J7_DUAL_ENCODER_POSITIVE_054_DEG_KP20"
ENCODER_KP20_RECOVERY_AUTHORIZATION = "RIGHT_J7_DUAL_ENCODER_POSITIVE_054_DEG_KP20_RECOVERY"
LEFT_ENCODER_TRIAL_AUTHORIZATION = "LEFT_J7_DUAL_ENCODER_NEGATIVE_054_DEG"
TEST_SPEED = 0.005
TEST_STEP = 0.00025
TX_FAULTS = ("write_net_down", "write_no_buffer", "write_other")
STARTUP_POSITION_TOLERANCE = 0.02
GRIPPER_DIAGNOSTIC_WINDOW_S = 0.06
GRIPPER_DIAGNOSTIC_INTERVAL_S = 0.01
GRIPPER_DIAGNOSTIC_SAMPLES = 3


def report(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def controller(motion=False, *, travel=TEST_TRAVEL, encoders=False, side="right"):
    node = dora.Node()
    direction = -1 if side == "left" else 1
    epoch = None
    started = last_feedback = None
    pending = False
    received = 0
    bus_baseline = None
    neutral = None
    ready_deadline = time.monotonic() + 2
    aligned = False
    target_offset = 0.0
    last_target = time.monotonic()
    completed = False
    rejected_position = None
    measured_travel = None
    try:
        for event in node:
            if event["type"] != "INPUT":
                if event["type"] in {"STOP", "INPUT_CLOSED"}:
                    raise RuntimeError("Feedback trial interrupted")
                continue
            now = time.monotonic()
            if encoders and started is not None and now - started >= ENCODER_TRIAL_WINDOW_S:
                completed = received > 0
                break
            if started is not None and now - started > (3 if motion else 2):
                raise TimeoutError("Trial exceeded its observation window")
            metadata = event["metadata"]
            if event["id"] == "status":
                status = event["value"][0].as_py()
                generation = metadata["start_epoch"]
                if status == "stopped" and generation == 0 and started is None:
                    started = now
                    node.send_output("command", pa.array(["start"]))
                elif status == "started" and generation == 1 and epoch is None:
                    epoch = generation
                elif motion and status == "aligned" and generation == epoch and not aligned:
                    aligned = True
                else:
                    raise RuntimeError("Unexpected driver status or epoch")
            elif event["id"] == "state":
                sample = event["value"][0].as_py()
                stamp = metadata["observation_timestamp"]
                positions = sample["qpos"]
                bus = sample["bus"]
                if (not pending or epoch is None or metadata["start_epoch"] != epoch
                        or not 0 <= time.time_ns() - stamp <= 100_000_000
                        or sample["motor_status"] != ["ENABLED"] * 8
                        or not bus["carrier"] or (bus_baseline is not None and bus != bus_baseline)
                        or any(len(sample[key]) != 8 or not all(map(math.isfinite, sample[key]))
                               for key in ("qpos", "qvel", "qtorque"))):
                    raise RuntimeError("Invalid or stale motor feedback")
                bus_baseline = bus
                if neutral is None:
                    neutral = positions
                for axis, (origin, actual) in enumerate(zip(neutral, positions, strict=True)):
                    expected = origin + (direction * target_offset if motion and axis == TEST_AXIS else 0)
                    if abs(actual - expected) > (0.01 if motion else 0.02):
                        rejected_position = dict(axis=axis + 1, expected_rad=expected,
                                                 measured_rad=actual, observation_timestamp=stamp,
                                                 start_epoch=epoch, qpos=positions)
                        raise RuntimeError(f"Unexpected position at axis {axis + 1}")
                last_feedback = now
                pending = False
                received += 1
                report("trial_feedback", observation_timestamp=stamp, start_epoch=epoch, **sample)
                if not motion and received == 10:
                    completed = True
                    break
                if motion:
                    measured = direction * (positions[TEST_AXIS] - neutral[TEST_AXIS])
                    if not encoders and (measured >= START_TEST_MEASURED if travel == START_TEST_TRAVEL else
                            target_offset == travel and abs(measured - travel) <= 0.002):
                        measured_travel = measured
                        completed = True
                        break
                    if aligned:
                        target_offset = min(travel, target_offset + min(
                            TEST_STEP, 0.8 * TEST_SPEED * (now - last_target)))
                    target = list(neutral)
                    target[TEST_AXIS] += direction * target_offset
                    issued = time.time_ns()
                    node.send_output("target", pa.array([{"qpos": target}], type=pa.struct([
                        ("qpos", pa.list_(pa.float32()))])),
                        {"start_epoch": epoch, "issued_ns": issued, "deadline_ns": issued + 50_000_000})
                    last_target = now
                    report("trial_target", observation_timestamp=stamp, qpos=target)
            elif started is None:
                if now > ready_deadline:
                    raise TimeoutError("No initial stopped status from the official driver")
            else:
                timeout = 0.15 if last_feedback is not None else 1.5
                reference = last_feedback if last_feedback is not None else started
                if now - reference > timeout:
                    raise TimeoutError("No fresh feedback from the official driver")
                if epoch is not None and not pending:
                    node.send_output("request", pa.array([True]))
                    pending = True
    finally:
        stop_requested_ns = time.time_ns()
        if started is not None:
            try:
                node.send_output("command", pa.array(["stop"]))
            finally:
                if rejected_position is not None:
                    report("trial_position_rejected", **rejected_position)
                if measured_travel is not None:
                    report("trial_motion_measured", delta_rad=measured_travel)
                report("trial_stop_requested", samples=received, stop_requested_ns=stop_requested_ns)
        else:
            report("trial_stop_requested", samples=received)
    if not completed:
        raise RuntimeError("Trial incomplete")


def driver(motion=False, *, travel=TEST_TRAVEL, encoders=False, side="right", j7_kp=None,
           record_recovery=False):
    import openarm_can as can
    import openarm_driver
    import numpy as np

    import guarded_official_node

    base = openarm_driver.SingleArmDriver
    interface = {"right": "can0", "left": "can1"}[side]
    direction = -1 if side == "left" else 1
    if not os.path.exists(f"/sys/class/net/{interface}/device"):
        raise RuntimeError(f"The expected physical CAN interface {interface} is absent")

    class FeedbackDriver(base):
        def __init__(self, arm_side, config, **kwargs):
            if j7_kp is not None:
                kps = config.get_default_kps().copy()
                kps[TEST_AXIS] = j7_kp
                kwargs["kps"] = kps
            super().__init__(arm_side, config, **kwargs)

        def check_transmit(self, phase, *, log=False):
            # The official Dora state message omits write_no_buffer and write_other.
            bus = self.openarm.get_bus_status()
            faults = {name: getattr(bus, name).count for name in TX_FAULTS}
            if log or any(faults.values()):
                report("trial_can_transmit", phase=phase, faults=faults,
                       writes_ok=bus.writes_ok, last_write_errno=bus.last_write_errno)
            if any(faults.values()):
                raise RuntimeError(f"CAN transmit failed during {phase}: {faults}")

        def start(self):
            # The pinned driver reads once after enable_all and once after its
            # first refresh_all. Inspect each read without replacing startup.
            if encoders:
                self.enable_started_ns = time.monotonic_ns()
                report("trial_encoder_enable_begin", started_ns=self.enable_started_ns)
            self.startup_phases = iter(("enable", "initial_refresh"))
            try:
                return super().start()
            finally:
                self.startup_phases = None
                self.initial_refresh_axes = self.initial_refresh_before = None

        def set_latest_state(self, timeout_us=300):
            timing = getattr(self, "stop_timing", None)
            first_stop_read = timing is not None and "disable_returned_ns" not in timing
            if first_stop_read:
                # The pinned stop implementation calls this immediately after disable_all returns.
                timing["disable_returned_ns"] = time.time_ns()
            super().set_latest_state(timeout_us=timeout_us)
            if first_stop_read:
                timing["initial_read_ns"] = time.time_ns()
                timing["initial_qpos_unverified"] = [float(p) if math.isfinite(p) else None
                                                     for p in self.latest_state["qpos"]]
                if record_recovery:
                    for delay in (0, 0.01, 0.02, 0.04, 0.08):
                        if delay:
                            time.sleep(delay)
                        if not self.read_recovery_encoders("disable_dispatched"):
                            break
            phases = getattr(self, "startup_phases", None)
            phase = next(phases, None) if phases is not None else None
            if phase is not None:
                self.check_transmit(phase, log=True)
            if phase == "enable":
                # The pinned driver sends its first refresh immediately after this read.
                self.openarm.recv_all(0)
                self.initial_refresh_axes = self.feedback_axes()
                self.initial_refresh_before = [c.get_link_stats(i).responses
                                               for c, i in self.initial_refresh_axes]
            elif phase == "initial_refresh":
                state = self.wait_for_feedback(self.initial_refresh_axes, self.initial_refresh_before)
                _, bus = self.get_health()
                _, _, rejected_axes = self.startup_position(state)
                if bus != self.pre_enable_bus or not self.openarm.is_link_running() or rejected_axes:
                    raise RuntimeError("Initial refresh position or bus check failed")

        def fetch_state(self, refresh=True):
            if not self.started:
                return super().fetch_state(refresh=refresh)
            if encoders:
                self.check_encoder_trial_time()
            state = self.fresh_state(refresh, deadline=time.monotonic() + 0.025 if encoders else None)
            self.check_transmit("running_feedback")
            if encoders:
                status, bus = self.get_health()
                positions = state["qpos"]
                other_axes = np.arange(len(positions)) != TEST_AXIS
                if (status != ["ENABLED"] * 8 or bus != self.pre_enable_bus
                        or not self.openarm.is_link_running() or not np.isfinite(positions).all()
                        or np.any(np.abs(positions[other_axes] - self.pre_enable_position[other_axes]) > 0.01)):
                    report("trial_encoder_feedback_rejected", status=status, bus=bus,
                           qpos=[float(p) if math.isfinite(p) else None for p in positions])
                    raise RuntimeError("Unexpected position, status or CAN fault during encoder trial")
                self.read_j7_encoders("running")
                self.check_encoder_trial_time()
            self.last_fresh_ns = time.monotonic_ns()
            if motion and not hasattr(self, "motion_origin"):
                self.motion_origin = state["qpos"].astype(np.float32).astype(float)
                self.motion_last = self.motion_origin.copy()
                self.motion_last_ns = self.last_fresh_ns
            return state

        def feedback_axes(self):
            return [(self._health_collections[collection], index)
                    for _, collection, index in self._health_axes]

        def fresh_state(self, refresh=True, *, deadline=None, transmit_phase=None):
            axes = self.feedback_axes()
            self.openarm.recv_all(0)
            before = [collection.get_link_stats(index).responses for collection, index in axes]
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Feedback time limit reached before refresh")
            if refresh:
                self.openarm.refresh_all()
            if transmit_phase is not None:
                self.check_transmit(transmit_phase, log=True)
            return self.wait_for_feedback(axes, before, deadline=deadline)

        def wait_for_feedback(self, axes, before, *, deadline=None):
            if deadline is None:
                deadline = time.monotonic() + 0.06
            malformed_baseline = getattr(self, "mode_readback_malformed", (0,) * len(axes))
            while True:
                stats = [collection.get_link_stats(index) for collection, index in axes]
                if any(axis.malformed_frames != old for axis, old in zip(stats, malformed_baseline, strict=True)):
                    raise RuntimeError("Malformed motor feedback")
                remaining_us = int((deadline - time.monotonic()) * 1e6)
                if remaining_us <= 0:
                    break
                if all(axis.responses > previous for axis, previous in zip(stats, before, strict=True)):
                    return self.latest_state
                self.set_latest_state(timeout_us=min(3000, remaining_us))
            missing = [i for i, (axis, previous) in enumerate(zip(stats, before, strict=True), 1)
                       if axis.responses <= previous]
            if not missing:
                raise TimeoutError("Fresh feedback arrived after the deadline")
            raise RuntimeError(f"Axes without fresh feedback: {missing}")

        def pre_enable_check(self):
            self.check_transmit("before_pre_enable")
            state = self.fresh_state()
            self.check_transmit("pre_enable_feedback", log=True)
            status, bus = self.get_health()
            if (status != ["DISABLED"] * 8 or not bus["carrier"]
                    or not self.openarm.is_link_running()
                    or len(state["qpos"]) != 8 or not all(map(math.isfinite, state["qpos"]))):
                raise RuntimeError("Pre-enable feedback check failed")
            self.confirm_gripper_mode()
            _, bus_after_mode_check = self.get_health()
            if bus_after_mode_check != bus or not self.openarm.is_link_running():
                raise RuntimeError("CAN link changed while reading back the gripper mode")
            self.pre_enable_position = state["qpos"].copy()
            self.pre_enable_bus = bus
            if encoders:
                if any(count for name, count in bus.items() if name != "carrier"):
                    raise RuntimeError("CAN fault before encoder trial")
                self.encoder_arm = can.OpenArm(interface, True)
                self.encoder_arm.init_arm_motors([can.MotorType.DM4310], [7], [0x17], [])
                self.encoder_arm.set_callback_mode_all(can.CallbackMode.PARAM)
                self.read_j7_encoders("disabled")
            if motion:
                limits = self.config.get_joint_limits(self.arm_side)
                end = self.pre_enable_position.copy()
                end[TEST_AXIS] += direction * travel
                if (np.any(np.minimum(self.pre_enable_position, end) < limits[:, 0])
                        or np.any(np.maximum(self.pre_enable_position, end) > limits[:, 1])):
                    raise RuntimeError("Insufficient range for the approved J7 trial")
            report("trial_pre_enable", qpos=list(self.pre_enable_position))

        def check_encoder_trial_time(self):
            if time.monotonic_ns() - self.enable_started_ns >= 2_950_000_000:
                raise TimeoutError(f"{side.capitalize()} J7 encoder trial deadline")

        def check_trial_event(self):
            if encoders and self.started:
                self.check_encoder_trial_time()
                last_feedback = getattr(self, "last_fresh_ns", self.enable_started_ns)
                if time.monotonic_ns() - last_feedback >= 100_000_000:
                    raise TimeoutError(f"{side.capitalize()} J7 trial lost fresh feedback")

        def read_j7_encoders(self, phase):
            devices = self.encoder_arm.get_arm()
            self.openarm.set_callback_mode_all(can.CallbackMode.IGNORE)
            try:
                self.encoder_arm.recv_all(0)
                reading = {}
                for name in ("p_m", "xout"):
                    register = getattr(can.MotorVariable, name).value
                    stats = devices.get_link_stats(0)
                    responses_before = stats.responses
                    malformed_before = stats.malformed_frames
                    sent_ns = time.monotonic_ns()
                    self.encoder_arm.query_param_all(register)
                    deadline = sent_ns + ENCODER_READ_TIMEOUT_NS
                    while time.monotonic_ns() < deadline:
                        remaining_us = (deadline - time.monotonic_ns()) // 1000
                        self.encoder_arm.recv_all(max(0, min(remaining_us, 1000)))
                        received_ns = time.monotonic_ns()
                        if stats.responses == responses_before:
                            continue
                        value = devices.get_motors()[0].get_param(register)
                        if (stats.responses != responses_before + 1
                                or stats.malformed_frames != malformed_before
                                or not math.isfinite(value) or received_ns >= deadline):
                            raise RuntimeError(f"Invalid {side} J7 {name} feedback")
                        reading[name] = dict(rad=value, sent_ns=sent_ns, observed_ns=received_ns)
                        break
                    else:
                        raise TimeoutError(f"No fresh {side} J7 {name} feedback")
                bus = self.encoder_arm.get_bus_status()
                if (not self.encoder_arm.is_link_running()
                        or any(getattr(bus, name).count for name in self.pre_enable_bus if name != "carrier")):
                    raise RuntimeError(f"CAN fault while reading {side} J7 encoders")
                report("trial_j7_encoder_reading", phase=phase, **reading)
            finally:
                # Parameter replies share J7's state CAN ID.
                self.openarm.recv_all(0)
                self.openarm.set_callback_mode_all(can.CallbackMode.STATE)

        def read_recovery_encoders(self, phase):
            try:
                self.read_j7_encoders(phase)
            except (RuntimeError, OSError) as exc:
                report("trial_j7_recovery_incomplete", phase=phase, reason=str(exc))
                return False
            return True

        def confirm_gripper_mode(self):
            collection, index = self.feedback_axes()[-1]
            register = can.MotorVariable.CTRL_MODE.value
            expected = can.ControlMode.POS_FORCE if self.gripper_posforce else can.ControlMode.MIT
            self.openarm.recv_all(0)
            stats = collection.get_link_stats(index)
            responses_before = stats.responses
            if collection.get_motors()[index].get_param(register) != -1:
                raise RuntimeError("Gripper mode readback is already cached")

            self.openarm.set_callback_mode_all(can.CallbackMode.PARAM)
            try:
                query = self.openarm.get_gripper() if self.gripper_posforce else self.openarm.get_arm()
                query.query_param_all(register)
                self.check_transmit("gripper_mode_readback")
                deadline = time.monotonic() + 0.06
                while time.monotonic() < deadline:
                    if stats.responses > responses_before:
                        actual = collection.get_motors()[index].get_param(register)
                        # A delayed state frame can share this reply ID without
                        # populating the control-mode register.
                        if actual == expected.value:
                            self.openarm.recv_all(0)
                            self.mode_readback_malformed = tuple(
                                devices.get_link_stats(i).malformed_frames for devices, i in self.feedback_axes()
                            )
                            report("trial_gripper_mode_confirmed", mode=expected.name)
                            return
                        if actual != -1:
                            report("trial_gripper_mode_mismatch", expected=expected.name,
                                   actual=actual if math.isfinite(actual) else None)
                            raise RuntimeError("Gripper control mode did not match the driver configuration")
                    remaining_us = int((deadline - time.monotonic()) * 1e6)
                    if remaining_us > 0:
                        self.openarm.recv_all(min(3000, remaining_us))
            finally:
                self.openarm.set_callback_mode_all(can.CallbackMode.STATE)
            raise TimeoutError("No fresh gripper mode response before enable")

        def startup_position(self, state):
            positions = [float(p) if math.isfinite(p) else None for p in state["qpos"]]
            changes = [None if actual is None else actual - float(origin)
                       for origin, actual in zip(self.pre_enable_position, positions, strict=True)]
            rejected_axes = [
                axis for axis, change in enumerate(changes, 1)
                if change is None or abs(change) > (
                    0.01 if encoders and axis != TEST_AXIS + 1 else STARTUP_POSITION_TOLERANCE
                )
            ]
            return positions, changes, rejected_axes

        def inspect_gripper_startup(self, baseline_bus):
            started = time.monotonic()
            deadline = started + GRIPPER_DIAGNOSTIC_WINDOW_S
            reason = "gripper_still_disabled"
            for sample in range(1, GRIPPER_DIAGNOSTIC_SAMPLES + 1):
                if time.monotonic() + GRIPPER_DIAGNOSTIC_INTERVAL_S >= deadline:
                    reason = "deadline"
                    break
                time.sleep(GRIPPER_DIAGNOSTIC_INTERVAL_S)
                try:
                    state = self.fresh_state(deadline=deadline)
                    self.check_transmit("gripper_startup_diagnostic")
                except (RuntimeError, TimeoutError) as exc:
                    report("trial_gripper_startup_incomplete", sample=sample, reason=str(exc))
                    return False
                status, bus = self.get_health()
                positions, changes, rejected_axes = self.startup_position(state)
                link_running = self.openarm.is_link_running()
                checked_at = time.monotonic()
                elapsed_ms = round((checked_at - started) * 1000, 2)
                report("trial_gripper_startup_sample", sample=sample,
                       elapsed_ms=elapsed_ms, motor_status=status, bus=bus,
                       link_running=link_running, qpos=positions,
                       change_rad=changes, rejected_position_axes=rejected_axes)
                if checked_at >= deadline:
                    reason = "deadline"
                elif bus != baseline_bus or not link_running:
                    reason = "bus_or_link_changed"
                elif rejected_axes:
                    reason = "position_changed"
                elif status == ["ENABLED"] * 8:
                    report("trial_gripper_startup_confirmed", sample=sample, elapsed_ms=elapsed_ms)
                    return True
                elif status == ["ENABLED"] * 7 + ["DISABLED"]:
                    continue
                else:
                    reason = "unexpected_motor_status"
                break
            report("trial_gripper_startup_incomplete", reason=reason,
                   elapsed_ms=round((time.monotonic() - started) * 1000, 2))
            return False

        def _on_start(self):
            state = self.fresh_state(transmit_phase="startup_feedback")
            status, bus = self.get_health()
            positions, changes, rejected_axes = self.startup_position(state)
            bus_ok = (bus["carrier"] and bus == self.pre_enable_bus
                      and self.openarm.is_link_running())
            if status == ["ENABLED"] * 8 and bus_ok and not rejected_axes:
                return True
            snapshot = dict(motor_status=status, bus=bus, qpos=positions, bus_ok=bus_ok,
                            change_rad=changes, rejected_position_axes=rejected_axes)
            if status == ["ENABLED"] * 7 + ["DISABLED"] and bus_ok and not rejected_axes:
                report("trial_gripper_startup_wait", **snapshot)
                return self.inspect_gripper_startup(bus)
            report("trial_startup_rejected", **snapshot)
            return False

        def _on_stop(self):
            return True

        def send_position(self, position):
            if not motion:
                raise RuntimeError("Position commands are disabled in the feedback trial")
            target = np.asarray(position, dtype=float)
            now = time.monotonic_ns()
            limits = self.config.get_joint_limits(self.arm_side)
            if (not hasattr(self, "motion_origin") or now - self.last_fresh_ns > 100_000_000
                    or target.shape != (8,) or not np.isfinite(target).all()
                    or np.any(target < limits[:, 0]) or np.any(target > limits[:, 1])):
                raise RuntimeError("Invalid position target or stale driver feedback")
            delta = target - self.motion_origin
            step = direction * (target[TEST_AXIS] - self.motion_last[TEST_AXIS])
            if (np.any(np.delete(delta, TEST_AXIS) != 0)
                    or not -1e-7 <= direction * delta[TEST_AXIS] <= travel + 1e-7
                    or not -1e-7 <= step <= min(TEST_STEP, TEST_SPEED * (now - self.motion_last_ns) / 1e9) + 1e-5):
                raise RuntimeError("Position target violates single-joint motion bounds")
            self.check_transmit("before_position")
            if encoders:
                self.check_encoder_trial_time()
                _, bus = self.get_health()
                if bus != self.pre_enable_bus or not self.openarm.is_link_running():
                    raise RuntimeError(f"CAN fault before sending {side} J7 target")
            dispatch_ns = time.monotonic_ns()
            accepted = super().send_position(target)
            self.check_transmit("position")
            if not accepted:
                raise RuntimeError("Official driver rejected the position target")
            self.motion_last = target
            self.motion_last_ns = now
            report("trial_driver_target", requested_rad=target.tolist(),
                   applied_rad=self.last_command.tolist(), dispatch_ns=dispatch_ns)
            return True

        def stop(self):
            timing = {"stop_entered_ns": time.time_ns()}
            self.stop_timing = timing
            try:
                super().stop()
                state = self.fresh_state()
                observed_ns = time.time_ns()
                status, bus = self.get_health()
                details = dict(observation_timestamp=observed_ns,
                               qpos=[float(p) if math.isfinite(p) else None for p in state["qpos"]])
                if status != ["DISABLED"] * 8 or not bus["carrier"]:
                    report("trial_disable_unconfirmed", motor_status=status, bus=bus, **details)
                    raise RuntimeError("Eight disabled motors could not be confirmed after stop")
                report("trial_disabled_confirmed", motor_status=status, bus=bus, **details)
                if record_recovery:
                    self.read_recovery_encoders("disabled_confirmed")
            finally:
                self.stop_timing = None
                report("trial_stop_timing", **timing)

    openarm_driver.SingleArmDriver = FeedbackDriver
    sys.argv = [sys.argv[0], "--side", side, "--can-interface", interface,
                "--config", "openarm_pedestal", "--stop", "--no-start-on-startup"]
    try:
        guarded_official_node.run()
    finally:
        openarm_driver.SingleArmDriver = base
