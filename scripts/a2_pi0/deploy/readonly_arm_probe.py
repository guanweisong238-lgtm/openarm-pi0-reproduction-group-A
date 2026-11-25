#!/usr/bin/env python3
"""Read identity registers and fresh motor telemetry on one authorized CAN FD interface."""
from __future__ import annotations

import argparse
from importlib import resources
import json
import math
import os
import time

import openarm_can as can
from openarm_driver import Config


IDENTITY = ("ESC_ID", "MST_ID", "hw_ver", "sw_ver", "sub_ver", "SN", "CTRL_MODE")
BUS_FAULTS = ("bus_off", "error_passive", "error_warning", "ack_error", "tx_timeout",
              "tx_overflow", "rx_overflow", "write_net_down", "write_no_buffer", "write_other")


def counts(bus, names):
    return {name: getattr(bus, name).count for name in names}


def receive(arm, ready):
    deadline = time.monotonic() + 0.06
    while time.monotonic() < deadline:
        arm.recv_all(3000)
        if ready() or any(counts(arm.get_bus_status(), BUS_FAULTS).values()):
            break


def inspect(interface):
    bundled = resources.files("openarm_driver").joinpath("configs", "openarm_pedestal.yaml")
    with resources.as_file(bundled) as path:
        config = Config(path)
    send_ids = config.get_send_ids()
    recv_ids = config.get_recv_ids()
    motor_types = [getattr(can.MotorType, name) for name in config.get_motor_types()]
    arm = can.OpenArm(interface, True)

    # Register every axis, including J8, in the library's host-side table only.
    # Empty control_modes emits no mode command, unlike init_gripper_motor().
    arm.init_arm_motors(motor_types, send_ids, recv_ids, [])
    devices = arm.get_arm()
    arm.set_callback_mode_all(can.CallbackMode.STATE)

    def responses():
        return [devices.get_link_stats(index).responses for index in range(len(send_ids))]

    before = responses()
    requested_ns = time.time_ns()
    arm.refresh_all()
    receive(arm, lambda: all(after > old for after, old in zip(responses(), before, strict=True)))
    observed_ns = time.time_ns()
    motors = devices.get_motors()

    def feedback(index, motor):
        stats = devices.get_link_stats(index)
        values = [motor.get_position(), motor.get_velocity(), motor.get_torque()]
        fresh = (stats.responses > before[index] and not stats.malformed_frames
                 and all(map(math.isfinite, values)))
        return {
            "index": index + 1, "query_id": send_ids[index], "reply_id": recv_ids[index],
            "fresh": fresh, "responses": stats.responses - before[index],
            "response_age_ms": round(stats.seconds_since_response() * 1000, 3) if fresh else None,
            "motor_position_rad": values[0] if fresh else None,
            "motor_velocity_rad_s": values[1] if fresh else None,
            "motor_torque_nm": values[2] if fresh else None,
            "motor_status": can.motor_error_to_string(motor.get_error_code()) if fresh else None,
            "identity": {},
        }

    axes = [feedback(index, motor) for index, motor in enumerate(motors)]
    enabled = any(axis["fresh"] and motor.is_enabled() for axis, motor in zip(axes, motors, strict=True))
    if not enabled and any(axis["fresh"] for axis in axes):
        arm.set_callback_mode_all(can.CallbackMode.PARAM)
        for name in IDENTITY:
            if any(counts(arm.get_bus_status(), BUS_FAULTS).values()):
                break
            register = getattr(can.MotorVariable, name).value
            arm.query_param_all(register)
            receive(arm, lambda: all(motor.get_param(register) >= 0 for motor in devices.get_motors()))
            for axis, motor in zip(axes, devices.get_motors(), strict=True):
                value = motor.get_param(register)
                axis["identity"][name] = int(value) if math.isfinite(value) and value >= 0 else None

    faults = counts(arm.get_bus_status(), BUS_FAULTS)
    complete = all(
        axis["fresh"] and axis["identity"].get("ESC_ID") == axis["query_id"]
        and axis["identity"].get("MST_ID") == axis["reply_id"]
        and all(axis["identity"].get(name) is not None for name in ("hw_ver", "sw_ver", "SN"))
        for axis in axes
    ) and not any(faults.values())
    print(json.dumps({
        "interface": interface, "started_ns": requested_ns, "observed_ns": observed_ns,
        "enabled_detected": enabled, "complete": complete, "axes": axes, "bus_faults": faults,
        "unmatched_frames": arm.get_unmatched_frames(),
    }, allow_nan=False), flush=True)
    return 3 if enabled else (0 if complete else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface")
    parser.add_argument("--confirm-readonly", action="store_true")
    args = parser.parse_args()
    if not args.confirm_readonly or os.geteuid() == 0:
        parser.error("Read-only authorization and an unprivileged process are required")
    return inspect(args.interface)


if __name__ == "__main__":
    raise SystemExit(main())
