#!/usr/bin/env python3
"""Dora-only simulation of the single-joint trajectory; never opens CAN."""
import json
import os
import sys
import time

import dora
import numpy as np
import pyarrow as pa

from small_motion_mock import trajectory


def log(event, **values):
    print(json.dumps({"event": event, "monotonic_ns": time.monotonic_ns(), **values},
                     allow_nan=False), flush=True)


def receiver(node):
    feedback = np.array([0.2, 0.1, 0, 1, 0, 0.1, 0, -0.4], dtype=float)
    count = 0
    disconnected = os.environ.get("SMALL_MOTION_CASE") == "disconnect"
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "target":
            feedback = np.asarray(event["value"][0].as_py()["qpos"], dtype=float)
            count += 1
            log("mock_feedback", count=count, qpos=feedback.tolist())
        elif not disconnected or count < 5:
            node.send_output("state", pa.array([{"qpos": feedback.tolist()}]),
                             {"sample_ns": time.monotonic_ns(), "start_epoch": 1})
    log("mock_receiver_exit", received=count)


def controller(node):
    initial = feedback = None
    sample_ns = epoch = None
    started_ns = last_send_ns = None
    sent = 0
    reason = "input_closed"
    official_mock = os.environ.get("SMALL_MOTION_OFFICIAL_MOCK") == "1"
    start_requested = False
    started = not official_mock
    last_request_ns = 0
    request_pending = False
    start_request_ns = None
    expected_epoch = None
    try:
        for event in node:
            if event["type"] == "STOP":
                reason = "dora_stop"
                break
            if event["type"] == "INPUT_CLOSED":
                reason = "input_closed:" + event["id"]
                break
            if event["type"] != "INPUT":
                continue
            now = time.monotonic_ns()
            if event["id"] == "status":
                status = event["value"][0].as_py()
                started = status in {"started", "aligned"}
                status_epoch = event["metadata"].get("start_epoch")
                if started and expected_epoch is None:
                    expected_epoch = status_epoch
                elif initial is not None and (not started or status_epoch != expected_epoch):
                    reason = "status_or_epoch_changed"
                    log("mock_reject", reason=reason)
                    break
                log("official_mock_status", status=status,
                    start_epoch=status_epoch)
                continue
            if event["id"] == "state":
                request_pending = False
                state = event["value"][0].as_py()
                if official_mock:
                    bus = state["bus"]
                    if (state["motor_status"] != ["ENABLED"] * 8 or not bus["carrier"]
                            or any(bus[key] != 0 for key in bus if key != "carrier")):
                        reason = "feedback_health"
                        log("mock_reject", reason=reason)
                        break
                feedback = state["qpos"]
                if official_mock:
                    age_ns = time.time_ns() - event["metadata"]["observation_timestamp"]
                    sample_ns = now - age_ns
                else:
                    sample_ns = event["metadata"]["sample_ns"]
                epoch = event["metadata"]["start_epoch"]
                if initial is None:
                    initial = np.asarray(feedback, dtype=float)
                    started_ns = now
                continue
            if official_mock and not start_requested:
                node.send_output("command", pa.array(["start"]))
                start_requested = True
                start_request_ns = now
                continue
            if official_mock and initial is None and now - start_request_ns > 2_000_000_000:
                reason = "startup_feedback_timeout"
                log("mock_reject", reason=reason)
                break
            if official_mock and started and not request_pending and now - last_request_ns >= 20_000_000:
                node.send_output("request", pa.array([True]))
                last_request_ns = now
                request_pending = True
            if initial is None or (last_send_ns and now - last_send_ns < 20_000_000):
                continue
            elapsed = (now - started_ns) / 1e9
            if elapsed > 1.0:
                reason = "trajectory_complete"
                break
            try:
                target = trajectory(initial, feedback, (now - sample_ns) / 1e9, epoch,
                                    expected_epoch if official_mock else 1,
                                    joint=0, delta=0.01, elapsed_s=elapsed)
            except ValueError as exc:
                reason = str(exc)
                log("mock_reject", reason=reason)
                break
            issued_ns = time.time_ns()
            node.send_output("target", pa.array([{"qpos": target.tolist()}]),
                             {"start_epoch": epoch, "issued_ns": issued_ns,
                              "deadline_ns": issued_ns + 50_000_000})
            sent += 1
            last_send_ns = now
            log("mock_dispatch", count=sent, elapsed_s=elapsed, qpos=target.tolist())
    finally:
        if official_mock and start_requested:
            node.send_output("command", pa.array(["stop"]))
        log("mock_controller_exit", reason=reason, sent=sent)


def main():
    # Initialize Arrow before starting the feedback freshness clock.
    pa.array([{"qpos": [0.0] * 8}])
    node = dora.Node()
    if sys.argv[1] == "controller":
        controller(node)
    elif sys.argv[1] == "receiver":
        receiver(node)
    else:
        raise ValueError("Expected controller or receiver")


if __name__ == "__main__":
    main()
