#!/usr/bin/env python3
"""Run the installed official Dora node with an in-memory driver, without CAN."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".cache/openarm-policy-driver/lib/python3.11/site-packages"))
import dora_openarm.main as official
import guarded_official_node as guarded


class MockDriver:
    active = None

    def __init__(self, *args, **kwargs):
        # Synthetic gripper angle satisfies the original right-arm alignment trigger.
        self.position = np.array([0.2, 0.1, 0, 1, 0, 0.1, 0, 0], dtype=float)
        self.sent = 0
        self.requests = 0
        self.started = False
        MockDriver.active = self

    def start(self):
        if os.environ.get("SMALL_MOTION_FAULT") == "start_rejected":
            print(json.dumps({"event": "official_mock_start_rejected"}), flush=True)
            return False
        self.started = True
        print(json.dumps({"event": "official_mock_start"}), flush=True)
        return True

    def stop(self):
        self.started = False
        print(json.dumps({"event": "official_mock_stop", "sent": self.sent,
                          "monotonic_ns": time.monotonic_ns()}), flush=True)

    def fetch_state(self, refresh=True):
        self.requests += 1
        if os.environ.get("SMALL_MOTION_FAULT") == "startup_no_feedback":
            time.sleep(2.5)
        if os.environ.get("SMALL_MOTION_FAULT") == "feedback_delay" and self.requests == 8:
            time.sleep(0.3)
        if os.environ.get("SMALL_MOTION_FAULT") == "gripper_drift" and self.requests == 8:
            self.position[-1] += 0.04
        return {"qpos": self.position, "qvel": np.zeros(8), "qtorque": np.zeros(8),
                "tmos": np.zeros(8, dtype=int), "trotor": np.zeros(8, dtype=int)}

    def get_health(self):
        names = ("bus_off", "error_passive", "error_warning", "ack_error",
                 "tx_overflow", "rx_overflow", "write_net_down")
        status = "ENABLED" if self.started else "DISABLED"
        return [status] * 8, {"carrier": True, **dict.fromkeys(names, 0)}

    def fetch_position(self, refresh=True):
        return self.position.copy()

    def send_position(self, position):
        self.position = np.asarray(position, dtype=float)
        self.sent += 1
        print(json.dumps({"event": "official_mock_target", "count": self.sent,
                          "monotonic_ns": time.monotonic_ns()}), flush=True)
        return True


official.openarm_driver.SingleArmDriver = MockDriver
if os.environ.get("SMALL_MOTION_FAULT") == "driver_failure":
    original_build_state = official.build_state_output
    calls = 0

    def changed_status_state(state, health):
        global calls
        calls += 1
        if calls == 8:
            raise RuntimeError("simulated driver failure")
        return original_build_state(state, health)

    official.build_state_output = changed_status_state
sys.argv = [sys.argv[0], "--side", "right", "--config", "openarm_pedestal",
            "--align-trigger", "gripper"]
try:
    guarded.run()
finally:
    active = MockDriver.active
    if active is not None and active.started:
        print(json.dumps({"event": "official_mock_exception_cleanup"}), flush=True)
        active.stop()
