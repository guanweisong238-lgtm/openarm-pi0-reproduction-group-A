#!/usr/bin/env python3
"""Guard the pinned official Dora node's action input and Python cleanup.

This module is not a complete robot dataflow. Running it with a start command
can enable hardware, and must only follow the separately approved motion plan.
"""
import json
import os
import time

import dora
import dora_openarm.main as official
import openarm_driver
from left_tracking_trace import LeftTrackingTrace

MAX_ACTION_LIFETIME_NS = 100_000_000


class GuardedNode:
    def __init__(self, inner, active=()):
        self.inner = inner
        self.active = active

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __iter__(self):
        for event in self.inner:
            for driver in self.active:
                check = getattr(driver, "check_trial_event", None)
                if driver.cleanup_required and check is not None:
                    check()
            if event.get("type") == "INPUT" and event.get("id") == "move_position":
                metadata = event.get("metadata", {})
                issued = metadata.get("issued_ns")
                deadline = metadata.get("deadline_ns")
                epoch = metadata.get("start_epoch")
                now = time.time_ns()
                if (type(issued) is not int or type(deadline) is not int
                        or type(epoch) is not int or epoch < 1
                        or not issued <= now <= deadline
                        or not 0 < deadline - issued <= MAX_ACTION_LIFETIME_NS):
                    print(json.dumps({"event": "action_expired_or_invalid", "time_ns": now}), flush=True)
                    continue
            yield event


def run():
    original_node = dora.Node
    original_driver = openarm_driver.SingleArmDriver
    trace_left = os.environ.get("OPENARM_TRACE_LEFT_TRACKING") == "1"
    active = []

    class TrackedDriver(LeftTrackingTrace, original_driver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cleanup_required = False
            active.append(self)

        def start(self):
            pre_enable = getattr(self, "pre_enable_check", None)
            if pre_enable is not None:
                pre_enable()
            self.cleanup_required = True
            if not super().start():
                raise RuntimeError("Official driver start failed")
            self.tracking_trace = trace_left and self.arm_side == "left_arm"
            return True

        def stop(self):
            self.tracking_trace = False
            self._tracking_dispatch = None
            super().stop()
            self.cleanup_required = False

    dora.Node = lambda: GuardedNode(original_node(), active)
    openarm_driver.SingleArmDriver = TrackedDriver
    try:
        official.main()
    finally:
        try:
            for driver in active:
                if driver.cleanup_required:
                    driver.stop()
        finally:
            dora.Node = original_node
            openarm_driver.SingleArmDriver = original_driver


if __name__ == "__main__":
    run()
