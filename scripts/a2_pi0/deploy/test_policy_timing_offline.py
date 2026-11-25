"""Regression checks for the first dispatch after asynchronous inference."""
from concurrent.futures import Future
import unittest
from unittest.mock import Mock, patch

import numpy as np

import dora_policy_node as policy


class PolicyTimingTest(unittest.TestCase):
    def setUp(self):
        self.now = 10_000_000_000
        self.wall_offset = 100_000_000_000
        self.node = Mock()
        self.controller = policy.Execution(self.node, Mock(), "transport")
        self.controller.loading = None
        self.controller.next_infer = self.now + 2_000_000_000
        self.controller.seq = 1
        self.state = np.zeros(16, dtype=np.float32)
        self.targets = np.repeat(self.state[None, :], 50, axis=0)
        self.dispatches = []
        self.enterContext(patch.object(policy.time, "monotonic_ns", lambda: self.now))
        self.enterContext(patch.object(policy.time, "time_ns", lambda: self.now + self.wall_offset))
        self.enterContext(patch.object(policy, "log", self.record))

    def record(self, name, **fields):
        if name in ("dispatch", "reject"):
            self.dispatches.append((name, fields))

    def tick(self):
        stamp = self.now + self.wall_offset
        self.controller.frames = {name: (stamp, b"") for name in policy.CAMERAS}
        self.controller.status = {side: ("aligned", 1) for side in policy.SIDES}
        self.controller.feedback = dict(
            zip(policy.SIDES, [(stamp, self.state[:8]), (stamp, self.state[8:])], strict=True)
        )
        self.controller.tick()

    def complete_inference(self, delay_ms=291):
        started = self.now - delay_ms * 1_000_000
        future = Future()
        future.set_result((self.targets, delay_ms, None))
        self.controller.pending = (future, started, started, started + self.wall_offset)

    def test_delayed_inference_dispatches_step_zero_then_advances_at_30_hz(self):
        self.complete_inference()
        self.tick()
        self.now += 34_000_000
        self.tick()

        self.assertEqual([data["step"] for _, data in self.dispatches], [0, 1])
        self.assertEqual(self.dispatches[0][1]["sent_ns"] - self.dispatches[0][1]["obs_timestamp_ns"]
                         + self.wall_offset, 291_000_000)

    def test_replan_waits_for_dispatch_interval_and_starts_at_zero(self):
        self.complete_inference()
        self.tick()
        self.controller.seq = 2
        self.now += 5_000_000
        self.complete_inference()
        self.tick()
        self.assertEqual(len(self.dispatches), 1)

        self.now += 39_000_000
        self.tick()
        self.assertEqual([(data["seq"], data["step"]) for _, data in self.dispatches],
                         [(1, 0), (2, 0)])

    def test_expiry_is_measured_from_observation_not_first_dispatch(self):
        self.complete_inference()
        self.tick()
        self.now += 1_060_000_000
        self.tick()
        self.assertEqual(self.dispatches[-1][1]["reason"], "plan_expired")

    def test_replan_still_checks_jump_from_last_dispatched_target(self):
        self.complete_inference()
        self.tick()
        self.controller.seq = 2
        self.targets = self.targets.copy()
        self.targets[:, 0] = 0.08
        self.now += 34_000_000
        self.complete_inference()
        self.tick()
        self.assertEqual(self.dispatches[-1][1]["reason"], "dispatch_step_too_large")


if __name__ == "__main__":
    unittest.main()
