"""Hardware-free regression tests for episode 0 readback and stop behavior."""
from types import SimpleNamespace
import sys
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from openarm_driver import Config

import demo0_replay as replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.config = Config("openarm_pedestal")
        self.limits = replay.arm_limits(self.config)
        self.pose = np.array([self.config.get_start_config()["moves"][-1]["position"][f"{side}_arm"]
                              for side in replay.SIDES])

    def sample(self, index):
        return {"stamp": time.time_ns(), "q": self.pose[index].tolist()}

    def fixture(self, stage, _limits):
        return np.tile(self.pose[:, :7].ravel(), (2, 1)), {"sha256": stage}

    def test_read_uses_only_disabled_state_queries(self):
        sockets = [(Mock(), Mock()), (Mock(), Mock())]
        with patch.object(replay, "open_probe", side_effect=sockets) as opened, \
             patch.object(replay, "fresh", side_effect=[self.sample(0), self.sample(1)]) as fresh, \
             patch.object(replay, "ReplayDriver") as driver:
            np.testing.assert_array_equal(replay.read_pose(), self.pose)
        self.assertEqual([call.args[0] for call in opened.call_args_list], ["right", "left"])
        self.assertEqual([call.args for call in fresh.call_args_list], sockets)
        driver.assert_not_called()
        for bus, _ in sockets:
            bus.disable_all.assert_not_called()

    def test_approach_preserves_ends_and_limits(self):
        start = self.pose[:, :7].ravel()
        target = start.copy()
        target[3] -= .4
        path, info = replay.approach_path(start, target, self.limits)
        np.testing.assert_array_equal(path[0], start)
        np.testing.assert_array_equal(path[-1], target)
        self.assertLessEqual(info["peak_speed_rad_s"], .09)
        self.assertLessEqual(info["peak_accel_rad_s2"], .10)

    def test_stale_plan_and_moved_robot_rejected(self):
        with patch.object(replay, "load_replay", side_effect=self.fixture):
            plan = replay.make_plan(self.pose, self.limits)
            replay.validate_plan(plan, self.pose, self.limits)
            moved = self.pose.copy()
            moved[0, 0] += .02
            with self.assertRaisesRegex(ValueError, "0.01 rad"):
                replay.validate_plan(plan, moved, self.limits)
            plan["read_ns"] -= replay.PLAN_AGE_NS+1
            with self.assertRaisesRegex(ValueError, "5 分钟"):
                replay.validate_plan(plan, self.pose, self.limits)

    def test_feedback_fault_precedes_dispatch(self):
        path = np.tile(self.pose[:, :7].ravel(), (2, 1))
        for unexpected in (TimeoutError("left feedback missing"), self.sample(1) | {"q": [3.]*8}):
            with self.subTest(unexpected=unexpected):
                right, left = Mock(), Mock()
                right.read.return_value = self.sample(0)
                left.read.side_effect = [unexpected]
                with self.assertRaises(RuntimeError if isinstance(unexpected, dict) else TimeoutError):
                    replay.play([right, left], path, self.pose[:, 7], "test")
                right.send_position.assert_not_called()
                left.send_position.assert_not_called()

    def test_failed_second_enable_requests_both_disables_before_readback(self):
        events = []
        drivers = []
        for index, side in enumerate(replay.SIDES):
            driver = Mock(arm_side=side)
            driver.start.return_value = index == 0
            driver.openarm.disable_all.side_effect = lambda side=side: events.append((side, "disable"))
            driver.read.side_effect = lambda _, index=index, side=side: (
                events.append((side, "read")) or self.sample(index))
            drivers.append(driver)
        gravity = SimpleNamespace(Bus=lambda bus: bus, CAP=(), Gravity=lambda side: None)
        with patch.dict(sys.modules, {"vr_gravity": gravity}), \
             patch.object(replay, "ReplayDriver", side_effect=drivers), \
             patch.object(replay, "play") as play:
            with self.assertRaisesRegex(RuntimeError, "使能检查失败"):
                replay.execute("approach", np.empty((0, 14)), self.pose, self.config)
        play.assert_not_called()
        self.assertEqual(events[-4:], [("right", "disable"), ("left", "disable"),
                                       ("right", "read"), ("left", "read")])


if __name__ == "__main__":
    unittest.main()
