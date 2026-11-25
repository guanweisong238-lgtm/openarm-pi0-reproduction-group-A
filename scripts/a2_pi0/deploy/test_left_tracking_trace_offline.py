"""Exercise the installed driver's checked targets without opening CAN."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from openarm_driver.driver import SingleArmDriver
from openarm_driver.safety import JointVelocityChecker

from left_tracking_trace import LeftTrackingTrace, TX_FAULTS


class TracedDriver(LeftTrackingTrace, SingleArmDriver):
    pass


class LeftTrackingTraceTest(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.replies = [0] * 8
        self.params = []
        self.fresh_axes = [3]
        self.fail_write = False
        self.bus = SimpleNamespace(writes_ok=0, **{key: SimpleNamespace(count=0) for key in TX_FAULTS})
        arm = SimpleNamespace(
            get_link_stats=lambda i: SimpleNamespace(responses=self.replies[i]),
            mit_control_all=self.write,
        )
        gripper = SimpleNamespace(set_position=self.write_gripper)
        driver = self.driver = object.__new__(TracedDriver)
        driver.openarm = SimpleNamespace(get_bus_status=lambda: self.bus, get_arm=lambda: arm,
                                         get_gripper=lambda: gripper, recv_all=self.receive)
        driver._health_collections = [arm, gripper]
        driver._log_health_if_changed = lambda: None
        driver._iter_motors = lambda: iter([
            SimpleNamespace(get_position=lambda: 0.12, get_velocity=lambda: 0.01,
                            get_torque=lambda: 0.02, get_state_tmos=lambda: 25,
                            get_state_trotor=lambda: 25) for _ in range(8)
        ])
        driver.joint_offsets = np.full(8, 0.1)
        driver.kps = driver.kds = np.ones(8)
        driver.num_mit_motors = 7
        driver.gripper_posforce = True
        driver.gripper_posforce_limits = (1.0, 1.0)
        driver.safety_checker = JointVelocityChecker(np.ones(8))
        driver.last_command = np.zeros(8)
        driver.last_command_time_s = 0.0
        driver._safety_stop_reason = None
        driver._last_safety_warning_time_s = None
        driver.tracking_trace = True
        self.enterContext(patch("left_tracking_trace.report", side_effect=self.capture))

    def capture(self, event, **fields):
        self.rows.append((event, fields))

    def receive(self, timeout_us):
        for axis in self.fresh_axes:
            self.replies[axis] += 1

    def write(self, params):
        self.params = params
        self.bus.writes_ok += len(params) - int(self.fail_write)
        self.bus.write_no_buffer.count += int(self.fail_write)

    def write_gripper(self, *_args, **_kwargs):
        self.bus.writes_ok += 1

    def test_checked_command_and_fresh_feedback_share_joint_coordinates(self):
        self.assertTrue(self.driver.send_position(np.full(8, 0.08)))
        (_, command), (_, feedback) = self.rows
        self.assertEqual(command["original_rad"], [0.08, 0.08])
        self.assertEqual(command["driver_rad"], [0.04, 0.04])
        self.assertEqual(command["driver_gap_rad"], [0.04, 0.04])
        self.assertAlmostEqual(self.params[3].q, 0.14)
        self.assertEqual(feedback["dispatch_ns"], command["dispatch_ns"])
        self.assertLessEqual(command["dispatch_ns"], feedback["read_started_ns"])
        self.assertAlmostEqual(feedback["measured_rad"][0], 0.02)
        self.assertAlmostEqual(feedback["tracking_gap_rad"][0], 0.02)
        self.assertIsNone(feedback["tracking_gap_rad"][1])

    def test_failed_write_does_not_get_a_tracking_error(self):
        self.driver.send_position(np.zeros(8))
        self.fail_write = True
        self.driver.send_position(np.zeros(8))
        (_, command), (_, feedback) = self.rows[-2:]
        self.assertEqual(command["tx_fault_delta"]["write_no_buffer"], 1)
        self.assertNotIn("tracking_gap_rad", feedback)

    def test_rejection_does_not_turn_previous_target_into_new_command(self):
        self.driver.send_position(np.zeros(8))
        self.driver._safety_stop_reason = "test stop"
        self.assertFalse(self.driver.send_position(np.ones(8)))
        _, rejected = self.rows[-1]
        self.assertFalse(rejected["accepted"])
        self.assertNotIn("driver_rad", rejected)

    def test_independent_state_read_with_no_reply_does_not_repeat_feedback(self):
        self.driver.send_position(np.zeros(8))
        self.fresh_axes = []
        self.driver.set_latest_state()
        self.assertEqual(len(self.rows), 2)
        self.fresh_axes = [4]
        self.driver.set_latest_state()
        _, feedback = self.rows[-1]
        self.assertEqual(feedback["new_replies"], [0, 1])
        self.assertIsNone(feedback["measured_rad"][0])
        self.assertAlmostEqual(feedback["measured_rad"][1], 0.02)


if __name__ == "__main__":
    unittest.main()
