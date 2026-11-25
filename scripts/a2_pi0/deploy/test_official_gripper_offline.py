"""Check the installed driver's gripper command mapping without opening CAN."""
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".cache/openarm-policy-driver/lib/python3.11/site-packages"))
from openarm_driver import Config
from openarm_driver.driver import SingleArmDriver, _create_default_checker


class GripperCommandTest(unittest.TestCase):
    def test_gripper_target_uses_configured_offset_on_both_arms(self):
        config = Config("openarm_pedestal")
        for side, gripper_position in (("right_arm", -0.4), ("left_arm", 0.4)):
            with self.subTest(side=side):
                driver = SingleArmDriver.__new__(SingleArmDriver)
                driver.arm_side = side
                driver.config = config
                driver.joint_offsets = config.get_joint_offsets(side)
                driver.num_mit_motors = 7
                driver.kps = config.get_default_kps()
                driver.kds = config.get_default_kds()
                driver.gripper_posforce = True
                driver.gripper_posforce_limits = config.get_gripper_posforce_limits()
                arm = SimpleNamespace(mit_control_all=Mock())
                gripper = SimpleNamespace(set_position=Mock())
                driver.openarm = SimpleNamespace(get_arm=lambda: arm, get_gripper=lambda: gripper)
                driver.safety_checker = _create_default_checker(side, config)
                driver._safety_stop_reason = None
                driver.set_latest_state = Mock()
                initial = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, gripper_position])
                driver.last_command = initial.copy()
                driver.last_command_time_s = time.monotonic() - 0.03
                target = initial.copy()
                target[0] += 0.001

                self.assertTrue(driver.send_position(target))
                arm.mit_control_all.assert_called_once()
                args, _ = gripper.set_position.call_args
                self.assertAlmostEqual(args[0], gripper_position + driver.joint_offsets[-1])


if __name__ == "__main__":
    unittest.main()
