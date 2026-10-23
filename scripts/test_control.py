import argparse
from contextlib import nullcontext
import time

import mujoco.viewer
import numpy as np
from simulation.env import OpenArmEnv


def run_control_test(headless=False):
    env = OpenArmEnv()
    home = env.reset()["ctrl"].copy()
    left = env.actuator_names.index("left_joint4_ctrl")
    right = env.actuator_names.index("right_joint4_ctrl")
    left_grip = env.actuator_names.index("left_finger1_ctrl")
    right_grip = env.actuator_names.index("right_finger1_ctrl")
    context = nullcontext(None) if headless else mujoco.viewer.launch_passive(env.model, env.data)
    with context as viewer:
        def advance(seconds, action):
            steps = round(seconds / env.control_dt)
            if not np.isclose(steps * env.control_dt, seconds):
                raise ValueError("Duration must match an integer number of control steps")
            obs = env.get_observation()
            for _ in range(steps):
                if viewer is not None and not viewer.is_running():
                    raise InterruptedError("Viewer closed before all control checks completed")
                start = time.perf_counter()
                obs = env.step(action)
                for key in ("joint_pos", "joint_vel", "gripper_pos", "gripper_vel"):
                    if not np.isfinite(obs[key]).all():
                        raise AssertionError(f"Non-finite state: {key}")
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0.0, env.control_dt - (time.perf_counter() - start)))
            return obs

        advance(1.0, home)
        cases = [
            ("T1 left arm", [(left, 0.15)]),
            ("T2 right arm", [(right, 0.15)]),
            ("T3 both arms", [(left, 0.15), (right, 0.15)]),
            ("T4 left gripper", [(left_grip, 0.35)]),
            ("T5 right gripper", [(right_grip, -0.35)]),
        ]
        for label, targets in cases:
            baseline = advance(1.0, home)["joint_pos"].copy()
            action = home.copy()
            for index, delta in targets:
                action[index] += delta
            obs = advance(2.0, action)
            for index, delta in targets:
                response = float(obs["joint_pos"][index] - baseline[index])
                if response * delta <= 0 or abs(response) < 0.02:
                    raise AssertionError(f"{label}: missing or reversed response {response}")
                print(f"{label}: {env.actuator_names[index]} delta={response:.6f} rad")
            returned = advance(2.0, home)
            for index, _ in targets:
                if abs(returned["joint_pos"][index] - baseline[index]) > 0.05:
                    raise AssertionError(f"{label}: did not return near the baseline")

        obs = env.reset()
        if obs["timestamp"] != 0.0 or not np.array_equal(obs["ctrl"], home):
            raise AssertionError("T6: reset did not restore time and controls")
        advance(1.0, home)
        print("PASS: T1-T6 control response, return, finite state and reset checks")
        if viewer is not None:
            print("All checks completed. Close the viewer to exit.")
            while viewer.is_running():
                try:
                    advance(0.1, home)
                except InterruptedError:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify v2 arm/gripper responses with optional viewer")
    parser.add_argument("--headless", action="store_true", help="Run numerical checks without a viewer")
    args = parser.parse_args()
    try:
        run_control_test(args.headless)
    except (InterruptedError, KeyboardInterrupt) as exc:
        print(f"Stopped: {exc}")
        raise SystemExit(130)
