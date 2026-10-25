"""Re-execute saved policy actions in physics, with an optional interactive viewer."""
import argparse
import json
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("MUJOCO_GL", "glfw")
from drawer_simulation import DrawerSimulation
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path, help="Model closed_loop.npz")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    with np.load(args.rollout, allow_pickle=False) as saved:
        actions = saved["requested"].copy()
        expected = saved["qpos"].copy()
        expected_before = saved["qpos_before"].copy()
    env = DrawerSimulation()
    maximum = 0.0

    def step(index):
        nonlocal maximum
        maximum = max(maximum, float(np.max(np.abs(env.data.qpos - expected_before[index]))))
        env.step(actions[index])
        maximum = max(maximum, float(np.max(np.abs(env.data.qpos - expected[index]))))
        if maximum >= 1e-6:
            raise RuntimeError(f"Model action replay diverged: {maximum}")

    def finish():
        print(json.dumps({"mode": "recorded_model_actions", "steps": len(actions),
                          "simulated_seconds": len(actions) / 30,
                          "maximum_qpos_error": maximum,
                          "final_opening_mm": -float(env.data.qpos[env.drawer_qpos]) * 1000,
                          "source": str(args.rollout.resolve())}), flush=True)

    if args.headless:
        for index in range(len(actions)):
            step(index)
        finish()
        env.close()
        return

    import mujoco.viewer
    paused, restart = threading.Event(), threading.Event()

    def on_key(key):
        if key == 32:
            paused.clear() if paused.is_set() else paused.set()
        elif key in (ord("R"), ord("r")):
            restart.set()

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key,
                                         show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.35, -0.24, 0.57]
                viewer.cam.distance = 0.95
                viewer.cam.azimuth = -140
                viewer.cam.elevation = -25
            viewer.sync()
            print("GUI_READY: recorded MODEL actions; Space=pause, R=restart; holds final frame", flush=True)
            index = 0
            resume_at = time.perf_counter() + 3
            while viewer.is_running():
                began = time.perf_counter()
                with viewer.lock():
                    if restart.is_set():
                        env.reset()
                        index, maximum = 0, 0.0
                        resume_at = began + 1
                        restart.clear()
                    if index < len(actions) and not paused.is_set() and began >= resume_at:
                        step(index)
                        index += 1
                        if index == len(actions):
                            finish()
                status = "Finished (R to replay)" if index == len(actions) else "Paused" if paused.is_set() else "Playing"
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                  mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "MODEL action replay\nTime\nDrawer opening\nStatus\nSpace / R",
                                  f"Saved policy predictions\n{index / 30:.1f} / {len(actions) / 30:.1f} s\n"
                                  f"{-env.data.qpos[env.drawer_qpos] * 1000:.1f} mm\n{status}\nPause / Restart"))
                viewer.sync()
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - began)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
