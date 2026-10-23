from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
from simulation.env import OpenArmEnv

ROOT = Path(__file__).resolve().parents[1]


def validate(path):
    env = OpenArmEnv()
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        "timestamps", "states", "actions", "images_left", "images_right",
        "terminal_state", "terminal_time", "scene_path", "model_commit",
        "actuator_names", "camera_names", "units", "robot_version", "control_dt",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    times = data["timestamps"].astype(np.float64)
    states, actions = data["states"], data["actions"]
    count = len(times)
    assert count > 0 and times.shape == (count,)
    assert states.shape == actions.shape == (count, env.model.nu)
    assert data["terminal_state"].shape == (env.model.nu,)
    for key in ["timestamps", "states", "actions", "terminal_state", "terminal_time"]:
        assert np.isfinite(data[key]).all(), key
    assert np.isclose(times[0], 0.0)
    assert np.all(np.diff(times) > 0)
    assert np.allclose(np.diff(times), env.control_dt, atol=1e-6, rtol=0)
    assert np.isclose(float(data["control_dt"]), env.control_dt)
    assert np.isclose(float(data["terminal_time"]), count * env.control_dt)
    assert data["actuator_names"].tolist() == env.actuator_names
    assert data["camera_names"].tolist() == ["camera_wrist_left", "camera_wrist_right"]
    assert str(data["units"]) == "rad" and str(data["robot_version"]) == "v2"
    assert str(data["scene_path"]) == "openarm_mujoco/v2/pedestal/pedestal.xml"
    expected_commit = (ROOT / "docs/openarm-commit.txt").read_text().strip()
    actual_commit = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(ROOT / "openarm_mujoco"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert str(data["model_commit"]) == expected_commit == actual_commit
    for key in ["images_left", "images_right"]:
        assert data[key].shape == (count, 240, 320, 3), key
        assert data[key].dtype == np.uint8, key
    for action in actions:
        assert np.allclose(action, env.check_action(action), atol=1e-6, rtol=0)
    env.reset()
    errors = []
    for state, action in zip(states, actions):
        errors.append(float(np.abs(env.get_observation()["joint_pos"] - state).max()))
        actual = env.step(action)["ctrl"]
        assert np.allclose(actual, action, atol=1e-6, rtol=0)
    terminal_error = float(np.abs(env.get_observation()["joint_pos"] - data["terminal_state"]).max())
    assert max(errors) < 1e-5
    assert terminal_error < 1e-5
    return {"frames": count, "terminal_time": float(env.data.time),
            "max_state_error": max(errors), "terminal_state_error": terminal_error,
            "model_commit": actual_commit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate A1 NPZ fields, action bounds and state replay")
    parser.add_argument("path", nargs="?", type=Path, default=ROOT / "outputs/sample_episode.npz")
    args = parser.parse_args()
    print(json.dumps(validate(args.path), indent=2))
