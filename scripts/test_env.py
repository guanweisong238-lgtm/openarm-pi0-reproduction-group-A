import numpy as np
from simulation.env import OpenArmEnv

env = OpenArmEnv()
obs = env.reset()
action = obs["ctrl"].copy()
assert obs["timestamp"] == 0.0
for _ in range(100):
    obs = env.step(action)
    assert np.isfinite(obs["joint_pos"]).all()
    assert np.isfinite(obs["joint_vel"]).all()
assert obs["joint_pos"].shape == (env.model.nu,)
assert obs["gripper_pos"].shape == (2,)
assert np.isfinite(obs["gripper_pos"]).all()
assert np.isfinite(obs["gripper_vel"]).all()
assert np.isclose(obs["timestamp"], 100 * env.control_dt)
print("Environment test passed:", obs["timestamp"])
