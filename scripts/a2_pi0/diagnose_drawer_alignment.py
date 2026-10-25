"""Measure policy alignment errors with forward kinematics and paired image inputs.

FK target positions are diagnostic setpoints, not predictions of physical tracking.
No policy actions, observations, or training parameters are modified by this script.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from drawer_simulation import DrawerSimulation
import mujoco

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "logs/a2_pi0/drawer_eval_20261024-100952-541431"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--image-ablation", action="store_true")
    args = parser.parse_args()
    env = DrawerSimulation()
    model, scratch = env.model, mujoco.MjData(env.model)
    site = model.site("right_ee_control_point").id
    handle = model.geom("drawer_handle").id
    expert = np.load(env.episode_root / "episode.npz", allow_pickle=False)
    offline = np.load(args.eval_dir / "offline_predictions.npz", allow_pickle=False)
    closed = np.load(args.eval_dir / "closed_loop.npz", allow_pickle=False)

    def geometry(qpos):
        scratch.qpos[:] = qpos
        mujoco.mj_kinematics(model, scratch)
        tip = scratch.site_xpos[site] + scratch.site_xmat[site].reshape(3, 3) @ np.array([0, 0, -.16])
        return tip.copy(), scratch.geom_xpos[handle].copy()

    def target_point(qpos, action):
        q = qpos.copy()
        q[env.qpos_ids] = action
        return geometry(q)[0]

    target = np.array([target_point(q, a[0]) for q, a in zip(expert["qpos"], offline["target"])])
    prediction = np.array([target_point(q, a[0]) for q, a in zip(expert["qpos"], offline["prediction"])])
    teacher_points = np.array([geometry(q)[0] for q in expert["qpos"]])
    actual = np.array([geometry(q)[0] for q in closed["qpos"]])
    handles = np.array([geometry(q)[1] for q in closed["qpos"]])
    distance = np.linalg.norm(actual - handles, axis=1)
    nearest = int(distance.argmin())
    clipping = np.abs(closed["requested"] - closed["applied"]) > 1e-9
    phases = {}
    for name, start, stop in (("approach", 0, 150), ("grasp", 150, 192)):
        error = prediction[start:stop] - target[start:stop]
        phases[name] = {"first_action_mean_signed_xyz_mm": (error.mean(0)*1000).tolist(),
                        "first_action_rmse_xyz_mm": (np.sqrt((error**2).mean(0))*1000).tolist()}
    snapshots = []
    for index in (3, 29, 59, 149, 359, 719):
        if index >= len(actual):
            continue
        desired = target_point(closed["qpos_before"][index], closed["applied"][index])
        snapshots.append({"step": index+1, "time_s": (index+1)/30,
            "actual_tip_xyz_m": actual[index].tolist(),
            "tip_minus_handle_xyz_mm": ((actual[index]-handles[index])*1000).tolist(),
            "commanded_tip_xyz_m": desired.tolist(),
            "actual_minus_commanded_xyz_mm": ((actual[index]-desired)*1000).tolist()})
    result = {"scope": "same-training-trajectory diagnosis, no independent validation",
        "coordinate_frame": "MuJoCo world XYZ; grip point = right control site + R*[0,0,-0.16]",
        "offline_fk": phases, "closed_loop_snapshots": snapshots,
        "closest_approach": {"step": nearest+1, "time_s": (nearest+1)/30,
            "distance_mm": float(distance[nearest]*1000),
            "tip_minus_handle_xyz_mm": ((actual[nearest]-handles[nearest])*1000).tolist()},
        "right_arm_clipped_frames": clipping[:, :7].sum(0).tolist(),
        "initial_offline_vs_live_block_max_rad": float(np.abs(offline["prediction"][0]-closed["action_blocks"][0]).max()),
        "teacher_grasp_tip_mean_xyz_m": teacher_points[150:192].mean(0).tolist()}

    if args.image_ablation:
        from drawer_training_config import DATA_ROOT, get_drawer_config
        from local_data_loader import EpisodeSubsetDataset
        from openpi.policies import policy_config
        import jax
        if jax.default_backend() != "gpu":
            raise RuntimeError("Image ablation requires GPU")
        policy = policy_config.create_trained_policy(get_drawer_config(),
            ROOT / "checkpoints/pi0_openarm_drawer_lora/overfit_roundtrip_001/2000",
            sample_kwargs={"num_steps": 10})
        dataset = EpisodeSubsetDataset(repo_id="local/openarm_drawer_trial", root=DATA_ROOT,
            episodes=[0], video_backend="pyav", delta_timestamps={"action": [i/30 for i in range(16)]})
        noise = np.random.default_rng(42).standard_normal((16, 32)).astype(np.float32)
        comparisons = []
        for index in (0, 4, 30, 60, 120, 150, 174):
            sample = dataset[index]
            decoded = {key: np.asarray(sample[key]) for key in (
                "observation.state", "observation.images.ceiling",
                "observation.images.wrist_left", "observation.images.wrist_right")}
            decoded["prompt"] = sample["task"]
            raw = dict(decoded)
            for camera in ("ceiling", "wrist_left", "wrist_right"):
                raw[f"observation.images.{camera}"] = expert[f"images_{camera}"][index]
            a = policy.infer(decoded, noise=noise)["actions"]
            b = policy.infer(raw, noise=noise)["actions"]
            repeated = policy.infer(raw, noise=noise)["actions"]
            row = {"frame": index, "repeat_max_rad": float(np.abs(b-repeated).max()),
                "decoded_reproduction_max_rad": float(np.abs(a-offline["prediction"][index]).max()),
                "raw_vs_decoded_right_arm_mae_rad": float(np.abs(b[:, :7]-a[:, :7]).mean()),
                "decoded_first_target_error_xyz_mm": ((target_point(expert["qpos"][index], a[0])-target[index])*1000).tolist(),
                "raw_first_target_error_xyz_mm": ((target_point(expert["qpos"][index], b[0])-target[index])*1000).tolist()}
            if index == 0:
                row["raw_reproduces_initial_live_max_rad"] = float(np.abs(b-closed["action_blocks"][0]).max())
            comparisons.append(row)
            print(json.dumps(row), flush=True)
        result["paired_image_ablation"] = comparisons

    np.savez_compressed(args.eval_dir / "alignment_geometry.npz", actual=actual, handle=handles,
                        teacher=teacher_points, offline_target=target, offline_prediction=prediction)
    output = args.eval_dir / "alignment_diagnosis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    main()
