"""固定检查帧、三噪声的离线动作/FK评估；不训练、不运行闭环。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from drawer_training_config import ROOT, get_drawer_config
from train_drawer_stage_fit import NOISE_SEEDS, anchor_frames, fk_probe, rotation_degrees


def measure(predictions, targets, frames, qpos, probe, episode_length):
    errors, positions, attitudes = [], [], []
    displacements, first_action_errors, joint_maes = [], [], []
    for prediction, target, frame in zip(predictions, targets, frames):
        valid = min(len(prediction), episode_length - int(frame))
        action_error = prediction[:valid] - target[:valid]
        errors.append(np.abs(action_error))
        first_action_errors.append(action_error[0].tolist())
        joint_maes.append(np.abs(action_error).mean(axis=0).tolist())
        point, rotation = probe(qpos[int(frame)], prediction[0])
        reference, reference_rotation = probe(qpos[int(frame)], target[0])
        positions.append(float(np.linalg.norm(point - reference) * 1000))
        # 世界坐标，预测减示教；负Z表示预测夹持点低于示教夹持点。
        displacements.append(((point - reference) * 1000).tolist())
        attitudes.append(rotation_degrees(rotation, reference_rotation))
    error = np.concatenate(errors)
    return {
        "right_arm_mae_rad": float(error[:, :7].mean()),
        "right_gripper_mae_rad": float(error[:, 7].mean()),
        "left_arm_gripper_mae_rad": float(error[:, 8:16].mean()),
        "position_error_mm_first_action_mean": float(np.mean(positions)),
        "position_error_mm_first_action_worst_frame": float(np.max(positions)),
        "attitude_error_deg_first_action_mean": float(np.mean(attitudes)),
        "attitude_error_deg_first_action_worst_frame": float(np.max(attitudes)),
        "per_frame_position_mm_first_action": positions,
        "per_frame_attitude_deg_first_action": attitudes,
        "per_frame_displacement_xyz_mm_first_action": displacements,
        # 16维顺序：右臂7、右夹爪、左臂7、左夹爪；正值表示预测角度更大。
        "per_frame_signed_joint_error_rad_first_action": first_action_errors,
        "per_frame_joint_mae_rad_action_horizon": joint_maes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="CPU：示教与自身比较，验证误差为零")
    args = parser.parse_args()
    if not args.dry_run and (args.checkpoint is None or args.output is None):
        parser.error("评估需要 --checkpoint 和 --output")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"不覆盖已有报告：{args.output}")
    anchors = anchor_frames()
    environment, probe = fk_probe()
    try:
        episode_path = environment.episode_root / "episode.npz"
        with np.load(episode_path, allow_pickle=False) as episode:
            qpos, teacher = episode["qpos"].copy(), episode["action"].copy()
        if args.dry_run:
            for frames in anchors.values():
                targets = [teacher[np.minimum(int(frame) + np.arange(16), len(teacher) - 1)]
                           for frame in frames]
                metrics = measure(targets, targets, frames, qpos, probe, len(teacher))
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, 0, atol=1e-5, err_msg=key)
            print("PASS：全部阶段示教自比较，动作/位置/姿态误差为零；末尾填充不参与MAE")
            return

        import flax.nnx as nnx
        import jax
        from openpi.models import model as model_lib
        from drawer_lossless_dataset import transformed_dataset
        from local_data_loader import collate_numpy

        if jax.default_backend() != "gpu":
            raise RuntimeError("模型评估需要用户终端的 JAX GPU")
        protocol = json.loads((args.checkpoint / "input_protocol.json").read_text())
        if protocol["image_source"] != "original_npz_uint8_rgb":
            raise ValueError("评估仅接受原始无损RGB检查点")
        cfg = get_drawer_config(lossless=True)
        dataset, data_config = transformed_dataset(cfg)
        if len(dataset) != len(teacher):
            raise ValueError("变换数据集与原始轨迹长度不一致")
        params = nnx.state(cfg.model.load(model_lib.restore_params(args.checkpoint / "params")))
        graphdef = nnx.graphdef(nnx.eval_shape(cfg.model.create, jax.random.key(cfg.seed)))

        @jax.jit
        def sample(current, observation, noise):
            return nnx.merge(graphdef, current).sample_actions(
                jax.random.key(0), observation, noise=noise, num_steps=10)

        stats = data_config.norm_stats["actions"]
        mean = np.asarray(stats.mean, dtype=np.float32)
        scale = np.asarray(stats.std, dtype=np.float32) + 1e-6
        batches = {}
        for frames in anchors.values():
            for frame in frames:
                batch = collate_numpy([dataset[int(frame)]])
                target = np.asarray(batch.pop("actions"))[0, :, :16] * scale + mean
                batches[int(frame)] = model_lib.Observation.from_dict(batch), target
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "scope": "same-training-trajectory anchors; not validation or closed-loop success",
            "image_source": "original_npz_uint8_rgb", "denoising_steps": 10,
            "schema_version": 2,
            "displacement_convention": "prediction minus teacher, world XYZ, millimeters",
            "joint_error_convention": "prediction minus teacher, radians; right7, right_gripper, left7, left_gripper",
            "source_episode_sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest(),
            "anchor_frames": {name: frames.tolist() for name, frames in anchors.items()},
            "by_seed": {},
        }
        for seed in NOISE_SEEDS:
            noise = np.random.default_rng(seed).standard_normal((1, 16, 32)).astype(np.float32)
            phases = {}
            for name, frames in anchors.items():
                predictions, targets = [], []
                for frame in frames:
                    observation, target = batches[int(frame)]
                    prediction = np.asarray(sample(params, observation, noise))[0, :, :16]
                    predictions.append(prediction * scale + mean)
                    targets.append(target)
                phases[name] = measure(predictions, targets, frames, qpos, probe, len(dataset))
            result["by_seed"][str(seed)] = phases
            metrics = phases["approach"]
            print(f"seed={seed} approach " + " ".join(
                f"{key}={metrics[key]:.6f}" for key in (
                    "right_arm_mae_rad", "right_gripper_mae_rad",
                    "position_error_mm_first_action_mean",
                    "position_error_mm_first_action_worst_frame",
                    "attitude_error_deg_first_action_mean")), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"评估报告：{args.output.resolve()}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
