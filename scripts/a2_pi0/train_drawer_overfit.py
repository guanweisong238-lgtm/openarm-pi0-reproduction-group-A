"""抽屉单轨迹过拟合实验：统计、缓存、固定训练帧拟合诊断、LoRA训练与保存。

只有episode 0，无验证集；所有诊断明确标为训练轨迹拟合，不声称泛化。
"""
from drawer_training_config import (ROOT, DATA_ROOT, EPISODE_ROOT, NORM_ROOT, get_drawer_config)
import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize


def prepare_stats():
    info = json.loads((DATA_ROOT / "meta/info.json").read_text())
    report = json.loads((EPISODE_ROOT / "report.json").read_text())
    if info["total_episodes"] != 1 or info["splits"] != {"train": "0:1"}:
        raise ValueError("本实验要求独立的单轨迹训练集")
    if not report["success_open_then_close"] or report["weld_frames"] != 0:
        raise ValueError("本实验要求已通过往返回放的无绑定示教")
    rows = []
    for path in sorted((DATA_ROOT / "data").rglob("*.parquet")):
        rows.extend(pq.read_table(path, columns=["episode_index", "observation.state", "action"]).to_pylist())
    if len(rows) != 462 or any(row["episode_index"] != 0 for row in rows):
        raise ValueError("示教帧数或轨迹编号不匹配")
    values = {"state": np.array([r["observation.state"] for r in rows], dtype=np.float64),
              "actions": np.array([r["action"] for r in rows], dtype=np.float64)}
    stats = {}
    for key, array in values.items():
        if array.shape != (462, 16) or not np.isfinite(array).all():
            raise ValueError("状态或动作不合法")
        stats[key] = normalize.NormStats(mean=array.mean(0), std=np.maximum(array.std(0), 1e-6),
                                         q01=np.quantile(array, .01, axis=0), q99=np.quantile(array, .99, axis=0))
    if (NORM_ROOT / "norm_stats.json").exists():
        existing = normalize.load(NORM_ROOT)
        for key in stats:
            np.testing.assert_allclose(existing[key].mean, stats[key].mean, atol=1e-12)
            np.testing.assert_allclose(existing[key].std, stats[key].std, atol=1e-12)
    else:
        NORM_ROOT.mkdir(parents=True, exist_ok=True)
        normalize.save(NORM_ROOT, stats)
    provenance = {"dataset": str(DATA_ROOT.relative_to(ROOT)), "train_episode_ids": [0],
        "frames": 462, "normalization": "mean_std", "std_floor": 1e-6,
        "weighting": "each original frame once", "validation_episode_ids": [],
        "source_report_sha256": hashlib.sha256((EPISODE_ROOT / "report.json").read_bytes()).hexdigest()}
    (NORM_ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--exp-name", default="overfit_roundtrip_001")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--lossless", action="store_true", help="Read original uint8 RGB arrays without video decoding")
    parser.add_argument("--diagnose-checkpoint", type=Path, help="Measure an existing checkpoint without updates or saving weights")
    args = parser.parse_args()
    if args.steps < 2 or not args.exp_name or args.exp_name in (".", "..") or any(x in args.exp_name for x in "/\\"):
        raise ValueError("步数或实验名称不合法")
    source = prepare_stats()
    from local_data_loader import OpenArmLoader, collate_numpy
    from openpi.models import model as model_lib
    import jax
    cfg = get_drawer_config(args.exp_name, args.steps, lossless=args.lossless)
    if cfg.checkpoint_dir.exists() and not args.prepare_only and args.diagnose_checkpoint is None:
        raise FileExistsError(cfg.checkpoint_dir)
    if args.lossless:
        from drawer_lossless_dataset import transformed_dataset
        transformed, data_config = transformed_dataset(cfg)
    else:
        loader = OpenArmLoader(cfg, split="train", shuffle=False, data_root=DATA_ROOT)
        transformed, data_config = loader._loader.dataset, loader.data_config()
    source = dict(source, image_source="original_npz_uint8_rgb" if args.lossless else "lerobot_lossy_video",
                  source_episode_sha256=hashlib.sha256((EPISODE_ROOT / "episode.npz").read_bytes()).hexdigest())
    began = time.perf_counter()
    cached = [transformed[i] for i in range(len(transformed))]
    print(f"缓存完成：{len(cached)}帧，{time.perf_counter()-began:.1f}秒", flush=True)
    def batch(index):
        value = collate_numpy([cached[int(index)]])
        actions = value.pop("actions")
        return model_lib.Observation.from_dict(value), actions
    first = batch(0)
    assert first[0].state.shape == (1, 32) and first[1].shape == (1, 16, 32)
    print(f"PASS：抽屉统计和训练输入检查；归一化={NORM_ROOT}", flush=True)
    if args.prepare_only:
        return
    if jax.default_backend() != "gpu":
        raise RuntimeError("训练需要JAX GPU")
    import flax.nnx as nnx
    import jax.numpy as jnp
    from train_smoke import load_parameters, make_train_step, save_checkpoint
    from drawer_training_objective import flow_squared_errors
    from openpi.training import optimizer as optimizer_lib
    output = ROOT / "logs/a2_pi0" / f"drawer_overfit_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")])
    (output / "config.txt").write_text(repr(cfg) + "\n")
    (output / "dataset.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"RUN_DIRECTORY={output}", flush=True)
    if args.diagnose_checkpoint is not None:
        protocol = json.loads((args.diagnose_checkpoint / "input_protocol.json").read_text())
        if protocol["image_source"] != source["image_source"]:
            raise ValueError("检查点图像来源与诊断输入不一致")
        params = nnx.state(cfg.model.load(model_lib.restore_params(args.diagnose_checkpoint / "params")))
    else:
        params = load_parameters(cfg)
    graphdef = nnx.graphdef(nnx.eval_shape(cfg.model.create, jax.random.key(cfg.seed)))
    if args.diagnose_checkpoint is None:
        tx = optimizer_lib.create_optimizer(cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None)
        opt_state = tx.init(params.filter(cfg.trainable_filter))
        update = make_train_step(cfg, graphdef, tx)
    @jax.jit
    def fit_loss(params, obs, actions, rng):
        errors = flow_squared_errors(nnx.merge(graphdef, params), rng, obs, actions)
        return errors.mean((0, 1))
    @jax.jit
    def sample_fit(params, obs, noise):
        return nnx.merge(graphdef, params).sample_actions(jax.random.key(0), obs,
                                                        noise=noise, num_steps=10)
    anchor_indices = np.unique(np.r_[np.linspace(0, 461, 12, dtype=int), 150, 174, 177, 191, 192, 281, 312, 401, 432])
    anchors = [jax.device_put(batch(i)) for i in anchor_indices]
    sample_noises = {seed: np.random.default_rng(seed).standard_normal((1,16,32)).astype(np.float32)
                     for seed in (42, 7, 99)}
    action_scale = np.asarray(data_config.norm_stats["actions"].std)+1e-6
    fit_records = []
    def diagnose(step):
        per_dim = np.array([np.asarray(fit_loss(params, *sample, jax.random.key(12000+i)))
                            for i,sample in enumerate(anchors)])
        losses = per_dim.mean(-1).tolist()
        if not np.isfinite(losses).all():
            raise RuntimeError("拟合损失出现非有限值")
        record = {"step": step, "train_flow_loss": float(np.mean(losses)),
                  "frame_indices": anchor_indices.tolist(), "losses": losses, "independent_validation": False}
        record["train_flow_loss_by_group"] = {
            "right_arm": float(per_dim[:,:7].mean()), "right_gripper": float(per_dim[:,7].mean()),
            "left_arm_gripper": float(per_dim[:,8:16].mean()), "padding16": float(per_dim[:,16:].mean())}
        if step%1000 == 0 or step == args.steps or args.diagnose_checkpoint is not None:
            sampled = {}
            for seed,noise in sample_noises.items():
                errors = []
                for frame,(obs,actions) in zip(anchor_indices,anchors):
                    prediction = np.asarray(sample_fit(params,obs,noise))[0,:,:16]
                    target = np.asarray(actions)[0,:,:16]
                    valid = min(cfg.model.action_horizon,len(cached)-int(frame))
                    errors.append(np.abs(prediction[:valid]-target[:valid])*action_scale)
                if not all(np.isfinite(error).all() for error in errors):
                    raise RuntimeError("采样动作拟合检查出现非有限值")
                combined = np.concatenate(errors)
                sampled[str(seed)] = {"right_arm_mae_rad": float(combined[:,:7].mean()),
                    "right_gripper_mae_rad": float(combined[:,7].mean()),
                    "right_arm_mae_per_frame_rad": [float(error[:,:7].mean()) for error in errors]}
            record["sampled_action_fit"] = {"num_steps":10,"noise_seeds":list(sample_noises),
                "scope":"training anchors only, tail padding excluded; not task success", "by_seed":sampled}
            logging.info("sample fit step=%d right_arm_MAE=%s",step,
                         {seed:round(value["right_arm_mae_rad"],6) for seed,value in sampled.items()})
        fit_records.append(record)
        with (output / "fit_diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        logging.info("fit step=%d train_flow_loss=%.6f", step, record["train_flow_loss"])
    if args.diagnose_checkpoint is not None:
        checkpoint_step = int(args.diagnose_checkpoint.name) if args.diagnose_checkpoint.name.isdigit() else 0
        diagnose(checkpoint_step)
        (output / "diagnostic_summary.json").write_text(json.dumps({
            "checkpoint": str(args.diagnose_checkpoint.resolve()), "additional_training_steps":0,
            "image_source": source["image_source"], "fit":fit_records[-1]},indent=2)+"\n")
        print(f"PASS：仅诊断，未更新或保存权重；记录={output}",flush=True)
        return
    diagnose(0)
    rng = jax.random.key(cfg.seed + 1)
    order_rng = np.random.default_rng(cfg.seed)
    order = np.empty(0, dtype=int)
    for step in range(1, args.steps + 1):
        if (step-1) % len(cached) == 0:
            order = order_rng.permutation(len(cached))
        value = jax.device_put(batch(order[(step-1) % len(cached)]))
        rng, key = jax.random.split(rng)
        began = time.perf_counter()
        params, opt_state, metrics = update(params, opt_state, key, *value)
        metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        if not all(np.isfinite(v) for v in metrics.values()) or metrics["update_norm"] == 0:
            raise RuntimeError(f"训练更新异常：{metrics}")
        record = {"step": step, **metrics, "seconds": time.perf_counter()-began}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if step == 1 or step % 20 == 0:
            logging.info("step=%d loss=%.6f seconds=%.3f", step, metrics["loss"], record["seconds"])
        if step % 250 == 0 or step == args.steps:
            diagnose(step)
        if step % 1000 == 0 or step == args.steps:
            checkpoint = save_checkpoint(cfg, params, opt_state, data_config, step=step)
            (checkpoint / "input_protocol.json").write_text(json.dumps(source, indent=2)+"\n")
            logging.info("checkpoint=%s", checkpoint)
    summary = {"experiment": args.exp_name, "steps": args.steps, "frames": len(cached),
               "checkpoint": str(checkpoint), "initial_train_flow_loss": fit_records[0]["train_flow_loss"],
               "final_train_flow_loss": fit_records[-1]["train_flow_loss"], "independent_validation": False,
               "closed_loop_tested": False, "image_source": source["image_source"]}
    summary["final_sampled_action_fit"] = fit_records[-1]["sampled_action_fit"]
    summary["overfit_success_verified"] = False
    summary["overfit_acceptance_note"] = "Training completion alone is not overfit success; inspect sampled errors and closed-loop task result."
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS：抽屉单轨迹训练完成，检查点={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
