"""Sampling audit and controlled short fits, starting from the same lossless checkpoint.

Micro-fit uses eight known training frames; these are optimization diagnostics,
not task success or an independent validation set. No checkpoints are overwritten.
"""
from drawer_training_config import ROOT, NORM_ROOT, get_drawer_config
import argparse
import dataclasses
from functools import partial
from datetime import datetime
import json
import math
import time
import numpy as np
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
import optax
from openpi.models import model as model_lib
from openpi.shared import nnx_utils, normalize
from drawer_lossless_dataset import transformed_dataset
from local_data_loader import collate_numpy
from drawer_training_objective import flow_squared_errors

FRAMES = (0, 4, 30, 60, 90, 120, 150, 174)
PROJECTIONS = r"(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)/.*"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "lora_aug", "lora_noaug", "projections_noaug"), default="audit")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU required")
    cfg = get_drawer_config(lossless=True)
    checkpoint = ROOT / "checkpoints/pi0_openarm_drawer_lora_lossless/overfit_roundtrip_lossless_001/2000"
    output = ROOT / "logs/a2_pi0" / f"drawer_training_audit_{args.mode}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True)
    print(f"AUDIT_DIRECTORY={output}", flush=True)
    dataset, _ = transformed_dataset(cfg)
    batches = []
    for frame in FRAMES:
        value = collate_numpy([dataset[frame]])
        actions = value.pop("actions")
        batches.append(jax.device_put((model_lib.Observation.from_dict(value), actions)))
    model = cfg.model.load(model_lib.restore_params(checkpoint / "params"))
    graph, params = nnx.split(model)
    del model
    stats = normalize.load(NORM_ROOT)["actions"]
    mean, scale = np.asarray(stats.mean), np.asarray(stats.std)+1e-6
    noise = np.random.default_rng(42).standard_normal((1, 16, 32)).astype(np.float32)

    @partial(jax.jit, static_argnames=("num_steps",))
    def sample(p, obs, initial_noise, num_steps=10):
        return nnx.merge(graph, p).sample_actions(jax.random.key(0), obs,
            noise=initial_noise, num_steps=num_steps)

    def sample_metrics(p, *, num_steps=10, seed=42):
        n = noise if seed == 42 else np.random.default_rng(seed).standard_normal((1,16,32)).astype(np.float32)
        prediction = np.array([np.asarray(sample(p, obs, n, num_steps))[0, :, :16]*scale+mean for obs, _ in batches])
        targets = np.array([np.asarray(a)[0,:,:16]*scale+mean for _, a in batches])
        error = abs(prediction-targets)
        return {"right_arm_mae_rad": float(error[:,:,:7].mean()),
                "right_gripper_mae_rad": float(error[:,:,7].mean()),
                "right_arm_mae_per_frame_rad": error[:,:,:7].mean((1,2)).tolist()}, prediction

    result = {"mode": args.mode, "frames": FRAMES, "checkpoint": str(checkpoint),
              "scope": "same-training-frame optimization diagnostic; no task success claim"}
    baseline, before = sample_metrics(params)
    result["before"] = baseline
    np.save(output / "before.npy", before)
    print("BEFORE "+json.dumps(baseline), flush=True)

    if args.mode == "audit":
        result["sampling"] = {"10_steps_seed42": baseline}
        for steps, seed in ((50,42), (10,7), (10,99)):
            metrics, _ = sample_metrics(params, num_steps=steps, seed=seed)
            result["sampling"][f"{steps}_steps_seed{seed}"] = metrics
            print(f"SAMPLING steps={steps} seed={seed} "+json.dumps(metrics), flush=True)
        bf16 = nnx_utils.state_map(params, nnx_utils.PathRegex(".*lora.*"),
                                  lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        metrics, converted = sample_metrics(bf16)
        result["bf16_reload"] = dict(metrics, max_output_change_rad=float(abs(converted-before).max()))
        del bf16

        @jax.jit
        def loss_breakdown(p, obs, actions, key):
            model = nnx.merge(graph,p)
            error = flow_squared_errors(model, key, obs, actions)
            official = jnp.mean(model.compute_loss(key, obs, actions, train=False))
            return error.mean((0,1)), official
        breakdown = []
        for frame, (obs, actions) in zip(FRAMES, batches):
            per_dim, official = jax.device_get(loss_breakdown(params,obs,actions,jax.random.key(12000+frame)))
            np.testing.assert_allclose(per_dim.mean(), official, rtol=1e-5, atol=1e-6)
            breakdown.append({"frame": frame, "all32": float(official),
                "right_arm": float(per_dim[:7].mean()), "right_gripper": float(per_dim[7]),
                "left_arm_gripper": float(per_dim[8:16].mean()), "padding16": float(per_dim[16:].mean())})
        result["flow_loss_breakdown"] = breakdown
        flat = flatten_dict(params.to_pure_dict(), sep="/")
        frozen = flatten_dict(params.filter(cfg.freeze_filter).to_pure_dict(), sep="/")
        result["frozen_projection_parameters"] = {k: math.prod(v.shape) for k,v in frozen.items() if not k.startswith("PaliGemma/")}
        result["all_parameters"] = sum(math.prod(v.shape) for v in flat.values())
        result["trainable_parameters"] = sum(math.prod(v.shape) for v in jax.tree.leaves(params.filter(cfg.trainable_filter)))
    else:
        if args.mode == "projections_noaug":
            allowed = nnx.Any(nnx_utils.PathRegex(".*lora.*"), nnx_utils.PathRegex(PROJECTIONS))
            cfg = dataclasses.replace(cfg, freeze_filter=nnx.Not(allowed))
            params = nnx_utils.state_map(params, nnx_utils.PathRegex(PROJECTIONS),
                                         lambda p: p.replace(p.value.astype(jnp.float32)))
            result["before_after_projection_fp32_cast"], _ = sample_metrics(params)
        tx = cfg.optimizer.create(2.5e-5)
        opt_state = tx.init(params.filter(cfg.trainable_filter))
        augment = args.mode == "lora_aug"

        @partial(jax.jit, donate_argnums=(0,1))
        def update(p, optimizer, obs, actions, key):
            model = nnx.merge(graph, p)
            model.train()
            def loss_fn(m):
                return flow_squared_errors(m,key,obs,actions,augment=augment).mean()
            loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0,cfg.trainable_filter))(model)
            trainable = p.filter(cfg.trainable_filter)
            updates, optimizer = tx.update(grads,optimizer,trainable)
            nnx.update(model,optax.apply_updates(trainable,updates))
            return nnx.state(model), optimizer, loss, optax.global_norm(grads)

        order = np.random.default_rng(123).integers(0,len(batches),size=args.steps)
        for step,index in enumerate(order,1):
            obs, actions = batches[index]
            params,opt_state,loss,grad = update(params,opt_state,obs,actions,jax.random.key(20000+step))
            loss, grad = float(loss),float(grad)
            if not np.isfinite([loss,grad]).all() or grad == 0:
                raise RuntimeError("Invalid gradient/loss")
            if step%20 == 0:
                print(f"UPDATE step={step} loss={loss:.6f} grad_norm={grad:.4f}",flush=True)
        result["steps"] = args.steps
        result["after"], after = sample_metrics(params)
        np.save(output / "after.npy", after)
        print("AFTER "+json.dumps(result["after"]), flush=True)
    (output / "report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("AUDIT_COMPLETE",flush=True)


if __name__ == "__main__":
    main()
