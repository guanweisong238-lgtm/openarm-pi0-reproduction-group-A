"""Validate the official pi0 ALOHA simulation checkpoint with synthetic input."""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_aloha_sim"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Check checkpoint size and MD5 without downloading")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set paths before importing ML libraries. Keep caches in this project.
    os.environ["OPENPI_DATA_HOME"] = str(ROOT / "weights" / "openpi")
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(ROOT / ".cache" / "jax")
    # Public GCS assets need no credentials or experimental bucket-type probing.
    os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = ROOT / "logs" / "a2_pi0" / run_id
    run_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log")],
    )
    try:
        import openpi
        from checkpoint_download import ensure_checkpoint

        logging.info("Python: %s", sys.executable)
        logging.info("OpenPI: %s", openpi.__file__)
        if not Path(openpi.__file__).resolve().is_relative_to((ROOT / ".venv").resolve()):
            raise RuntimeError("Run with this project's .venv/bin/python")
        logging.info("Logs: %s", run_dir)
        logging.info("Checkpoint source: %s", CHECKPOINT)
        checkpoint_dir = ensure_checkpoint(ROOT / "weights" / "openpi", verify_only=args.verify_only)
        logging.info("Checkpoint downloaded: %s", checkpoint_dir)
        if args.download_only or args.verify_only:
            return

        import jax
        import numpy as np
        from openpi.policies import policy_config
        from openpi.training import config

        if jax.default_backend() != "gpu":
            raise RuntimeError("JAX GPU is required for this inference check")
        cfg = config.get_config("pi0_aloha_sim")
        logging.info("Loading model; first inference also requires JAX compilation")
        policy = policy_config.create_trained_policy(cfg, checkpoint_dir)
        rng = np.random.default_rng(args.seed)
        observation = {
            "state": np.ones(14, dtype=np.float32),
            "images": {
                name: rng.integers(0, 256, (3, 224, 224), dtype=np.uint8)
                for name in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
            },
            "prompt": "Transfer cube",
        }
        # Explicit model noise makes the synthetic smoke check reproducible.
        noise = rng.standard_normal((cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
        elapsed = []
        for index in range(2):
            start = time.perf_counter()
            result = policy.infer(observation, noise=noise)
            actions = np.asarray(result["actions"])
            elapsed.append(time.perf_counter() - start)
            if actions.shape != (cfg.model.action_horizon, 14):
                raise RuntimeError(f"Unexpected action shape: {actions.shape}")
            if not np.isfinite(actions).all():
                raise RuntimeError("Actions contain NaN or infinity")
            logging.info("Inference %s passed: shape=%s elapsed=%.3fs", index + 1, actions.shape, elapsed[-1])

        np.save(run_dir / "actions.npy", actions)
        summary = {
            "config": cfg.name,
            "checkpoint_source": CHECKPOINT,
            "checkpoint_path": str(checkpoint_dir),
            "seed": args.seed,
            "input": "synthetic random images and constant robot state",
            "action_shape": list(actions.shape),
            "finite": True,
            "first_call_seconds_including_compilation": elapsed[0],
            "second_call_seconds": elapsed[1],
            "limitation": "Smoke check only; does not verify OpenArm compatibility or task success.",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logging.info("PASS: synthetic inference check; results saved to %s", run_dir)
    except Exception:
        logging.exception("Check failed; retain this log for diagnosis")
        raise


if __name__ == "__main__":
    main()
