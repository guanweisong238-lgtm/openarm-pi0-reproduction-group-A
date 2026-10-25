"""按清单录制已通过数值筛选的RGB示教；检查重现性并记录逐条状态。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit_drawer_expansion import audit
from collect_drawer_trial import ROOT


def collect(profile, output_root):
    numeric = ROOT / profile["numeric"]
    status = {"name": profile["name"], "split": profile["split"]}
    source = audit(numeric)
    if source["status"] != "numeric_only":
        raise ValueError(f"数值轨迹未通过：{numeric}")
    output = output_root / profile["name"]
    command = [sys.executable, str(ROOT / "scripts/a2_pi0/collect_drawer_trial.py"),
               "--mode", "contact", "--expert", "contact", "--drawer-offset-x", "0.10",
               "--pull-distance", "0.10", "--return-drawer", "--wrist-view", "side",
               "--contact-pitch", "15", "--initial-right-jitter", str(profile["jitter"]),
               "--seed", str(profile["seed"]), "--drawer-offset-y", str(profile["drawer_y"]),
               "--approach-offset", *map(str, profile["approach"]), "--output", str(output)]
    environment = os.environ.copy()
    environment.update(MUJOCO_GL="egl", LIBGL_ALWAYS_SOFTWARE="1",
                       __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
                       LP_NUM_THREADS="4")
    with (output_root / (profile["name"] + ".log")).open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        return dict(status, status="process_failed", exit_code=result.returncode)
    row = audit(output)
    status.update(row)
    with np.load(numeric / "episode.npz", allow_pickle=False) as reference, np.load(
            output / "episode.npz", allow_pickle=False) as rgb:
        for key in ("state", "action", "qpos", "drawer", "finger_contact_counts", "finger_normal_forces"):
            np.testing.assert_array_equal(rgb[key], reference[key], err_msg=key)
    status["numeric_rgb_equal"] = True
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    result = {"name": config["name"], "config": str(args.config), "episodes": [],
              "training_integrated": False}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect, profile, output): profile for profile in config["episodes"]}
        for future in as_completed(futures):
            profile = futures[future]
            try:
                status = future.result()
            except Exception as error:
                status = {"name": profile["name"], "split": profile["split"],
                          "status": "verification_failed", "error": str(error)}
            result["episodes"].append(status)
            (output / "batch_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(status["name"], status["status"], flush=True)
    if any(row["status"] != "rgb_candidate" for row in result["episodes"]):
        raise SystemExit("批次存在失败轨迹，见batch_report.json；不加入成功数据集")
    print("PASS：全部RGB轨迹通过审核和数值版逐元素对齐", flush=True)


if __name__ == "__main__":
    main()
