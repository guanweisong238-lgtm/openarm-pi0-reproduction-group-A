"""审核扩充试采；区分数值试验、RGB成功候选、失败轨迹和未完成目录。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from drawer_lossless_dataset import LosslessDrawerDataset


def audit(directory):
    report = json.loads((directory / "report.json").read_text())
    problems = []
    for key in ("success_open_then_close", "replay_verified"):
        if not report.get(key):
            problems.append(key + " is false")
    if report.get("mode") != "contact" or report.get("weld_frames") != 0:
        problems.append("not weld-free contact")
    row = {"path": str(directory), "frames": report["frames"],
           "peak_open_mm": report["peak_open_m"] * 1000,
           "initial_randomization": report.get("initial_right_randomization"),
           "pitch_degrees": report["expert_parameters"]["pitch_degrees"]}
    with np.load(directory / "episode.npz", allow_pickle=False) as episode:
        for key in ("state", "action"):
            if episode[key].shape != (report["frames"], 16) or not np.isfinite(episode[key]).all():
                problems.append("invalid " + key)
        # initial qpos grouping identifies same-start variants; it is not a generalization split.
        row["initial_qpos_sha256"] = hashlib.sha256(episode["qpos"][0].tobytes()).hexdigest()
        row["action_sha256"] = hashlib.sha256(episode["action"].tobytes()).hexdigest()
        counts = episode["finger_contact_counts"]
        forces = episode["finger_normal_forces"]
        bilateral = np.all(counts > 0, axis=1) & np.all(forces > .01, axis=1)
        row["bilateral_contact_frames"] = int(bilateral.sum())
        if not bilateral.any():
            problems.append("no bilateral finger-handle contact")
        if report["image_shape"] is not None:
            cameras = {}
            for name in ("ceiling", "wrist_left", "wrist_right"):
                images = episode["images_" + name]
                if images.dtype != np.uint8 or images.shape != (report["frames"], 240, 320, 3):
                    problems.append("invalid RGB " + name)
                if not np.any(images) or not np.any(images[0] != images[-1]):
                    problems.append("black or temporally constant RGB " + name)
                cameras[name] = {"mean_brightness": float(images.mean())}
            row["cameras"] = cameras
    if report["image_shape"] is not None:
        dataset = LosslessDrawerDataset(directory)
        for index in range(len(dataset)):
            sample = dataset[index]
            valid = min(dataset.horizon, len(dataset) - index)
            np.testing.assert_array_equal(sample["action"][:valid], dataset.action[index:index + valid])
            assert sample["action_is_pad"].sum() == dataset.horizon - valid
    row["problems"] = problems
    row["status"] = ("rejected" if problems else
                     "numeric_only" if report["image_shape"] is None else "rgb_candidate")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/drawer_expansion"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [audit(path.parent) for path in sorted(args.root.glob("*/*/report.json"))]
    unfinished = [str(path) for path in args.root.glob("*/*")
                  if path.is_dir() and not (path / "report.json").exists()]
    result = {"episodes": rows, "unfinished_directories": unfinished,
              "note": "RGB candidates are not automatically included in training. Numeric/RGB copies and same-start variants must not be split across train/validation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for status in ("rgb_candidate", "numeric_only", "rejected"):
        print(status, sum(row["status"] == status for row in rows))
    print("unfinished", len(unfinished))
    print("report", args.output)


if __name__ == "__main__":
    main()
