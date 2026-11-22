"""Compare raw and blended handoffs on a saved recorded-observation replay."""

import argparse
import json
from pathlib import Path

import numpy as np

ARM = np.r_[0:7, 8:15]
GRIPPER = [7, 15]


def blend_handoffs(data, frames):
    anchors = data["anchor_indices"]
    windows = data["predicted_windows"]
    raw = data["predicted_ik_target"]
    blended = raw.copy()
    lengths = []
    if frames:
        for index, timestamp in enumerate(anchors[1:], start=1):
            offset = timestamp - anchors[index - 1]
            end = anchors[index + 1] if index + 1 < len(anchors) else len(raw)
            length = min(frames, end - timestamp, windows.shape[1] - offset)
            if length < 1:
                raise ValueError("adjacent windows must overlap")
            lengths.append(int(length))
            weights = (1 - np.cos(np.pi * np.arange(1, length + 1) / length))[:, None] / 2
            old = windows[index - 1, offset:offset + length][:, ARM]
            new = windows[index, :length][:, ARM]
            blended[timestamp:timestamp + length, ARM] = (1 - weights) * old + weights * new
            np.testing.assert_array_equal(blended[timestamp + length - 1], raw[timestamp + length - 1])
    np.testing.assert_array_equal(blended[:, GRIPPER], raw[:, GRIPPER])
    return blended, lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blend-frames", type=int, choices=(0, 4, 8, 12), default=8)
    args = parser.parse_args()
    report = json.loads((args.replay / "report.json").read_text())
    with np.load(args.replay / "full_episode.npz", allow_pickle=False) as archive:
        data = dict(archive)
    anchors = data["anchor_indices"]
    windows = data["predicted_windows"]
    raw = data["predicted_ik_target"]
    target = data["recorded_ik_target"]
    measured = data["observed_state"]
    previous = windows[np.arange(len(anchors) - 1), np.diff(anchors)]
    times = anchors[1:]
    old_error = np.abs((previous - measured[times])[:, ARM]).max(axis=1)
    new_error = np.abs((windows[1:, 0] - measured[times])[:, ARM]).max(axis=1)
    disagreement = np.abs((windows[1:, 0] - previous)[:, ARM]).max(axis=1)
    diagnostics = {
        "episode_id": report["episode_id"],
        "checkpoint": report["checkpoint"],
        "handoffs": len(times),
        "old_to_measured_median_rad": float(np.median(old_error)),
        "new_to_measured_median_rad": float(np.median(new_error)),
        "same_time_disagreement_median_rad": float(np.median(disagreement)),
        "methods": {},
        "gripper_events_note": "Largest positive and negative right-gripper target changes are timing proxies, not verified contact events.",
    }
    grip_change = np.diff(target[:, 7])
    event_frames = [int(grip_change.argmax()) + 1, int(grip_change.argmin()) + 1]
    for frames in (0, 4, 8, 12):
        blended, transition_lengths = blend_handoffs(data, frames)
        step = np.abs(np.diff(blended[:, ARM], axis=0)).max(axis=1)
        error = np.abs(blended[:, ARM] - target[:, ARM])
        diagnostics["methods"][str(frames)] = {
            "shortened_transitions": sum(length < frames for length in transition_lengths),
            "boundary_median_rad": float(np.median(step[times - 1])),
            "peak_single_step_rad": float(step.max()),
            "steps_over_0_1_rad": int(np.count_nonzero(step > 0.1)),
            "arm_mae_rad": float(error.mean()),
            "arm_mae_near_gripper_events_rad": {
                str(frame): float(error[max(0, frame - 8):frame + 9].mean())
                for frame in event_frames
            },
        }
        if frames == args.blend_frames:
            selected = blended
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "handoff_metrics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    report["model_display"] = f"MODEL IK TARGET ({args.blend_frames}-frame handoff)"
    report["handoff_blend_frames"] = args.blend_frames
    error = np.abs(selected - target)
    report["mean_absolute_error_rad"] = float(error.mean())
    report["per_joint_mae_rad"] = error.mean(axis=0).tolist()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    data["predicted_ik_target"] = selected
    np.savez_compressed(args.output / "full_episode.npz", **data)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
