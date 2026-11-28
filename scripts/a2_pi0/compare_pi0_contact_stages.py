#!/usr/bin/env python3
"""Compare two real-observation π0 replays at the same demonstrated task events."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from real_task_c_phases import grip_intervals


SCENE = Path(__file__).resolve().parents[2] / "assets/real_replay_scene/empty_recovery_scene.xml"
FPS = 30


class GraspPoint:
    def __init__(self):
        self.model = mj.MjModel.from_xml_path(str(SCENE))
        self.data = mj.MjData(self.model)
        self.joints = {
            side: [self.model.joint(f"openarm_{side}_joint{j}").id for j in range(1, 8)]
            for side in ("right", "left")
        }

    def __call__(self, target, side):
        start = 0 if side == "right" else 8
        self.data.qpos[self.model.jnt_qposadr[self.joints[side]]] = target[start:start + 7]
        mj.mj_forward(self.model, self.data)
        hand = self.data.body(f"openarm_{side}_ee_base_link")
        return hand.xpos + hand.xmat.reshape(3, 3) @ np.array([-.013, 0, -.12])


def analyze(demonstration, prediction, point):
    left_demo = grip_intervals(demonstration[:, 15], False)
    right_demo = grip_intervals(demonstration[:, 7], True)
    left_model = grip_intervals(prediction[:, 15], False)
    right_model = grip_intervals(prediction[:, 7], True)
    pull_start = max(left_demo, key=lambda pair: pair[1] - pair[0])[0]
    grasp_start, release = max(right_demo, key=lambda pair: pair[1] - pair[0])

    def pull_distance(targets):
        a = point(targets[pull_start], "left")[0]
        b = point(targets[pull_start + FPS], "left")[0]
        return float((a - b) * 1000)

    def first_close(intervals, expected):
        return (intervals[0][0] - expected) / FPS if intervals else None

    grasp_error = None
    if right_model:
        expected = point(demonstration[grasp_start], "right")
        measured = point(prediction[right_model[0][0]], "right")
        grasp_error = float(np.linalg.norm(measured - expected) * 1000)

    closed = prediction[grasp_start:release, 7] > -.305
    return {
        "left_close_delay_s": first_close(left_model, left_demo[0][0]),
        "demo_left_pull_mm_first_second": pull_distance(demonstration),
        "model_left_pull_mm_first_second": pull_distance(prediction),
        "right_close_delay_s": first_close(right_model, right_demo[0][0]),
        "right_grasp_point_error_mm": grasp_error,
        "right_closed_fraction_during_demo_hold": float(closed.mean()),
        "right_sustained_closed_intervals_overlapping_demo_hold": sum(
            start < release and end > grasp_start for start, end in right_model
        ),
    }


def load_replay(root, episode):
    directory = root / f"episode_{episode}" / "stride_20"
    metadata = json.loads((directory / "report.json").read_text())
    with np.load(directory / "full_episode.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in (
            "timestamp_ns", "observed_state", "recorded_ik_target",
            "anchor_indices", "predicted_ik_target",
        )}
    return metadata, arrays


def summarize(rows):
    summary = {}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = {
            "median": float(np.median(values)) if values else None,
            "episode_count": len(values),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    point = GraspPoint()
    rows = []
    for episode in (13, 19, 20, 24):
        old_metadata, old = load_replay(args.baseline, episode)
        new_metadata, new = load_replay(args.candidate, episode)
        for field in ("episode_id", "split", "fps", "source_samples_path",
                      "replan_stride_samples", "seed_rule", "num_flow_steps"):
            if old_metadata[field] != new_metadata[field]:
                raise ValueError(f"Comparison protocol mismatch in episode {episode}: {field}")
        for key in ("timestamp_ns", "observed_state", "recorded_ik_target", "anchor_indices"):
            np.testing.assert_array_equal(old[key], new[key])
        demo = old["recorded_ik_target"]
        rows.append({
            "episode": episode,
            "replans": int(len(old["anchor_indices"])),
            "baseline": analyze(demo, old["predicted_ik_target"], point),
            "candidate": analyze(demo, new["predicted_ik_target"], point),
        })
    report = {
        "baseline_replay": str(args.baseline),
        "candidate_replay": str(args.candidate),
        "baseline_checkpoint": old_metadata["checkpoint"],
        "candidate_checkpoint": new_metadata["checkpoint"],
        "protocol": "Matched real recorded images and feedback; identical fixed seed and 20-frame replan stride.",
        "gripper_events": "At least 10 continuous frames beyond the closure threshold (left < 0.305, right > -0.305 rad).",
        "geometry": "Approximate OpenArm grasp point from FK of pre-driver IK commands. Positive left distance is along nominal drawer pull (-x).",
        "episodes": rows,
        "baseline": summarize([row["baseline"] for row in rows]),
        "candidate": summarize([row["candidate"] for row in rows]),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("baseline", "candidate")}, indent=2))


if __name__ == "__main__":
    main()
