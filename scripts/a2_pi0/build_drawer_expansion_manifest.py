"""审核原示教和扩充批次，生成独立的训练/验证清单；不改训练配置和归一化统计。"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_drawer_expansion import audit
from collect_drawer_trial import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    requested = [("data/drawer_trial/contact_roundtrip_rgb_001", "train"),
                 ("data/drawer_expansion/pilot_rgb/pitch12", "train"),
                 ("data/drawer_expansion/pilot_rgb/pitch18", "train")]
    requested += [(str(Path(config["output_root"]) / p["name"]), p["split"])
                  for p in config["episodes"]]
    rows, groups, actions = [], {}, set()
    for relative, split in requested:
        directory = ROOT / relative
        row = audit(directory)
        if row["status"] != "rgb_candidate":
            raise ValueError(f"RGB轨迹未通过审核：{relative}")
        provenance = json.loads((directory / "report.json").read_text())
        # 把同一机器人初态、场景和相机条件的多种专家路径保持在同一split。
        overrides = provenance["scene_overrides"]
        scene = {"initial_qpos": row["initial_qpos_sha256"],
                 "scene_sha256": provenance["scene_sha256"],
                 "scene_overrides": {"x": overrides.get("drawer_offset_x_m", 0.0),
                                     "y": overrides.get("drawer_offset_y_m", 0.0),
                                     "radius": overrides.get("drawer_handle_radius_m")},
                 "cameras": provenance["camera_local_poses"]}
        group = hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()
        if group in groups and groups[group] != split:
            raise ValueError("相同初始条件跨训练/验证集合")
        groups[group] = split
        if row["action_sha256"] in actions:
            raise ValueError("重复动作轨迹，不应计为独立示教")
        actions.add(row["action_sha256"])
        rows.append({"path": relative, "split": split, "frames": row["frames"],
                     "condition_group": group,
                     "episode_sha256": hashlib.sha256((directory / "episode.npz").read_bytes()).hexdigest(),
                     "scene_overrides": provenance["scene_overrides"],
                     "expert_parameters": provenance["expert_parameters"],
                     "initial_right_randomization": provenance.get("initial_right_randomization")})
    result = {"name": config["name"], "episodes": rows,
              "split_note": "Episode/initial-condition grouped; small local validation, not broad generalization. Numeric copies are excluded.",
              "training_integrated": False,
              "normalization_note": "No existing statistics modified. Multi-episode training requires explicit configuration and training-only statistics."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        print(split, "episodes", len(selected), "frames", sum(row["frames"] for row in selected))
    print("manifest", args.output)


if __name__ == "__main__":
    main()
