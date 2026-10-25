"""按批次清单逐条审阅示教；一次一个窗口，N/关窗下一条，Q退出整组。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from collect_drawer_trial import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/drawer_expansion/dataset_v002.json")
    parser.add_argument("--batch-config", type=Path, default=ROOT / "configs/drawer-expansion-batch002.json")
    parser.add_argument("--headless", action="store_true", help="无窗口依次校验全部选中轨迹")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    batch = json.loads(args.batch_config.read_text())
    rows = {row["path"]: row for row in manifest["episodes"]}
    selected = []
    for profile in batch["episodes"]:
        relative = str(Path(batch["output_root"]) / profile["name"])
        row = rows[relative]
        if row["split"] != profile["split"]:
            raise ValueError("批次和数据清单的split不一致")
        episode = ROOT / relative / "episode.npz"
        if hashlib.sha256(episode.read_bytes()).hexdigest() != row["episode_sha256"]:
            raise ValueError(f"轨迹哈希不符：{episode}")
        selected.append((row, episode))
    env = os.environ.copy()
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    for index, (row, episode) in enumerate(selected, 1):
        label = f"{index}/{len(selected)} {row['split'].upper()}"
        print(f"REVIEW {label}: {episode.parent.name}", flush=True)
        command = [sys.executable, str(ROOT / "scripts/a2_pi0/replay_drawer_trial.py"),
                   "--episode", str(episode), "--label", label]
        command += ["--headless"] if args.headless else ["--review-controls"]
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode == 20:
            print("REVIEW_STOPPED：用户退出整组审阅", flush=True)
            return
        if result.returncode:
            raise SystemExit(result.returncode)
    print("REVIEW_FINISHED：全部轨迹结束；此消息不代表用户已逐条人工验收", flush=True)


if __name__ == "__main__":
    main()
