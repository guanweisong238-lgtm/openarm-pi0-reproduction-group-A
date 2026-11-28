"""Task C start targets from episode 0 of real_20261128_202032."""
from dataclasses import dataclass
import json

import numpy as np

from dp_paths import HERE, ROOT

DEMO = HERE / "task_start.json"
DEMO_ID, DEMO_FRAME = 0, 0


@dataclass(frozen=True)
class Release:
    arch: str = "π0"
    checkpoint = ROOT / json.loads((HERE / "pi0_config.json").read_text())["checkpoint"]
    pose_sha256: str = "b618c802bcaf0589cf6eef0c05229c8e67822527a8c574cee369296ae9985210"


RELEASE = Release()


def start_row():
    data = json.loads(DEMO.read_text())
    if (data["episode"], data["frame"]) != (DEMO_ID, DEMO_FRAME):
        raise ValueError("示范起点编号不符")
    return np.asarray(data["qpos_rad"], dtype="<f4")
