#!/usr/bin/env bash
set -euo pipefail
deploy=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$deploy/../../.." && pwd)
pilot="$deploy/dp_control/dp_pilot"
model_python="$root/.venv/bin/python"
driver_python=${OPENARM_PI0_DRIVER_PYTHON:-"$root/.cache/openarm-policy-driver/bin/python"}

case "${1:-}" in
  -h|--help) echo "用法：$0 [--check | --no-gravity]"; exit 0 ;;
  --check|--no-gravity|'') (( $# <= 1 )) || exit 2 ;;
  *) echo "未知参数：$1" >&2; exit 2 ;;
esac

"$model_python" -B - "$pilot" <<'PY'
import sys
import tkinter

sys.path.insert(0, sys.argv[1])
from dp_paths import CAMERA_DIR, ROOT, SOURCE_LOCAL
from pi0_observations import CONFIG
from task_reference import RELEASE

for path in (CAMERA_DIR / "service.py", CAMERA_DIR / "bridge.py",
             SOURCE_LOCAL / "camera_bandwidth/check_four_cameras.py",
             RELEASE.checkpoint / "checkpoint.json", RELEASE.checkpoint / "lora_params.npz",
             ROOT / CONFIG["run"] / "config.json"):
    if not path.is_file():
        raise SystemExit(f"缺少部署文件 {path}")
sys.path.insert(0, str(CAMERA_DIR))
from bridge import Client
import cv2
PY
"$driver_python" -B - "$pilot" <<'PY'
import sys

import openarm_can
from openarm_driver import Config, SingleArmDriver

sys.path.insert(0, sys.argv[1])
from dp_paths import HERE
from motion_arm import task_start_pose

sys.path.insert(0, str(HERE.parent))
from vr_gravity import Gravity

config = Config("openarm_pedestal")
for side in ("left", "right"):
    task_start_pose(side, config.get_joint_limits(f"{side}_arm"))
    Gravity(side).torque([0.] * 8)
PY

if [[ "${1:-}" == --check ]]; then
  echo '离线检查通过：控制依赖、四路相机模块、示范起点和重力模型。'
  exit 0
fi

# Reuse the source workbench's CAN FD checks without changing the interfaces.
python3 - <<'PY_CAN'
import json
import os
import subprocess

parent = os.environ.get("OPENARM_PI0_CAN_PARENT", "3-2.1:1.0")
rows = {row["ifname"]: row for row in json.loads(subprocess.check_output(
    ["ip", "-j", "-details", "link", "show", "type", "can"]))}
for name in ("can0", "can1"):
    row = rows.get(name, {})
    if (row.get("parentbus"), row.get("parentdev")) != ("usb", parent):
        raise SystemExit(f"{name} 未检测到现场核对的 USB 双路适配器 {parent}")
    if "UP" not in row["flags"]:
        raise SystemExit(f"{name} 尚未准备，请先完成现场 CAN 配置")
    cfg = row["linkinfo"]["info_data"]
    bit, data = cfg.get("bittiming", {}), cfg.get("data_bittiming", {})
    if not (cfg.get("ctrlmode") == ["FD"] and cfg.get("state") == "ERROR-ACTIVE"
            and cfg.get("restart_ms") == 0 and bit.get("bitrate") == 1000000
            and data.get("bitrate") == 5000000 and bit.get("sample_point") == .75
            and data.get("sample_point") == .75 and data.get("sjw") == 2):
        raise SystemExit(f"{name} 的 CAN FD 状态或参数异常")
PY_CAN

[[ -n "${DISPLAY:-}" ]] || { echo '请在本机图形桌面的终端运行。' >&2; exit 1; }
mkdir -p -m 700 "$HOME/.cache/openarm-dp-field"
exec 9>"$HOME/.cache/openarm-dp-field/motion-launch.lock"
flock -n 9 || { echo '已有双臂运动工作台。' >&2; exit 1; }
exec "$model_python" -B "$pilot/live_motion.py" "$@"
