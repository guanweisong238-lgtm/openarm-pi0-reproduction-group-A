"""Paths for the copied controller and the existing deployment runtimes."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parents[1]
ROOT = DEPLOY.parents[2]
SOURCE = Path(os.environ.get("OPENARM_DP_SOURCE", ROOT.parent / "openarm_diffusion_policy")).resolve()
SOURCE_LOCAL = SOURCE / "vr_teleoperation/local"
CAMERA_DIR = SOURCE_LOCAL / "four_camera"
MODEL = SOURCE / "openarm_v2/vendor/openarm_mujoco/v2/pedestal/pedestal.xml"
SDK_PY = Path(os.environ.get(
    "OPENARM_PI0_DRIVER_PYTHON", ROOT / ".cache/openarm-policy-driver/bin/python"
)).absolute()
