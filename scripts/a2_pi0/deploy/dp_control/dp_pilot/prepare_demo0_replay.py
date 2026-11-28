"""Audit recorded joint angles and prepare offline replay files for episode 0."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from dp_paths import ROOT, SDK_PY


SIDES = ("right", "left")
ARM_AXES = np.r_[0:7, 8:15]
AXES = [f"{side}_J{joint}" for side in SIDES for joint in range(1, 8)]
RAW = ROOT / "data/real_20261128_202032/episodes/0/obs/arms"
ALIGNED = ROOT / "data/processed/real_20261128_202032_alignment_v001/episode_0/samples.npz"
SPEED_CAP = 0.20
ACCEL_CAP = 0.50
REPLAY_HZ = 50


def driver_limits():
    code = """
import json
from openarm_driver import Config
c = Config('openarm_pedestal')
print(json.dumps([c.get_joint_limits(side + '_arm').tolist() for side in ('right', 'left')]))
"""
    return np.asarray(json.loads(subprocess.check_output([SDK_PY, "-B", "-c", code]))).reshape(16, 2)


def load_checked(limits):
    with np.load(ALIGNED, allow_pickle=False) as data:
        stamps = data["timestamp_ns"]
        q = data["state"].astype(float)
        if not len(stamps) or q.shape != (len(stamps), 16) or np.any(np.diff(stamps) <= 0):
            raise ValueError("inconsistent aligned observations")
        sources = {}
        for index, side in enumerate(SIDES):
            path = RAW / side / "state.parquet"
            table = pq.read_table(path, columns=["timestamp", "qpos"])
            raw_stamps = table["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
            raw_q = np.stack(table["qpos"].to_pylist())
            rows = np.searchsorted(raw_stamps, stamps, side="right") - 1
            if (np.any(np.diff(raw_stamps) <= 0) or np.any(rows < 0)
                    or not np.array_equal(raw_stamps[rows], data[f"obs_timestamp_ns_{side}"])
                    or not np.array_equal(raw_q[rows].astype(np.float32), q[:, index*8:(index+1)*8])):
                raise ValueError(f"{side}: aligned angles or timestamps differ from raw observations")
            arm_limits = limits[index*8:index*8+7]
            arms = raw_q[:, :7]
            if not np.isfinite(raw_q).all() or np.any((arms < arm_limits[:, 0]) | (arms > arm_limits[:, 1])):
                raise ValueError(f"{side}: raw observations are invalid or beyond the driver arm limits")
            ages = (stamps - raw_stamps[rows]) / 1e6
            sources[side] = {"rows": len(raw_q), "max_age_ms": float(ages.max()),
                             "first_second_max_age_ms": float(ages[stamps - stamps[0] <= 1e9].max()),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return (stamps - stamps[0]) / 1e9, q, sources


def phase_path(t, q):
    tangents = PchipInterpolator(t, q, axis=0).derivative()(t)
    tangents[[0, -1]] = 0
    return CubicHermiteSpline(t, q, tangents, axis=0)


def peak_derivatives(path):
    cubic, quadratic, linear, _ = path.c
    dt = np.diff(path.x)[:, None]
    # Velocity extrema are at the ends or where acceleration crosses zero.
    turn = np.divide(-quadratic, 3*cubic, out=np.zeros_like(cubic), where=cubic != 0)
    points = np.stack((np.zeros_like(cubic), np.broadcast_to(dt, cubic.shape), np.clip(turn, 0, dt)))
    speed = np.max(np.abs((3*cubic*points + 2*quadratic)*points + linear))
    accel = np.max(np.abs(np.stack((2*quadratic, 6*cubic*dt + 2*quadratic))))
    return float(speed), float(accel)


def prepare_path(seconds, q, limits):
    path = phase_path(seconds, q)
    speed, accel = peak_derivatives(path)
    scale = max(1, speed/SPEED_CAP, np.sqrt(accel/ACCEL_CAP))
    ticks = int(np.ceil(seconds[-1]*scale*REPLAY_HZ))
    time = np.arange(ticks+1)/REPLAY_HZ
    scale = time[-1]/seconds[-1]
    angles = path(time/scale)
    if not np.all((angles >= limits[:, 0]) & (angles <= limits[:, 1])):
        raise ValueError("replay targets exceed the driver arm limits")
    summary = dict(source_duration_s=float(seconds[-1]), duration_s=float(time[-1]),
                   rate_hz=REPLAY_HZ, samples=len(time), peak_speed_rad_s=speed/scale,
                   peak_accel_rad_s2=accel/scale**2,
                   max_displacement_deg=float(np.rad2deg(np.max(abs(angles-angles[0])))))
    return time, angles, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    limits = driver_limits()
    seconds, q, sources = load_checked(limits)
    arm_q = q[:, ARM_AXES]
    arm_limits = limits[ARM_AXES]
    first_second = seconds <= 1.0
    preview = prepare_path(seconds[first_second], arm_q[first_second], arm_limits)
    full = prepare_path(seconds, arm_q, arm_limits)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, (time, angles, _) in (("first_second_offline.npz", preview), ("full_demo_offline.npz", full)):
        np.savez_compressed(args.out / name, time_s=time, q_rad=angles, axes=AXES, offline_only=True)
    report = dict(
        field_execution_approved=False, dataset="real_20261128_202032", episode=0,
        contact_during_capture={"reported_contact": False, "source": "user"},
        raw_sources=sources,
        aligned_sha256=hashlib.sha256(ALIGNED.read_bytes()).hexdigest(), axis_order=AXES,
        source_frames=len(seconds), source_duration_s=float(seconds[-1]),
        recorded_peak_sample_speed_rad_s=float(np.max(abs(np.diff(arm_q, axis=0)/np.diff(seconds)[:, None]))),
        angle_limits_rad=limits[ARM_AXES].tolist(), angle_min_rad=arm_q.min(0).tolist(),
        angle_max_rad=arm_q.max(0).tolist(), start_rad=arm_q[0].tolist(),
        target_speed_cap_rad_s=SPEED_CAP, target_accel_cap_rad_s2=ACCEL_CAP,
        preview=preview[2], full=full[2])
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
