#!/usr/bin/env python3
"""Replay recorded camera bytes and emulate only the pinned Dora driver's wire interface."""
import hashlib
import json
import os
from pathlib import Path
import time

import dora
import numpy as np
import pyarrow as pa

from dora_policy_node import BUS_TYPE, CAMERAS, CONFIG, QPOS_TYPE, ROOT, SIDES, TRANSFORMS, log

STATE_TYPE = pa.struct([
    ("qpos", pa.list_(pa.float32())),
    ("qvel", pa.list_(pa.float32())),
    ("qtorque", pa.list_(pa.float32())),
    ("tmos", pa.list_(pa.int32())),
    ("trotor", pa.list_(pa.int32())),
    ("motor_status", pa.list_(pa.string())),
    ("bus", BUS_TYPE),
])
HEALTHY_BUS = {name: 0 for name in BUS_TYPE.names}
HEALTHY_BUS["carrier"] = True


def recorded_sample():
    if directory := os.environ.get("OPENARM_MOCK_SNAPSHOT_DIR"):
        snapshot = Path(directory)
        capture = json.loads((snapshot / "capture.json").read_text())
        state = np.asarray(
            [capture["qpos_rad"][f"{side}_arm"] for side in SIDES], dtype=np.float32
        )
        if state.shape != (2, 8) or not np.isfinite(state).all():
            raise ValueError("Snapshot must contain eight finite positions per arm")
        images = {
            name: pa.array(np.frombuffer((snapshot / f"{name}.jpg").read_bytes(), dtype=np.uint8))
            for name in CAMERAS
        }
        return state.flatten(), images, {name: int(capture["frame_ns"][name]) for name in CAMERAS}, None

    run = json.loads((ROOT / CONFIG["run"] / "config.json").read_text())
    candidate = ROOT / run["candidate"]
    manifest = (candidate / "manifest.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != run["candidate_manifest_sha256"]:
        raise ValueError("Replay manifest hash mismatch")
    replay_index = int(os.environ.get("OPENARM_MOCK_REPLAY_INDEX", CONFIG["replay_index"]))
    index = replay_index
    if index < 0:
        raise ValueError("Replay index must be nonnegative")
    for entry in json.loads(manifest)["entries"]:
        if entry["split"] != "validation":
            continue
        if index >= entry["full_window_samples"]:
            index -= entry["full_window_samples"]
            continue
        path = ROOT / entry["samples_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["samples_sha256"]:
            raise ValueError("Replay sample hash mismatch")
        with np.load(path, allow_pickle=False) as samples:
            state = samples["state"][index]
            images = {
                name: pa.array(np.frombuffer(
                    (ROOT / str(samples[f"image_path_{name}"][index])).read_bytes(), dtype=np.uint8))
                for name in CAMERAS
            }
            stamps = {name: int(samples[f"image_timestamp_ns_{name}"][index]) for name in CAMERAS}
            return state, images, stamps, entry["episode_id"]
    raise IndexError(replay_index)


def main():
    node = dora.Node()
    mode = os.environ.get("OPENARM_MOCK_CASE", "model")
    state, images, recorded_stamps, episode = recorded_sample()
    feedback = dict(zip(SIDES, np.split(state, 2), strict=True))
    epochs = {"right": 41, "left": 57}
    counts = dict.fromkeys(SIDES, 0)
    last_target_ns = 0
    frames = 0
    stopped = False
    zeros = [0] * 8

    def status(side, value):
        node.send_output(f"status_{side}", pa.array([value]),
                         {"start_epoch": epochs[side], "timestamp": time.time_ns()})

    for side in SIDES:
        status(side, "aligned")
    log("mock_ready", case=mode, episode=episode, recorded_image_ns=recorded_stamps)
    stopped_by = "input_stream_closed"
    try:
        for event in node:
            if event["type"] == "STOP":
                stopped_by = "dora_stop"
                break
            if event["type"] != "INPUT":
                continue
            ident = event["id"]
            now = time.time_ns()
            if ident == "camera_tick":
                if mode != "stale" or frames < 8:
                    for name, image in images.items():
                        node.send_output(f"camera_{name}", image,
                                         {"timestamp_ns": now, "encoding": "jpeg",
                                          "image_transform": TRANSFORMS[name],
                                          "recorded_timestamp_ns": recorded_stamps[name]})
                    frames += 1
                if last_target_ns and now - last_target_ns > 180_000_000 and not stopped:
                    stopped = True
                    log("mock_watchdog", targets=counts)
            else:
                operation, side = ident.split("_", 1)
                if operation == "request":
                    if mode == "disconnect" and side == "left" and counts[side] >= 4:
                        continue
                    epoch = epochs[side]
                    if mode == "epoch" and side == "left" and counts[side] >= 3:
                        epoch += 1
                    value = pa.array([{"qpos": feedback[side], "qvel": zeros, "qtorque": zeros,
                                       "tmos": zeros, "trotor": zeros, "motor_status": ["ENABLED"] * 8,
                                       "bus": HEALTHY_BUS}], type=STATE_TYPE)
                    node.send_output(f"state_{side}", value,
                                     {"start_epoch": epoch, "observation_timestamp": now})
                else:
                    meta = event["metadata"]
                    value = event["value"]
                    if (stopped or value.type != QPOS_TYPE or len(value) != 1
                            or meta.get("start_epoch") != epochs[side]):
                        log("mock_rejected", side=side, metadata_epoch=meta.get("start_epoch"))
                        continue
                    target = np.asarray(value[0].as_py()["qpos"], dtype=np.float32)
                    if target.shape != (8,) or not np.isfinite(target).all():
                        raise ValueError("Unexpected target in mock endpoint")
                    # Instant feedback is a wire-test fixture, without dynamics.
                    feedback[side] = target
                    counts[side] += 1
                    last_target_ns = now
                    log("mock_target", side=side, seq=meta["seq"], step=meta["step"],
                        epoch=meta["start_epoch"], obs_timestamp_ns=meta["obs_timestamp_ns"],
                        qpos_rad=target.tolist())
                    if mode == "status_stop" and min(counts.values()) == 3:
                        status("left", "stopped")
    finally:
        log("mock_exit", source=stopped_by, targets=counts)


if __name__ == "__main__":
    main()
