#!/usr/bin/env python3
"""Forward the existing four-camera service's JPEG snapshots over Dora."""
import os
from pathlib import Path
import sys

import dora
import numpy as np
import pyarrow as pa

from dora_policy_node import CAMERAS, ROOT, TRANSFORMS, log


def main():
    source_dir = Path(os.environ.get(
        "OPENARM_FOUR_CAMERA_DIR",
        ROOT.parent / "openarm_diffusion_policy/vr_teleoperation/local/four_camera",
    ))
    sys.path.insert(0, str(source_dir))
    from bridge import Client

    node = dora.Node()
    client = Client(Path(os.environ["OPENARM_CAMERA_SOCKET"]))
    previous = {}
    previous_health = None
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] != "tick":
                continue
            status, frames = client.request("snapshot")
            if status.get("error"):
                raise RuntimeError(status["error"])
            ready = bool(status.get("ready") and status.get("basic_image_ready"))
            if ready != previous_health:
                log("camera_socket_health", ready=ready, reason=status.get("snapshot_error"))
                previous_health = ready
            if not ready:
                continue
            by_name = {frame.metadata["camera"]: frame for frame in frames}
            if len(frames) != len(CAMERAS) or set(by_name) != set(CAMERAS):
                raise ValueError("Expected four distinct camera JPEGs")
            for name, frame in by_name.items():
                meta = frame.metadata
                stamp = meta.get("timestamp_ns")
                if (type(stamp) is not int or meta.get("encoding") != "jpeg"
                        or meta.get("image_transform") != TRANSFORMS[name]):
                    raise ValueError(f"Unexpected camera payload: {name}")
                if stamp > previous.get(name, 0):
                    node.send_output(f"camera_{name}", pa.array(np.frombuffer(frame.image, np.uint8)), {
                        "timestamp_ns": stamp, "image_transform": meta["image_transform"],
                        "encoding": "jpeg",
                    })
                    previous[name] = stamp
    finally:
        client.close()
        log("camera_socket_exit", last_camera_times_ns=previous)


if __name__ == "__main__":
    main()
