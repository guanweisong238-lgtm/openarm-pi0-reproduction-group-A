"""Download public checkpoint objects and validate their stored bytes."""

import base64
import hashlib
import logging
from pathlib import Path
import time
from urllib.parse import quote

import filelock
import fsspec
import requests


PREFIX = "openpi-assets/checkpoints/pi0_aloha_sim"


def matches(path: Path, info: dict) -> bool:
    if not path.is_file() or path.stat().st_size != info["size"]:
        return False
    if not info.get("md5Hash"):
        raise RuntimeError(f"Remote MD5 unavailable: {info['name']}")
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode() == info["md5Hash"]


def ensure_checkpoint(cache_root: Path, *, verify_only: bool = False) -> Path:
    target = cache_root / PREFIX
    target.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(target) + ".lock"):
        fs = fsspec.filesystem("gs", token="anon", requests_timeout=30)
        entries = fs.find(PREFIX, detail=True)
        objects = [
            (name, info) for name, info in entries.items()
            if info["type"] == "file" and not name.endswith("/")
        ]
        if not objects:
            raise RuntimeError("Remote checkpoint listing is empty")
        invalid = []
        for name, info in objects:
            path = target / name.removeprefix(PREFIX + "/")
            if matches(path, info):
                logging.info("Verified size + MD5: %s", path.relative_to(target))
            else:
                invalid.append((name, info, path))
        logging.info("Checkpoint audit: %d verified, %d missing or invalid", len(objects) - len(invalid), len(invalid))
        if verify_only and invalid:
            raise RuntimeError(f"Checkpoint incomplete: {len(invalid)} files require download")

        with requests.Session() as session:
            for name, info, path in invalid:
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + ".download-part")
                url = "https://storage.googleapis.com/" + quote(name, safe="/")
                for attempt in range(1, 4):
                    try:
                        logging.info("Downloading %s (%d bytes), attempt %d/3", name, info["size"], attempt)
                        received = 0
                        next_report = 256 * 1024 * 1024
                        # Bind to the generation whose checksum was listed above.
                        params = {"generation": info["generation"]}
                        with session.get(url, params=params, stream=True, timeout=(15, 120)) as response:
                            response.raise_for_status()
                            with partial.open("wb") as output:
                                # Preserve stored bytes even if GCS sets Content-Encoding.
                                for block in response.raw.stream(8 * 1024 * 1024, decode_content=False):
                                    output.write(block)
                                    received += len(block)
                                    if received >= next_report:
                                        logging.info("%s: %.1f / %.1f MiB", path.name, received / 2**20, info["size"] / 2**20)
                                        next_report = received + 256 * 1024 * 1024
                        if not matches(partial, info):
                            raise RuntimeError(f"Size or MD5 mismatch: {name}")
                        partial.replace(path)
                        logging.info("Downloaded and verified: %s", path.relative_to(target))
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        logging.exception("Download attempt failed; retrying")
                        time.sleep(1)
        logging.info("Checkpoint verified: %d files, size + MD5 checks passed", len(objects))
    return target
