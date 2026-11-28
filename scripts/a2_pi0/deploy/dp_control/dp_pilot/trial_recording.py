"""Per-run evidence. File I/O happens off the inference and motor threads."""
from collections import Counter
import json
from queue import Empty, Full, Queue
import threading
import time

from dp_paths import ROOT

RECORDINGS = ROOT / "logs/openarm-pi0-trials"
VIEWS = ("global","chest","wrist_left","wrist_right")


class TrialRecording:
    @classmethod
    def prepare(cls, session_id, meta):
        return cls(RECORDINGS/session_id, meta)

    def __init__(self, path, meta):
        self.path = path
        path.mkdir(parents=True, exist_ok=False)
        for view in VIEWS: (path/view).mkdir()
        (path/"manifest.json").write_text(json.dumps(
            dict(meta, type="pi0_field_diagnostic", hz=10, views=VIEWS,
                 joint_order="left J1..J7, gripper; right J1..J7, gripper",
                 units="rad", task_success=None), ensure_ascii=False, indent=2))
        self.queue = Queue(maxsize=256)
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.start_ns = None
        self.waypoint_saved = False
        self.stop_info = None
        self.dropped = Counter()
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def start(self, stamp):
        with self.lock:
            if self.error or self.closed.is_set(): return False
            self.start_ns = stamp
            return True

    def put(self, row, frames=()):
        with self.lock:
            if self.start_ns is None or self.closed.is_set() or self.error: return
            if row["anchor_ns"] <= self.start_ns: return
            try: self.queue.put_nowait((row,frames))
            except Full: self.dropped[row["kind"]] += 1

    def waypoint(self,row,frames):
        with self.lock:
            if self.waypoint_saved or self.start_ns is not None or self.closed.is_set() or self.error: return
            try: self.queue.put_nowait((dict(row,kind="waypoint_check"),frames))
            except Full: self.dropped["waypoint_check"] += 1
            else: self.waypoint_saved=True

    def finish(self, stamp, reason):
        with self.lock:
            self.stop_info = {"stop_ns":stamp,"reason":reason}
            self.closed.set()

    def _write(self):
        counts = Counter()
        try:
            with (self.path/"timeline.jsonl").open("x",buffering=1) as out:
                while True:
                    # After finish(), put() can no longer append to this queue.
                    if self.closed.is_set() and self.queue.empty(): break
                    try: row, frames = self.queue.get(timeout=.1)
                    except Empty: continue
                    for f in frames:
                        name = f.metadata["camera"]
                        if name not in VIEWS: raise ValueError("unexpected camera")
                        if row["kind"]=="waypoint_check":
                            (self.path/"waypoint").mkdir(exist_ok=True)
                            dest=self.path/"waypoint"/f"{name}.jpg"
                        else: dest=self.path/name/f"{row['anchor_ns']}.jpg"
                        with dest.open("xb") as im:
                            im.write(f.image)
                    out.write(json.dumps(row,allow_nan=False)+"\n")
                    counts[row["kind"]] += 1
        except Exception as exc:
            self.error = str(exc)
        try:
            summary = dict(self.stop_info or {}, start_ns=self.start_ns, counts=counts,
                           dropped=self.dropped, error=self.error,
                           finalized=self.closed.is_set(), task_success=None)
            (self.path/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        except OSError as exc:
            self.error = str(exc)
