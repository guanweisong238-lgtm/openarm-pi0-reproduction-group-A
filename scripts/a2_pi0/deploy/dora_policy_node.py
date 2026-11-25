#!/usr/bin/env python3
"""Run the Task C policy against the hardware-free OpenArm protocol replay."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time

import dora
import numpy as np
import pyarrow as pa

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = json.loads((HERE / "mock_policy_config.json").read_text())
CAMERAS = ("global", "chest", "wrist_left", "wrist_right")
SIDES = ("right", "left")
TRANSFORMS = {name: "identity-v1" if name == "chest" else "rot180-v1" for name in CAMERAS}
PERIOD_NS = 1_000_000_000 / 30
LATENESS_NS = 15_000_000
MIN_SEND_INTERVAL_NS = int(PERIOD_NS) - 4_000_000
MAX_AGE_NS = CONFIG["observation_max_age_ms"] * 1_000_000
TIMEOUT_NS = CONFIG["inference_timeout_ms"] * 1_000_000
QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])
BUS_TYPE = pa.struct(
    [("carrier", pa.bool_())] + [(key, pa.int64()) for key in (
        "bus_off", "error_passive", "error_warning", "ack_error", "tx_overflow",
        "rx_overflow", "net_down",
    )]
)
LIMITS = np.concatenate([CONFIG["target_limits_rad"][side] for side in SIDES])
VELOCITY = np.tile(CONFIG["joint_velocity_rad_s"], 2)
FEEDBACK_GAP = np.tile(CONFIG["feedback_max_gap_rad"], 2)


def log(event, **fields):
    print(json.dumps(dict(event=event, monotonic_ns=time.monotonic_ns(), **fields),
                     allow_nan=False), flush=True)


def load_model(mode):
    if mode != "model":
        return None
    from policy_server import PolicyService

    logging.getLogger("jax").setLevel(logging.WARNING)
    service = PolicyService(ROOT / CONFIG["checkpoint"], ROOT / CONFIG["run"])
    log("model_ready", checkpoint=CONFIG["checkpoint"], warmup_s=service.warmup())
    return service


def infer(model, mode, frames, state):
    if model is not None:
        from policy_server import decode_jpeg

        images = {name: decode_jpeg(blob) for name, blob in frames.items()}
        return model.infer(images, state, seed=CONFIG["seed"], include_raw=True)

    # Explicit transport fixtures. Their values are never labeled as model output.
    started = time.monotonic_ns()
    time.sleep({"timeout": 0.9, "stale": 0.45}.get(mode, 0.21))
    targets = np.repeat(state[None, :], 50, axis=0)
    # Nominal open targets even when the measured motor angle is slightly past them.
    targets[:, 7] = LIMITS[7, 0]
    targets[:, 15] = LIMITS[15, 1]
    targets[:, 0] += np.linspace(0, 0.0245, 50, dtype=np.float32)
    if mode == "nan":
        targets[0, 0] = np.nan
    if mode == "dimension":
        targets = targets[:, :-1]
    if mode == "range":
        targets[20, 0] = 4
    if mode == "jump":
        targets[:, 0] += 0.3
    return targets, (time.monotonic_ns() - started) / 1e6, None


@dataclass
class ActionPlan:
    seq: int
    observation_ns: int
    observation_monotonic_ns: int
    targets: np.ndarray
    first_dispatch_ns: int | None = None


class Execution:
    def __init__(self, node, pool, mode):
        self.node = node
        self.pool = pool
        self.mode = mode
        self.loading = pool.submit(load_model, mode)
        self.frames = {}
        self.status = {}
        self.feedback = {}
        self.bus = {}
        self.pending = self.plan = self.last_target = None
        self.last_sent = self.last_request = self.next_infer = 0
        self.seq = 0
        self.model = None
        self.blocked = self.paused = self.waiting_for_fresh = False

    def reject(self, reason, **details):
        if self.blocked:
            return
        seq = self.plan.seq if self.plan else self.seq
        log("reject", reason=reason, pending_seq=self.seq, active_seq=seq, **details)
        self.blocked = True
        self.plan = None
        if self.pending:
            self.pending[0].cancel()

    def receive(self, event):
        ident, value = event["id"], event["value"]
        meta = event.get("metadata", {})
        if ident.startswith("camera_"):
            name = ident.removeprefix("camera_")
            stamp = meta.get("timestamp_ns")
            if (meta.get("image_transform") != TRANSFORMS[name] or meta.get("encoding") != "jpeg"
                    or value.type != pa.uint8() or value.null_count or len(value) < 4
                    or type(stamp) is not int):
                raise ValueError(f"camera_format:{name}")
            if name not in self.frames or stamp > self.frames[name][0]:
                self.frames[name] = stamp, value.to_numpy().tobytes()
            return

        side = ident.split("_", 1)[1]
        epoch = meta.get("start_epoch")
        if type(epoch) is not int or epoch < 0:
            raise ValueError(f"epoch_invalid:{side}")
        if ident.startswith("status_"):
            current = (value[0].as_py(), epoch)
            if current[0] == "aligned" and epoch == 0:
                raise ValueError(f"epoch_invalid:{side}")
            if self.seq and current != self.status.get(side):
                raise ValueError(f"status_or_epoch_changed:{side}")
            if self.status.get(side) != current:
                self.feedback.pop(side, None)
                self.bus.pop(side, None)
            self.status[side] = current
            return

        # Arrow schema used by the pinned dora-openarm state output.
        if self.status.get(side) != ("aligned", epoch):
            if self.seq:
                raise ValueError(f"state_epoch_mismatch:{side}")
            return
        state = value[0].as_py()
        qpos = np.asarray(state["qpos"], dtype=np.float32)
        stamp = meta.get("observation_timestamp")
        if qpos.shape != (8,) or not np.isfinite(qpos).all() or type(stamp) is not int:
            raise ValueError(f"feedback_format:{side}")
        if side in self.feedback and stamp <= self.feedback[side][0]:
            return
        bus = state["bus"]
        if not bus["carrier"] or state["motor_status"] != ["ENABLED"] * 8:
            raise ValueError(f"feedback_health:{side}")
        if side in self.bus and bus != self.bus[side]:
            raise ValueError(f"bus_counters_changed:{side}")
        self.bus[side] = bus
        self.feedback[side] = stamp, qpos

    def tick(self):
        if self.blocked:
            return
        now = time.monotonic_ns()
        # Request state separately so a missing action cannot suppress feedback.
        if now - self.last_request >= PERIOD_NS:
            for side in SIDES:
                if self.status.get(side, (None,))[0] == "aligned":
                    self.node.send_output(f"request_{side}", pa.array([True]))
            self.last_request = now

        if self.loading is not None:
            if not self.loading.done():
                return
            self.model = self.loading.result()
            self.loading = None

        wall = time.time_ns()
        if len(self.feedback) != 2 or len(self.frames) != 4:
            return
        stamps = {f"camera_{name}": self.frames[name][0] for name in CAMERAS}
        stamps.update({f"state_{side}": self.feedback[side][0] for side in SIDES})
        oldest, newest = min(stamps.values()), max(stamps.values())
        if not 0 <= wall - newest <= wall - oldest <= MAX_AGE_NS:
            if self.seq:
                self.reject("observation_stale_or_future", source_times_ns=stamps)
            elif not self.waiting_for_fresh:
                self.waiting_for_fresh = True
                log("waiting_for_fresh_observation", source_times_ns=stamps)
            return

        state = np.concatenate([self.feedback[side][1] for side in SIDES])
        if self.pending:
            future, started, observation_monotonic_ns, observation_ns = self.pending
            if now - started > TIMEOUT_NS:
                self.reject("inference_timeout", obs_timestamp_ns=observation_ns,
                            elapsed_ms=(now - started) / 1e6)
                return
            if future.done():
                self.pending = None
                targets, infer_ms, normalized = future.result()
                log("inference", seq=self.seq, obs_timestamp_ns=observation_ns,
                    infer_ms=infer_ms, total_ms=(now - started) / 1e6, backend=self.mode,
                    model_normalized=normalized.tolist() if normalized is not None else None,
                    target_shape=list(targets.shape),
                    targets_rad=targets.tolist() if np.isfinite(targets).all() else None)
                if targets.shape != (50, 16) or not np.isfinite(targets).all():
                    self.reject("action_shape_or_nonfinite", obs_timestamp_ns=observation_ns)
                    return
                bad = np.argwhere((targets < LIMITS[:, 0]) | (targets > LIMITS[:, 1]))
                if bad.size:
                    step, axis = map(int, bad[0])
                    self.reject("target_out_of_range", obs_timestamp_ns=observation_ns,
                                step=step, axis=axis, value=float(targets[step, axis]),
                                allowed_rad=LIMITS[axis].tolist())
                    return
                if (np.abs(np.diff(targets, axis=0)) > VELOCITY / 30 + 1e-6).any():
                    self.reject("plan_step_too_large", obs_timestamp_ns=observation_ns)
                    return
                self.plan = ActionPlan(self.seq, observation_ns, observation_monotonic_ns, targets)

        now = time.monotonic_ns()
        if plan := self.plan:
            if now - plan.observation_monotonic_ns > CONFIG["max_plan_age_ms"] * 1_000_000:
                self.reject("plan_expired", obs_timestamp_ns=plan.observation_ns)
                return
            if now - self.last_sent >= MIN_SEND_INTERVAL_NS:
                # Start each accepted plan at step 0 on its first eligible tick.
                # The observation age remains anchored to the inference input.
                if plan.first_dispatch_ns is None:
                    plan.first_dispatch_ns = now
                elapsed = now - plan.first_dispatch_ns
                step = int(elapsed // PERIOD_NS)
                due = plan.first_dispatch_ns + round(step * PERIOD_NS)
                if step >= len(plan.targets):
                    self.reject("plan_expired", obs_timestamp_ns=plan.observation_ns)
                    return
                if 0 <= now - due <= LATENESS_NS and due > self.last_sent:
                    self.send_target(state, plan.targets[step], now, due, plan.seq, step, plan.observation_ns, oldest)
                    if self.blocked:
                        return

        if not self.pending and now >= self.next_infer:
            wall = time.time_ns()
            if wall - oldest > MAX_AGE_NS:
                self.reject("observation_stale_or_future", source_times_ns=stamps)
                return
            if newest - oldest > CONFIG["observation_skew_ms"] * 1_000_000:
                return
            frames = {name: self.frames[name][1] for name in CAMERAS}
            self.seq += 1
            self.pending = (self.pool.submit(infer, self.model, self.mode, frames, state),
                            now, now - (wall - newest), newest)
            self.next_infer = now + round(CONFIG["replan_steps"] * PERIOD_NS)
            log("observation", seq=self.seq, obs_timestamp_ns=newest, source_times_ns=stamps,
                feedback_rad=state.tolist(), epochs={key: value[1] for key, value in self.status.items()})

    def send_target(self, state, target, now, due, seq, step, observation_ns, oldest):
        if time.time_ns() - oldest > MAX_AGE_NS:
            self.reject("observation_stale_or_future", obs_timestamp_ns=observation_ns)
            return
        gap = np.abs(target - state)
        allowed_gap = FEEDBACK_GAP if self.last_target is not None else np.minimum(FEEDBACK_GAP, VELOCITY / 30)
        if (gap > allowed_gap).any():
            self.reject("target_feedback_gap", obs_timestamp_ns=observation_ns,
                        step=step, gap_rad=gap.tolist())
            return
        if self.last_target is not None:
            # Do not grant more than one nominal step after a dropped or delayed tick.
            dt = min(now - self.last_sent, PERIOD_NS) / 1e9
            if (np.abs(target - self.last_target) > VELOCITY * dt + 1e-6).any():
                self.reject("dispatch_step_too_large", obs_timestamp_ns=observation_ns, step=step)
                return
        for index, side in enumerate(SIDES):
            arm_target = target[index * 8:(index + 1) * 8]
            self.node.send_output(
                f"target_{side}", pa.array([{"qpos": arm_target}], type=QPOS_TYPE),
                {"start_epoch": self.status[side][1], "seq": seq, "step": step,
                 "obs_timestamp_ns": observation_ns},
            )
        self.last_sent, self.last_target = now, target
        log("dispatch", seq=seq, step=step, obs_timestamp_ns=observation_ns,
            sent_ns=now, due_ns=due, sent_target_rad=target.tolist(), feedback_rad=state.tolist(),
            feedback_ns={side: value[0] for side, value in self.feedback.items()})
        if self.mode == "pause" and not self.paused and step >= 9:
            self.paused = True
            time.sleep(0.08)


def main():
    mode = os.environ.get("OPENARM_MOCK_CASE", "model")
    if mode not in {"model", "transport", "pause", "stale", "disconnect", "timeout",
                    "nan", "dimension", "range", "jump", "epoch", "status_stop"}:
        raise ValueError(f"Unknown mock case: {mode}")
    node = dora.Node()
    with ThreadPoolExecutor(max_workers=1) as pool:
        controller = Execution(node, pool, mode)
        stopped_by = "input_stream_closed"
        try:
            for event in node:
                if event["type"] == "STOP":
                    stopped_by = "dora_stop"
                    break
                if event["type"] == "INPUT_CLOSED" and event["id"] != "tick":
                    controller.reject("input_closed", input=event["id"])
                if event["type"] != "INPUT" or controller.blocked:
                    continue
                try:
                    if event["id"] == "tick":
                        controller.tick()
                    else:
                        controller.receive(event)
                except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
                    controller.reject(str(exc), input=event["id"])
        finally:
            controller.plan = None
            if controller.pending:
                controller.pending[0].cancel()
            log("policy_exit", source=stopped_by, last_sent_ns=controller.last_sent)


if __name__ == "__main__":
    main()
