"""Adapt four-camera snapshots and π0 plans to the copied 10 Hz workbench."""
from contextlib import closing
import json
import multiprocessing as mp
import sys
import time

import numpy as np

from dp_paths import CAMERA_DIR, DEPLOY, HERE, ROOT

CONFIG = json.loads((HERE / "pi0_config.json").read_text())
VIEWS = {"global": "camera_port_1", "chest": "camera_port_2_4",
         "wrist_left": "wrist_port_2_3", "wrist_right": "wrist_port_2_2"}
TICK_NS = 100_000_000
INFERENCE_TIMEOUT_NS = CONFIG["inference_timeout_ms"] * 1_000_000
PLAN_AGE_NS = CONFIG["max_plan_age_ms"] * 1_000_000
REPLAN_NS = round(CONFIG["replan_steps"] * 1e9 / 30)


def swap_arms(values):
    return np.concatenate((values[..., 8:16], values[..., :8]), axis=-1)


def _model_process(connection):
    try:
        sys.path.insert(0, str(DEPLOY))
        from policy_server import PolicyService, decode_jpeg

        model = PolicyService(ROOT / CONFIG["checkpoint"], ROOT / CONFIG["run"])
        model.warmup()
        connection.send(("ready",))
        while True:
            images, feedback = connection.recv()
            try:
                targets, _ = model.infer(
                    {name: decode_jpeg(jpeg) for name, jpeg in images.items()},
                    swap_arms(feedback), seed=CONFIG["seed"],
                )
                response = ("targets", time.monotonic_ns(), swap_arms(targets))
            except Exception as exc:
                response = ("error", f"{type(exc).__name__}: {exc}")
            connection.send(response)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class Pi0Policy:
    def __init__(self):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_model_process, args=(child,), daemon=True)
        self.pending = None
        try:
            self.process.start()
            child.close()
            if not self.connection.poll(180):
                raise TimeoutError("π0 预热超时")
            message = self.connection.recv()
            if message != ("ready",):
                raise RuntimeError(*message[1:])
        except BaseException:
            self.close()
            raise
        finally:
            child.close()

    def submit(self, owner, anchor, images, feedback):
        if self.pending is not None:
            raise RuntimeError("π0 仍有推理在途")
        self.pending = owner, anchor, time.monotonic_ns()
        self.connection.send((images, feedback))

    def poll(self, owner):
        if not self.process.is_alive():
            raise RuntimeError("π0 推理进程已退出")
        if self.pending is None:
            return None
        request_owner, observation, started = self.pending
        if not self.connection.poll():
            if time.monotonic_ns() - started > INFERENCE_TIMEOUT_NS:
                raise TimeoutError("π0 推理超时")
            return None
        message = self.connection.recv()
        self.pending = None
        if request_owner is not owner:
            return None
        if message[0] != "targets":
            raise RuntimeError(*message[1:])
        _, finished, targets = message
        if not 0 <= finished - started <= INFERENCE_TIMEOUT_NS:
            raise TimeoutError("π0 推理超时")
        if targets.shape != (50, 16) or not np.isfinite(targets).all():
            raise ValueError("π0 动作必须为有限的 50×16 绝对关节目标")
        return Plan(observation, targets)

    def close(self):
        self.connection.close()
        if self.process.pid is not None:
            self.process.join(1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()


class Plan:
    def __init__(self, observation_ns, targets):
        self.observation_ns = observation_ns
        self.targets = targets
        self.first_dispatch_ns = None

    def at(self, anchor):
        if not 0 <= anchor - self.observation_ns <= PLAN_AGE_NS:
            raise TimeoutError("π0 动作块已过期")
        if self.first_dispatch_ns is None:
            self.first_dispatch_ns = anchor
        step = (anchor - self.first_dispatch_ns) * 30 // 1_000_000_000
        if step >= len(self.targets):
            raise TimeoutError("π0 动作块已执行完")
        return self.targets[step]


def read_snapshot(session, camera, anchor):
    status, frames = camera.request(f"dp_snapshot {anchor}")
    if (status.get("dp_host_anchor_ns") != anchor or status.get("mock") is not False
            or not status.get("ready") or not status.get("basic_image_ready")
            or status.get("snapshot_error")):
        raise ValueError(f"四路真实相机快照未就绪 {status.get('snapshot_error') or status.get('error')}")

    images, stamps = {}, []
    for frame in frames:
        meta = frame.metadata
        view = meta["camera"]
        if (view not in VIEWS or view in images or meta.get("source_camera") != VIEWS[view]
                or meta.get("image_transform") != ("identity-v1" if view == "chest" else "rot180-v1")
                or meta.get("transform_applied_at") != "four-camera-service-before-store"):
            raise ValueError("相机机位或图像方向异常")
        images[view] = frame.image
        stamps.append(meta["timestamp_ns"])
    if images.keys() != VIEWS.keys():
        raise ValueError("四路相机快照不完整")

    arms = session.at(anchor)
    feedback = []
    for side in ("left", "right"):
        stamp, q = arms[side]
        q = np.asarray(q, dtype=np.float32)
        if q.shape != (8,) or not np.isfinite(q).all():
            raise ValueError(f"{side} 反馈格式异常")
        stamps.append(stamp)
        feedback.append(q)
    if any(type(stamp) is not int or not 0 <= anchor - stamp < TICK_NS for stamp in stamps):
        raise ValueError("观测时间戳异常或过期")
    if max(stamps) - min(stamps) > CONFIG["observation_skew_ms"] * 1_000_000:
        raise ValueError("四相机与关节反馈时间跨度过大")
    return frames, arms, images, np.concatenate(feedback)


def predict_live(session, path, policy, on_prediction, on_observation):
    if str(CAMERA_DIR) not in sys.path:
        sys.path.insert(0, str(CAMERA_DIR))
    from bridge import Client

    owner = object()
    next_inference = 0
    permit_after = None
    plan = None
    anchor = time.time_ns() + TICK_NS
    with closing(Client(path)) as camera:
        while not session.locked.is_set():
            if session.locked.wait(max(0, (anchor - time.time_ns()) / 1e9)):
                return
            frames, arms, images, feedback = read_snapshot(session, camera, anchor)
            if not 0 <= time.time_ns() - anchor < TICK_NS:
                raise TimeoutError("四路观测超过工作台节拍")
            on_observation(anchor, frames, feedback, arms)
            new_plan = policy.poll(owner)

            if session.permit_after != permit_after:
                permit_after = session.permit_after
                plan = None
                next_inference = 0
            if new_plan is not None and new_plan.observation_ns > (permit_after or 0):
                plan = new_plan
            if plan is not None and anchor > (permit_after or 0):
                on_prediction(anchor, plan.at(anchor), feedback, plan.observation_ns)
            if policy.pending is None and anchor >= next_inference and anchor > (permit_after or 0):
                policy.submit(owner, anchor, images, feedback)
                next_inference = anchor + REPLAN_NS
            anchor += TICK_NS
