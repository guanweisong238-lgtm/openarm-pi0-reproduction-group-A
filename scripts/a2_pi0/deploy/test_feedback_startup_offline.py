"""Exercise the official start/stop path with simulated CAN and Dora events."""
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa

import feedback_trial

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".cache/openarm-policy-driver/lib/python3.11/site-packages"))
from openarm_driver.driver import SingleArmDriver, _BUS_COUNTER_NAMES
import openarm_can


def sample(gripper="ENABLED", **fields):
    bus = dict.fromkeys(_BUS_COUNTER_NAMES, 0) | {"carrier": True} | fields.pop("bus", {})
    defaults = dict(status=["ENABLED"] * 7 + [gripper], bus=bus, link=True,
                    position=0., missing=False, late=False, read_delay=0)
    return SimpleNamespace(**(defaults | fields))


class FeedbackStartupTest(unittest.TestCase):
    def run_trial(self, initial, extra=(), *, accepted=False, motion=False, running_fault=None,
                  startup_fault=None, initial_refresh=None, gripper_mode=4, encoders=False,
                  missing_encoder=None, delayed_gripper_state=False, side="right", j7_kp=None,
                  record_recovery=False):
        enabled = sample()
        disabled = sample(status=["DISABLED"] * 8)
        # The official start performs one refresh before calling the startup hook.
        running = ([running_fault] if running_fault is not None
                   else [enabled] * 10 if accepted or missing_encoder is not None else [])
        frames = deque([disabled, initial_refresh or initial, initial, *extra, *running])
        now = 100.
        responses = [0] * 8
        calls = []
        outputs = []
        current = disabled
        pending = None
        pending_reads = 0
        mode_value = -1
        mode_pending = False
        gripper_state_seen = False
        gripper_mode_malformed = 0
        encoder_responses = 0
        encoder_pending = None
        encoder_value = {}

        def sleep(seconds):
            nonlocal now
            now += seconds

        def refresh_all():
            nonlocal pending, pending_reads, current
            self.assertIsNone(pending, "New request sent before the previous read finished")
            calls.append("refresh")
            pending = frames.popleft()
            pending_reads = pending.read_delay
            if calls.count("refresh") == 3 and startup_fault == "startup_feedback":
                current = sample(bus={"write_no_buffer": 2})

        def disable_all():
            nonlocal current, pending
            calls.append("disable")
            frames.clear()
            frames.append(sample(status=["DISABLED"] * 8, position=0.03))
            pending = None
            current = disabled

        def enable_all():
            nonlocal current
            calls.append("enable")
            if startup_fault == "enable":
                current = sample(bus={"write_no_buffer": 3})

        def query_gripper_mode(register):
            nonlocal mode_pending
            self.assertEqual(register, 10)
            calls.append("query_gripper_mode")
            mode_pending = True

        def receive(timeout):
            nonlocal mode_value, mode_pending, gripper_state_seen, gripper_mode_malformed
            sleep(timeout / 1e6)
            if timeout and mode_pending and gripper_mode is not None:
                responses[7] += 1
                if delayed_gripper_state and not gripper_state_seen:
                    gripper_state_seen = True
                    gripper_mode_malformed += 1
                    return
                mode_value = gripper_mode
                mode_pending = False

        def query_encoder(register):
            nonlocal encoder_pending
            encoder_pending = register
            calls.append(f"query_encoder_{register}")

        def receive_encoder(timeout):
            nonlocal encoder_responses, encoder_pending
            sleep(timeout / 1e6)
            if encoder_pending is None:
                return
            if "enable" in calls and encoder_pending == missing_encoder:
                return
            encoder_value[encoder_pending] = -0.12 - (encoder_pending - 80) * 0.01
            encoder_pending = None
            encoder_responses += 1

        class LiveLinkStats:
            def __init__(self, response_count, malformed_count=lambda: 0):
                self.response_count = response_count
                self.malformed_count = malformed_count

            @property
            def responses(self):
                return self.response_count()

            @property
            def malformed_frames(self):
                return self.malformed_count()

        encoder_devices = SimpleNamespace(
            get_link_stats=lambda _: LiveLinkStats(lambda: encoder_responses),
            get_motors=lambda: [SimpleNamespace(get_param=lambda register: encoder_value[register])],
        )
        encoder_arm = SimpleNamespace(
            init_arm_motors=lambda *args: None, set_callback_mode_all=lambda mode: None,
            get_arm=lambda: encoder_devices, recv_all=receive_encoder,
            query_param_all=query_encoder, is_link_running=lambda: True,
            get_bus_status=lambda: SimpleNamespace(**{
                key: SimpleNamespace(count=0) for key in _BUS_COUNTER_NAMES
            }),
        )

        def set_latest_state(driver, timeout_us=300):
            nonlocal current, pending, pending_reads
            sleep(timeout_us / 1e6)
            if pending is not None and pending_reads:
                pending_reads -= 1
            elif pending is not None:
                current, pending = pending, None
                for axis in range(7 if current.missing else 8):
                    responses[axis] += 1
                if current.late:
                    sleep(0.06)
            driver.latest_state = {key: np.full(8, current.position if key == "qpos" else 0.)
                                   for key in ("qpos", "qvel", "qtorque", "tmos", "trotor")}

        def init(driver, *args, **kwargs):
            driver.started = False
            driver.arm_side = args[0]
            self.assertEqual(driver.arm_side, f"{side}_arm")
            self.assertEqual(kwargs["can_interface"], "can1" if side == "left" else "can0")
            driver.config = args[1]
            if j7_kp is not None:
                expected_kps = driver.config.get_default_kps().copy()
                expected_kps[feedback_trial.TEST_AXIS] = j7_kp
                np.testing.assert_array_equal(kwargs["kps"], expected_kps)
            else:
                self.assertNotIn("kps", kwargs)
            driver.gripper_posforce = True
            driver._health_axes = [(str(i), 0, i) for i in range(8)]
            collection = SimpleNamespace(
                get_link_stats=lambda i: LiveLinkStats(
                    lambda: responses[i], lambda: gripper_mode_malformed if i == 7 else 0),
                get_motors=lambda: [SimpleNamespace(get_param=lambda register: mode_value)] * 8,
                query_param_all=query_gripper_mode)
            driver._health_collections = [collection]
            driver.openarm = SimpleNamespace(
                recv_all=receive, refresh_all=refresh_all, get_gripper=lambda: collection,
                is_link_running=lambda: current.link, set_callback_mode_all=lambda mode: None,
                enable_all=enable_all, disable_all=disable_all,
                get_bus_status=lambda: SimpleNamespace(
                    **{key: SimpleNamespace(count=current.bus.get(key, 0))
                       for key in feedback_trial.TX_FAULTS}, writes_ok=0, last_write_errno=0))

        def event(port, value):
            return {"type": "INPUT", "id": port, "value": pa.array([value]), "metadata": {}}

        def events():
            yield event("command", "start")
            yield from (event("request_state", True) for _ in range(10))
            yield event("command", "stop")

        class Node:
            def __iter__(self):
                return events()

            def send_output(self, *args):
                outputs.append(args)

        logs = []

        def report(name, **fields):
            calls.append(name)
            logs.append((name, fields))

        argv = sys.argv[:]
        with (patch.object(SingleArmDriver, "__init__", init),
              patch.object(SingleArmDriver, "set_latest_state", set_latest_state),
              patch.object(SingleArmDriver, "get_health", lambda self: (current.status, current.bus)),
              patch.object(feedback_trial.os.path, "exists", return_value=True),
              patch.object(feedback_trial.time, "monotonic", side_effect=lambda: now),
              patch.object(feedback_trial.time, "monotonic_ns", side_effect=lambda: int(now * 1e9)),
              patch.object(feedback_trial.time, "sleep", side_effect=sleep),
              patch.object(feedback_trial.dora, "Node", return_value=Node()),
              patch.object(feedback_trial.time, "time_ns", side_effect=lambda: int(now * 1e9)),
              patch.object(openarm_can, "OpenArm", return_value=encoder_arm) as create_encoder_arm,
              patch.object(feedback_trial, "report", side_effect=report)):
            try:
                if gripper_mode is None:
                    with self.assertRaisesRegex(TimeoutError, "No fresh gripper mode response"):
                        feedback_trial.driver(motion)
                elif gripper_mode != 4:
                    with self.assertRaisesRegex(RuntimeError, "Gripper control mode did not match"):
                        feedback_trial.driver(motion)
                elif startup_fault is not None or running_fault is not None:
                    phase = startup_fault or "running_feedback"
                    with self.assertRaisesRegex(RuntimeError, f"CAN transmit failed during {phase}"):
                        feedback_trial.driver(motion)
                elif initial_refresh is not None and initial_refresh.missing:
                    with self.assertRaisesRegex(RuntimeError, r"Axes without fresh feedback: \[8\]"):
                        feedback_trial.driver(motion)
                elif initial_refresh is not None and initial_refresh.position != 0:
                    with self.assertRaisesRegex(RuntimeError, "Initial refresh position or bus check failed"):
                        feedback_trial.driver(motion)
                elif missing_encoder is not None:
                    name = "p_m" if missing_encoder == 80 else "xout"
                    with self.assertRaisesRegex(TimeoutError, f"No fresh right J7 {name} feedback"):
                        feedback_trial.driver(motion, encoders=True)
                elif accepted:
                    feedback_trial.driver(motion, encoders=encoders, side=side, j7_kp=j7_kp,
                                          record_recovery=record_recovery)
                    if encoders:
                        create_encoder_arm.assert_called_once_with("can1" if side == "left" else "can0", True)
                else:
                    with self.assertRaisesRegex(RuntimeError, "Official driver start failed"):
                        feedback_trial.driver(motion)
            finally:
                sys.argv = argv

        if gripper_mode != 4:
            self.assertEqual(calls.count("query_gripper_mode"), 1)
            self.assertNotIn("enable", calls)
            self.assertEqual([port for port, _, _ in outputs], ["status"])
            return logs

        self.assertEqual(calls.count("enable"), 1)
        self.assertLess(calls.index("trial_gripper_mode_confirmed"), calls.index("enable"))
        self.assertEqual(calls.count("disable"), 1)
        self.assertLess(calls.index("disable"), calls.index("trial_disabled_confirmed"))
        self.assertLess(calls.index("disable"), calls.index("trial_stop_timing"))
        if initial_refresh is not None and (initial_refresh.missing or initial_refresh.position != 0):
            self.assertEqual(calls[:calls.index("disable")].count("refresh"), 2)
        self.assertLessEqual({port for port, _, _ in outputs}, {"status", "state"})
        statuses = [value[0].as_py() for port, value, _ in outputs if port == "status"]
        expected = ["stopped", "started", "stopped"] if accepted else ["stopped"]
        if running_fault is not None or missing_encoder is not None:
            expected.append("started")
        self.assertEqual(statuses, expected)
        states = [(value, meta) for port, value, meta in outputs if port == "state"]
        self.assertEqual(len(states), 10 if accepted else 0)
        for value, meta in states:
            self.assertEqual(value[0].as_py()["motor_status"], ["ENABLED"] * 8)
            self.assertEqual(meta["start_epoch"], 1)
        return logs

    def test_encoder_feedback_provides_fresh_pairs(self):
        for side in ("right", "left"):
            with self.subTest(side=side):
                logs = self.run_trial(sample(), motion=True, accepted=True, encoders=True,
                                      delayed_gripper_state=True, side=side)
                pairs = [fields for event, fields in logs if event == "trial_j7_encoder_reading"]
                self.assertEqual([pair["phase"] for pair in pairs], ["disabled"] + ["running"] * 10)
                for pair in pairs:
                    self.assertLess(pair["p_m"]["observed_ns"], pair["xout"]["observed_ns"])

    def test_missing_either_encoder_stops_before_publishing_feedback(self):
        for register in (80, 81):
            with self.subTest(register=register):
                logs = self.run_trial(sample(), motion=True, missing_encoder=register)
                pairs = [fields["phase"] for event, fields in logs if event == "trial_j7_encoder_reading"]
                self.assertEqual(pairs, ["disabled"])

    def test_gain_override_only_changes_right_j7_position_gain(self):
        for kp in (15.0, 20.0):
            with self.subTest(kp=kp):
                self.run_trial(sample(), motion=True, accepted=True, encoders=True, j7_kp=kp)

    def test_recovery_samples_continue_after_disable(self):
        logs = self.run_trial(sample(), motion=True, accepted=True, encoders=True, j7_kp=20.0,
                              record_recovery=True)
        pairs = [fields for event, fields in logs if event == "trial_j7_encoder_reading"]
        self.assertEqual([pair["phase"] for pair in pairs[-6:]],
                         ["disable_dispatched"] * 5 + ["disabled_confirmed"])
        disabled_at = dict(logs)["trial_stop_timing"]["disable_returned_ns"]
        self.assertGreaterEqual(pairs[-6]["p_m"]["sent_ns"], disabled_at)

    def test_missing_or_mismatched_gripper_mode_blocks_enable(self):
        for gripper_mode in (None, 1):
            with self.subTest(gripper_mode=gripper_mode):
                logs = dict(self.run_trial(sample(), motion=True, gripper_mode=gripper_mode))
                self.assertNotIn("trial_gripper_mode_confirmed", logs)

    def test_waits_for_initial_refresh_before_startup_confirmation(self):
        for motion in (False, True):
            with self.subTest(motion=motion):
                self.run_trial(sample(), initial_refresh=sample(read_delay=2), accepted=True, motion=motion)

    def test_initial_refresh_missing_reply_or_position_shift_blocks_startup(self):
        for initial_refresh in (sample(missing=True), sample(position=0.03)):
            with self.subTest(initial_refresh=initial_refresh):
                self.run_trial(sample(), initial_refresh=initial_refresh, motion=True)

    def test_startup_confirmation_send_failure_blocks_start_and_targets(self):
        logs = dict(self.run_trial(sample(), motion=True, startup_fault="startup_feedback"))
        self.assertEqual(logs["trial_can_transmit"]["phase"], "startup_feedback")
        self.assertEqual(logs["trial_can_transmit"]["faults"]["write_no_buffer"], 2)
        self.assertNotIn("trial_gripper_startup_wait", logs)

    def test_all_enabled_immediately(self):
        logs = self.run_trial(sample(), accepted=True)
        self.assertNotIn("trial_gripper_startup_wait", [name for name, _ in logs])
        self.assert_stop_timing(dict(logs))

    def test_gripper_enabled_on_third_fresh_sample(self):
        waiting = sample("DISABLED")
        for motion in (False, True):
            with self.subTest(motion=motion):
                logs = self.run_trial(waiting, [waiting, waiting, sample()], accepted=True, motion=motion)
                confirmed = [kw for name, kw in logs if name == "trial_gripper_startup_confirmed"]
                self.assertEqual(len(confirmed), 1)
                self.assertEqual(confirmed[0]["sample"], 3)
                self.assertLess(confirmed[0]["elapsed_ms"], 60)

    def test_invalid_confirmation(self):
        waiting = sample("DISABLED")
        for reason, extra, samples in (
            ("gripper_still_disabled", [waiting] * 3 + [sample()], 3),
            ("Axes without fresh feedback: [8]", [sample(missing=True)], 0),
            ("position_changed", [sample(position=0.03)], 1),
            ("bus_or_link_changed", [sample(bus={"carrier": True, "bus_off": 1})], 1),
            ("bus_or_link_changed", [sample(link=False)], 1),
            ("CAN transmit failed", [sample(bus={"carrier": True, "write_no_buffer": 1})], 0),
            ("unexpected_motor_status", [sample(status=["DISABLED"] + ["ENABLED"] * 7)], 1),
            ("after the deadline", [sample(late=True)], 0),
        ):
            for motion in (False, True):
                with self.subTest(reason=reason, extra=extra, motion=motion):
                    logs = self.run_trial(waiting, extra, motion=motion)
                    self.assertEqual(sum(name == "trial_gripper_startup_sample" for name, _ in logs), samples)
                    rejected = [kw["reason"] for name, kw in logs if name == "trial_gripper_startup_incomplete"]
                    self.assertEqual(len(rejected), 1)
                    self.assertIn(reason, rejected[0])

    def test_invalid_initial_state_does_not_wait(self):
        for initial in (sample("DISABLED", position=0.03), sample("DISABLED", link=False),
                        sample("DISABLED", bus={"carrier": True, "bus_off": 1}),
                        sample(status=["DISABLED"] + ["ENABLED"] * 7)):
            for motion in (False, True):
                with self.subTest(initial=initial, motion=motion):
                    logs = dict(self.run_trial(initial, initial_refresh=sample(), motion=motion))
                    self.assertIn("trial_startup_rejected", logs)
                    self.assertNotIn("trial_gripper_startup_wait", logs)

    def test_motion_can_write_failure_before_any_position_command(self):
        fault = sample(bus={"write_no_buffer": 1})
        logs = dict(self.run_trial(sample(), motion=True, running_fault=fault))
        self.assertEqual(logs["trial_can_transmit"]["phase"], "running_feedback")
        self.assert_stop_timing(logs)
        self.assertNotIn("trial_driver_target", logs)

    def assert_stop_timing(self, logs):
        stop = logs["trial_stop_timing"]
        verified = logs["trial_disabled_confirmed"]
        self.assertLessEqual(stop["stop_entered_ns"], stop["disable_returned_ns"])
        self.assertLessEqual(stop["disable_returned_ns"], stop["initial_read_ns"])
        self.assertLess(stop["initial_read_ns"], verified["observation_timestamp"])
        self.assertEqual(stop["initial_qpos_unverified"], [0.] * 8)
        self.assertEqual(verified["qpos"], [0.03] * 8)

    def test_motion_stops_on_enable_write_failure_before_start_is_published(self):
        logs = dict(self.run_trial(sample(), motion=True, startup_fault="enable"))
        self.assertEqual(logs["trial_can_transmit"]["phase"], "enable")
        self.assertEqual(logs["trial_can_transmit"]["faults"]["write_no_buffer"], 3)
        self.assertNotIn("trial_driver_target", logs)

    def test_motion_position_rejection_sends_stop_before_logging(self):
        outputs = []

        def event(port, value, **metadata):
            return {"type": "INPUT", "id": port, "value": pa.array([value]), "metadata": metadata}

        def events():
            yield event("status", "stopped", start_epoch=0)
            yield event("status", "started", start_epoch=1)
            yield event("tick", True)
            initial = [0.] * 8
            state = dict(qpos=initial, qvel=initial, qtorque=initial,
                         motor_status=["ENABLED"] * 8, bus={"carrier": True})
            yield event("state", state, start_epoch=1, observation_timestamp=100)
            yield event("tick", True)
            state["qpos"] = [0.] * 5 + [0.02, 0., 0.]
            yield event("state", state, start_epoch=1, observation_timestamp=101)
            self.fail("Controller processed more input after rejection")

        class Node:
            def __iter__(self):
                return events()

            def send_output(self, port, value, *args):
                outputs.append((port, value[0].as_py()))

        with (patch.object(feedback_trial.dora, "Node", return_value=Node()),
              patch.object(feedback_trial.time, "time_ns", return_value=102),
              patch.object(feedback_trial, "report", side_effect=lambda name, **kw: outputs.append((name, kw)))):
            with self.assertRaisesRegex(RuntimeError, "Unexpected position at axis 6"):
                feedback_trial.controller(motion=True)

        self.assertEqual(sum(port == "target" for port, _ in outputs), 1)
        stop, rejected, requested = outputs[-3:]
        self.assertEqual(stop, ("command", "stop"))
        self.assertEqual(rejected[0], "trial_position_rejected")
        self.assertEqual(rejected[1]["qpos"], [0.] * 5 + [0.02, 0., 0.])
        self.assertEqual(rejected[1]["observation_timestamp"], 101)
        self.assertEqual(requested, ("trial_stop_requested", {"samples": 1, "stop_requested_ns": 102}))

    def test_start_probe_caps_target_and_stops_on_measurable_movement(self):
        for stop_at in (12, 105):
            with self.subTest(stop_at=stop_at):
                outputs = []
                now = 100.0

                def event(port, value, **metadata):
                    return {"type": "INPUT", "id": port, "value": pa.array([value]), "metadata": metadata}

                def events():
                    nonlocal now
                    yield event("status", "stopped", start_epoch=0)
                    yield event("status", "started", start_epoch=1)
                    for index in range(stop_at + 1):
                        now += 0.025
                        yield event("tick", True)
                        qpos = [0.0] * 8
                        if index == stop_at:
                            qpos[6] = 0.0021
                        state = dict(qpos=qpos, qvel=[0.0] * 8, qtorque=[0.0] * 8,
                                     motor_status=["ENABLED"] * 8, bus={"carrier": True})
                        yield event("state", state, start_epoch=1, observation_timestamp=int(now * 1e9))
                        if index == 0:
                            yield event("status", "aligned", start_epoch=1)
                    self.fail("Controller did not stop on wrist movement")

                class Node:
                    def __iter__(self):
                        return events()

                    def send_output(self, port, value, *args):
                        outputs.append((port, value[0].as_py()))

                with (patch.object(feedback_trial.dora, "Node", return_value=Node()),
                      patch.object(feedback_trial.time, "monotonic", side_effect=lambda: now),
                      patch.object(feedback_trial.time, "time_ns", side_effect=lambda: int(now * 1e9)),
                      patch.object(feedback_trial, "report", side_effect=lambda name, **kw: outputs.append((name, kw)))):
                    feedback_trial.controller(motion=True, travel=feedback_trial.START_TEST_TRAVEL)
                targets = [value["qpos"] for name, value in outputs if name == "target"]
                self.assertEqual(len(targets), stop_at)
                self.assertTrue(all(q[:6] == [0.0] * 6 and q[7] == 0.0 for q in targets))
                self.assertLessEqual(max(q[6] for q in targets), feedback_trial.START_TEST_TRAVEL + 1e-8)
                self.assertTrue(all(0 <= b[6] - a[6] <= 0.005 * 0.025 + 1e-8
                                    for a, b in zip(targets, targets[1:])))
                if stop_at == 105:
                    self.assertAlmostEqual(targets[-1][6], feedback_trial.START_TEST_TRAVEL)
                self.assertEqual(outputs[-3], ("command", "stop"))
                self.assertEqual(outputs[-2][0], "trial_motion_measured")

    def test_left_encoder_trial_targets_only_mirrored_j7(self):
        outputs = []
        now = 100.0

        def event(port, value, **metadata):
            return {"type": "INPUT", "id": port, "value": pa.array([value]), "metadata": metadata}

        def events():
            nonlocal now
            yield event("status", "stopped", start_epoch=0)
            yield event("status", "started", start_epoch=1)
            zero = [0.0] * 8
            state = dict(qpos=zero, qvel=zero, qtorque=zero, motor_status=["ENABLED"] * 8,
                         bus={"carrier": True})
            for index in range(120):
                now += 0.025
                yield event("tick", True)
                yield event("state", state, start_epoch=1, observation_timestamp=int(now * 1e9))
                if index == 0:
                    yield event("status", "aligned", start_epoch=1)
            self.fail("The encoder trial did not stop within its time limit")

        class Node:
            def __iter__(self):
                return events()

            def send_output(self, port, value, *args):
                outputs.append((port, value[0].as_py()))

        with (patch.object(feedback_trial.dora, "Node", return_value=Node()),
              patch.object(feedback_trial.time, "monotonic", side_effect=lambda: now),
              patch.object(feedback_trial.time, "time_ns", side_effect=lambda: int(now * 1e9)),
              patch.object(feedback_trial, "report")):
            feedback_trial.controller(motion=True, encoders=True, side="left",
                                      travel=feedback_trial.ENCODER_TRIAL_TRAVEL)

        targets = [value["qpos"] for name, value in outputs if name == "target"]
        self.assertEqual(targets[0], [0.0] * 8)
        self.assertTrue(all(q[:6] + [q[7]] == [0.0] * 7 for q in targets))
        self.assertAlmostEqual(targets[-1][6], -feedback_trial.ENCODER_TRIAL_TRAVEL)
        self.assertTrue(all(-0.005 * 0.025 - 1e-8 <= b[6] - a[6] <= 0
                            for a, b in zip(targets, targets[1:])))
        self.assertEqual(outputs[-1], ("command", "stop"))


if __name__ == "__main__":
    unittest.main()
