"""Verify the adapter with synthetic values and the copied dual-arm session."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
import time

import numpy as np

from live_motion import MotionSession
from pi0_observations import PLAN_AGE_NS, TICK_NS, VIEWS, Plan, read_snapshot, swap_arms


class Pi0AdapterTests(TestCase):
    def test_observation_and_action_joint_order(self):
        anchor = time.time_ns()
        arms = {"left": (anchor - 10, list(range(8))),
                "right": (anchor - 20, list(range(8, 16)))}
        frames = [SimpleNamespace(
            image=view.encode(), metadata={
                "camera": view, "source_camera": source,
                "image_transform": "identity-v1" if view == "chest" else "rot180-v1",
                "transform_applied_at": "four-camera-service-before-store",
                "timestamp_ns": anchor - 100,
            }) for view, source in reversed(list(VIEWS.items()))]
        camera = Mock()
        status = dict(dp_host_anchor_ns=anchor, mock=False, ready=True, basic_image_ready=True)
        camera.request.return_value = status, frames
        session = Mock()
        session.at.return_value = arms
        _, _, images, q = read_snapshot(session, camera, anchor)
        self.assertEqual(images, {k: k.encode() for k in VIEWS})
        np.testing.assert_array_equal(q, np.arange(16))
        np.testing.assert_array_equal(swap_arms(q), np.r_[8:16, 0:8])
        np.testing.assert_array_equal(swap_arms(np.array([np.r_[8:16, 0:8]]))[0], q)

        for key, value in (("source_camera", "wrong-port"),
                           ("image_transform", "unverified"),
                           ("timestamp_ns", anchor - TICK_NS)):
            with self.subTest(key=key):
                saved = frames[0].metadata[key]
                frames[0].metadata[key] = value
                with self.assertRaises(ValueError):
                    read_snapshot(session, camera, anchor)
                frames[0].metadata[key] = saved

        status["mock"] = True
        with self.assertRaises(ValueError):
            read_snapshot(session, camera, anchor)

    def test_plan_uses_elapsed_30_hz_index_and_expires(self):
        anchor = time.time_ns()
        targets = np.repeat(np.arange(50)[:, None], 16, axis=1)
        plan = Plan(anchor, targets)
        first = anchor + 4 * TICK_NS
        self.assertEqual(plan.at(first)[0], 0)
        self.assertEqual(plan.at(first + TICK_NS)[0], 3)
        self.assertEqual(plan.at(first + 3 * TICK_NS)[0], 9)
        with self.assertRaises(TimeoutError):
            plan.at(anchor + PLAN_AGE_NS + 1)

    def test_fresh_post_click_observation_is_required_to_dispatch_both_arms(self):
        session = MotionSession()
        session.phase = "ready"
        session.ready = {"left", "right"}
        session.limits = {side: np.tile([-1, 1], (8, 1)) for side in session.ready}
        session.send = Mock()
        self.assertTrue(session.begin())
        targets = [0.01] * 8 + [0.02] * 8
        feedback = [0.0] * 16
        session.accept(time.time_ns(), targets, feedback, session.permit_after - 1)
        session.send.assert_not_called()
        now = time.time_ns()
        session.accept(now, targets, feedback, now)
        self.assertEqual([(call.args[0], call.kwargs["q"]) for call in session.send.call_args_list],
                         [("left", targets[:8]), ("right", targets[8:])])
