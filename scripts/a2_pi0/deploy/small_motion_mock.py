#!/usr/bin/env python3
"""Exercise a bounded single-joint trajectory without accessing robot hardware."""
import json
import math

import numpy as np


def trajectory(initial, feedback, sample_age_s, epoch, expected_epoch, *, joint, delta, elapsed_s):
    """Return the next simulated target or reject the entire simulated run."""
    initial = np.asarray(initial, dtype=float)
    feedback = np.asarray(feedback, dtype=float)
    if initial.shape != (8,) or feedback.shape != (8,):
        raise ValueError("feedback_dimension")
    if not np.isfinite(initial).all() or not np.isfinite(feedback).all():
        raise ValueError("feedback_nonfinite")
    if not math.isfinite(sample_age_s) or not 0 <= sample_age_s <= 0.1:
        raise ValueError("feedback_stale")
    if type(epoch) is not int or epoch != expected_epoch:
        raise ValueError("epoch_changed")
    if type(joint) is not int or not 0 <= joint < 7:
        raise ValueError("joint_invalid")
    if not math.isfinite(delta) or abs(delta) > 0.01:
        raise ValueError("requested_range_exceeded")
    if not math.isfinite(elapsed_s) or not 0 <= elapsed_s <= 1.0:
        raise ValueError("trajectory_time_invalid")
    target = initial.copy()
    target[joint] += delta * elapsed_s
    if np.max(np.abs(target - feedback)) > 0.02:
        raise ValueError("feedback_deviation")
    return target


def main():
    # Fixture values are synthetic and only exercise scheduling and checks.
    initial = np.array([0.2, 0.1, 0.0, 1.0, 0.0, 0.1, 0.0, -0.4])
    feedback = initial.copy()
    previous = initial.copy()
    for step in range(51):
        target = trajectory(initial, feedback, 0.02, 1, 1, joint=0, delta=0.01,
                            elapsed_s=step / 50)
        assert np.max(np.abs(target - previous)) <= 0.000201
        assert np.array_equal(target[1:], initial[1:])
        feedback = target.copy()  # Mock receiver echoes the target immediately.
        previous = target
    print(json.dumps({"event": "mock_trajectory_complete", "steps": 51,
                      "initial": initial.tolist(), "final": feedback.tolist()}))

    failures = [
        ("feedback_dimension", dict(feedback=[0.0] * 7)),
        ("feedback_nonfinite", dict(feedback=[float("nan")] * 8)),
        ("feedback_stale", dict(sample_age_s=0.11)),
        ("epoch_changed", dict(epoch=2)),
        ("requested_range_exceeded", dict(delta=0.02)),
        ("feedback_deviation", dict(feedback=initial + 0.03)),
    ]
    for expected, changes in failures:
        args = dict(initial=initial, feedback=initial, sample_age_s=0.02, epoch=1,
                    expected_epoch=1, joint=0, delta=0.01, elapsed_s=0.5)
        args.update(changes)
        try:
            trajectory(**args)
        except ValueError as exc:
            if str(exc) != expected:
                raise
            print(json.dumps({"event": "mock_rejected", "reason": expected}))
        else:
            raise AssertionError(f"Expected rejection: {expected}")
    print(json.dumps({"event": "mock_exit", "hardware_access": False}))


if __name__ == "__main__":
    main()
