"""Opt-in trace of the pinned driver path for left J4 and J5."""

import json
import math
import time


AXES = (3, 4)
TX_FAULTS = ("write_net_down", "write_no_buffer", "write_other")


def joint_values(values):
    result = []
    for i in AXES:
        try:
            value = float(values[i])
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            value = math.nan
        result.append(value if math.isfinite(value) else None)
    return result


def gap(first, second):
    return [a - b if a is not None and b is not None else None
            for a, b in zip(first, second, strict=True)]


def report(event, **fields):
    print(json.dumps({"event": event, "side": "left", "joints": [4, 5], **fields}),
          flush=True)


class LeftTrackingTrace:
    # Set by the lifecycle wrapper only after successful start.
    tracking_trace = False
    _tracking_dispatch = None
    _tracking_feedback = None

    def _reply_counts(self):
        arm = self._health_collections[0]
        return [arm.get_link_stats(i).responses for i in AXES]

    def _write_counts(self):
        bus = self.openarm.get_bus_status()
        return [bus.writes_ok, *(getattr(bus, fault).count for fault in TX_FAULTS)]

    def set_latest_state(self, timeout_us=300):
        if not self.tracking_trace:
            return super().set_latest_state(timeout_us)
        before = self._reply_counts()
        started_ns = time.time_ns()
        result = super().set_latest_state(timeout_us)
        observed_ns = time.time_ns()
        after = self._reply_counts()
        replies = [new - old for old, new in zip(before, after, strict=True)]
        if any(count > 0 for count in replies):
            positions = joint_values(self.latest_state["qpos"])
            velocities = joint_values(self.latest_state["qvel"])
            torques = joint_values(self.latest_state["qtorque"])
            fields = {"measured_rad": positions, "velocity_rad_s": velocities, "torque_nm": torques}
            sample = dict(read_started_ns=started_ns, observed_ns=observed_ns,
                          new_replies=replies,
                          **{name: [value if count > 0 else None
                                    for value, count in zip(values, replies, strict=True)]
                             for name, values in fields.items()})
            if self._tracking_feedback is None:
                self._report_feedback(sample)
            else:
                self._tracking_feedback.append(sample)
        return result

    def _report_feedback(self, sample):
        command = self._tracking_dispatch
        if command is not None and command["dispatch_ns"] <= sample["read_started_ns"]:
            target = command["driver_rad"]
            sample.update(
                dispatch_ns=command["dispatch_ns"],
                driver_rad=target,
                tracking_gap_rad=gap(target, sample["measured_rad"]),
            )
        report("left_tracking_feedback", **sample)

    def send_position(self, position):
        if not self.tracking_trace:
            return super().send_position(position)
        requested_ns = time.time_ns()
        requested = joint_values(position)
        before = self._write_counts()
        self._tracking_feedback = []
        accepted = None
        try:
            accepted = super().send_position(position)
            return accepted
        finally:
            pending = self._tracking_feedback
            self._tracking_feedback = None
            after = self._write_counts()
            writes, *faults = [new - old for old, new in zip(before, after, strict=True)]
            current = dict(requested_ns=requested_ns, original_rad=requested, accepted=accepted,
                           writes_ok_delta=writes, tx_fault_delta=dict(zip(TX_FAULTS, faults)))
            if accepted:
                target = joint_values(self.last_command)
                current.update(
                    dispatch_ns=self.last_command_dispatch_timestamp_ns,
                    driver_rad=target,
                    driver_gap_rad=gap(requested, target),
                )
            elif accepted is False:
                current["safety_stop_reason"] = self._safety_stop_reason
            if accepted and not any(faults) and writes == self.num_mit_motors + self.gripper_posforce:
                self._tracking_dispatch = current
            elif accepted is not False or writes or any(faults):
                # A partial/unknown write makes the active motor target unknown.
                self._tracking_dispatch = None
            report("left_tracking_command", **current)
            for sample in pending:
                self._report_feedback(sample)
