"""One official OpenArm SDK owner per arm. Status queries and disable only."""
import json
import math
import select
import sys
import time


class MotorStatusMismatch(ValueError):
    def __init__(self, expected, codes):
        from openarm_can import motor_error_to_string
        self.codes = tuple(codes)
        details = ", ".join(f"{'夹爪(J8)' if i==8 else f'J{i}'}={code}({motor_error_to_string(code)})"
                            for i, code in enumerate(codes, 1) if code != expected)
        super().__init__(f"期望 {expected}({motor_error_to_string(expected)})，实际 {details}；八轴={list(codes)}")


def await_enable(read, check, *, clock=time.monotonic, sleep=time.sleep):
    """Only poll fresh states; never resend enable or position during transition."""
    deadline = clock() + .150
    enabled = set()
    while True:
        check()
        try:
            sample = read()
        except MotorStatusMismatch as exc:
            on = {i for i,c in enumerate(exc.codes) if c == 1}
            if len(exc.codes) != 8 or not set(exc.codes) <= {0,1} or not enabled <= on:
                raise
            enabled = on
            if clock() >= deadline:
                raise TimeoutError(f"使能确认超时（未下发回位目标）：{exc}") from exc
            sleep(.005)
            continue
        check()
        if clock() >= deadline:
            raise TimeoutError("八轴使能确认到达过晚（未下发回位目标）")
        return sample


def open_probe(side):
    import openarm_can as oa
    from openarm_driver import Config

    cfg = Config("openarm_pedestal")
    label = side + "_arm"
    bus = oa.OpenArm(cfg.get_can_interface(label), True)
    # Eight local receive descriptors, including the gripper. Empty modes avoids
    # the device mode-setting writes made by init_gripper_motor/SingleArmDriver.
    bus.init_arm_motors([getattr(oa.MotorType, name) for name in cfg.get_motor_types()],
                        cfg.get_send_ids(), cfg.get_recv_ids(), [])
    bus.set_callback_mode_all(oa.CallbackMode.STATE)
    return bus, cfg.get_joint_offsets(label)


def _counters(groups):
    # get_link_stats returns reference_internal; freeze values before recv_all.
    return [(s.responses, s.malformed_frames) for s in
            (coll.get_link_stats(i) for coll, size in groups for i in range(size))]


def before_refresh(groups):
    try:
        before = _counters(groups)
    except IndexError as exc:
        raise ValueError(f"电机注册数量不符（实际/期望）："
                         f"{[(len(c.get_motors()), n) for c, n in groups]}") from exc
    return before, time.time_ns(), time.monotonic() + .035


def fresh(bus, offsets, groups=None, expected_code=0, *, pending=None):
    groups = groups if groups is not None else [(bus.get_arm(), 8)]
    if pending is None:
        pending = before_refresh(groups)
        bus.refresh_all()
    before, start, limit = pending
    missing = list(range(1, 9))
    while time.monotonic() < limit:
        stats = _counters(groups)
        malformed = [i for i, (a, b) in enumerate(zip(before, stats), 1) if b[1] != a[1]]
        if malformed:
            raise ValueError(f"malformed CAN reply: axes {malformed}")
        missing = [i for i, (a, b) in enumerate(zip(before, stats), 1) if b[0] <= a[0]]
        if not missing:
            break
        bus.recv_all(500)
    else:
        raise ValueError(f"missing fresh eight-axis replies: axes {missing}")
    # get_motors() returns value copies: take this snapshot only after recv_all.
    snapshots = [coll.get_motors() for coll, _ in groups]
    sizes = [len(snapshot) for snapshot in snapshots]
    if sizes != [n for _, n in groups] or sum(sizes) != 8:
        raise ValueError(f"电机注册数量不符：实际 {sizes}，期望 {[n for _, n in groups]}（共八轴）")
    motors = [m for snapshot in snapshots for m in snapshot]
    running = bus.is_link_running()
    healthy = bus.is_bus_healthy()
    if not running or not healthy:
        reason = f"CAN 链路 running={running}，SDK healthy={healthy}"
        if not healthy:
            try:
                # Read the same socket's latched counters; never clear them.
                state = bus.get_bus_status()
                faults = {key: getattr(state, key).count for key in (
                    "bus_off", "error_passive", "tx_overflow", "rx_overflow", "ack_error",
                    "tx_timeout", "write_net_down", "write_no_buffer", "write_other")}
                reason += "；" + (", ".join(f"{k}={v}" for k, v in faults.items() if v)
                                  or "未发现非零故障计数")
                reason += f"；last_write_errno={state.last_write_errno}"
            except Exception as exc:
                reason += f"；故障计数读取失败：{exc}"
        raise ValueError(reason)
    codes = [int(m.get_error_code()) for m in motors]
    q = [m.get_position()-float(offset) for m, offset in zip(motors, offsets)]
    if len(q) != 8 or not all(map(math.isfinite, q)):
        raise ValueError("invalid joint feedback")
    if any(code != expected_code for code in codes):
        raise MotorStatusMismatch(expected_code, codes)
    return {"stamp": start, "q": q}


def stop_probe(bus, offsets):
    # Flush pending replies before the request. Confirmation requires a new
    # eight-axis refresh; a successful disable_all() call is not an ACK.
    bus.disable_all()
    bus.recv_all(1000)
    return fresh(bus, offsets)


def emit(kind, **fields):
    print(json.dumps({"kind": kind, **fields}, allow_nan=False), flush=True)


def run(side):
    bus, offsets = open_probe(side)
    try:
        while not select.select([sys.stdin], [], [], .012)[0]:
            emit("feedback", **fresh(bus, offsets))
        # A stop message or loss of the PC parent's pipe both latch this worker.
    except Exception as exc:
        try:
            emit("fault", reason=str(exc))
        except BrokenPipeError:
            pass
    finally:
        # Do not let a failing UI/log pipe prevent the actual disable attempt.
        try:
            stop_probe(bus, offsets)
        except Exception as exc:
            emit("unconfirmed", reason=str(exc))
        else:
            emit("disabled")


if __name__ == "__main__":
    if sys.argv[1:] not in (["--field-session", "left"], ["--field-session", "right"]):
        sys.exit("Use the PC field-verification panel")
    run(sys.argv[2])
