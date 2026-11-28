"""Production profile metadata; importable without robot or simulation libraries."""
DEFAULT_PERCENT = 100
PROFILE = "pedestal-original-pd-first-frame-v2"
RAMP_S = 10.
CAP_NM = (6., 6., 2., 6., 1., 1., 1.)


def description(percent=DEFAULT_PERCENT, *, diagnostic=False):
    return dict(profile="pedestal-original-pd-gravity-v1" if diagnostic else PROFILE,
                max_percent=percent, ramp_seconds=RAMP_S if diagnostic else 0.,
                onset="after_alignment" if diagnostic else "first_enabled_position_command",
                ff_cap_nm=list(CAP_NM), joints="J1-J7", pd="openarm_pedestal")
