"""Infer placement and drawer-operation intervals from recorded Task C commands."""
import numpy as np

PHASES = ('open_drawer', 'approach_cube', 'grasp', 'carry', 'release', 'after_release', 'close_drawer', 'pull_drawer')


def grip_intervals(values, closed_above):
    """Return half-open intervals of sustained closing using the recorded command scale."""
    closed = values > -0.305 if closed_above else values < 0.305
    edges = np.flatnonzero(np.diff(np.r_[False, closed, False]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 10]


def phase_frames(actions, style):
    """Return approximate phase labels and gripper-command event frames."""
    right = grip_intervals(actions[:, 7], True)
    left = grip_intervals(actions[:, 15], False)
    if not right:
        raise ValueError('No sustained right gripper closure in demonstration')
    grasp, release = max(right, key=lambda pair: pair[1] - pair[0])
    limits = {'right_grasp': grasp, 'right_release': release}
    initial_left = [pair for pair in left if pair[0] < grasp - 8]
    if not initial_left or release + 9 >= len(actions):
        raise ValueError('Unable to locate drawer grasp or right release')
    left_grasp, left_release = max(initial_left, key=lambda pair: pair[1] - pair[0])
    limits.update(left_grasp=left_grasp, left_release_handle=left_release)
    if style == 'release_then_push':
        regrasp = [a for a, _ in left if a > grasp]
        if not regrasp:
            raise ValueError('Unable to locate second left gripper closure')
        limits['left_regrasp'] = regrasp[0]
        opening_end = left_release
        closing_start = regrasp[0] - 15
    elif style in ('hold_then_push', 'unspecified'):
        # Use left arm movement after the initial grip and after the right release.
        joints = actions[:, 8:15]
        motion = np.linalg.norm(np.diff(joints, axis=0), axis=1)
        motion = np.convolve(motion, np.ones(9) / 9, mode='same')
        early = motion[left_grasp:grasp - 8]
        moving = np.flatnonzero(early >= max(0.007, 0.3 * early.max()))
        opening_end = left_grasp + int(moving[-1]) + 1 if moving.size else left_grasp + 20
        if style == 'hold_then_push':
            late = motion[release:left_release]
            moving = np.flatnonzero(late >= max(0.007, 0.3 * late.max())) if late.size else []
            closing_start = release + int(moving[0]) if len(moving) else release + 9
        else:
            # The batch has no annotated closing styles. Group the remainder after release.
            closing_start = len(actions)
    else:
        raise ValueError(f'Unsupported Task C operation style: {style}')
    # Leave a short right-arm approach interval even if the arms move concurrently.
    approach_start = max(left_grasp, min(opening_end, grasp - 24))
    close = min(len(actions), max(release + 9, closing_start)) if style == 'unspecified' else min(
        len(actions) - 1, max(release + 9, closing_start)
    )
    edges = [0, approach_start, grasp - 8, grasp + 9, release - 8, release + 9, close, len(actions)]
    if any(a > b for a, b in zip(edges, edges[1:])):
        raise ValueError(f'Phase timestamps out of order: {edges}')
    labels = np.empty(len(actions), dtype=np.int8)
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        labels[a:b] = i
    if style == 'unspecified':
        labels[left_grasp:approach_start] = PHASES.index('pull_drawer')
        limits['pull_end'] = approach_start
    return labels, limits
