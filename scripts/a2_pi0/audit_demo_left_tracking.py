"""Trace zero-inference-delay Task C demo rejects to raw left-arm records."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from audit_taskc_gripper_range import RAW
from eval_gripper_replay import FPS, STRIDE, load_data, policy, replay_guards


def read_raw(episode, stream):
    columns = ['timestamp', 'qpos'] + (['qvel'] if stream == 'obs' else [])
    table = pq.read_table(RAW / str(episode) / stream / 'arms/left/state.parquet', columns=columns)
    ts = table['timestamp'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    if (np.diff(ts) <= 0).any():
        raise ValueError(f'Non-increasing {stream} timestamps in episode {episode}')
    return ts, *(np.stack(table[name].to_pylist()) for name in columns[1:])


def previous(ts, query):
    i = np.searchsorted(ts, query, side='right') - 1
    if np.any(i < 0) or np.any(query > ts[-1]):
        raise ValueError('Query outside recorded data')
    return i


def audit(entry, source):
    ep = entry['episode_id']
    with np.load(policy.ROOT / entry['samples_path'], allow_pickle=False) as z:
        state, act, grid = z['state'], z['actions'], z['timestamp_ns']
        ti, oi = z['action_index_left'][:, 0], z['obs_index_left']
        ta, to = z['action_timestamp_ns_left'][:, 0], z['obs_timestamp_ns_left']
    at, aq = read_raw(ep, 'action')
    ot, oq, ov = read_raw(ep, 'obs')
    np.testing.assert_array_equal(ti, previous(at, grid))
    np.testing.assert_array_equal(oi, previous(ot, grid))
    np.testing.assert_array_equal(at[ti], ta)
    np.testing.assert_array_equal(ot[oi], to)
    np.testing.assert_array_equal(aq[ti], act[:, 0, 8:])
    np.testing.assert_array_equal(oq[oi], state[:, 8:])
    stop = replay_guards(act[::STRIDE], state, 0)
    if stop['reason'] != 'target_feedback_gap':
        return dict(episode=ep, stop=stop)
    ms = round(stop['at_s'] * 1000)
    frame = (stop['block'] - 1) * STRIDE + stop['step']
    feedback_frame = ms * FPS // 1000
    np.testing.assert_array_equal(np.abs(act[frame, 0] - state[feedback_frame]), stop['gap_rad'])
    axis = int(np.argmax(stop['gap_rad'])) - 8
    if axis < 0:
        raise ValueError(f'First reject in episode {ep} is not dominated by left arm')
    wall = grid[0] + ms * 1_000_000
    target = aq[ti[frame], axis]
    gate = float(policy.FEEDBACK_GAP[8 + axis])
    sync_obs = previous(ot, at[ti[frame]])
    wall_obs = previous(ot, wall)
    wall_action = previous(at, wall)
    future = np.flatnonzero((ot >= at[ti[frame]]) & (ot <= at[ti[frame]] + 300_000_000))
    catch = future[np.abs(oq[future, axis] - target) <= gate]
    at_idx = np.flatnonzero((at >= at[ti[frame]] - 300_000_000) & (at <= at[ti[frame]]))
    at_idx = at_idx[at[at_idx] >= ot[0]]
    recent = at_idx[at[at_idx] >= ta[frame] - 100_000_000]
    target_rates = np.abs(np.diff(aq[recent, axis])) / (np.diff(at[recent]) / 1e9)
    offsets = np.arange(0, 201, 4)
    sample_t = at[at_idx, None] + offsets[None, :] * 1_000_000
    lag_loss = np.abs(aq[at_idx, axis, None] - oq[previous(ot, sample_t), axis]).mean(axis=0)
    best = int(lag_loss.argmin())
    return dict(episode=ep, joint=f'J{axis+1}' if axis < 7 else 'gripper', stop=stop,
                source_session=source['source_session'], source_episode=int(source['source_episode_id']),
                target_timestamp_ns=int(ta[frame]),
                target_frame=frame, feedback_frame=feedback_frame,
                target_minus_feedback_stamp_ms=float((ta[frame] - to[feedback_frame]) / 1e6),
                target_rad=float(target), replay_feedback_rad=float(oq[oi[feedback_frame], axis]),
                raw_feedback_at_target_rad=float(oq[sync_obs, axis]),
                raw_gap_at_target_stamp_rad=float(abs(target - oq[sync_obs, axis])),
                raw_gap_at_replay_tick_rad=float(abs(aq[wall_action, axis] - oq[wall_obs, axis])),
                raw_feedback_velocity_rad_s=float(ov[sync_obs, axis]),
                raw_target_change_previous_100ms_rad=float(target - aq[previous(at, ta[frame] - 100_000_000), axis]),
                raw_peak_target_rate_previous_100ms_rad_s=float(target_rates.max()),
                replay_config_velocity_rad_s=float(policy.VELOCITY[8 + axis]),
                ms_until_feedback_within_gate=None if not catch.size else float((ot[catch[0]] - ta[frame]) / 1e6),
                pre_stop_300ms_lag_fit=dict(offset_ms=int(offsets[best]),
                                            mae_at_zero_rad=float(lag_loss[0]), mae_at_offset_rad=float(lag_loss[best])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads((RAW.parent / 'episode_mapping.json').read_text())['episodes']
    sources = {int(row['merged_episode_id']): row for row in mapping}
    result = [audit(entry, sources[entry['episode_id']]) for entry in load_data() if entry['split'] == 'validation']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
