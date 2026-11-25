#!/usr/bin/env python3
"""Analyze phase-aligned Task C demonstrations and export smoothed review candidates."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from compare_real_replay_handoffs import ARM
from real_task_c_phases import PHASES, phase_frames
from score_real_task_c_replays import FPS, ROOT, SCENE

LABELS = [f'{side} J{i}' for side in ('R', 'L') for i in range(1, 9)]
UNITS = ('deg', 'deg/s', 'deg/s²')
SIGNALS = ('position', 'velocity', 'acceleration')
MAP = ROOT / 'data/real_20261122_camera_adjusted_new_block/episode_mapping.json'
CANDIDATE = ROOT / 'data/processed/real_camera_adjusted_candidate_v001/manifest.json'


def derivatives(positions):
    velocity = np.diff(positions, axis=0) * FPS
    acceleration = np.diff(velocity, axis=0) * FPS
    return positions, velocity, acceleration


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_episodes(episode_range):
    mapping = json.loads(MAP.read_text())['episodes']
    sessions = {int(row['merged_episode_id']): row['source_session'] for row in mapping}
    episodes = []
    for entry in json.loads(CANDIDATE.read_text())['entries']:
        episode = entry['episode_id']
        if episode_range is not None and not episode_range[0] <= episode <= episode_range[1]:
            continue
        with np.load(ROOT / entry['samples_path'], allow_pickle=False) as archive:
            timestamps = archive['timestamp_ns']
            state, target = archive['state'].astype(float), archive['actions'][:, 0].astype(float)
        np.testing.assert_allclose(np.diff(timestamps), 1e9 / FPS, atol=2, rtol=0)
        labels, _ = phase_frames(target, entry['operation_style'])
        episodes.append(dict(episode=episode, session=sessions[episode], style=entry['operation_style'],
                             split=entry['split'], timestamps=timestamps, state=state, target=target, labels=labels))
    return episodes


def phase_data(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        for index in range(len(PHASES)):
            frames = np.flatnonzero(episode['labels'] == index)
            if len(frames) < 8:
                continue
            groups[episode['style'], index].append((episode, frames))
    return groups


def statistics(groups):
    rows = []
    for (style, phase), pieces in sorted(groups.items()):
        sessions = sorted({episode['session'] for episode, _ in pieces})
        for session in sessions:
            selected = [(e, f) for e, f in pieces if e['session'] == session]
            for source in ('state', 'target'):
                series = [derivatives(episode[source][frames]) for episode, frames in selected]
                for index, name in enumerate(SIGNALS):
                    values = np.rad2deg(np.concatenate([seq[index] for seq in series]))
                    metrics = np.percentile(values, [5, 50, 95], axis=0)
                    for j, joint in enumerate(LABELS):
                        rows.append(dict(style=style, phase=PHASES[phase], session=session, source=source,
                                         signal=name, unit=UNITS[index], joint=joint, episodes=len(selected),
                                         samples=len(values), p05=metrics[0, j], median=metrics[1, j], p95=metrics[2, j]))
    return rows


def plot_phases(groups, out, cohort):
    all_sessions = sorted({episode['session'] for pieces in groups.values() for episode, _ in pieces})
    colors = {session: plt.get_cmap('tab10')(i) for i, session in enumerate(all_sessions)}
    phase_grid = np.linspace(0, 1, 101)
    with PdfPages(out / 'demonstration_trajectories_and_distributions.pdf') as pdf:
        for (style, phase), pieces in sorted(groups.items()):
            for signal, name in enumerate(SIGNALS):
                fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharex=True)
                for j, ax in enumerate(axes.flat):
                    for session in all_sessions:
                        selected = [(episode, frames) for episode, frames in pieces if episode['session'] == session]
                        if not selected:
                            continue
                        series = [np.rad2deg(derivatives(e['state'][f, j])[signal]) for e, f in selected]
                        aligned = np.stack([np.interp(phase_grid, np.linspace(0, 1, len(v)), v) for v in series])
                        if signal == 0:
                            ax.plot(phase_grid, aligned.T, color=colors[session], alpha=.13, linewidth=.55)
                        low, mid, high = np.percentile(aligned, [5, 50, 95], axis=0)
                        ax.fill_between(phase_grid, low, high, color=colors[session], alpha=.12)
                        ax.plot(phase_grid, mid, color=colors[session], label=f'{session[-6:]} (n={len(series)})')
                    ax.set_title(LABELS[j])
                    ax.grid(alpha=.2)
                handles, names = axes.flat[0].get_legend_handles_labels()
                fig.legend(handles, names, loc='upper right', ncol=3)
                fig.suptitle(f'{cohort} | {style} / {PHASES[phase]} / joint {name} ({UNITS[signal]})', x=.02, ha='left')
                fig.supxlabel('Fraction of the recorded phase')
                fig.tight_layout(rect=(0, .025, 1, .94))
                pdf.savefig(fig)
                plt.close(fig)


def candidates(groups, out):
    flags, phase_flags = [], defaultdict(list)
    for (style, phase), pieces in sorted(groups.items()):
        if len(pieces) < 5:
            continue
        for axis_set, side in ((slice(0, 7), 'right'), (slice(8, 15), 'left')):
            values = np.asarray([
                [np.percentile(np.abs(signal).max(axis=1), 95)
                 for signal in derivatives(e['target'][frames, axis_set])[1:]]
                for e, frames in pieces
            ])
            center = np.median(values, axis=0)
            mad = np.median(np.abs(values - center), axis=0)
            threshold = center + 4.5 * 1.4826 * mad
            for (episode, frames), row in zip(pieces, values, strict=True):
                which = (mad > 1e-9) & (row > threshold)
                if which.any():
                    flags.append(dict(episode=episode['episode'], session=episode['session'], style=style,
                                      phase=PHASES[phase], side=side, start=int(frames[0]), end=int(frames[-1]),
                                      signals=[name for name, flagged in zip(SIGNALS[1:], which) if flagged]))
                    phase_flags[episode['episode']].append((episode, frames, axis_set))
    edits = []
    target_dir = out / 'smoothing_candidates'
    target_dir.mkdir(exist_ok=True)
    for episode_id, edits_to_apply in sorted(phase_flags.items()):
        episode = edits_to_apply[0][0]
        original = episode['target']
        filtered = original.copy()
        for _, frames, axes in edits_to_apply:
            raw = original[frames, axes]
            candidate = savgol_filter(raw, 7, 2, axis=0)
            candidate[:3], candidate[-3:] = raw[:3], raw[-3:]
            before = np.percentile(np.abs(np.diff(raw, n=2, axis=0)), 95)
            after = np.percentile(np.abs(np.diff(candidate, n=2, axis=0)), 95)
            accepted = after < before and np.max(np.abs(candidate - raw)) <= np.deg2rad(2)
            if accepted:
                filtered[frames, axes] = candidate
            edits.append(dict(episode=episode_id, phase=PHASES[episode['labels'][frames[0]]],
                              arm='right' if axes.start == 0 else 'left', accepted=bool(accepted),
                              acceleration_p95_before_deg_s2=float(np.rad2deg(before) * FPS**2),
                              acceleration_p95_after_deg_s2=float(np.rad2deg(after) * FPS**2),
                              max_edit_deg=float(np.rad2deg(np.max(np.abs(candidate - raw))))))
        if np.any(original != filtered):
            np.testing.assert_array_equal(original[:, [7, 15]], filtered[:, [7, 15]])
            np.savez_compressed(target_dir / f'episode_{episode_id}.npz',
                                timestamp_ns=episode['timestamps'], original_target_rad=original,
                                candidate_target_rad=filtered, phase_index=episode['labels'])
    return flags, edits


def pose_matches(groups):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    jids = [model.joint(f'openarm_{side}_joint{i}').id for side in ('right', 'left') for i in range(1, 8)]
    addresses = model.jnt_qposadr[jids]
    sites = [model.site(f'{side}_ee_control_point').id for side in ('right', 'left')]
    rows = []
    for (style, phase), pieces in sorted(groups.items()):
        info, joints, xyz, rotations = [], [], [], []
        for episode, frames in pieces:
            for progress_index in range(0, len(frames), 6):
                frame = frames[progress_index]
                q = episode['state'][frame, ARM]
                data.qpos[addresses] = q
                mujoco.mj_kinematics(model, data)
                info.append((episode['episode'], episode['session'], int(frame), progress_index / (len(frames) - 1)))
                joints.append(q)
                xyz.append(data.site_xpos[sites].copy())
                rotations.append(Rotation.from_matrix(data.site_xmat[sites].reshape(2, 3, 3)).as_quat())
        xyz, rotations, joints = map(np.asarray, (xyz, rotations, joints))
        for side, name in enumerate(('right', 'left')):
            relatives = cKDTree(xyz[:, side]).query_ball_point(xyz[:, side], .02)
            for i, near in enumerate(relatives):
                epi, session, frame, progress = info[i]
                for group in ('same_session', 'different_session'):
                    eligible = [j for j in near if epi != info[j][0] and abs(progress - info[j][3]) <= .10
                                and ((session == info[j][1]) == (group == 'same_session'))]
                    if not eligible:
                        continue
                    dots = np.abs(rotations[eligible, side] @ rotations[i, side])
                    angles = np.rad2deg(2 * np.arccos(np.clip(dots, 0, 1)))
                    distances = np.linalg.norm(xyz[eligible, side] - xyz[i, side], axis=1)
                    score = np.where(angles <= 10, distances / .02 + angles / 10, np.inf)
                    best = score.argmin()
                    if not np.isfinite(score[best]):
                        continue
                    j = eligible[best]
                    diff = np.abs(np.rad2deg(joints[i, side*7:side*7+7] - joints[j, side*7:side*7+7]))
                    row = dict(style=style, phase=PHASES[phase], arm=name, relation=group, episode=epi,
                               frame=frame, session=session, match_episode=info[j][0], match_frame=info[j][2],
                               match_session=info[j][1], position_mm=distances[best]*1000,
                               rotation_deg=angles[best], max_joint_difference_deg=diff.max())
                    row.update({f'j{k+1}_deg': value for k, value in enumerate(diff)})
                    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episode-range', type=int, nargs=2, help='Inclusive episode ID bounds')
    parser.add_argument('--cohort', default='Source sessions')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(args.episode_range)
    if not episodes:
        parser.error('No episodes in the requested range')
    groups = phase_data(episodes)
    write_csv(args.out / 'demonstration_joint_statistics.csv', statistics(groups))
    flags, edits = candidates(groups, args.out)
    matches = pose_matches(groups)
    write_csv(args.out / 'demonstration_pose_matches.csv', matches)
    plot_phases(groups, args.out, args.cohort)
    by_group = defaultdict(list)
    for row in matches:
        by_group[row['relation']].append(row['max_joint_difference_deg'])
    counts = defaultdict(int)
    for e in episodes:
        counts[e['session'], e['style']] += 1
    summary = dict(episodes=len(episodes), episode_ids=[e['episode'] for e in episodes],
                   cohort=args.cohort, grouping='source session within selected cohort',
                   session_style_counts=[dict(session=s, style=t, episodes=n) for (s, t), n in sorted(counts.items())],
                   pose_match=dict(position_mm=20, orientation_deg=10, phase_fraction=.10,
                                   summary={group: dict(matches=len(values), median_deg=float(np.median(values)),
                                                        p95_deg=float(np.percentile(values, 95)), max_deg=float(max(values)))
                                            for group, values in sorted(by_group.items())}),
                   flagged=flags, smoothing_review=edits)
    (args.out / 'demonstration_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print({k: v for k, v in summary.items() if k not in ('flagged', 'smoothing_review')})


if __name__ == '__main__':
    main()
