"""Offline J4 hold comparison; no hardware imports or I/O."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq
import yaml

from audit_taskc_gripper_range import RAW, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    session = ROOT / 'logs/openarm-demo24-joints-only-20261125'
    pose = json.loads((session / 'outcome.json').read_text())['last_hold_sample']
    cfg = ROOT / '.cache/openarm-policy-driver/lib/python3.11/site-packages/openarm_driver/configs/openarm_pedestal.yaml'
    kp = yaml.safe_load(cfg.read_text())['control_gains']['kps'][3]
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    for side in ('right', 'left'):
        for i, angle in enumerate(pose[side + '_arm']['measured_deg'][:7], start=1):
            data.qpos[model.joint(f'openarm_{side}_joint{i}').qposadr] = np.deg2rad(angle)
    mujoco.mj_forward(model, data)
    report = dict(selection='Disjoint 0.5 s windows; target and feedback J4 range <= 0.5 deg each; '
                            'each measured J1-J7 within 10 deg of live pose through window. '
                            'Samples on observation clock, using most recent target. Median per episode first.',
                  model=str(args.model), model_note='Zero velocity, model masses, fingers at XML default, no added payload/contact.', sides={})
    for side in ('right', 'left'):
        live = pose[side + '_arm']
        ref = np.array(live['measured_deg'][:7]); windows = []
        for ep in range(49):
            folder = RAW / str(ep)
            def load(stream, columns):
                t = pq.read_table(folder / stream / 'arms' / side / 'state.parquet', columns=['timestamp', *columns])
                return (t['timestamp'].to_numpy().astype('datetime64[ns]').astype(np.int64),
                        *(np.stack(t[k].to_pylist()) for k in columns))
            at, a = load('action', ['qpos'])
            ot, o, tq = load('obs', ['qpos', 'qtorque'])
            t0, end = max(ot[0], at[0]), min(ot[-1], at[-1])
            for t in np.arange(t0, end - 500_000_000, 500_000_000):
                s = slice(*ot.searchsorted([t, t + 500_000_000]))
                if len(ot[s]) < 100 or np.diff(ot[s]).max() > 15_000_000:
                    continue
                idx = at.searchsorted(ot[s], side='right') - 1
                if np.max(ot[s] - at[idx]) > 15_000_000:
                    continue
                obs = np.rad2deg(o[s, :7]); target = np.rad2deg(a[idx, 3])
                if abs(obs-ref).max() > 10 or max(np.ptp(target), np.ptp(obs[:, 3])) > .5:
                    continue
                windows.append(dict(episode=ep, offset_s=(t-t0)/1e9, gap_deg=float(np.median(target-obs[:, 3])),
                                    torque_nm=float(np.median(tq[s, 3]))))
        ids = sorted({w['episode'] for w in windows})
        med = {k: [float(np.median([w[k] for w in windows if w['episode']==ep])) for ep in ids]
               for k in ('gap_deg','torque_nm')}
        gravity = float(data.qfrc_bias[model.joint(f'openarm_{side}_joint4').dofadr[0]])
        report['sides'][side] = dict(live_gap_deg=-live['signed_error_deg'][3], nominal_kp=kp,
                                     model_gravity_nm=gravity, gravity_only_gap_deg=float(np.rad2deg(gravity/kp)),
                                     n_episodes=len(ids), n_windows=len(windows),
                                     episode_median_quartiles={k:np.percentile(v,[25,50,75]).tolist() if v else None for k,v in med.items()},
                                     windows=windows)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
