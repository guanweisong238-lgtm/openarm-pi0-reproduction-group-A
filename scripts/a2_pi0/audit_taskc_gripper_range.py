"""Read local Task C parquet and cached plans only; emit JSON to stdout."""
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'logs/openarm-gripper-followup-20261125'
RAW = ROOT / 'data/real_20261122_camera_adjusted_new_block/episodes'
SIDES = {'right': 7, 'left': 15}


def stats(values, bounds, mask=None):
    # Rows are plans, columns are steps. Never round samples before comparison.
    x = values.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite input')
    if mask is None:
        mask = np.ones(x.shape, dtype=bool)
    lo, hi = bounds
    below, above = (x < lo) & mask, (x > hi) & mask
    valid = x[mask]
    result = dict(n=valid.size, min_rad=float(valid.min()), max_rad=float(valid.max()),
                  below=int(below.sum()), above=int(above.sum()))
    if values.ndim == 2:
        # Nearest representable in-range endpoints in the cached prediction dtype.
        a, b = values.dtype.type(lo), values.dtype.type(hi)
        if float(a) < lo:
            a = np.nextafter(a, values.dtype.type(np.inf))
        if float(b) > hi:
            b = np.nextafter(b, values.dtype.type(-np.inf))
        clipped = np.clip(values, a, b).astype(np.float64)[mask]
        changed = int((clipped != valid).sum())
        remaining = int(((clipped < lo) | (clipped > hi)).sum())
        if remaining or changed != result['below'] + result['above']:
            raise ValueError('Clip invariant failed')
        result.update(plans=len(x), bad_plans=int((below | above).any(axis=1).sum()),
                      clip_changed=changed, post_clip_oob=remaining,
                      max_clip_rad=float(np.max(np.abs(clipped - valid))))
    return result


def main():
    config = json.loads((ROOT / 'scripts/a2_pi0/deploy/mock_policy_config.json').read_text())
    bounds = {s: config['target_limits_rad'][s][7] for s in SIDES}
    run = json.loads((ROOT / config['run'] / 'config.json').read_text())
    manifest = (ROOT / 'data/processed' / Path(run['candidate']).name / 'manifest.json').read_bytes()
    if sha256(manifest).hexdigest() != run['candidate_manifest_sha256'] or config['replan_steps'] != 20:
        raise ValueError('Unexpected manifest or replan stride')
    entries = json.loads(manifest)['entries']
    ids = sorted(e['episode_id'] for e in entries)
    if ids != list(range(49)):
        raise ValueError('Incomplete raw episode roster')
    raw = {}
    for kind in ('action', 'obs'):
        raw[kind] = {}
        for side in SIDES:
            parts = []
            for episode in ids:
                p = RAW / str(episode) / kind / 'arms' / side / 'state.parquet'
                col = pq.read_table(p, columns=['qpos'])['qpos']
                if not np.all(pc.list_value_length(col).to_numpy() == 8):
                    raise ValueError(f'Unexpected qpos width in {p}')
                parts.append(pc.list_element(col, 7).to_numpy())
            raw[kind][side] = stats(np.concatenate(parts), bounds[side])

    validation = {}
    for e in entries:
        if e['split'] == 'validation':
            with np.load(ROOT / e['samples_path'], allow_pickle=False) as data:
                validation[e['episode_id']] = len(data['state'])
    windows, masks, hashes = [], [], {}
    for seed in (1001, 1002, 1003):
        for ep, frames in sorted(validation.items()):
            p = CACHE / f'seed{seed}/episode_{ep}.npz'
            hashes[str(p.relative_to(CACHE))] = sha256(p.read_bytes()).hexdigest()
            with np.load(p, allow_pickle=False) as data:
                anchors, w = data['anchors'], data['windows']
            np.testing.assert_array_equal(anchors, np.arange(0, frames, 20))
            if w.shape != (len(anchors), 50, 16) or w.dtype != np.float32:
                raise ValueError(f'Unexpected cached output in {p}')
            windows.append(w)
            masks.append(anchors[:, None] + np.arange(20) < frames)
    w = np.concatenate(windows)
    report = dict(checkpoint=config['checkpoint'], bounds_rad=bounds, raw=raw,
                  validation_frames=validation, cache_sha256=hashes, prediction={})
    for name, part, mask in [('full50', w, None), ('prefix20', w[:, :20], None),
                             ('prefix20_recorded', w[:, :20], np.concatenate(masks))]:
        report['prediction'][name] = {s: stats(part[:, :, j], bounds[s], mask) for s, j in SIDES.items()}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
