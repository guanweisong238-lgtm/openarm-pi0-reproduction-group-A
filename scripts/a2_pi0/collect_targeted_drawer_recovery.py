"""Collect isolated expert corrections from train-only failure snapshots; retain rejected attempts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from drawer_success_training_config import ROOT, MANIFEST


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extra-recovery', type=Path, action='append', default=[],
                   help='Also include a separately collected and verified alternative expert recovery')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.diagnosis.read_text())
    train = {r['episode_id'] for r in json.loads(MANIFEST.read_text())['episodes'] if r['split']=='train'}
    entries, attempts = [], []
    for row in rows:
        eid = row['episode_id']
        if eid not in train or row['split'] != 'train':
            raise ValueError('Held-out parents forbidden')
        # Cover initial alignment, actual early deviation, and failed closest approach.
        for snapshot in row['snapshots']:
            step = int(Path(snapshot).stem.split('_')[-1])
            if step not in {0, 30, row['closest_step']-1}:
                continue
            dest = args.output / f'episode{eid:03d}_step{step:04d}'
            report_path = dest/'report.json'
            log = args.output/f'episode{eid:03d}_step{step:04d}.log'
            cmd = [sys.executable, str(ROOT/'scripts/a2_pi0/collect_drawer_recovery.py'),
                   '--snapshot', str(ROOT/snapshot), '--episode-id', str(eid), '--output', str(dest),
                   '--skip-intermediate-dwells', '--pitch-degrees', '15', '--image-stride', '4']
            if not report_path.exists():
                print('COLLECT', eid, step, flush=True)
                with log.open('w') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, MUJOCO_GL='glfw'))
                if result.returncode:
                    attempts.append(dict(parent_episode_id=eid, step=step, status='error', log=str(log)))
                    print('ERROR retained', log, flush=True)
                    continue
            report = json.loads(report_path.read_text())
            if (report['parent_episode_id'] != eid or report['parent_split'] != 'train' or
                report['snapshot_sha256'] != hashlib.sha256((ROOT/snapshot).read_bytes()).hexdigest()):
                raise ValueError('Existing output identity mismatch')
            accepted = bool(report['train_ready'] and report['judge']['success'] and report['replay_verified']
                            and report['gl_backend']=='glfw')
            attempts.append(dict(parent_episode_id=eid, step=step, accepted=accepted,
                                 path=str(dest.resolve().relative_to(ROOT)), judge=report['judge']))
            if accepted:
                entries.append(dict(episode_id=f'recovery_{eid}_{step}', parent_episode_id=eid, split='train',
                                    frames=report['frames'], path=str(dest.resolve().relative_to(ROOT)),
                                    episode_sha256=hashlib.sha256((dest/'episode.npz').read_bytes()).hexdigest()))
            print('RECOVERY_RESULT', eid, step, 'accepted=', accepted, flush=True)
            (args.output/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
    for dest in args.extra_recovery:
        report=json.loads((dest/'report.json').read_text())
        eid=report['parent_episode_id']
        if (eid not in train or report['parent_split']!='train' or not report['train_ready']
            or not report['judge']['success'] or not report['replay_verified'] or report['gl_backend']!='glfw'):
            raise ValueError(f'Unverified alternative recovery: {dest}')
        if report['snapshot_sha256']!=hashlib.sha256(Path(report['snapshot']).read_bytes()).hexdigest():
            raise ValueError('Alternative snapshot mismatch')
        step=report['initial_control_step_offset']
        pitch=report['expert_parameters']['pitch_degrees']
        path=str(dest.resolve().relative_to(ROOT))
        if any(e['path']==path for e in entries):raise ValueError('Duplicate recovery')
        entries.append(dict(episode_id=f'recovery_{eid}_{step}_pitch{pitch:g}',parent_episode_id=eid,
                            split='train',frames=report['frames'],path=path,
                            episode_sha256=hashlib.sha256((dest/'episode.npz').read_bytes()).hexdigest()))
        attempts.append(dict(parent_episode_id=eid,step=step,pitch_degrees=pitch,accepted=True,
                             path=path,judge=report['judge'],reason='Separately verified alternative expert recovery'))
    doc = dict(schema='targeted_recovery_v001', parent_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
               episodes=entries, attempts=attempts, model_success_evidence=False,
               scope='Expert recovery supervision from recorded train9/10/11 failures; no held-out states')
    (args.output/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
    print('COLLECTION_DONE', len(entries), '/', len(attempts), flush=True)


if __name__=='__main__':
    main()
