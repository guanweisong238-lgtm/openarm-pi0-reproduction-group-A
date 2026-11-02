"""Independently assess every completed trial with the original assessor and frozen combined case manifests."""
import argparse,hashlib,json,time
from pathlib import Path
from drawer_success_training_config import ROOT
import assess_drawer_open as assessor


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--follow',action='store_true')
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((args.run/'run_config.json').read_text());episodes=[]
    for name in cfg['manifests']:
        path=Path(name)
        if cfg.get('manifest_sha256',{}).get(name,digest(path))!=digest(path):raise ValueError('Frozen case manifest changed')
        episodes.extend(json.loads(path.read_text())['episodes'])
    expected=set(range(13))|set(range(100,120))
    if {r['episode_id'] for r in episodes}!=expected or len(episodes)!=33:raise ValueError('Expected the frozen33 starts')
    manifest=args.output.with_suffix('.assessment_manifest.json')
    manifest.write_text(json.dumps(dict(episodes=episodes,scope='assessment only; never a training manifest'),indent=2)+'\n')
    # The original assessor takes its manifest from a module-level constant. Rebind
    # only in this audit process; neither the module file nor experiment is edited.
    assessor.MANIFEST=manifest
    seen={}
    while True:
        try:rows=json.loads((args.run/'index.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if not args.follow:raise
            time.sleep(2);continue
        for row in rows:
            eid=row['episode_id']
            if eid in seen:continue
            trial=args.run/row['trial'];summary=json.loads((trial/'summary.json').read_text())
            if Path(summary['checkpoint']).resolve()!=Path(cfg['checkpoint']).resolve():raise ValueError('Mixed model checkpoints')
            if summary['task']!='open' or summary['noise_seed']!=42 or summary['action_chunk']!=4 or summary['gripper_mode']!='continuous':raise ValueError('Trial configuration differs')
            result=assessor.assess(trial)
            if bool(result['model_pull_success'])!=bool(row['success']):raise ValueError('Independent assessor disagrees with index')
            seen[eid]=result;print('INDEPENDENT_REPLAY',eid,result['model_pull_success'],flush=True)
        complete=set(seen)==expected
        report=dict(complete=complete,trials=len(seen),success=sum(r['model_pull_success'] for r in seen.values()),
            target_achieved=bool(complete and all(r['model_pull_success'] and r['evidence_valid'] for r in seen.values())),
            checkpoint=cfg['checkpoint'],checkpoint_protocol_sha256=cfg['checkpoint_protocol_sha256'],
            run=str(args.run.resolve()),run_config_sha256=digest(args.run/'run_config.json'),
            assessor_source_sha256=digest(Path(assessor.__file__)),results=[seen[i] for i in sorted(seen)])
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
        if complete or not args.follow:break
        time.sleep(5)
    print('INDEPENDENT_AUDIT_DONE',len(seen),report['success'],flush=True)


if __name__=='__main__':main()
