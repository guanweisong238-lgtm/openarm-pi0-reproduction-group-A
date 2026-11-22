#!/usr/bin/env python3
"""Convert and independently verify every present episode; never overwrite samples."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/real_data_20261121_193807/episodes'
DEST=ROOT/'data/processed/real_alignment_prototype_v001'
SCRIPT=ROOT/'scripts/a2_pi0'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path,default=RAW.parent)
parser.add_argument('--output',type=Path,default=DEST)
args=parser.parse_args();RAW=args.dataset.resolve()/'episodes';DEST=args.output.resolve()
if DEST.exists():raise SystemExit(f'Output already exists: {DEST}')
DEST.mkdir(parents=True)
episodes=sorted((p for p in RAW.iterdir() if p.is_dir() and p.name.isdigit()),key=lambda p:int(p.name))
rows=[];errors=[]
for ep in episodes:
 eid=int(ep.name);out=DEST/f'episode_{eid}'
 try:
  if not out.exists():
   p=subprocess.run([sys.executable,str(SCRIPT/'prepare_real_episode.py'),'--episode',str(eid),'--dataset',str(RAW.parent),'--output',str(out)],cwd=ROOT,text=True,capture_output=True)
   if p.returncode:raise RuntimeError('conversion: '+p.stderr.strip()[-2000:])
  report=json.loads((out/'report.json').read_text())
  if report['episode_id']!=str(eid):raise ValueError('mismatched episode id')
  p=subprocess.run([sys.executable,str(SCRIPT/'check_real_alignment.py'),'--episode',str(eid),'--dataset',str(RAW.parent),'--output',str(out)],cwd=ROOT,text=True,capture_output=True)
  if p.returncode:raise RuntimeError('audit: '+p.stderr.strip()[-2000:])
  age=report['source_age_ms']; cams=('global','chest','wrist_left','wrist_right')
  rows.append({'episode_id':eid,'samples':report['sample_count'],'full_windows':report['full_windows'],'padded_windows':report['padded_windows'],
   'max_camera_age_ms':round(max(age[c]['max_ms'] for c in cams),4),'camera_matches_over_50ms':sum(age[c]['over_50ms'] for c in cams),
   'max_obs_age_ms':round(max(age['obs_'+s]['max_ms'] for s in ('left','right')),4),'max_action_age_ms':round(max(age['action_'+s]['max_ms'] for s in ('left','right')),4),'audit':'pass'})
  print(f'episode {eid}: verified {report["sample_count"]} samples',flush=True)
 except Exception as e:
  errors.append({'episode_id':eid,'error':str(e)});print('FAILED',eid,str(e),flush=True)
if rows:
 target=DEST/'batch_audit.csv'
 with target.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report={'episodes_present':[int(p.name) for p in episodes],'verified_episodes':[r['episode_id'] for r in rows],
 'samples':sum(r['samples'] for r in rows),'full_windows':sum(r['full_windows'] for r in rows),'padded_windows':sum(r['padded_windows'] for r in rows),
 'max_camera_age_ms':max((r['max_camera_age_ms'] for r in rows),default=None),'camera_matches_over_50ms':sum(r['camera_matches_over_50ms'] for r in rows),
 'max_obs_age_ms':max((r['max_obs_age_ms'] for r in rows),default=None),'max_action_age_ms':max((r['max_action_age_ms'] for r in rows),default=None),
 'errors':errors,'human_task_labels':'not established for full batch','split':'not assigned',
 'action_padding':'excluded or masked required; training loss does not yet apply action_valid',
 'scripts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SCRIPT/'prepare_real_episode.py',SCRIPT/'check_real_alignment.py',SCRIPT/'prepare_real_batch.py')}}
(DEST/'batch_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if errors:sys.exit(1)
