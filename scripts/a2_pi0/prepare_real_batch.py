#!/usr/bin/env python3
"""Convert and independently verify every present episode; never overwrite samples."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path

import yaml
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/real_data_20261121_193807/episodes'
DEST=ROOT/'data/processed/real_alignment_prototype_v001'
SCRIPT=ROOT/'scripts/a2_pi0'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path,default=RAW.parent)
parser.add_argument('--output',type=Path,default=DEST)
parser.add_argument('--metadata-success-only',action='store_true',help='Use present episodes labeled successful for the single task in metadata.yaml')
args=parser.parse_args();RAW=args.dataset.resolve()/'episodes';DEST=args.output.resolve()
if DEST.exists():raise SystemExit(f'Output already exists: {DEST}')
present=sorted((p for p in RAW.iterdir() if p.is_dir() and p.name.isdigit()),key=lambda p:int(p.name))
episodes=present
selection={}
if args.metadata_success_only:
 meta_path=RAW.parent/'metadata.yaml'
 meta=yaml.safe_load(meta_path.read_text())
 if len(meta['tasks'])!=1 or not str(meta['tasks'][0].get('prompt','')).strip():
  raise ValueError('expected a single task with a prompt')
 labels={int(row['id']):row for row in meta['episodes']}
 if len(labels)!=len(meta['episodes']) or set(int(p.name) for p in present)-labels.keys():
  raise ValueError('duplicate labels or missing episode metadata')
 if any(type(row['success']) is not bool or row.get('task_index')!=0 for row in labels.values()):
  raise ValueError('expected boolean success labels for task 0')
 present_ids={int(p.name) for p in present}
 episodes=[p for p in present if labels[int(p.name)]['success']]
 selection={'raw_dataset':str(RAW.parent.relative_to(ROOT)),
            'metadata_sha256':hashlib.sha256(meta_path.read_bytes()).hexdigest(),
            'prompt':meta['tasks'][0]['prompt'],
            'missing_episode_ids':sorted(labels.keys()-present_ids),
            'excluded_failed_episode_ids':sorted(i for i in present_ids if not labels[i]['success']),
            'selected_episode_ids':[int(p.name) for p in episodes]}
 if not episodes:raise ValueError('no present episodes labeled successful')
DEST.mkdir(parents=True)
rows=[];errors=[]
for ep in episodes:
 eid=int(ep.name);out=DEST/f'episode_{eid}'
 try:
  if not out.exists():
   cmd=[sys.executable,str(SCRIPT/'prepare_real_episode.py'),'--episode',str(eid),'--dataset',str(RAW.parent),'--output',str(out)]
   if selection:cmd+=['--prompt',selection['prompt']]
   p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
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
report={'episodes_present':[int(p.name) for p in present],'episodes_selected':[int(p.name) for p in episodes],'verified_episodes':[r['episode_id'] for r in rows],
 'samples':sum(r['samples'] for r in rows),'full_windows':sum(r['full_windows'] for r in rows),'padded_windows':sum(r['padded_windows'] for r in rows),
 'max_camera_age_ms':max((r['max_camera_age_ms'] for r in rows),default=None),'camera_matches_over_50ms':sum(r['camera_matches_over_50ms'] for r in rows),
 'max_obs_age_ms':max((r['max_obs_age_ms'] for r in rows),default=None),'max_action_age_ms':max((r['max_action_age_ms'] for r in rows),default=None),
 'errors':errors,'human_task_labels':'not established for full batch','split':'not assigned',
 'action_padding':'excluded or masked required; training loss does not yet apply action_valid',
 'scripts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SCRIPT/'prepare_real_episode.py',SCRIPT/'check_real_alignment.py',SCRIPT/'prepare_real_batch.py')}}
if selection:report['metadata_selection']=selection
(DEST/'batch_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if errors:sys.exit(1)
