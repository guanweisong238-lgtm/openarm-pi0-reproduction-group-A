#!/usr/bin/env python3
"""Create an exploratory episode split and training-only statistics for full windows."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from openpi.shared import normalize
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/processed/real_alignment_prototype_v001'
DEST=ROOT/'data/processed/real_training_candidate_v001'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 global SOURCE,DEST
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--output',type=Path,default=DEST)
 parser.add_argument('--stratify-at',type=int,help='Split episode IDs into groups below and at/above this ID')
 args=parser.parse_args();SOURCE=args.source.resolve();DEST=args.output.resolve()
 if DEST.exists():raise SystemExit(f'{DEST} already exists; never overwrite a candidate')
 audit_path=SOURCE/'batch_audit.json';audit=json.loads(audit_path.read_text())
 ids=sorted(audit['verified_episodes'])
 selected=audit.get('episodes_selected',audit['episodes_present'])
 assert not audit['errors'] and ids==sorted(selected) and len(ids)>=5
 metadata=audit.get('metadata_selection')
 if metadata and ids!=sorted(metadata['selected_episode_ids']):
  raise ValueError('batch audit and metadata selection disagree')
 # Repeatable episode-level split, not evidence of independent initial conditions.
 rng=np.random.default_rng(42)
 if args.stratify_at is None:
  val=set(rng.permutation(ids).tolist()[:4]);split_method='numpy default_rng(42) permutation; first 4 episodes validation'
 else:
  groups=[[i for i in ids if i<args.stratify_at],[i for i in ids if i>=args.stratify_at]]
  if min(map(len,groups))<2:raise ValueError('each stratum needs at least two episodes')
  val=set()
  for group in groups:val.update(rng.permutation(group).tolist()[:max(1,round(len(group)*0.2))])
  split_method=f'numpy default_rng(42), independently permuted groups below and at/above {args.stratify_at}; 20% validation per group, rounded'
 train=set(ids)-val
 accum={k:{'sum':np.zeros(16,dtype=np.float64),'sum_sq':np.zeros(16,dtype=np.float64),'count':0} for k in ('state','actions')};entries=[]
 for eid in ids:
  path=SOURCE/f'episode_{eid}'/'samples.npz';report=SOURCE/f'episode_{eid}'/'report.json'
  with np.load(path,allow_pickle=False) as z:
   valid=z['action_valid']; idx=np.flatnonzero(valid.all(axis=1)); state=z['state'][idx];actions=z['actions'][idx]
   assert state.shape==(len(idx),16) and actions.shape==(len(idx),50,16) and valid.shape==(len(z['timestamp_ns']),50)
   assert len(idx)==len(valid)-49 and np.array_equal(idx,np.arange(len(idx)))
   assert np.isfinite(state).all() and np.isfinite(actions).all()
   if eid in train:
    for key,a in [('state',state),('actions',actions)]:
     a=a.reshape(-1,16).astype(np.float64);s=accum[key];s['sum']+=a.sum(0);s['sum_sq']+=np.square(a).sum(0);s['count']+=len(a)
   entries.append({'episode_id':eid,'split':'validation' if eid in val else 'train','operation_style':('release_then_push' if eid<args.stratify_at else 'hold_then_push') if args.stratify_at is not None else 'unspecified','samples_path':str(path.relative_to(ROOT)),'source_report_path':str(report.relative_to(ROOT)),'samples_sha256':sha(path),'report_sha256':sha(report),'eligible_start_index_exclusive':int(len(idx)),'full_window_samples':len(idx),'excluded_padded_samples':len(valid)-len(idx),'human_task_review':'see_review_decision'})
 stats={};counts={}
 for name,s in accum.items():
  mean=s['sum']/s['count'];std=np.sqrt(np.maximum(s['sum_sq']/s['count']-mean**2,0))
  if not np.isfinite(std).all() or np.any(std<1e-6):raise ValueError(f'{name} has near-constant or invalid dimension: {std}')
  stats[name]=normalize.NormStats(mean=mean,std=std);counts[name]=s['count']
 manifest={'dataset_status':'exploratory_candidate_not_approved','source':str(SOURCE.relative_to(ROOT)),'batch_audit_sha256':sha(audit_path),
  'split_method':split_method,
  'split_warning':'episode split does not establish independent scenes or generalization',
  'task_review_warning':'consult review_decision.json for the actual review scope and limitations',
  'frequency_hz':30,'action_horizon':50,'action_semantics':'absolute IK target before driver safety processing','vector_order':'right 8 then left 8',
  'sample_selection':'only complete 50-step windows; discard all 49 padded starts per episode',
  'normalization':'population mean and std on train only; each selected state once and each of 50 targets for every selected action window (overlap counted)',
  'normalization_counts':counts,'entries':entries,'generator_sha256':sha(Path(__file__))}
 if metadata:
  manifest.update(prompt=metadata['prompt'],metadata_selection=metadata,
                  dataset_status='exploratory_capture_labeled_success')
  for entry in entries:entry['human_task_review']='capture_label_only'
 DEST.mkdir(parents=True,exist_ok=False);(DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');normalize.save(DEST,stats)
 if metadata:
  review={'decision':'use_capture_labeled_success_for_exploratory_training',
          'episode_ids':ids,'source':metadata['raw_dataset'],'metadata_sha256':metadata['metadata_sha256'],
          'scope':'Present episodes labeled successful for the sole recorded task, with alignment audit passed.',
          'limitations':'Capture labels are not an independent review of task success, image quality or exposure synchronization.'}
  (DEST/'review_decision.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'train_ids':sorted(train),'validation_ids':sorted(val),'train_samples':sum(e['full_window_samples'] for e in entries if e['split']=='train'),'validation_samples':sum(e['full_window_samples'] for e in entries if e['split']=='validation'),'excluded_padded_samples':sum(e['excluded_padded_samples'] for e in entries),'normalization_counts':counts,'minimum_std':{k:float(v.std.min()) for k,v in stats.items()}},indent=2))
if __name__=='__main__':main()
