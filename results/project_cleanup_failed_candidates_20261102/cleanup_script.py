from pathlib import Path
import hashlib,json,shutil,datetime
root=Path.cwd().resolve()
out=root/'results/project_cleanup_failed_candidates_20261102'
base=root/'checkpoints/pi0_openarm_drawer_endpoint_v001'
relative=['feedback_endpoint_017/2000','feedback_endpoint_018_parity/500','feedback_endpoint_019_feedback/1000','feedback_endpoint_019_feedback/2000','feedback_endpoint_021_contact_expanded/2000','feedback_endpoint_021_contact_expanded/4000','feedback_endpoint_022_contact_expanded/2000','feedback_endpoint_022_contact_expanded/4000']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
fingerprint=json.loads((root/'results/drawer_grasp_improvement_20261102/checkpoint_after_final100.json').read_text())
keep=Path(fingerprint['checkpoint'])
assert all(sha(keep/p)==h for p,h in fingerprint['files'].items()),'recommended checkpoint changed'
log_before=(root/'optimization_log.md').read_bytes()
assert hashlib.sha256(log_before[:19915]).hexdigest()=='03bd60097df4d027192efed15412acf916f1a4c5a52eb12d07c97a7230e6e21a'
shutil.copy2(root/'checkpoints/README.md',out/'checkpoints_before.md')
shutil.copy2('/tmp/cleanup_failed_candidates_20261102.py',out/'cleanup_script.py')
items=[]
for rel in relative:
 path=base/rel
 assert path.is_dir() and not path.is_symlink() and path.resolve().is_relative_to(base)
 assert (path/'input_protocol.json').is_file() and (path/'params').is_dir()
 files=[]
 for p in sorted(path.rglob('*')):
  assert not p.is_symlink(),f'symlink: {p}'
  if not p.is_file():continue
  entry={'path':str(p.relative_to(path)),'bytes':p.stat().st_size}
  # Keep all small protocol/config/statistics and tensor metadata, not usable weights.
  if p.stat().st_size<1024*1024:
   target=out/'metadata'/rel/p.relative_to(path)
   target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
   entry['archived_sha256']=sha(p)
   assert sha(target)==entry['archived_sha256']
  files.append(entry)
 items.append({'checkpoint':str(path.relative_to(root)),'files':files,'bytes':sum(x['bytes'] for x in files)})
plan={'created_at':datetime.datetime.now().astimezone().isoformat(),'free_before_bytes':shutil.disk_usage(root).free,'items':items,'reason':'Only explicitly rejected candidates; keep all seven remaining checkpoints, all data, logs and evaluation evidence. Archived metadata is not a weight backup.','recommended_checkpoint':str(keep.relative_to(root)),'recommended_inference_files_verified_before':len(fingerprint['files']),'optimization_log_before_sha256':hashlib.sha256(log_before).hexdigest()}
write('cleanup_plan.json',plan)
for item in items:
 path=root/item['checkpoint'];shutil.rmtree(path)
 item['deleted']=not path.exists()
 print('Removed',item['checkpoint'],flush=True)
assert all(sha(keep/p)==h for p,h in fingerprint['files'].items())
result={'completed_at':datetime.datetime.now().astimezone().isoformat(),'deleted_checkpoints':len(items),'deleted_file_bytes':sum(x['bytes'] for x in items),'free_before_bytes':plan['free_before_bytes'],'free_after_bytes':shutil.disk_usage(root).free,'recommended_inference_files_verified_after':len(fingerprint['files']),'recommended_checkpoint_unchanged':True,'original_log_unchanged':(root/'optimization_log.md').read_bytes()==log_before,'remaining_complete_checkpoints':sorted(str(p.parent.relative_to(root)) for p in (root/'checkpoints').rglob('input_protocol.json') if (p.parent/'params').is_dir()),'metadata_only_not_restorable_without_retraining':True}
write('cleanup_result.json',result)
print(json.dumps(result,ensure_ascii=False),flush=True)
