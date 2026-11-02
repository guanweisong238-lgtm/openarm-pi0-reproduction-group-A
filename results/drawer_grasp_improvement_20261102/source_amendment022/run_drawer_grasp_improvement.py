"""Authorized cleanup-following training curriculum, model selection, and untouched paired test.

This driver never trains on validation/evaluation starts. Existing outputs are
checked and reused on restart; completed policy failures are never retried.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).parent
RESULT=ROOT/'results/drawer_grasp_improvement_20261102'
LOGS=ROOT/'logs/a2_pi0/grasp_improvement_20261102'
BASE=ROOT/'checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000'
ORIGINAL=ROOT/'data/processed/openarm_drawer_success_v003/manifest.json'
DEV=ROOT/'data/processed/openarm_drawer_eval_v003_selection/manifest.json'
TEST=ROOT/'data/processed/openarm_drawer_eval_v004_final/manifest.json'
ENV=dict(os.environ,MUJOCO_GL='glfw',DRAWER_EVAL_GL='glfw',LIBGL_ALWAYS_SOFTWARE='1',__GLX_VENDOR_LIBRARY_NAME='mesa')


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def status(phase,**kw):
    write(RESULT/'progress.json',dict(status='running',phase=phase,updated_at=now(),**kw))
    print('PHASE',phase,json.dumps(kw),flush=True)
def call(name,argv):
    LOGS.mkdir(parents=True,exist_ok=True)
    with (LOGS/(name+'.log')).open('a') as log:
        subprocess.run([sys.executable,*map(str,argv)],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)


def prepare_cases(path,seed,start,counts):
    if path.exists():return
    rng=np.random.default_rng(seed);profiles=[]
    for group,count in counts.items():
        ranges=([.09,.11],[-.03,.03],[.005,.012],[.01,.04]) if group=='near' else ([.08,.12],[-.05,.05],[.004,.014],[.04,.08])
        for i in range(count):
            profiles.append(dict(name=f'{group}_{i:03}',evaluation_group=group,
                drawer_offset_x=float(rng.uniform(*ranges[0])),drawer_offset_y=float(rng.uniform(*ranges[1])),
                handle_radius=float(rng.uniform(*ranges[2])),right_jitter=float(rng.uniform(*ranges[3])),
                seed=int(rng.integers(1_000_000,2_000_000_000))))
    config=ROOT/'configs'/f'{path.parent.name}.json'
    if config.exists():raise FileExistsError(config)
    write(config,dict(name=path.parent.name,seed=seed,episode_id_start=start,starts=profiles,
        purpose='Selection only' if path==DEV else 'Final test; never used for training or checkpoint selection'))
    call('prepare_'+path.parent.name,[SCRIPTS/'build_drawer_eval_starts.py','--config',config,'--output',path.parent])
    manifest=read(path)
    for row,profile in zip(manifest['episodes'],profiles,strict=True):
        row['evaluation_group']=profile['evaluation_group']
        row['evaluation_role']='selection' if path==DEV else 'final_test'
        rp=ROOT/row['path']/'report.json';report=read(rp)
        for key in ('stages','contact_metrics','success_open_then_close','opened_8cm_held_1s_during_episode',
                    'success_8cm_held_1s','final_closed_5mm_held_1s','final_open_m','peak_open_m',
                    'replay_max_qpos_error','replay_verified','upstream_returned_success'):
            report.pop(key,None)
        report.update(evaluation_group=profile['evaluation_group'],evaluation_role=row['evaluation_role'])
        write(rp,report)
    write(path,manifest)


def lock_cases():
    lock=RESULT/'locked_evaluation_inputs.json'
    if not lock.exists():
        paths=[]
        for manifest in (DEV,TEST):
            paths.append(manifest)
            for row in read(manifest)['episodes']:
                paths += [ROOT/row['path']/'report.json',ROOT/row['path']/'episode.npz']
        # Exact scene+state duplication would invalidate an unseen-start claim.
        seen=set()
        manifests=[ORIGINAL,ROOT/'data/processed/openarm_drawer_eval_v001/manifest.json',
                   ROOT/'data/processed/openarm_drawer_eval_v002/manifest.json',DEV,TEST]
        duplicates=[]
        for manifest in manifests:
            for row in read(manifest)['episodes']:
                rp=ROOT/row['path'];scene=read(rp/'report.json').get('scene_overrides',{})
                with np.load(rp/'episode.npz') as raw:
                    key=hashlib.sha256(raw['initial_integration_state'].tobytes()+raw['initial_eq_data'].tobytes()+
                        json.dumps([scene.get('drawer_offset_x_m',0),scene.get('drawer_offset_y_m',0),scene.get('drawer_handle_radius_m')]).encode()).hexdigest()
                if key in seen and manifest in (DEV,TEST):duplicates.append(row['episode_id'])
                seen.add(key)
        if duplicates:raise ValueError(f'Duplicate new evaluation starts {duplicates}')
        write(lock,dict(locked_at=now(),exact_new_duplicates=duplicates,files={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    for name,digest in read(lock)['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Locked evaluation input changed: {name}')


def evaluate(name,checkpoint,manifest,ids=None):
    out=LOGS/name;rows=read(manifest)['episodes']
    ids=ids or [r['episode_id'] for r in rows]
    existing=read(out/'index.json') if (out/'index.json').exists() else []
    if [r['episode_id'] for r in existing]!=ids[:len(existing)]:raise ValueError('Unexpected evaluation prefix')
    if existing:
        cfg=read(out/'run_config.json')
        if Path(cfg['checkpoint']).resolve()!=checkpoint.resolve():raise ValueError('Mixed checkpoints')
    for offset,eid in enumerate(ids[len(existing):],len(existing)):
        status('evaluation',stage=name,episode_id=eid,completed=offset,total=len(ids))
        call(name,[SCRIPTS/'run_drawer_eval_batch.py','--checkpoint',checkpoint,'--weight-precision','checkpoint',
             '--manifest',manifest,'--episode-ids',eid,'--seconds','24','--task','open','--action-chunk','4','--seed','42',
             '--append','--trial-offset',offset,'--output',out])
    records=read(out/'index.json')
    if len(records)!=len(ids) or not all(r['evidence_valid'] for r in records):raise ValueError('Invalid evaluation evidence')
    groups={r['episode_id']:r.get('evaluation_group','original') for r in rows}
    summary=dict(stage=name,checkpoint=str(checkpoint),trials=len(records),success=sum(r['success'] for r in records),
        groups={g:dict(trials=sum(groups[r['episode_id']]==g for r in records),
                       success=sum(r['success'] for r in records if groups[r['episode_id']]==g)) for g in sorted(set(groups[i] for i in ids))},
        output=str(out))
    write(out/'stage_summary.json',summary);print('STAGE_DONE',json.dumps(summary),flush=True)
    return summary


def fingerprint(checkpoint):
    paths=[checkpoint/'input_protocol.json']
    paths+=sorted(p for base in ('params','assets') for p in (checkpoint/base).rglob('*') if p.is_file())
    return {str(p.relative_to(checkpoint)):sha(p) for p in paths}


def replay(name,manifest):
    import assess_drawer_open as assessor
    assessor.MANIFEST=manifest;out=LOGS/name;cfg=read(out/'run_config.json')
    for source,digest in cfg['runtime_source_sha256'].items():
        if sha(SCRIPTS/source)!=digest:raise ValueError('Runtime source changed')
    for source,digest in cfg['manifest_sha256'].items():
        if sha(Path(source))!=digest:raise ValueError('Evaluation manifest changed')
    audited=[]
    for row in read(out/'index.json'):
        result=assessor.assess(out/row['trial'])
        if not result['evidence_valid'] or result['model_pull_success']!=row['success']:raise ValueError('Independent replay mismatch')
        audited.append(result)
    write(RESULT/(name+'_independent_replay.json'),dict(complete=True,results=audited))


def main():
    RESULT.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    protocol=RESULT/'protocol.json'
    if not protocol.exists():
        write(protocol,dict(created_at=now(),baseline=str(BASE),training_scene_seed=2026110204,query_seed=2026110205,
            selection_seed=2026110206,final_test_seed=2026110207,selection_counts=dict(near=8,expanded=8),
            final_counts=dict(near=70,expanded=30),training=dict(scenes=48,x_range=[.08,.12],y_range=[-.055,.055],radius_range=[.004,.014],
                jitter=.08,samples_per_scene=96,contact_focus_probability=.4,new_scene_sampling_weight=2,rehearsal_sampling_weight=4),
            courses=[dict(name='feedback_endpoint_021_contact_expanded',lr=3e-6),dict(name='feedback_endpoint_022_contact_expanded',lr=1e-5)],
            steps=4000,save_every=2000,selection='Require original train12/10/9/11 all pass; rank by selection16 success, near success; prefer baseline on ties. If selected candidate fails original33, try next ranked candidate. Course022 only if021 provides no eligible selection improvement. Final100 never selects weights.',
            exclusions='Original validation2/3/6, eval100..119, eval200..299, selection300..315, final400..499 never used for gradients or expert labels.',
            criterion='Original strict contact/pulling/80mm for30steps, fixed42/H4/continuous/checkpoint precision/GLFW, isolated per trial, 24s.',
            cleanup_report='results/project_cleanup_20261102/README.md',diagnosis='results/drawer_generalization_failure_diagnosis_20261102/README.md'))
    if not (RESULT/'source').exists():
        (RESULT/'source').mkdir()
        for name in (Path(__file__).name,'collect_drawer_randomized_training.py','collect_drawer_augmented_queries.py',
                     'merge_drawer_rehearsal.py','train_drawer_feedback_endpoint.py'):
            (RESULT/'source'/name).write_bytes((SCRIPTS/name).read_bytes())
    status('prepare_locked_evaluation_sets')
    prepare_cases(DEV,2026110206,300,dict(near=8,expanded=8))
    prepare_cases(TEST,2026110207,400,dict(near=70,expanded=30))
    lock_cases()
    status('waiting_for_independent_training_scenes')
    collection_log=ROOT/'logs/a2_pi0/drawer_randomized_train_v002.log'
    started=time.monotonic()
    while 'AUGMENT_DONE' not in collection_log.read_text():
        if 'Traceback' in collection_log.read_text() or time.monotonic()-started>7200:raise RuntimeError('Training scene collection incomplete')
        time.sleep(5)
    scene_manifest=ROOT/'data/drawer_recovery/randomized_train_v002/manifest.json'
    if len(read(scene_manifest)['episodes'])<20:raise RuntimeError('Insufficient valid training references')
    status('collect_contact_focused_feedback')
    expert=ROOT/'data/drawer_feedback_queries/v006_expanded_contact'
    if not (expert/'READY.json').exists():
        call('queries',[SCRIPTS/'collect_drawer_augmented_queries.py','--manifest',scene_manifest,
            '--base',ROOT/'data/drawer_feedback_queries/v004_randomized_feedback','--output',expert,
            '--samples-per-scene','96','--seed','2026110205','--contact-focus-probability','.4','--scene-sampling-weight','2'])
    merged=ROOT/'data/drawer_feedback_queries/v007_expanded_contact_rehearsal'
    if not (merged/'READY.json').exists():
        call('merge',[SCRIPTS/'merge_drawer_rehearsal.py','--base',expert,'--rehearsal',ROOT/'data/drawer_feedback_queries/rehearsal016_train','--output',merged])
    # Verify real per-query parent IDs, labels, and scene provenance before gradients.
    with np.load(merged/'physical_queries.npz') as raw:
        parents=set(raw['parent_episode_id'].tolist())
    allowed={r['episode_id'] for r in read(ORIGINAL)['episodes'] if r['split']=='train'}
    if not parents<=allowed:raise ValueError('Held-out training parent')
    write(RESULT/'training_data_audit.json',dict(parent_ids=sorted(parents),allowed_train_ids=sorted(allowed),
        queries=read(merged/'READY.json')['frames'],metadata_sha256=sha(merged/'READY.json'),heldout_parent_ids_used=False))
    call('prepare_training',[SCRIPTS/'train_drawer_feedback_endpoint.py','--data',merged,'--checkpoint',BASE,
        '--exp-name','feedback_endpoint_021_contact_expanded','--prepare-only'])
    baseline=evaluate('baseline_selection16',BASE,DEV)
    candidates=[dict(checkpoint=str(BASE),selection=baseline,source='baseline')]
    chosen=BASE
    for course in read(protocol)['courses']:
        exp=course['name'];cp_root=ROOT/'checkpoints/pi0_openarm_drawer_endpoint_v001'/exp
        training_data=merged;learning_rate=course['lr']
        if exp=='feedback_endpoint_022_contact_expanded':
            # Amendment after observed TRAIN-only regression, before final-test inference.
            # Preserve successful020 behavior on all10 original train starts,
            # including9/10/11 absent from the older016 rehearsal corpus.
            rehearsal=ROOT/'data/drawer_feedback_queries/rehearsal020_all_train'
            if not (rehearsal/'READY.json').exists():
                status('collect020_train_rehearsal')
                call('rehearsal020',[SCRIPTS/'collect_drawer_policy_rehearsal.py','--run',ROOT/'logs/a2_pi0/repair_screen_020_2000/full_33',
                    '--output',rehearsal,'--policy-label','recorded_policy020','--sampling-weight','12'])
            training_data=ROOT/'data/drawer_feedback_queries/v008_expanded_contact_rehearsal020'
            if not (training_data/'READY.json').exists():
                call('merge020',[SCRIPTS/'merge_drawer_rehearsal.py','--base',merged,'--rehearsal',rehearsal,'--output',training_data])
            learning_rate=3e-6
            write(RESULT/'amendment022.json',dict(reason='021/2000 regressed on original train9/11;016 rehearsal omitted these previous failures. Preserve020 successes on all10 original TRAIN parents.',
                made_before_final_test=True,training_data=str(training_data),lr=learning_rate,policy020_rehearsal_weight=12,
                original_protocol_retained=True,final_test_and_selection_criteria_unchanged=True))
            audit_raw=read(training_data/'READY.json')
            with np.load(training_data/'physical_queries.npz') as raw:
                if not set(raw['parent_episode_id'].tolist())<=allowed:raise ValueError('Held-out rehearsal parent')
                labels,counts=np.unique(raw['label_source'],return_counts=True)
            write(RESULT/'022_training_data_audit.json',dict(queries=audit_raw['frames'],metadata_sha256=sha(training_data/'READY.json'),
                labels=dict(zip(labels.tolist(),counts.tolist())),heldout_parents_used=False))
            source=RESULT/'source_amendment022';source.mkdir(exist_ok=True)
            for filename in (Path(__file__).name,'collect_drawer_policy_rehearsal.py','merge_drawer_rehearsal.py'):
                (source/filename).write_bytes((SCRIPTS/filename).read_bytes())
        status('training',experiment=exp,lr=learning_rate)
        if not (cp_root/'4000/input_protocol.json').exists():
            call(exp,[SCRIPTS/'train_drawer_feedback_endpoint.py','--data',training_data,'--checkpoint',BASE,
                '--exp-name',exp,'--steps','4000','--lr',learning_rate,'--eval-every','500',
                '--endpoint-path','cached','--cartesian-weight','.5','--gripper-weight','4','--save-every','2000'])
        for step in (2000,4000):
            cp=cp_root/str(step)
            gate=evaluate(f'{exp}_{step}_train_gate',cp,ORIGINAL,[12,10,9,11])
            entry=dict(checkpoint=str(cp),gate=gate,source=exp,step=step)
            if gate['success']==4:entry['selection']=evaluate(f'{exp}_{step}_selection16',cp,DEV)
            candidates.append(entry);write(RESULT/'candidates.json',candidates)
        def score(entry):
            s=entry.get('selection')
            if not s:return (-1,-1)
            return (s['success'],s['groups']['near']['success'])
        eligible=[c for c in candidates if score(c)>score(candidates[0])]
        eligible.sort(key=score,reverse=True)
        for entry in eligible:
            cp=Path(entry['checkpoint']);name=cp.parent.name+'_'+cp.name+'_original33'
            combined=RESULT/'original33_manifest.json'
            if not combined.exists():
                rows=read(ORIGINAL)['episodes']+read(ROOT/'data/processed/openarm_drawer_eval_v001/manifest.json')['episodes']
                write(combined,dict(episodes=rows,purpose='historical development regression screening'))
            regression=evaluate(name,cp,combined)
            entry['original33']=regression;write(RESULT/'candidates.json',candidates)
            if regression['success']==33:
                replay(name,combined);chosen=cp;break
        if chosen!=BASE:break
    # Lock chosen weights before any inference on the final test100.
    frozen_path=RESULT/'frozen_final_models.json'
    models={'baseline020':BASE}
    if chosen!=BASE:models['selected_candidate']=chosen
    if frozen_path.exists():
        frozen=read(frozen_path)
        if frozen['chosen_checkpoint']!=str(chosen):raise ValueError('Final selection changed')
    else:
        frozen=dict(frozen_at=now(),chosen_checkpoint=str(chosen),models={name:dict(checkpoint=str(cp),files=fingerprint(cp)) for name,cp in models.items()})
        write(frozen_path,frozen)
    lock_cases();write(RESULT/'candidates.json',candidates)
    final={}
    for name,cp in models.items():
        if fingerprint(cp)!=frozen['models'][name]['files']:raise ValueError('Final model changed')
        final[name]=evaluate(name+'_final100',cp,TEST)
        replay(name+'_final100',TEST)
        if fingerprint(cp)!=frozen['models'][name]['files']:raise ValueError('Weights changed during test')
    lock_cases()
    write(RESULT/'final_comparison.json',dict(complete=True,completed_at=now(),chosen_checkpoint=str(chosen),
        training_improved_selection=chosen!=BASE,results=final,final_test_used_for_model_selection=False))
    lines=['# 扩范围抓取修复与独立验收（2026-11-02）','',
        '已完成清理、失败诊断、独立训练数据采集、训练/开发集筛选和最终新100起点评估。旧模型、原始数据、历史失败与optimization_log原文保留。','',
        f'开发集选定权重：`{chosen.relative_to(ROOT)}`。最终100个起点在训练前锁定，不参与选权重。','',
        '| 模型 | 原范围 | 扩大范围 | 合计 |','| --- | ---: | ---: | ---: |']
    for name,s in final.items():lines.append(f'| {name} | {s["groups"]["near"]["success"]}/70 | {s["groups"]["expanded"]["success"]}/30 | {s["success"]}/100 |')
    lines+=['','两种模型若都执行，使用完全相同的新起点，独立动作重放全部有效。若开发集没有合格改进，则保留020并只执行一份最终测试，不虚报新模型改进。',
        '最终测试结果无论好坏均保留，不再据此挑选其他权重。near/expanded定义与上一轮新100起点相同；合计仅代表预定70/30混合，同模拟器结果不代表真机。','',
        '详见protocol.json、training_data_audit.json、candidates.json、frozen_final_models.json、final_comparison.json与各独立重放报告。','']
    (RESULT/'README.md').write_text('\n'.join(lines))
    write(RESULT/'progress.json',dict(status='complete',completed_at=now(),final_comparison=read(RESULT/'final_comparison.json')))
    print('GRASP_IMPROVEMENT_COMPLETE',json.dumps(final),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(RESULT/'error.json',dict(at=now(),type=type(exc).__name__,message=str(exc)))
        raise
