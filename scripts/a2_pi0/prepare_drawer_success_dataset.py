"""将审核通过的13条完整RGB示教制作成独立快照，计算train-only统计。"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

from collect_drawer_trial import ROOT, write_json

NAME = 'pi0_openarm_drawer_success_v003_lora'
ASSET_ID = 'openarm_drawer_success_v003'
OUTPUT = ROOT/'data/processed/openarm_drawer_success_v003'
NORM = ROOT/'assets'/NAME/ASSET_ID
VALIDATION = {
    'data/drawer_expansion/diverse_rgb/cabinet_y_pos005',
    'data/drawer_expansion/diverse_rgb/initial_seed108',
    'data/drawer_human/session_20261024_152741/episode_002',
}


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    from openpi.shared import normalize
    audit_path=ROOT/'results/collection_audit_20261024/audit.json'
    audit=json.loads(audit_path.read_text())
    selected=[r for r in audit['episodes'] if r['classification'] in ('success_rgb_candidate','numeric_success_rgb_pending')]
    assert len(selected)==13
    assert VALIDATION <= {r['path'] for r in selected}
    if OUTPUT.exists() or NORM.exists():raise FileExistsError('数据或统计版本已存在，拒绝覆盖')
    # 先检查所有源数据已就绪；后台RGB未完成时不创建半成品版本。
    for row in selected:
        source=ROOT/row['path'];report=json.loads((source/'report.json').read_text())
        if report.get('human_recording') and report.get('rgb_export_status')!='complete':
            raise RuntimeError(f'RGB尚未完成：{source}')
        with np.load(source/'episode.npz',allow_pickle=False) as e:
            if not all('images_'+n in e.files for n in ('ceiling','wrist_left','wrist_right')):
                raise ValueError(f'缺少RGB：{source}')
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'BUILDING').write_text('Not ready until READY.json exists\n')
    rows=[];train_states=[];train_actions=[];seen_actions=set();initial_groups={}
    for index,quality in enumerate(selected):
        source=ROOT/quality['path'];original=json.loads((source/'report.json').read_text())
        dest=OUTPUT/'episodes'/f'episode_{index:03d}';dest.mkdir(parents=True)
        split='validation' if quality['path'] in VALIDATION else 'train'
        shutil.copy2(source/'episode.npz',dest/'episode.npz')
        source_hash=sha(source/'episode.npz')
        if sha(dest/'episode.npz')!=source_hash:raise ValueError('快照校验失败')
        with np.load(dest/'episode.npz',allow_pickle=False) as e:
            n=len(e['action']);assert n==quality['frames']
            action_hash=hashlib.sha256(e['action'].tobytes()).hexdigest()
            assert action_hash==quality['action_sha256']
            assert action_hash not in seen_actions,'重复动作轨迹'
            seen_actions.add(action_hash)
            assert e['action'].shape==e['state'].shape==(n,16)
            for key in ('state','action'):assert np.isfinite(e[key]).all()
            for camera in ('ceiling','wrist_left','wrist_right'):
                images=e['images_'+camera]
                assert images.dtype==np.uint8 and images.shape==(n,240,320,3)
                assert np.all(np.any(images,axis=(1,2,3)))
                del images
            if split=='train':
                train_states.append(e['state'].astype(np.float64))
                train_actions.append(e['action'].astype(np.float64))
            overrides=original.get('scene_overrides',{})
            condition={'qpos':e['qpos'][0].tolist(),'scene_sha256':original['scene_sha256'],
                'x':overrides.get('drawer_offset_x_m',0.),'y':overrides.get('drawer_offset_y_m',0.),
                'radius':overrides.get('drawer_handle_radius_m'),'cameras':original['camera_local_poses']}
            group=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest()
            assert group not in initial_groups or initial_groups[group]==split,'同初态跨集合'
            initial_groups[group]=split
        write_json(dest/'source_report.json',original)
        fresh_quality=dict(quality,rgb_complete=True,classification='success_rgb_candidate',
                           preparation_rgb_verified=True)
        if original.get('human_recording'):fresh_quality['rgb_status_snapshot']='complete'
        write_json(dest/'quality_audit.json',fresh_quality)
        report=dict(original)
        for key in ('lerobot_frames_verified','lerobot_path','clipped_frames','rgb_export_error'):
            report.pop(key,None)
        report.update(prompt='Open the drawer, then close it.',frames=n,fps=30,seconds=n/30,
            image_shape=[240,320,3],exported_lerobot=False,success_open_then_close=True,
            opened_8cm_held_1s_during_episode=True,final_closed_5mm_held_1s=True,
            weld_frames=quality['actual_weld_frames'],replay_verified=True,
            replay_max_qpos_error=quality['replay_error'],
            quality_contact_scope='whole handle assembly (bar + stems)',
            recorded_contact_scope=original.get('contact_scope','bar-only; see quality_audit for recomputed assembly contacts'),
            source_episode=quality['path'],source_episode_sha256=source_hash,
            contact_metrics={'bilateral_component_frames':quality['bilateral_frames'],
                             'longest_bilateral_seconds':quality['longest_bilateral_seconds']})
        if report.get('human_recording'):
            report['grasp_diagnostics']=[]
            if not original.get('contact_scope'):
                report['expert_parameters']={'pitch_degrees':None,'grasp_point_local_m':[0.,0.,-.16],
                    'orientation_policy':'fixed at initial actual orientation; see source_report for legacy metadata'}
        write_json(dest/'report.json',report)
        rows.append({'episode_id':index,'path':str(dest.relative_to(ROOT)),'split':split,'frames':n,
            'source_path':quality['path'],'source_episode_sha256':source_hash,'episode_sha256':source_hash,
            'action_sha256':action_hash,'condition_group':group,'human':quality.get('human',False),
            'warmup_frames':quality.get('warmup_frames',0),
            'initialization_group':'human_warm_start' if quality.get('human') else 'script_ready_start',
            'source_contact_scope':report['recorded_contact_scope'],
            'quality_contact_scope':report['quality_contact_scope']})
        print('SNAPSHOT',index,split,n,quality['path'],flush=True)
    stats={};near_constant={}
    for key,parts in [('state',train_states),('actions',train_actions)]:
        values=np.concatenate(parts)
        mean=values.mean(axis=0);raw_std=values.std(axis=0);std=np.maximum(raw_std,1e-6)
        stats[key]=normalize.NormStats(mean=mean,std=std,q01=np.quantile(values,.01,axis=0),q99=np.quantile(values,.99,axis=0))
        np.testing.assert_allclose((values-mean)/std*std+mean,values,atol=1e-7,rtol=1e-6)
        near_constant[key]=np.flatnonzero(raw_std<1e-6).tolist()
    normalize.save(NORM,stats)
    loaded=normalize.load(NORM)
    for key in stats:
        np.testing.assert_array_equal(loaded[key].mean,stats[key].mean)
        np.testing.assert_array_equal(loaded[key].std,stats[key].std)
    summary={split:{'episodes':sum(r['split']==split for r in rows),
                   'frames':sum(r['frames'] for r in rows if r['split']==split)} for split in ('train','validation')}
    manifest={'name':ASSET_ID,'schema_version':1,'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'episodes':rows,'splits':summary,'action_horizon':16,'fps':30,'action_dim':16,'model_action_dim':32,
        'task':'Open the drawer, then close it.','norm_stats_root':str(NORM.relative_to(ROOT)),
        'split_policy':'Previous two scripted holdouts preserved; entire older human session held out; newer human session train-only.',
        'normalization_policy':'train only, each original frame once, mean/std, std floor 1e-6',
        'trim_policy':'none; raw observations/actions preserved byte-for-byte',
        'limitations':['Local small-data validation, not broad generalization',
                       'Warm-start human episodes and ready-start script episodes retain different initialization conditions',
                       'Long idle periods retained; training sampler should be chosen explicitly'],
        'audit_source':str(audit_path.relative_to(ROOT)),'audit_sha256':sha(audit_path)}
    write_json(OUTPUT/'manifest.json',manifest)
    write_json(NORM/'provenance.json',{'manifest':str((OUTPUT/'manifest.json').relative_to(ROOT)),
        'manifest_sha256':sha(OUTPUT/'manifest.json'),'train_episode_ids':[r['episode_id'] for r in rows if r['split']=='train'],
        'validation_episode_ids':[r['episode_id'] for r in rows if r['split']=='validation'],
        'train_frames':summary['train']['frames'],'normalization':'mean_std','std_floor':1e-6,
        'weighting':'each original training frame once; validation and padded tail frames excluded',
        'near_constant_dims':near_constant})
    write_json(OUTPUT/'READY.json',{'episodes':len(rows),'splits':summary,'manifest_sha256':sha(OUTPUT/'manifest.json')})
    (OUTPUT/'BUILDING').unlink()
    print('READY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
