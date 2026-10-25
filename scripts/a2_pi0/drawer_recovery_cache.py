"""独立恢复示教缓存；沿用已冻结的原train统计，不覆盖v003缓存。"""
import hashlib,json
from pathlib import Path
import numpy as np
from drawer_success_training_config import ROOT,MANIFEST
from drawer_success_cache import CachedInputs,flatten,identity


def prepare_recovery(cfg,manifest_path):
    from openpi import transforms
    doc=json.loads(Path(manifest_path).read_text())
    if doc['parent_manifest_sha256']!=hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
        raise ValueError('恢复数据来源版本不一致')
    train={r['episode_id'] for r in json.loads(MANIFEST.read_text())['episodes'] if r['split']=='train'}
    key=hashlib.sha256(Path(manifest_path).read_bytes()+Path(__file__).read_bytes()+identity(cfg).encode()).hexdigest()
    root=ROOT/'.cache/drawer_recovery_inputs'/key[:20]
    if (root/'READY.json').exists():return CachedInputs(root)
    if root.exists():raise RuntimeError(f'恢复缓存未完成: {root}')
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    from drawer_success_training_config import ASSET_ID
    if data.asset_id!=ASSET_ID:raise ValueError('恢复缓存必须使用原绝对动作配置，再在运行时转换残差')
    transform=transforms.compose([*data.repack_transforms.inputs,*data.data_transforms.inputs,
        transforms.Normalize(data.norm_stats,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs])
    rows=doc['episodes'];total=sum(r['frames'] for r in rows);writers=[];fields=[];index=0
    root.mkdir(parents=True)
    for row in rows:
        path=ROOT/row['path'];report=json.loads((path/'report.json').read_text())
        if row['split']!='train' or report['parent_episode_id'] not in train or not report['train_ready']:
            raise ValueError('拒绝未经审核或validation来源恢复数据')
        if hashlib.sha256((path/'episode.npz').read_bytes()).hexdigest()!=row['episode_sha256']:
            raise ValueError('恢复文件SHA不一致')
        with np.load(path/'episode.npz') as archive:
            state=archive['state'];action=archive['action']
            indices=np.asarray(row.get('source_frame_indices',list(range(row['frames']))),int)
            if len(indices)!=row['frames'] or len(set(indices))!=len(indices) or indices.min()<0 or indices.max()>=len(state):
                raise ValueError('恢复片段采样索引非法')
            images={k:archive['images_'+k][indices] for k in report['camera_mapping']}
        for packed,local in enumerate(indices):
            raw={'observation.state':state[local].astype(np.float32),'prompt':report['prompt'],
                'action':action[np.minimum(local+np.arange(cfg.model.action_horizon),len(action)-1)].astype(np.float32)}
            raw.update({'observation.images.'+k:v[packed] for k,v in images.items()})
            leaves=list(flatten(transform(raw)))
            if not writers:
                for number,(p,value) in enumerate(leaves):
                    name=f'field_{number:02d}.npy'
                    dtype=np.float32 if value.dtype.kind=='f' else np.int32 if value.dtype.kind in 'iu' and value.dtype.itemsize>4 else value.dtype
                    writer=np.lib.format.open_memmap(root/name,mode='w+',dtype=dtype,shape=(total,*value.shape))
                    writers.append((p,writer));fields.append(dict(path=p,file=name,dtype=str(np.dtype(dtype))))
            values=dict(leaves)
            for p,writer in writers:writer[index]=values[p]
            index+=1
        print(f'RECOVERY_CACHE {path.name} frames={row["frames"]}',flush=True)
    for _,writer in writers:writer.flush()
    (root/'READY.json').write_text(json.dumps(dict(identity=key,episodes=rows,fields=fields,frames=total,
        normalization='frozen original v003 train-only absolute statistics; runtime delta conversion follows main trainer'),indent=2)+'\n')
    del writers
    return CachedInputs(root)
