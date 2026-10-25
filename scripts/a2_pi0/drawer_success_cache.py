"""将v003模型输入顺序写入磁盘memmap，避免随机训练反复解压整条RGB。"""
import hashlib
import json
from pathlib import Path

import numpy as np

from drawer_success_training_config import ROOT,MANIFEST,NORM_ROOT,transformed_dataset


def flatten(tree,prefix=()):
    for key,value in tree.items():
        path=prefix+(key,)
        if isinstance(value,dict):yield from flatten(value,path)
        else:yield path,np.asarray(value)


def unflatten(leaves):
    tree={}
    for path,value in leaves:
        node=tree
        for key in path[:-1]:node=node.setdefault(key,{})
        node[path[-1]]=value
    return tree


def identity(cfg):
    digest=hashlib.sha256()
    for path in (MANIFEST,NORM_ROOT/'norm_stats.json',Path(__file__),ROOT/'scripts/a2_pi0/openarm_policy.py',
                 ROOT/'scripts/a2_pi0/drawer_success_training_config.py'):
        digest.update(path.read_bytes())
    digest.update(repr(cfg.model).encode())
    return digest.hexdigest()


class CachedInputs:
    def __init__(self,path):
        self.root=path
        self.metadata=json.loads((path/'READY.json').read_text())
        self.rows=self.metadata['episodes']
        self.ends=np.cumsum([r['frames'] for r in self.rows])
        self.fields=[(tuple(row['path']),np.load(path/row['file'],mmap_mode='r')) for row in self.metadata['fields']]

    def __len__(self):return int(self.ends[-1])

    def locate(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        episode=int(np.searchsorted(self.ends,index,side='right'))
        start=0 if episode==0 else int(self.ends[episode-1])
        return episode,int(index)-start

    def __getitem__(self,index):
        self.locate(index)
        return unflatten((path,np.asarray(values[index])) for path,values in self.fields)


def prepare(cfg,split):
    key=identity(cfg)
    root=ROOT/'.cache/drawer_success_inputs'/key[:20]/split
    if (root/'READY.json').exists():
        cached=CachedInputs(root)
        if cached.metadata['identity']!=key:raise ValueError('cache identity mismatch')
        return cached
    if root.exists():raise RuntimeError(f'发现未完成缓存，请先移走后重试：{root}')
    dataset,_=transformed_dataset(cfg,split,cache_episodes=1)
    rows=dataset.dataset.episodes
    root.mkdir(parents=True)
    first=list(flatten(dataset[0]));fields=[];writers=[]
    for number,(path,value) in enumerate(first):
        # 模型使用32位数值输入；图像保持uint8，避免磁盘与主机内存翻倍。
        dtype=np.float32 if value.dtype.kind=='f' else np.int32 if value.dtype.kind in 'iu' and value.dtype.itemsize>4 else value.dtype
        name=f'field_{number:02d}.npy'
        writer=np.lib.format.open_memmap(root/name,mode='w+',dtype=dtype,shape=(len(dataset),*value.shape))
        writers.append((path,writer));fields.append({'path':path,'file':name,'dtype':str(np.dtype(dtype))})
    for index in range(len(dataset)):
        leaves=dict(flatten(dataset[index]))
        for path,writer in writers:writer[index]=leaves[path]
        if index==0 or (index+1)%500==0:print(f'CACHE {split} {index+1}/{len(dataset)}',flush=True)
    for _,writer in writers:writer.flush()
    metadata={'identity':key,'split':split,'episodes':rows,'fields':fields,'frames':len(dataset),
              'float_inputs':'float32','images':'uint8; no augmentation cached; training augmentation remains active'}
    (root/'READY.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    del writers,dataset
    return CachedInputs(root)
