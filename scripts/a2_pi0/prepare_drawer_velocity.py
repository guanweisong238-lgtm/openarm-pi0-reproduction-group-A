"""独立重放原13条示教提取实际速度，train-only新统计；不改原数据/缓存。"""
import hashlib,json
from pathlib import Path
import numpy as np
from drawer_success_training_config import ROOT,MANIFEST
from drawer_delta_config import DELTA_NORM_ROOT
from drawer_velocity_config import VELOCITY_NORM_ROOT
from drawer_simulation import DrawerSimulation


def prepare_velocity():
    import mujoco
    source=Path(__file__)
    key=hashlib.sha256(MANIFEST.read_bytes()+source.read_bytes()+
        (source.parent/'drawer_simulation.py').read_bytes()+mujoco.__version__.encode()).hexdigest()
    root=ROOT/'.cache/drawer_velocity_v001'/key[:20]
    rows=json.loads(MANIFEST.read_text())['episodes']
    if not (root/'READY.json').exists():
        if root.exists():raise RuntimeError(f'速度缓存未完成: {root}')
        root.mkdir(parents=True);meta={'identity':key,'splits':{},'mujoco_version':mujoco.__version__}
        for split in ('train','validation'):
            selected=[r for r in rows if r['split']==split];n=sum(r['frames'] for r in selected)
            writer=np.lib.format.open_memmap(root/f'{split}.npy',mode='w+',dtype=np.float32,shape=(n,16))
            index=0;error=0.
            for row in selected:
                env=DrawerSimulation(ROOT/row['path'])
                try:
                    with np.load(ROOT/row['path']/'episode.npz') as data:
                        for frame,action in enumerate(data['action']):
                            error=max(error,float(np.max(np.abs(env.data.qpos-data['qpos'][frame]))))
                            writer[index]=env.data.qvel[env.velocity_ids];index+=1;env.step(action)
                    print(f'VELOCITY_REPLAY {split} id={row["episode_id"]} frames={row["frames"]} max_qpos_error={error}',flush=True)
                finally:env.close()
            if index!=n or error>1e-9:raise ValueError('速度提取未复现原示教')
            writer.flush();del writer
            meta['splits'][split]={'episode_ids':[r['episode_id'] for r in selected],'frames':n,'replay_max_qpos_error':error}
        (root/'READY.json').write_text(json.dumps(meta,indent=2)+'\n')
    banks={split:np.load(root/f'{split}.npy',mmap_mode='r') for split in ('train','validation')}
    if not (VELOCITY_NORM_ROOT/'norm_stats.json').exists():
        values=np.asarray(banks['train'],np.float64)
        original=json.loads((DELTA_NORM_ROOT/'norm_stats.json').read_text())
        extras=dict(mean=values.mean(0),std=np.maximum(values.std(0),.1),
            q01=np.quantile(values,.01,axis=0),q99=np.quantile(values,.99,axis=0))
        for name,extra in extras.items():original['norm_stats']['state'][name]+=extra.tolist()
        VELOCITY_NORM_ROOT.mkdir(parents=True,exist_ok=False)
        (VELOCITY_NORM_ROOT/'norm_stats.json').write_text(json.dumps(original,indent=2)+'\n')
        (VELOCITY_NORM_ROOT/'provenance.json').write_text(json.dumps(dict(velocity_bank_identity=key,
            train_episode_ids=[r['episode_id'] for r in rows if r['split']=='train'],train_frames=len(values),
            velocity_std_floor_rad_per_second=.1,state_definition='positions16 then actual measured velocities16',
            positions_and_actions='unchanged original train-only delta statistics',validation_used_for_statistics=False),indent=2)+'\n')
    provenance=json.loads((VELOCITY_NORM_ROOT/'provenance.json').read_text())
    if provenance['velocity_bank_identity']!=key:raise ValueError('速度资产与缓存来源不一致')
    return banks,json.loads((root/'READY.json').read_text())


if __name__=='__main__':
    banks,meta=prepare_velocity();print('VELOCITY_READY',meta,flush=True)
