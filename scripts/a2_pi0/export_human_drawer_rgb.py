"""从人工示教数值记录重放生成RGB；控制期间无需同步慢速软件渲染。"""
import argparse
import os

os.environ['MUJOCO_GL']='egl'
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES','/usr/share/glvnd/egl_vendor.d/50_mesa.json')

import mujoco
import numpy as np

from collect_drawer_trial import write_json, write_preview
from drawer_simulation import DrawerSimulation
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-root',type=Path,required=True)
    args=parser.parse_args()
    env=DrawerSimulation(args.episode_root)
    report=dict(env.provenance)
    try:
        with np.load(args.episode_root/'episode.npz',allow_pickle=False) as episode:
            arrays={k:episode[k] for k in episode.files}
        if 'images_ceiling' in arrays:
            raise ValueError('RGB already exists')
        images={name:[] for name in ('ceiling','wrist_left','wrist_right')}
        error=0.
        for index,action in enumerate(arrays['action']):
            error=max(error,float(np.max(np.abs(env.data.qpos-arrays['qpos'][index]))))
            obs=env.observation()
            for name in images: images[name].append(obs['observation.images.'+name])
            env.step(action)
            if index%100==0: print('RGB',index,'/',len(arrays['action']),flush=True)
        error=max(error,float(np.max(np.abs(env.data.qpos-arrays['terminal_qpos']))))
        if error>=1e-6: raise RuntimeError(f'RGB replay error {error}')
        arrays.update({'images_'+k:np.asarray(v) for k,v in images.items()})
        temporary=args.episode_root/'episode_rgb_pending.npz'
        np.savez_compressed(temporary,**arrays)
        os.replace(temporary,args.episode_root/'episode.npz')
        report.update(image_shape=[240,320,3],rgb_export_status='complete',replay_verified=True,replay_max_qpos_error=error)
        report.pop('rgb_export_error',None)
        write_json(args.episode_root/'report.json',report)
        write_preview(args.episode_root)
        print('RGB_COMPLETE',args.episode_root,flush=True)
    except Exception as error:
        report.update(rgb_export_status='failed',rgb_export_error=str(error))
        write_json(args.episode_root/'report.json',report)
        raise
    finally:env.close()


if __name__=='__main__':main()
