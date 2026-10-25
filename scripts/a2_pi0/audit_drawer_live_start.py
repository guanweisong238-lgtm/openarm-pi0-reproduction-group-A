"""核对实时初始化与已导出训练/验证首帧；不执行动作。"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from drawer_simulation import DrawerSimulation
from drawer_success_training_config import ROOT, MANIFEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode-ids',type=int,nargs='+',default=[3,6])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    results = []
    for row in manifest['episodes']:
        if row['episode_id'] not in args.episode_ids:
            continue
        env = DrawerSimulation(ROOT / row['path'])
        try:
            live = env.observation()
            with np.load(ROOT / row['path'] / 'episode.npz') as data:
                first_state = data['state'][0].astype(np.float32)
                cameras = {}
                panels = []
                for name in ('ceiling', 'wrist_left', 'wrist_right'):
                    stored = data['images_' + name][0]
                    rendered = live['observation.images.' + name]
                    diff = abs(stored.astype(np.float32) - rendered.astype(np.float32))
                    cameras[name] = dict(exact=bool(np.array_equal(stored, rendered)),
                        mae_255=float(diff.mean()), max_error=float(diff.max()),
                        changed_pixel_fraction=float(np.any(diff > 0, axis=-1).mean()))
                    panels.append(np.concatenate((stored, rendered), axis=0))
                panel = np.concatenate(panels, axis=1)
                cv2.imwrite(str(args.output / f"episode_{row['episode_id']}_stored_top_live_bottom.png"),
                            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                results.append(dict(episode_id=row['episode_id'], split=row['split'],
                    initialization=row['initialization_group'],
                    initial_state_exact=bool(np.array_equal(first_state, live['observation.state'])),
                    initial_state_max_error_rad=float(abs(first_state-live['observation.state']).max()),
                    initial_qpos_max_error=float(abs(data['qpos'][0]-env.data.qpos).max()),
                    cameras=cameras))
        finally:
            env.close()
    result = dict(renderer=os.environ.get('MUJOCO_GL'), rows=results)
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
