"""Export unlabeled Task C review sheets from the hash-pinned train split only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from eval_gripper_replay import close_frame, load_data, policy


def main(out):
    entries = load_data()
    if len({e['episode_id'] for e in entries}) != len(entries):
        raise ValueError('Episode repeated in manifest')
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    cameras = ('global', 'wrist_right', 'chest', 'wrist_left')
    for entry in entries:
        if entry['split'] != 'train':
            continue
        ep = entry['episode_id']
        src = policy.ROOT / entry['samples_path']
        if hashlib.sha256(src.read_bytes()).hexdigest() != entry['samples_sha256']:
            raise ValueError(src)
        with np.load(src, allow_pickle=False) as data:
            end = entry['eligible_start_index_exclusive']
            t = data['timestamp_ns']
            close = close_frame(data['actions'][:end, 0])
            if close is None:
                raise ValueError(f'No sustained close: {ep}')
            frames = sorted({close + d for d in (-60,-30,-15,-9,-6,-3,0,3,6,15,30) if 0 <= close+d < end})
            paths = {c: data['image_path_'+c] for c in cameras}
            times = {c: data['image_timestamp_ns_'+c] for c in cameras}
            sheet = Image.new('RGB', (4*320, len(frames)*204))
            draw = ImageDraw.Draw(sheet)
            name = f'episode_{ep:02}.jpg'
            for row, i in enumerate(frames):
                views = {}
                for col, c in enumerate(cameras):
                    path = Path(str(paths[c][i]))
                    if path.parent.parts[-3:] != (str(ep), 'cameras', c) or not 0 <= int(t[i])-int(times[c][i]):
                        raise ValueError(path)
                    with Image.open(policy.ROOT/path) as img:
                        tile = ImageOps.contain(img.convert('RGB'), (320,180))
                    sheet.paste(tile, (col*320, row*204))
                    draw.text((col*320+2,row*204+182), f'ep{ep} f{i} {i-close:+}   {c}', fill='white')
                    views[c] = dict(path=str(path), timestamp_ns=int(times[c][i]))
                rows.append(dict(episode_id=ep, split='train', frame=i, timestamp_ns=int(t[i]),
                                 demo_close_frame=int(close), sheet=name, sheet_row=row, views=views,
                                 graspable=None, review_status='pending'))
            sheet.save(out/name, quality=91)
    (out/'review.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    main(parser.parse_args().out)
