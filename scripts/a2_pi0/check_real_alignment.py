#!/usr/bin/env python3
"""Audit every aligned index against its raw parquet row and camera timing hash."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/real_data_20261121_193807'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--episode',type=int,default=0)
parser.add_argument('--dataset',type=Path,default=RAW)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
RAW=args.dataset.resolve()
OUT=args.output or ROOT/'data/processed/real_alignment_prototype_v001'/f'episode_{args.episode}'
x=np.load(OUT/'samples.npz',allow_pickle=False); t=x['timestamp_ns']; n=len(t); grid=np.arange(n)[:,None]+np.arange(50); np.testing.assert_array_equal(x['action_valid'],grid<n)
assert x['state'].shape==(n,16) and x['actions'].shape==(n,50,16)
for typ in ('obs','action'):
 for arm,start in [('right',0),('left',8)]:
  df=pq.read_table(RAW/'episodes'/str(args.episode)/typ/'arms'/arm/'state.parquet').to_pandas();ts=df.timestamp.to_numpy().astype('datetime64[ns]').astype(np.int64); vals=np.stack(df.qpos)
  idx=x[f'{typ}_index_{arm}']; np.testing.assert_array_equal(x[f'{typ}_timestamp_ns_{arm}'],ts[idx]); np.testing.assert_array_equal(x['state'][:,start:start+8] if typ=='obs' else x['actions'][:,:,start:start+8],vals[idx]); targets=t if typ=='obs' else t[np.minimum(grid,n-1)]; assert np.all(ts[idx]<=targets); assert np.all((idx==len(ts)-1)|(ts[np.minimum(idx+1,len(ts)-1)]>targets))
hashes={}
with (RAW/'camera_timing'/f'episode_{args.episode}.jsonl').open() as f:
 for line in f:
  row=json.loads(line)
  if row.get('event')=='frame': hashes[(row['camera'],row['timestamp_ns'])]=row['jpeg_sha256']
for cam in ('global','chest','wrist_left','wrist_right'):
 ts=x[f'image_timestamp_ns_{cam}']; assert np.all(ts<=t)
 seen=set()
 for name,tm in zip(x[f'image_path_{cam}'],ts):
  p=ROOT/str(name); assert p.exists() and int(p.stem)==tm and p.parent==RAW/'episodes'/str(args.episode)/'cameras'/cam
  if str(p) not in seen:
   assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[(cam,int(tm))];seen.add(str(p))
 for i in (0,n//2,n-1):
  with Image.open(ROOT/str(x[f'image_path_{cam}'][i])) as im: im.load();assert im.size==(1280,720)
 print(cam,'verified unique image hashes',len(seen))
print('PASS',n,'samples, all state/action rows and unique images verified; 12 representative JPEGs decoded')
