"""Validate actual train and validation batches and trace them to source arrays."""
import argparse
from pathlib import Path
import numpy as np
from real_four_camera_loader import RealFourCameraDataset,batch_samples,DEFAULT
from real_four_camera_config import CAMERA_MAPPING

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,default=DEFAULT);args=parser.parse_args()
for split in ('train','validation'):
 dataset=RealFourCameraDataset(split,root=args.candidate)
 indices=sorted(set([0, min(100,len(dataset)-1),int(dataset.ends[0])-1,len(dataset)-1]))
 converted=[]
 for i in indices:
  raw=dataset.get_raw(i);sample=dataset[i]
  assert raw['actions'].shape==(50,16) and raw['state'].shape==(16,)
  for field in ('state','actions'):
   stats=dataset.stats[field];restored=sample[field][...,:16]*(stats.std+1e-6)+stats.mean
   np.testing.assert_allclose(restored,raw[field],rtol=1e-5,atol=1e-5)
   assert np.all(sample[field][...,16:]==0)
  assert sample['tokenized_prompt_mask'].sum()==29
  for target,src in CAMERA_MAPPING.items():
   im=sample['image'][target];assert im.shape==(224,224,3) and im.dtype==np.uint8
   assert raw['images'][src].shape==(720,1280,3)
   assert sample['image_mask'][target]
  converted.append(sample)
 obs,actions=batch_samples(converted)
 assert actions.shape==(len(indices),50,32) and obs.state.shape==(len(indices),32)
 assert len(obs.images)==4 and all(a.shape==(len(indices),224,224,3) and np.isfinite(a).all() and a.min()>=-1 and a.max()<=1 for a in obs.images.values())
 assert np.isfinite(actions).all() and obs.tokenized_prompt.shape==(len(indices),48)
 print(f'PASS {split}: {len(dataset)} eligible windows; checked {len(indices)} samples across boundaries; four RGB views, nontruncated 29-token prompt, normalization inverse, padding, finite observation/action')
