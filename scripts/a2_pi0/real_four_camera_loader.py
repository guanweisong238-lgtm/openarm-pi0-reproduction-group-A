"""Read four-camera real-data samples with the exploratory split and train-only stats."""
from collections import OrderedDict
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

# Establish project cache paths before OpenPI creates its tokenizer.
import real_training_support
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from real_four_camera_config import FourCameraInputs, FourCameraPi0Config, CAMERA_MAPPING

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'data/processed/real_training_candidate_v001'
PROMPT='用左手拉开抽屉，右手抓起方块并放入抽屉、释放，然后用左手推合抽屉。'

def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

class RealFourCameraDataset:
 def __init__(self,split,root=DEFAULT,cache_episodes=2):
  if split not in ('train','validation') or cache_episodes<1:raise ValueError('bad split/cache size')
  self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
  decision=json.loads((self.root/'review_decision.json').read_text())
  if decision['decision']!='assume_all_episodes_human_reviewed_for_this_exploratory_run':raise ValueError('missing review decision')
  self.entries=[x for x in self.manifest['entries'] if x['split']==split]
  if not self.entries:raise ValueError('empty split')
  self.ends=np.cumsum([x['full_window_samples'] for x in self.entries]);self.cache=OrderedDict();self.cache_size=cache_episodes
  self.stats=normalize.load(self.root);self.adapter=FourCameraInputs()
  self.resize=transforms.ResizeImages(224,224);self.pad=transforms.PadStatesAndActions(32)
  self.norm=transforms.Normalize(self.stats,strict=True)
  self.tokenizer=PaligemmaTokenizer(48)
  tokens=self.tokenizer._tokenizer.encode(PROMPT,add_bos=True)+self.tokenizer._tokenizer.encode('\n')
  if len(tokens)>48:raise ValueError(f'prompt would be truncated: {len(tokens)} tokens')
  self.prompt_tokens,self.prompt_mask=self.tokenizer.tokenize(PROMPT)
  self.prompt_tokens=np.asarray(self.prompt_tokens,dtype=np.int32)
  self.prompt_mask=np.asarray(self.prompt_mask,dtype=bool)
 def __len__(self):return int(self.ends[-1])
 def locate(self,index):
  if not 0<=index<len(self):raise IndexError(index)
  e=int(np.searchsorted(self.ends,index,side='right'));return e,int(index)-(int(self.ends[e-1]) if e else 0)
 def get_raw(self,index):
  e,i=self.locate(index);row=self.entries[e]
  if e not in self.cache:
   path=ROOT/row['samples_path']
   if file_sha(path)!=row['samples_sha256']:raise ValueError(f'source hash mismatch: {path}')
   with np.load(path,allow_pickle=False) as f:z={k:f[k] for k in f.files}
   count=row['eligible_start_index_exclusive']
   if count!=row['full_window_samples'] or not z['action_valid'][:count].all() or z['action_valid'][count:].all(axis=1).any():raise ValueError(f'bad action window selection: {path}')
   self.cache[e]=z
   if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
  self.cache.move_to_end(e);z=self.cache[e]
  images={}
  for cam in CAMERA_MAPPING.values():
   path=ROOT/str(z['image_path_'+cam][i])
   with Image.open(path) as im:images[cam]=np.asarray(im.convert('RGB'))
  return {'episode_id':row['episode_id'],'sample_index':i,'timestamp_ns':int(z['timestamp_ns'][i]),'images':images,'state':z['state'][i].copy(),'actions':z['actions'][i].copy(),'prompt':PROMPT}
 def __getitem__(self,index):
  raw=self.get_raw(index);data=self.adapter(raw);data=self.norm(data)
  data['state']=np.asarray(data['state'],dtype=np.float32);data['actions']=np.asarray(data['actions'],dtype=np.float32)
  data=self.resize(data);data=self.pad(data)
  data['image']={k:np.asarray(v,dtype=np.uint8) for k,v in data['image'].items()}
  data['tokenized_prompt']=self.prompt_tokens.copy();data['tokenized_prompt_mask']=self.prompt_mask.copy();data.pop('prompt')
  return data

def batch_samples(samples):
 if not samples:raise ValueError('empty batch')
 import jax
 batch=jax.tree.map(lambda *xs:np.stack(xs),*samples)
 actions=batch.pop('actions');obs=model_lib.Observation.from_dict(batch)
 return obs,actions
