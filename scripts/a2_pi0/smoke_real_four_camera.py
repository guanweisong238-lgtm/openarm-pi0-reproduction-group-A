#!/usr/bin/env python3
"""Synthetic four-camera pi0 smoke test with real base weights; no dataset or checkpoints."""
import dataclasses
import argparse
import json
import time

from real_training_support import get_train_config
from real_four_camera_config import FourCameraPi0Config, FOUR_CAMERA_KEYS
from real_training_support import load_parameters, make_train_step
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as model_lib
from openpi.training import optimizer as optimizer_lib


def report(stage, **values):
    stats = jax.devices()[0].memory_stats() or {}
    print(json.dumps({'stage':stage,'time_s':round(time.monotonic()-start,2),
        'gpu_bytes_in_use':stats.get('bytes_in_use'), 'gpu_peak_bytes_in_use':stats.get('peak_bytes_in_use'),**values}),flush=True)

start=time.monotonic()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--real', action='store_true', help='Use one real training sample instead of synthetic data')
args=parser.parse_args()
base=get_train_config()
model_cfg=FourCameraPi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50)
cfg=dataclasses.replace(base,model=model_cfg)
assert jax.default_backend() == 'gpu'
report('loading_pretrained_parameters', model=str(model_cfg), synthetic_data=not args.real)
params=load_parameters(cfg)
report('pretrained_parameters_loaded')
abstract=nnx.eval_shape(cfg.model.create,jax.random.key(cfg.seed))
graphdef=nnx.graphdef(abstract)
if args.real:
 from real_four_camera_loader import RealFourCameraDataset, batch_samples
 ds=RealFourCameraDataset('train')
 sample=ds.get_raw(0)
 report('real_sample_selected', episode_id=sample['episode_id'], sample_index=sample['sample_index'], timestamp_ns=sample['timestamp_ns'])
 obs,actions=batch_samples([ds[0]])
 obs,actions=jax.device_put((obs,actions))
else:
 # Values are synthetic and distinct; no dataset statistics are implied.
 images={key:jnp.full((1,224,224,3),(-0.6 + i*0.3),dtype=jnp.float32) for i,key in enumerate(FOUR_CAMERA_KEYS)}
 obs=model_lib.Observation(images=images,image_masks={key:jnp.ones((1,),dtype=bool) for key in images},
  state=jnp.zeros((1,32),dtype=jnp.float32),tokenized_prompt=jnp.zeros((1,48),dtype=jnp.int32),tokenized_prompt_mask=jnp.ones((1,48),dtype=bool))
 actions=jnp.zeros((1,50,32),dtype=jnp.float32)
@jax.jit
def forward(p,ob,ac):
 m=nnx.merge(graphdef,p)
 return m.compute_loss(jax.random.key(123),ob,ac,train=False)
report('forward_start')
loss=forward(params,obs,actions)
jax.block_until_ready(loss)
assert loss.shape==(1,50) and bool(jnp.isfinite(loss).all())
report('forward_pass', loss_mean=float(jnp.mean(loss)), loss_shape=list(loss.shape))
del loss,forward
jax.clear_caches()
tx=optimizer_lib.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None)
opt_state=tx.init(params.filter(cfg.trainable_filter))
step=make_train_step(cfg,graphdef,tx)
report('backward_start')
params,opt_state,metrics=step(params,opt_state,jax.random.key(124),obs,actions)
metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
assert all(jnp.isfinite(v) for v in metrics.values()) and metrics['grad_norm']>0 and metrics['update_norm']>0
report('backward_update_pass',**metrics)
