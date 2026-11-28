"""Shared configuration, base-weight loading, and LoRA update for task C."""
from pathlib import Path
import logging,os,math
ROOT=Path(__file__).resolve().parents[2]
os.environ['OPENPI_DATA_HOME']=str(ROOT/'weights/openpi')
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['JAX_COMPILATION_CACHE_DIR']=str(ROOT/'.cache/jax')
import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import optax
from openpi.models import pi0_config
from openpi.shared import array_typing,nnx_utils
from openpi.training import config,weight_loaders
BASE_PARAMS=ROOT/'weights/openpi/openpi-assets/checkpoints/pi0_base/params'

def get_train_config():
 return config.TrainConfig(name='pi0_openarm_lora',exp_name='smoke_001',
  model=pi0_config.Pi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=16),
  weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE_PARAMS)),
  freeze_filter=nnx.Not(nnx_utils.PathRegex('.*lora.*')),ema_decay=None,batch_size=1,num_workers=0,num_train_steps=10,log_interval=1,save_interval=10,wandb_enabled=False,
  assets_base_dir=str(ROOT/'assets'),checkpoint_base_dir=str(ROOT/'checkpoints'),overwrite=False,resume=False)

def initialize_parameters(cfg,rng,partial_params=None):
 model=cfg.model.create(rng)
 if partial_params is not None:
  graphdef,state=nnx.split(model);state.replace_by_pure_dict(partial_params);model=nnx.merge(graphdef,state)
 params=nnx.state(model)
 return nnx_utils.state_map(params,cfg.freeze_filter,lambda p:p.replace(p.value.astype(jnp.bfloat16)))

def check_freeze_rules(cfg,params):
 trainable=traverse_util.flatten_dict(params.filter(cfg.trainable_filter).to_pure_dict(),sep='/')
 frozen=traverse_util.flatten_dict(params.filter(cfg.freeze_filter).to_pure_dict(),sep='/')
 assert trainable and frozen and all('lora' in name for name in trainable) and all('lora' not in name for name in frozen)
 assert all(value.dtype==jnp.bfloat16 for value in frozen.values())
 logging.info('Parameters: total=%s trainable=%s',f"{sum(math.prod(x.shape) for x in frozen.values())+sum(math.prod(x.shape) for x in trainable.values()):,}",f"{sum(math.prod(x.shape) for x in trainable.values()):,}")

def load_parameters(cfg):
 rng=jax.random.key(cfg.seed);shape=jax.eval_shape(lambda key:initialize_parameters(cfg,key),rng);expected=shape.to_pure_dict();loaded=cfg.weight_loader.load(expected)
 array_typing.check_pytree_equality(expected=expected,got=loaded,check_shapes=True,check_dtypes=True)
 partial=traverse_util.unflatten_dict({k:v for k,v in traverse_util.flatten_dict(loaded).items() if not isinstance(v,jax.ShapeDtypeStruct)})
 del loaded
 params=jax.jit(lambda key,weights:initialize_parameters(cfg,key,weights),donate_argnums=(1,))(rng,partial)
 jax.block_until_ready(params);check_freeze_rules(cfg,params);return params

def make_train_step(cfg,graphdef,tx,*,continuity_weight=0.0,pair_stride=20,pair_overlap=8,left_gripper_weight=1.0,handoff_action_scale=None,old_forecast_weight=1.0):
 def step(params,opt_state,rng,observation,actions):
  model=nnx.merge(graphdef,params);model.train()
  def loss_fn(current_model):
   if old_forecast_weight!=1.0:
    flow=current_model.compute_weighted_forecast_loss(rng,observation,actions,weight=old_forecast_weight)
    return flow,{'flow_loss':flow}
   if handoff_action_scale is not None:
    flow,continuity,count=current_model.compute_handoff_loss(rng,observation,actions,stride=pair_stride,overlap=pair_overlap,action_scale=handoff_action_scale)
    return (flow+continuity_weight*continuity if continuity_weight else flow),{'flow_loss':flow,'continuity_loss':continuity,'eligible_arm_targets':count}
   if continuity_weight:
    flow,continuity=current_model.compute_paired_loss(rng,observation,actions,stride=pair_stride,overlap=pair_overlap)
    return flow+continuity_weight*continuity,{'flow_loss':flow,'continuity_loss':continuity}
   kwargs={'left_gripper_weight':left_gripper_weight} if left_gripper_weight!=1.0 else {}
   return jnp.mean(current_model.compute_loss(rng,observation,actions,train=True,**kwargs)),{}
  (loss,components),grads=nnx.value_and_grad(loss_fn,has_aux=True,argnums=nnx.DiffState(0,cfg.trainable_filter))(model)
  trainable=params.filter(cfg.trainable_filter);updates,next_opt_state=tx.update(grads,opt_state,trainable);updated=optax.apply_updates(trainable,updates)
  actual_delta=jax.tree.map(lambda new,old:new-old,updated,trainable);nnx.update(model,updated)
  return nnx.state(model),next_opt_state,{'loss':loss,'grad_norm':optax.global_norm(grads),'update_norm':optax.global_norm(actual_delta),**components}
 return jax.jit(step,donate_argnums=(0,1))
