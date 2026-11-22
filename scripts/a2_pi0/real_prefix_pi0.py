"""Experimental pi0 with a clean seven-action prefix; no robot control."""
import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import einops
from openpi.models import model as model_lib
from real_four_camera_config import FourCameraPi0Config
from real_four_camera_pi0 import Pi0,FOUR_CAMERA_KEYS,make_attn_mask

PREFIX=7

class PrefixPi0(Pi0):
 def compute_loss(self,rng,observation,actions,*,train=False):
  if actions.shape[-2]!=50:raise ValueError('expected 50-step training windows')
  prep_rng,noise_rng,time_rng=jax.random.split(rng,3)
  observation=model_lib.preprocess_observation(prep_rng,observation,train=train,image_keys=FOUR_CAMERA_KEYS)
  batch_shape=actions.shape[:-2];noise=jax.random.normal(noise_rng,actions.shape)
  t=jax.random.beta(time_rng,1.5,1,batch_shape)*.999+.001;t=t[...,None,None]
  # The recorded first seven actions condition suffix prediction; they are not targets to denoise.
  x=(t*noise+(1-t)*actions).at[...,:PREFIX,:].set(actions[...,:PREFIX,:]);u=noise-actions
  p,pm,pa=self.embed_prefix(observation);s,sm,sa,cond=self.embed_suffix(observation,x,t[...,0,0]);mask=jnp.concatenate([pm,sm],axis=1);ar=jnp.concatenate([pa,sa],axis=0)
  attention=make_attn_mask(mask,ar);positions=jnp.cumsum(mask,axis=1)-1
  (_,suffix),_=self.PaliGemma.llm([p,s],mask=attention,positions=positions,adarms_cond=[None,cond])
  prediction=self.action_out_proj(suffix[:,-self.action_horizon:]);loss=jnp.mean(jnp.square(prediction-u),axis=-1)
  # Existing optimizer averages all 50 positions; compensate for seven excluded prefix steps.
  return loss.at[...,:PREFIX].set(0)*(50/(50-PREFIX))

 def sample_actions(self,rng,observation,*,known_prefix=None,num_steps=10,noise=None):
  if known_prefix is None:
   # A first segment has no preceding actions; use the standard unconditional sampler.
   return super().sample_actions(rng,observation,num_steps=num_steps,noise=noise)
  if known_prefix.shape[-2:]!=(PREFIX,self.action_dim):raise ValueError('known_prefix must have shape (batch,7,action_dim)')
  obs=model_lib.preprocess_observation(None,observation,train=False,image_keys=FOUR_CAMERA_KEYS)
  b=obs.state.shape[0]
  if known_prefix.shape[0]!=b:raise ValueError('prefix and observation batch sizes differ')
  if noise is None:noise=jax.random.normal(rng,(b,self.action_horizon,self.action_dim))
  noise=noise.at[:,:PREFIX,:].set(known_prefix)
  prefix,mask,ar=self.embed_prefix(obs);attn=make_attn_mask(mask,ar);pos=jnp.cumsum(mask,axis=1)-1
  _,kv=self.PaliGemma.llm([prefix,None],mask=attn,positions=pos)
  dt=-1.0/num_steps
  def step(carry):
   x,t=carry;s,sm,sa,cond=self.embed_suffix(obs,x,jnp.broadcast_to(t,b));sa_mask=make_attn_mask(sm,sa);p_mask=einops.repeat(mask,'b p -> b s p',s=s.shape[1]);full=jnp.concatenate([p_mask,sa_mask],axis=-1)
   positions=jnp.sum(mask,axis=-1)[:,None]+jnp.cumsum(sm,axis=-1)-1
   (_,suffix),_=self.PaliGemma.llm([None,s],mask=full,positions=positions,kv_cache=kv,adarms_cond=[None,cond])
   v=self.action_out_proj(suffix[:,-self.action_horizon:]);return (x+dt*v).at[:,:PREFIX,:].set(known_prefix),t+dt
  x,_=jax.lax.while_loop(lambda carry:carry[1]>=-dt/2,step,(noise,1.0))
  return x

@dataclasses.dataclass(frozen=True)
class PrefixPi0Config(FourCameraPi0Config):
 def create(self,rng):return PrefixPi0(self,rngs=nnx.Rngs(rng))
