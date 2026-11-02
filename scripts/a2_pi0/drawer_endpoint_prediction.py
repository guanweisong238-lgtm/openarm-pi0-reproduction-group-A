"""Differentiable one-step predictor using the exact deployment KV-cache path."""
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask


def endpoint_prediction(model, observation):
    observation=model_lib.preprocess_observation(None,observation,train=False)
    noise=jnp.broadcast_to(jnp.asarray(np.random.default_rng(42).standard_normal(
        (model.action_horizon,model.action_dim)).astype(np.float32)),
        (observation.state.shape[0],model.action_horizon,model.action_dim))
    prefix,pmask,par=model.embed_prefix(observation)
    _,cache=model.PaliGemma.llm([prefix,None],mask=make_attn_mask(pmask,par),
                               positions=jnp.cumsum(pmask,axis=1)-1)
    suffix,smask,sar,cond=model.embed_suffix(observation,noise,jnp.ones(observation.state.shape[0]))
    mask=jnp.concatenate([jnp.broadcast_to(pmask[:,None,:],
        (pmask.shape[0],suffix.shape[1],pmask.shape[1])),make_attn_mask(smask,sar)],axis=-1)
    (_,out),_=model.PaliGemma.llm([None,suffix],mask=mask,
        positions=jnp.sum(pmask,axis=-1)[:,None]+jnp.cumsum(smask,axis=-1)-1,
        kv_cache=cache,adarms_cond=[None,cond])
    return noise-model.action_out_proj(out[:,-model.action_horizon:])
