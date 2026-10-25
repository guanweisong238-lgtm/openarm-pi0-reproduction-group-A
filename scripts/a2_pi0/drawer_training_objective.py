"""Unreduced pi0 flow error for drawer diagnostics, matching installed OpenPI.

dims 为 None 时保留全部 32 维（上游口径）；传入切片则只返回对应维度的误差，
供"只对真实右臂维度训练"的对照使用。未归约的误差形状为 (batch, horizon, dims)。
"""
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask


def sample_flow_time(rng, shape, mode='beta'):
    time = jax.random.beta(rng, 1.5, 1, shape) * .999 + .001
    if mode == 'beta':
        return time
    if mode != 'low_mix':
        raise ValueError(f'Unknown time sampling: {mode}')
    # 保留一半原分布，另一半采t∈[.05,.3)，单独的随机键不改变原noise。
    select = jax.random.bernoulli(jax.random.fold_in(rng, 1), .5, shape)
    low = jax.random.uniform(jax.random.fold_in(rng, 2), shape, minval=.05, maxval=.3)
    return jnp.where(select, low, time)


def flow_squared_errors(model, rng, observation, actions, *, augment=False, dims=None,
                        time_sampling='beta', return_time=False, return_prediction=False):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = model_lib.preprocess_observation(preprocess_rng, observation, train=augment)
    if time_sampling=='fixed_endpoint':
        code=np.random.default_rng(42).standard_normal(actions.shape[-2:]).astype(np.float32)
        noise=jnp.broadcast_to(jnp.asarray(code),actions.shape)
        time=jnp.ones(actions.shape[:-2])
    else:
        noise = jax.random.normal(noise_rng, actions.shape)
        time = sample_flow_time(time_rng, actions.shape[:-2], time_sampling)
    x_t = time[..., None, None] * noise + (1-time[..., None, None]) * actions
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, cond = model.embed_suffix(observation, x_t, time)
    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    attn = make_attn_mask(mask, jnp.concatenate([prefix_ar, suffix_ar], axis=0))
    positions = jnp.cumsum(mask, axis=1)-1
    (_, output), _ = model.PaliGemma.llm([prefix, suffix], mask=attn,
        positions=positions, adarms_cond=[None, cond])
    velocity = model.action_out_proj(output[:, -model.action_horizon:])
    error = (velocity-(noise-actions))**2
    error = error if dims is None else error[..., dims]
    if return_prediction:return error,x_t-time[...,None,None]*velocity
    return (error, time) if return_time else error
