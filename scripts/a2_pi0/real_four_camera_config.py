"""Four-camera pi0 configuration for real-data experiments, independent of old runs."""
import dataclasses
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.shared import array_typing as at
from real_four_camera_pi0 import FOUR_CAMERA_KEYS, Pi0
import numpy as np

def parse_image(value):
    """Convert an RGB image to contiguous HWC uint8 without rotating it."""
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.shape}")
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating images must be finite and in [0,1]")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"Unsupported image dtype {image.dtype}")
    return np.ascontiguousarray(image)

CAMERA_MAPPING = {
    'base_0_rgb': 'global',
    'chest_0_rgb': 'chest',
    'left_wrist_0_rgb': 'wrist_left',
    'right_wrist_0_rgb': 'wrist_right',
}

@dataclasses.dataclass(frozen=True)
class FourCameraPi0Config(pi0_config.Pi0Config):
    action_horizon: int = 50

    def create(self, rng):
        return Pi0(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        image = jax.ShapeDtypeStruct([batch_size, *model_lib.IMAGE_RESOLUTION, 3], jnp.float32)
        mask = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            obs = model_lib.Observation(
                images={name: image for name in FOUR_CAMERA_KEYS},
                image_masks={name: mask for name in FOUR_CAMERA_KEYS},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        return obs, jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

@dataclasses.dataclass(frozen=True)
class FourCameraInputs:
    """Map raw RGB views plus right-first state and actions to OpenPI input."""
    def __call__(self, sample):
        state = np.asarray(sample['state'], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError('state must be finite and have shape (16,)')
        prompt = sample['prompt']
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must be a nonempty string')
        result = {'state': state, 'prompt': prompt,
                  'image': {key: parse_image(sample['images'][source]) for key, source in CAMERA_MAPPING.items()},
                  'image_mask': {key: np.bool_(True) for key in CAMERA_MAPPING}}
        if 'actions' in sample:
            actions = np.asarray(sample['actions'], dtype=np.float32)
            if actions.shape != (50, 16) or not np.isfinite(actions).all():
                raise ValueError('actions must be finite and have shape (50, 16)')
            result['actions'] = actions
        return result
