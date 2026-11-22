"""Check that four cameras survive preprocessing and affect image-token computation."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace
from openpi.models import model as model_lib
from real_four_camera_config import FourCameraInputs, FourCameraPi0Config
from real_four_camera_pi0 import FOUR_CAMERA_KEYS, Pi0

cfg = FourCameraPi0Config(action_horizon=50)
spec, actions = cfg.inputs_spec(batch_size=1)
assert tuple(spec.images) == FOUR_CAMERA_KEYS
assert actions.shape == (1, 50, 32)
raw = {'images': {k: np.full((224,224,3), i * 30, np.uint8) for i,k in enumerate(('global','chest','wrist_left','wrist_right'))},
       'state': np.zeros(16, np.float32), 'actions': np.zeros((50,16),np.float32), 'prompt': 'Open drawer and put the block inside, then close the drawer.'}
inputs = FourCameraInputs()(raw)
assert tuple(inputs['image']) == FOUR_CAMERA_KEYS and inputs['actions'].shape == (50,16)

def observation(chest):
    images = {k: jnp.asarray(v[None], dtype=jnp.float32) / 127.5 - 1 for k,v in inputs['image'].items()}
    images['chest_0_rgb'] = chest
    return model_lib.Observation(images=images, image_masks={k:jnp.ones((1,),dtype=bool) for k in images}, state=jnp.zeros((1,32)))

# Exercise the real preprocessing and the actual Pi0.embed_prefix implementation,
# using a tiny differentiable stand-in for the large visual encoder.
class TinyVision:
    def __call__(self, image, train=False):
        return jnp.mean(image,axis=(1,2,3))[:,None,None], None
fake = SimpleNamespace(PaliGemma=SimpleNamespace(img=TinyVision()))
def visual_tokens(chest):
    obs = model_lib.preprocess_observation(None, observation(chest), image_keys=FOUR_CAMERA_KEYS)
    return Pi0.embed_prefix(fake, obs)[0]
chest = jnp.full((1,224,224,3), -0.5)
tokens = visual_tokens(chest)
grad = jax.grad(lambda c: visual_tokens(c).sum())(chest)
assert tokens.shape == (1,4,1), tokens.shape
assert float(jnp.sum(jnp.abs(grad))) > 0
print('PASS: four RGB inputs -> four preprocessed image tokens; chest gradient is nonzero')
print('PASS: model spec action horizon 50, 32 internal dimensions; real input action shape 50 x 16')
print('NOTE: tiny stand-in vision encoder only; not a pretrained-model forward/backward or robot-data training')
