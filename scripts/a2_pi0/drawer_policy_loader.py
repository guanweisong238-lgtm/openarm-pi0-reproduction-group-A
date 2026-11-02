"""Load trained policy without silently rounding its learned float32 parameters."""
from pathlib import Path
from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies.policy import Policy
from openpi.shared import normalize


def create_checkpoint_precision_policy(cfg,checkpoint,sample_kwargs):
    checkpoint=Path(checkpoint)
    model=cfg.model.load(model_lib.restore_params(checkpoint/'params'))
    model.eval()
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    stats=normalize.load(checkpoint/'assets'/data.asset_id)
    return Policy(model,
        transforms=[transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,
                    transforms.Normalize(stats,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,transforms.Unnormalize(stats,use_quantiles=data.use_quantile_norm),
                           *data.data_transforms.outputs],
        sample_kwargs=sample_kwargs,metadata=cfg.policy_metadata)
