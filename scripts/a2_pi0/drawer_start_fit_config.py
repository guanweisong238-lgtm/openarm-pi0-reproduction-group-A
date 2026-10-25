"""按检查点协议恢复起步诊断的输入语义；保持旧v003配置和缓存不变。"""
from drawer_success_training_config import SuccessDataConfig,get_success_config
import dataclasses
import numpy as np
from openpi import transforms


@dataclasses.dataclass(frozen=True)
class MaskLeftState(transforms.DataTransformFn):
    def __call__(self,data):
        result=dict(data)
        result['state']=np.asarray(data['state']).copy()
        result['state'][8:16]=0
        return result


@dataclasses.dataclass(frozen=True)
class MaskedSuccessDataConfig(SuccessDataConfig):
    def create(self,assets_dirs,model_config):
        data=super().create(assets_dirs,model_config)
        return dataclasses.replace(data,model_transforms=transforms.Group(
            inputs=[*data.model_transforms.inputs,MaskLeftState()],outputs=data.model_transforms.outputs))


def get_policy_config(protocol):
    cfg=get_success_config()
    if 'action_horizon' in protocol:
        cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,action_horizon=int(protocol['action_horizon'])))
    if protocol.get('state_representation')=='position_velocity16':
        from drawer_velocity_config import VelocityDeltaSuccessDataConfig
        return dataclasses.replace(cfg,data=VelocityDeltaSuccessDataConfig())
    if protocol.get('action_representation')=='right7_delta':
        from drawer_delta_config import DeltaSuccessDataConfig
        return dataclasses.replace(cfg,data=DeltaSuccessDataConfig(mask_left_state=bool(protocol.get('mask_left_normalized_state'))))
    return dataclasses.replace(cfg,data=MaskedSuccessDataConfig()) if protocol.get('mask_left_normalized_state') else cfg


def get_stats_location(protocol):
    from drawer_success_training_config import NORM_ROOT,ASSET_ID
    if protocol.get('state_representation')=='position_velocity16':
        from drawer_velocity_config import VELOCITY_NORM_ROOT,VELOCITY_ASSET_ID
        return VELOCITY_NORM_ROOT,VELOCITY_ASSET_ID
    if protocol.get('action_representation')=='right7_delta':
        from drawer_delta_config import DELTA_NORM_ROOT,DELTA_ASSET_ID
        return DELTA_NORM_ROOT,DELTA_ASSET_ID
    return NORM_ROOT,ASSET_ID
