"""Pi0现有32维状态槽：16个实际关节位置+16个实际关节速度。"""
import dataclasses
import numpy as np
from openpi import transforms
from openpi.shared import normalize
from drawer_success_training_config import ROOT,SuccessDataConfig
from drawer_delta_config import MASK
from openarm_policy import OpenArmInputs,OpenArmOutputs

VELOCITY_ASSET_ID='openarm_drawer_success_v003_right7_delta_velocity'
VELOCITY_NORM_ROOT=ROOT/'assets/pi0_openarm_drawer_velocity_v001'/VELOCITY_ASSET_ID


@dataclasses.dataclass(frozen=True)
class VelocityInputs(transforms.DataTransformFn):
    def __call__(self,data):
        result=OpenArmInputs()(data)
        velocity=np.asarray(data['observation.velocity'],np.float32)
        if velocity.shape!=(16,) or not np.isfinite(velocity).all():raise ValueError('需要有限16维实际关节速度')
        result['state']=np.concatenate([result['state'],velocity])
        return result


@dataclasses.dataclass(frozen=True)
class MaskLeftPositionVelocity(transforms.DataTransformFn):
    def __call__(self,data):
        result=dict(data);state=np.asarray(data['state']).copy()
        if state.shape!=(32,):raise ValueError('速度协议需要32维状态')
        state[8:16]=0;state[24:32]=0;result['state']=state
        return result


@dataclasses.dataclass(frozen=True)
class VelocityDeltaSuccessDataConfig(SuccessDataConfig):
    def create(self,assets_dirs,model_config):
        data=super().create(assets_dirs,model_config)
        return dataclasses.replace(data,asset_id=VELOCITY_ASSET_ID,norm_stats=normalize.load(VELOCITY_NORM_ROOT),
            data_transforms=transforms.Group(inputs=[VelocityInputs(),transforms.DeltaActions(MASK)],
                outputs=[transforms.AbsoluteActions(MASK),OpenArmOutputs()]),
            model_transforms=transforms.Group(inputs=[*data.model_transforms.inputs,MaskLeftPositionVelocity()],
                outputs=data.model_transforms.outputs))
