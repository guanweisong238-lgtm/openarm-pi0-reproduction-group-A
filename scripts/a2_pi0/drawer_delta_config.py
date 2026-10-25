"""右7关节残差动作的独立统计/策略配置；不修改原v003绝对动作资产。"""
from drawer_success_training_config import ROOT,MANIFEST,NORM_ROOT,SuccessDataConfig
from drawer_start_fit_config import MaskLeftState
import dataclasses
import hashlib
import json
import numpy as np
from openpi import transforms
from openpi.shared import normalize
from openarm_policy import OpenArmInputs,OpenArmOutputs

DELTA_ASSET_ID='openarm_drawer_success_v003_right7_delta'
DELTA_NORM_ROOT=ROOT/'assets/pi0_openarm_drawer_start_fit_delta_v001'/DELTA_ASSET_ID
MASK=(True,)*7+(False,)*9


def prepare_delta_stats():
    manifest=json.loads(MANIFEST.read_text());sha=hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if (DELTA_NORM_ROOT/'norm_stats.json').exists():
        provenance=json.loads((DELTA_NORM_ROOT/'provenance.json').read_text())
        if provenance['manifest_sha256']!=sha:raise ValueError('残差统计manifest不匹配')
        return
    arrays=[];ids=[]
    for row in manifest['episodes']:
        if row['split']!='train':continue
        with np.load(ROOT/row['path']/'episode.npz') as data:
            state=data['state'][:,:7];actions=data['action'][:,:7]
            indices=np.arange(len(state))[:,None]+np.arange(16)
            valid=indices<len(state)
            residual=actions[np.minimum(indices,len(state)-1)]-state[:,None,:]
            arrays.append(residual[valid]);ids.append(row['episode_id'])
    values=np.concatenate(arrays)
    original=json.loads((NORM_ROOT/'norm_stats.json').read_text())
    action=original['norm_stats']['actions']
    for key,value in dict(mean=values.mean(axis=0),std=np.maximum(values.std(axis=0),1e-6),
                          q01=np.quantile(values,.01,axis=0),q99=np.quantile(values,.99,axis=0)).items():
        action[key][:7]=value.tolist()
    DELTA_NORM_ROOT.mkdir(parents=True,exist_ok=False)
    (DELTA_NORM_ROOT/'norm_stats.json').write_text(json.dumps(original,indent=2)+'\n')
    (DELTA_NORM_ROOT/'provenance.json').write_text(json.dumps(dict(manifest_sha256=sha,train_episode_ids=ids,
        representation='right7_delta',definition='action[t+k,:7] - state[t,:7]',horizon=16,
        valid_chunk_entries=int(len(values)),padding_excluded=True,state_statistics='original full-train state statistics',
        other_action_statistics='original absolute gripper and left statistics'),indent=2)+'\n')


@dataclasses.dataclass(frozen=True)
class DeltaSuccessDataConfig(SuccessDataConfig):
    mask_left_state:bool=False

    def create(self,assets_dirs,model_config):
        data=super().create(assets_dirs,model_config)
        inputs=[*data.model_transforms.inputs]
        if self.mask_left_state:inputs.append(MaskLeftState())
        return dataclasses.replace(data,asset_id=DELTA_ASSET_ID,norm_stats=normalize.load(DELTA_NORM_ROOT),
            data_transforms=transforms.Group(inputs=[OpenArmInputs(),transforms.DeltaActions(MASK)],
                outputs=[transforms.AbsoluteActions(MASK),OpenArmOutputs()]),
            model_transforms=transforms.Group(inputs=inputs,outputs=data.model_transforms.outputs))
