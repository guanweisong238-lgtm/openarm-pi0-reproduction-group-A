"""13条成功示教的独立OpenPI配置与无损多轨迹数据入口；不启动训练。"""
import dataclasses

from openpi import transforms
from openpi.shared import normalize
from openpi.training import config

from drawer_training_config import ROOT, get_drawer_config
from drawer_multi_episode_dataset import MultiEpisodeDrawerDataset
from local_data_loader import TransformedDataset
from openarm_policy import OpenArmInputs, OpenArmOutputs

NAME='pi0_openarm_drawer_success_v003_lora'
ASSET_ID='openarm_drawer_success_v003'
DATA_ROOT=ROOT/'data/processed/openarm_drawer_success_v003'
MANIFEST=DATA_ROOT/'manifest.json'
NORM_ROOT=ROOT/'assets'/NAME/ASSET_ID


@dataclasses.dataclass(frozen=True)
class SuccessDataConfig(config.DataConfigFactory):
    def create(self,assets_dirs,model_config):
        if not (DATA_ROOT/'READY.json').exists():
            raise RuntimeError('成功数据版本未构建完成')
        return config.DataConfig(repo_id='local/openarm_drawer_success_v003',asset_id=ASSET_ID,
            norm_stats=normalize.load(NORM_ROOT),use_quantile_norm=False,
            data_transforms=transforms.Group(inputs=[OpenArmInputs()],outputs=[OpenArmOutputs()]),
            model_transforms=config.ModelTransformFactory()(model_config),
            action_sequence_keys=('action',),prompt_from_task=False)


def get_success_config(exp_name='multiepisode_001',steps=2000):
    return dataclasses.replace(get_drawer_config(exp_name,steps,lossless=True),name=NAME,
        data=SuccessDataConfig())


def transformed_dataset(cfg,split='train',cache_episodes=2):
    """显式按split读取；旧单轨迹训练脚本不会自动切换到此入口。"""
    raw=MultiEpisodeDrawerDataset(MANIFEST,split,horizon=cfg.model.action_horizon,
                                 cache_episodes=cache_episodes)
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    return TransformedDataset(raw,[*data.repack_transforms.inputs,*data.data_transforms.inputs,
        transforms.Normalize(data.norm_stats,use_quantiles=data.use_quantile_norm),
        *data.model_transforms.inputs]),data


if __name__=='__main__':
    cfg=get_success_config()
    for split in ('train','validation'):
        dataset,_=transformed_dataset(cfg,split,cache_episodes=1)
        sample=dataset[0]
        print(split,'frames',len(dataset),'actions',sample['actions'].shape,
              'state',sample['state'].shape,flush=True)
