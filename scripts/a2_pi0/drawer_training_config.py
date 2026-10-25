"""独立的抽屉往返单轨迹LoRA配置；与抓方块数据、统计和检查点分开。"""
from training_config import ROOT, get_train_config
import dataclasses
from openpi import transforms
from openpi.shared import normalize
from openpi.training import config, optimizer
from openarm_policy import OpenArmInputs, OpenArmOutputs

DATA_ROOT = ROOT / "data/drawer_trial/contact_roundtrip_rgb_001/lerobot_v21"
EPISODE_ROOT = DATA_ROOT.parent
NAME = "pi0_openarm_drawer_lora"
ASSET_ID = "openarm_drawer_roundtrip"
NORM_ROOT = ROOT / "assets" / NAME / ASSET_ID


@dataclasses.dataclass(frozen=True)
class DrawerDataConfig(config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return config.DataConfig(
            repo_id="local/openarm_drawer_trial", asset_id=ASSET_ID,
            norm_stats=normalize.load(NORM_ROOT), use_quantile_norm=False,
            data_transforms=transforms.Group(inputs=[OpenArmInputs()], outputs=[OpenArmOutputs()]),
            model_transforms=config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",), prompt_from_task=False)


def get_drawer_config(exp_name="overfit_roundtrip_001", steps=2000, lossless=False):
    return dataclasses.replace(get_train_config(), name=NAME + ("_lossless" if lossless else ""), exp_name=exp_name,
        data=DrawerDataConfig(), num_train_steps=steps, save_interval=1000, log_interval=20,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=min(100, steps // 10),
            peak_lr=2.5e-5, decay_steps=steps, decay_lr=2.5e-6))
