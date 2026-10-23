import dataclasses
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
os.environ["OPENPI_DATA_HOME"] = str(ROOT / "weights/openpi")
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = str(ROOT / ".cache/jax")
import flax.nnx as nnx

from openpi import transforms
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.shared import normalize
from openpi.training import config
from openpi.training import weight_loaders

from openarm_policy import OpenArmInputs, OpenArmOutputs


ROOT = Path(__file__).resolve().parents[2]
NAME = "pi0_openarm_lora"
DATA_ROOT = ROOT / "data/processed/openarm_pickup_v21"
NORM_ROOT = ROOT / "assets" / NAME / "openarm_pickup"

# 使用基础模型初始化；之后单独下载、校验。
BASE_PARAMS = (
    ROOT / "weights/openpi/openpi-assets/checkpoints/pi0_base/params"
)


@dataclasses.dataclass(frozen=True)
class OpenArmDataConfig(config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return config.DataConfig(
            repo_id="local/openarm_pickup",
            asset_id="openarm_pickup",
            norm_stats=normalize.load(NORM_ROOT),
            use_quantile_norm=False,
            data_transforms=transforms.Group(
                inputs=[OpenArmInputs()],
                outputs=[OpenArmOutputs()],
            ),
            model_transforms=config.ModelTransformFactory()(
                model_config
            ),
            action_sequence_keys=("action",),
            prompt_from_task=False,
        )


def get_train_config():
    model = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=16,
    )

    return config.TrainConfig(
        name=NAME,
        exp_name="smoke_001",
        model=model,
        data=OpenArmDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(BASE_PARAMS)
        ),

        # 冻结所有非 LoRA 参数，包括图像编码器。
        freeze_filter=nnx.Not(
            nnx_utils.PathRegex(".*lora.*")
        ),
        ema_decay=None,

        batch_size=1,
        num_workers=0,
        num_train_steps=10,
        log_interval=1,
        save_interval=10,
        wandb_enabled=False,

        assets_base_dir=str(ROOT / "assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
        overwrite=False,
        resume=False,
    )


if __name__ == "__main__":
    cfg = get_train_config()
    data = cfg.data.create(cfg.assets_dirs, cfg.model)

    assert DATA_ROOT.is_dir()
    assert cfg.model.action_dim == 32
    assert data.use_quantile_norm is False
    for key in ("state", "actions"):
        stats = data.norm_stats[key]
        assert stats.mean.shape == (16,)
        assert (stats.std > 0).all()

    print("配置:", cfg.name)
    print("数据目录:", DATA_ROOT)
    print("归一化统计:", NORM_ROOT)
    print("模型内部动作维度:", cfg.model.action_dim)
    print("预测长度:", cfg.model.action_horizon)
    print("Batch size:", cfg.batch_size)
    print("试跑步数:", cfg.num_train_steps)
    print("基础权重已就绪:", BASE_PARAMS.is_dir())
    print("检查点目录:", cfg.checkpoint_dir)
    print("PASS：训练配置和归一化统计检查通过")