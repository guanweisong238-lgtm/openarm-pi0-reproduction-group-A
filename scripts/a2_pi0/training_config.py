# 本文件负责：集中定义 OpenArm 的 π0 LoRA 试训练配置及项目缓存、数据和权重路径。
# 组合输入输出适配、归一化统计和模型预处理，指定 LoRA 模型变体及非 LoRA 参数冻结规则。
# 当前配置使用批量大小 1、预测长度 16、训练 10 步；本地数据划分由 local_data_loader.py 执行。
# 直接运行只校验并打印配置，不下载基础权重，也不启动训练循环。

import dataclasses
from pathlib import Path
import os

# 一、导入机器学习库前设置缓存和显存分配选项
# 此模块导入时即设置环境变量，因此数据加载器应先导入本模块。
# 关闭预分配可避免启动即占用大部分显存，但不保证 16 GB 足够完成训练。
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
# 二、实验标识和本地输入路径
# 原始机器人向量为 16 维；归一化后由模型变换补齐至内部的 32 维。
NAME = "pi0_openarm_lora"
DATA_ROOT = ROOT / "data/processed/openarm_pickup_v21"
NORM_ROOT = ROOT / "assets" / NAME / "openarm_pickup"

# 使用基础模型初始化；之后单独下载、校验。
BASE_PARAMS = (
    ROOT / "weights/openpi/openpi-assets/checkpoints/pi0_base/params"
)


@dataclasses.dataclass(frozen=True)
class OpenArmDataConfig(config.DataConfigFactory):
    """将 OpenArm 输入输出适配、已有统计和模型变换组装成数据配置。

    本类只描述变换协议；本地数据根目录和实际轨迹划分由 OpenArmLoader 选择。
    repo_id 是数据标识，不表示需要从 Hugging Face 下载数据。
    """

    def create(self, assets_dirs, model_config):
        """返回 OpenPI DataConfig，并加载固定 NORM_ROOT 中的统计。

        assets_dirs 为上游工厂接口参数，本实现不据此寻找统计文件。
        model_config 决定图像缩放、指令分词和补齐规则；统计文件缺失时直接报错。
        构造模型变换可能触发分词器资源的缓存读取或下载。
        """
        return config.DataConfig(
            repo_id="local/openarm_pickup",
            asset_id="openarm_pickup",
            norm_stats=normalize.load(NORM_ROOT),
            # 与 compute_openarm_norm_stats.py 的均值/标准差验收方式保持一致。
            use_quantile_norm=False,
            data_transforms=transforms.Group(
                inputs=[OpenArmInputs()],
                outputs=[OpenArmOutputs()],
            ),
            model_transforms=config.ModelTransformFactory()(
                model_config
            ),
            action_sequence_keys=("action",),
            # OpenArmInputs 已读取样本的 task；无需重复注入任务文本。
            prompt_from_task=False,
        )


# 三、最小 LoRA 试训练配置
def get_train_config():
    """返回用于验证训练链路的 10 步配置，不创建模型或启动训练。

    两个语言模型分支均使用 LoRA 变体；基础参数应由 BASE_PARAMS 初始化。
    预测 16 个时间步对应数据中的约 0.533 秒，不代表仿真控制频率已适配。
    此配置尚不能替代显存实测、参数冻结核验和训练后 checkpoint 重载验收。
    """
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
        # 试跑不维护 EMA 副本，以减少额外参数存储。
        ema_decay=None,

        batch_size=1,
        # 单进程解码便于定位数据异常，也避免额外进程复制状态。
        num_workers=0,
        num_train_steps=10,
        log_interval=1,
        save_interval=10,
        wandb_enabled=False,

        assets_base_dir=str(ROOT / "assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
        # 同名实验产物不自动覆盖；需要续训时应明确调整实验配置。
        overwrite=False,
        resume=False,
    )


# 四、独立配置检查入口
# 此处只创建配置和变换，不初始化大模型，也不验证基础 checkpoint 的完整性。
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
