# 本文件负责：验证 pi0_base 参数加载、LoRA 初始化和参数冻结范围。
# 使用与官方训练入口相同的思路：先推导参数形状，再加载基础参数并编译初始化。
# 冻结参数转为 bfloat16；基础 checkpoint 不含的 LoRA 参数保留随机初始化。
# 本脚本不创建优化器、不计算梯度、不更新权重；初始化成功不代表训练显存一定足够。

# 一、先加载配置，保证缓存和显存分配设置先于 JAX 生效
from training_config import BASE_PARAMS, ROOT, get_train_config

from datetime import datetime
import logging
import math

import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.shared import array_typing
from openpi.shared import nnx_utils


# 二、参数初始化函数
def initialize_parameters(cfg, rng, partial_params=None):
    """创建 LoRA 模型参数，并用基础 checkpoint 替换对应的参数。

    参数：
        cfg：OpenArm LoRA 训练配置。
        rng：JAX 随机数键，决定新建 LoRA 参数的初始值。
        partial_params：基础权重字典；缺少的 LoRA 参数沿用初始化值。

    返回：
        NNX 参数状态；冻结参数为 bfloat16，LoRA 参数保持初始化精度。
    """
    model = cfg.model.create(rng)

    if partial_params is not None:
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(partial_params)
        model = nnx.merge(graphdef, state)

    params = nnx.state(model)
    return nnx_utils.state_map(
        params,
        cfg.freeze_filter,
        lambda p: p.replace(p.value.astype(jnp.bfloat16)),
    )


def count_parameters(state):
    """按数组形状统计标量参数数量，不把完整模型复制回 CPU。"""
    return sum(
        math.prod(value.shape)
        for value in jax.tree.leaves(state.to_pure_dict())
    )


# 三、冻结规则检查
def check_freeze_rules(cfg, params):
    """确认可训练参数非空且全部属于 LoRA，冻结参数使用 bfloat16。"""
    trainable = params.filter(cfg.trainable_filter)
    frozen = params.filter(cfg.freeze_filter)

    flat_trainable = traverse_util.flatten_dict(
        trainable.to_pure_dict(), sep="/"
    )
    flat_frozen = traverse_util.flatten_dict(
        frozen.to_pure_dict(), sep="/"
    )

    assert flat_trainable, "没有找到可训练参数"
    assert flat_frozen, "没有找到冻结参数"
    assert all("lora" in name for name in flat_trainable), (
        "发现非 LoRA 参数进入可训练集合"
    )
    assert all("lora" not in name for name in flat_frozen), (
        "发现 LoRA 参数被意外冻结"
    )
    assert all(
        value.dtype == jnp.bfloat16
        for value in flat_frozen.values()
    ), "冻结参数没有全部转换为 bfloat16"

    total_count = count_parameters(params)
    trainable_count = count_parameters(trainable)

    logging.info("总参数数量：%s", f"{total_count:,}")
    logging.info("可训练参数数量：%s", f"{trainable_count:,}")
    logging.info(
        "可训练比例：%.4f%%",
        100 * trainable_count / total_count,
    )
    for name in list(flat_trainable)[:5]:
        logging.info("可训练参数示例：%s", name)


# 四、完整检查流程
def main():
    """加载基础权重并在 GPU 初始化，检查冻结规则和 LoRA 数值。"""
    log_dir = ROOT / "logs/a2_pi0"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    log_path = log_dir / f"check_lora_init_{run_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )

    cfg = get_train_config()
    assert BASE_PARAMS.is_dir(), f"基础参数目录不存在：{BASE_PARAMS}"
    assert jax.default_backend() == "gpu", "当前 JAX 未使用 GPU"

    rng = jax.random.key(cfg.seed)
    logging.info("GPU：%s", jax.devices())
    logging.info("日志：%s", log_path)

    # 只推导形状，避免先实际分配一个完整的随机初始化模型。
    logging.info("推导 LoRA 模型参数形状……")
    params_shape = jax.eval_shape(
        lambda key: initialize_parameters(cfg, key),
        rng,
    )
    check_freeze_rules(cfg, params_shape)

    # 加载器补入 LoRA 的形状占位；只有真实基础参数传给后续初始化。
    logging.info("读取并检查基础权重……")
    expected = params_shape.to_pure_dict()
    loaded = cfg.weight_loader.load(expected)
    array_typing.check_pytree_equality(
        expected=expected,
        got=loaded,
        check_shapes=True,
        check_dtypes=True,
    )

    flat_loaded = traverse_util.flatten_dict(loaded)
    partial = traverse_util.unflatten_dict({
        key: value
        for key, value in flat_loaded.items()
        if not isinstance(value, jax.ShapeDtypeStruct)
    })
    del loaded, flat_loaded

    # JIT 可以消除随即被基础参数覆盖的随机初始化计算；
    # 参数捐赠允许复用输入缓冲区，降低初始化时的额外显存需求。
    logging.info("在 GPU 编译并初始化参数……")
    init_fn = jax.jit(
        lambda key, weights: initialize_parameters(cfg, key, weights),
        donate_argnums=(1,),
    )
    params = init_fn(rng, partial)
    jax.block_until_ready(params)

    check_freeze_rules(cfg, params)

    # 仅把较小的 LoRA 参数复制回 CPU 检查，避免复制整个基础模型。
    trainable = params.filter(cfg.trainable_filter)
    assert all(
        np.isfinite(np.asarray(value)).all()
        for value in jax.tree.leaves(trainable.to_pure_dict())
    ), "LoRA 参数包含非有限数值"

    memory = jax.devices()[0].memory_stats() or {}
    for key in ("bytes_in_use", "peak_bytes_in_use"):
        if key in memory:
            logging.info("%s：%.2f GiB", key, memory[key] / 2**30)

    logging.info("PASS：基础权重加载、LoRA 初始化和冻结规则检查通过")
    logging.info("注意：尚未分配优化器及反向传播激活，训练峰值显存仍待验证")


# 五、命令行入口
if __name__ == "__main__":
    main()