# 本文件负责：在 OpenArm 训练集上进行 10 步 LoRA 试训练，验证训练链路。
# 复用已有配置、数据加载器和参数初始化函数，仅更新 LoRA 参数。
# 检查损失、梯度及实际参数变化，保存训练日志、完整模型参数和优化器状态。
# 本脚本只验证技术链路，不代表模型已经学会抓取；不支持自动续训或覆盖实验。

# 一、先加载配置，保证缓存及显存设置先于机器学习库导入
from training_config import ROOT, get_train_config
from check_lora_init import initialize_parameters, check_freeze_rules
from local_data_loader import OpenArmLoader

from datetime import datetime
import json
import logging
import time

import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from openpi.shared import array_typing
from openpi.shared import normalize
from openpi.training import optimizer as optimizer_lib


# 二、基础权重加载和 LoRA 初始化
def load_parameters(cfg):
    """加载基础参数，初始化缺失的 LoRA 参数，返回 GPU 上的 NNX 状态。"""
    rng = jax.random.key(cfg.seed)
    shape = jax.eval_shape(
        lambda key: initialize_parameters(cfg, key), rng
    )
    expected = shape.to_pure_dict()
    loaded = cfg.weight_loader.load(expected)

    array_typing.check_pytree_equality(
        expected=expected,
        got=loaded,
        check_shapes=True,
        check_dtypes=True,
    )

    partial = traverse_util.unflatten_dict({
        key: value
        for key, value in traverse_util.flatten_dict(loaded).items()
        if not isinstance(value, jax.ShapeDtypeStruct)
    })
    del loaded

    init_fn = jax.jit(
        lambda key, weights: initialize_parameters(cfg, key, weights),
        donate_argnums=(1,),
    )
    params = init_fn(rng, partial)
    jax.block_until_ready(params)
    check_freeze_rules(cfg, params)
    return params


# 三、单步训练：前向、反向、优化器更新
def make_train_step(cfg, graphdef, tx):
    """构造编译后的训练函数，仅对 cfg.trainable_filter 选中的参数求梯度。"""

    def step(params, opt_state, rng, observation, actions):
        model = nnx.merge(graphdef, params)
        model.train()

        def loss_fn(current_model):
            """计算当前批次的平均 flow matching 训练损失。"""
            return jnp.mean(
                current_model.compute_loss(
                    rng, observation, actions, train=True
                )
            )

        diff_state = nnx.DiffState(0, cfg.trainable_filter)
        loss, grads = nnx.value_and_grad(
            loss_fn, argnums=diff_state
        )(model)

        trainable = params.filter(cfg.trainable_filter)
        updates, next_opt_state = tx.update(
            grads, opt_state, trainable
        )
        updated = optax.apply_updates(trainable, updates)

        # 检查实际参数差值，而不只检查优化器给出的理论更新量。
        actual_delta = jax.tree.map(
            lambda new, old: new - old, updated, trainable
        )

        # 只把更新后的 LoRA 状态写回，基础模型参数保持冻结。
        nnx.update(model, updated)
        metrics = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(actual_delta),
        }
        return nnx.state(model), next_opt_state, metrics

    # 每步复用参数和优化器缓冲区，避免同时保留两套完整状态。
    return jax.jit(step, donate_argnums=(0, 1))


# 四、保存模型、优化器及归一化资产
def save_checkpoint(cfg, params, opt_state, data_config, *, step=None):
    """保存指定步数的产物，禁止覆盖已有步数检查点。

    params/ 使用 OpenPI restore_params 可读取的结构。
    optimizer/ 保存步数和优化器状态；本脚本尚未实现续训入口。
    """
    root = cfg.checkpoint_dir
    root.mkdir(parents=True, exist_ok=True)
    actual_step = cfg.num_train_steps if step is None else step
    checkpoint = root / str(actual_step)
    checkpoint.mkdir()

    with ocp.PyTreeCheckpointer() as writer:
        writer.save(
            checkpoint / "params",
            {"params": params.to_pure_dict()},
        )
    with ocp.PyTreeCheckpointer() as writer:
        writer.save(
            checkpoint / "optimizer",
            {
                "step": np.asarray(actual_step),
                "opt_state": opt_state,
            },
        )

    normalize.save(
        checkpoint / "assets" / data_config.asset_id,
        data_config.norm_stats,
    )
    (checkpoint / "config.txt").write_text(repr(cfg) + "\n")
    return checkpoint


# 五、试训练主流程与日志记录
def main():
    """运行配置指定的训练步数，数值检查通过后保存 checkpoint。"""
    cfg = get_train_config()
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(
            f"实验目录已存在，请修改 exp_name：{cfg.checkpoint_dir}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("本次试训练要求 JAX GPU")

    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    log_dir = ROOT / "logs/a2_pi0" / f"train_{run_id}"
    log_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "run.log"),
        ],
    )
    logging.info("日志目录：%s", log_dir)

    loader = OpenArmLoader(cfg, split="train", shuffle=True)
    iterator = iter(loader)
    logging.info("训练轨迹：%s；帧数：%d", loader.episode_ids, loader.num_frames)

    params = load_parameters(cfg)

    # 图结构只需推导形状，不再实际分配一套随机基础权重。
    abstract_model = nnx.eval_shape(
        cfg.model.create, jax.random.key(cfg.seed)
    )
    graphdef = nnx.graphdef(abstract_model)

    # 优化器只为可训练 LoRA 参数创建状态。
    tx = optimizer_lib.create_optimizer(
        cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None
    )
    opt_state = tx.init(params.filter(cfg.trainable_filter))
    train_step = make_train_step(cfg, graphdef, tx)

    records = []
    rng = jax.random.key(cfg.seed + 1)
    logging.info("开始试训练；首次梯度编译可能较慢……")

    for index in range(cfg.num_train_steps):
        observation, actions = next(iterator)
        batch = jax.device_put((observation, actions))
        rng, step_rng = jax.random.split(rng)

        start = time.perf_counter()
        params, opt_state, metrics = train_step(
            params, opt_state, step_rng, *batch
        )
        metrics = {
            key: float(value)
            for key, value in jax.device_get(metrics).items()
        }
        elapsed = time.perf_counter() - start

        if not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError(f"训练出现非有限数值：{metrics}")

        record = {
            "step": index + 1,
            **metrics,
            "seconds": elapsed,
        }
        records.append(record)
        with (log_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

        logging.info(
            "step=%d loss=%.6f grad_norm=%.6f "
            "update_norm=%.8f time=%.3fs",
            index + 1, metrics["loss"], metrics["grad_norm"],
            metrics["update_norm"], elapsed,
        )

    if not any(r["update_norm"] > 0 for r in records):
        raise RuntimeError("所有步骤参数变化均为零")

    memory = jax.devices()[0].memory_stats() or {}
    for key in ("bytes_in_use", "peak_bytes_in_use"):
        if key in memory:
            logging.info("%s：%.2f GiB", key, memory[key] / 2**30)

    logging.info("保存 checkpoint……")
    checkpoint = save_checkpoint(
        cfg, params, opt_state, loader.data_config()
    )
    logging.info("Checkpoint：%s", checkpoint)
    logging.info("PASS：10步LoRA试训练与checkpoint保存通过")


# 六、命令行入口
if __name__ == "__main__":
    main()
