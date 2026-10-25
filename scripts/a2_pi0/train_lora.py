# 本文件负责：在已验收数据上运行可配置步数的LoRA训练，并定期离线验证与保存。
# 复用试训练的参数初始化、冻结规则和优化步骤，独立实验目录不覆盖smoke检查点。
# 验证集24至29号轨迹每条固定抽取两帧，采用固定随机数计算flow matching损失。
# 验证损失包含模型内部32维及轨迹尾部填充，与抓取成功率或真实动作误差不同。
# 支持--train-episode限制训练轨迹，仍按帧和动作块采样。
# 本入口从pi0_base重新初始化，不支持从中断处恢复优化器与数据迭代器。

# 一、先加载项目配置，再导入机器学习依赖。
from training_config import ROOT, get_train_config
from train_smoke import load_parameters, make_train_step, save_checkpoint
from local_data_loader import OpenArmLoader, collate_numpy

import argparse
import dataclasses
from datetime import datetime
import json
import logging
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_lib
from openpi.training import optimizer as optimizer_lib


# 二、独立训练配置；沿用原有模型结构、数据统计和冻结规则。
def build_config(args):
    """构造首轮1000步实验配置，学习率预热100步后余弦衰减。"""
    base = get_train_config()
    schedule = optimizer_lib.CosineDecaySchedule(
        warmup_steps=min(100, max(1, args.steps // 10)),
        peak_lr=2.5e-5,
        decay_steps=args.steps,
        decay_lr=2.5e-6,
    )
    return dataclasses.replace(
        base, exp_name=args.exp_name, num_train_steps=args.steps,
        lr_schedule=schedule, save_interval=args.save_every, log_interval=10,
    )


# 三、固定验证观测，确保不同训练步数比较的是同一批数据与随机目标。
def validation_samples(cfg):
    """每条验证轨迹抽取首帧和中间帧，不使用训练数据重新计算统计。"""
    loader = OpenArmLoader(cfg, split="val", shuffle=False)
    transformed = loader._loader.dataset
    raw = transformed.dataset
    episode_ids = np.asarray([
        int(value) for value in raw.hf_dataset["episode_index"]
    ])
    samples, selected = [], []
    for episode in loader.episode_ids:
        indices = np.flatnonzero(episode_ids == episode)
        for index in (indices[0], indices[len(indices) // 2]):
            batch = collate_numpy([transformed[int(index)]])
            actions = batch.pop("actions")
            observation = model_lib.Observation.from_dict(batch)
            samples.append(jax.device_put((observation, actions)))
            selected.append({"episode": episode, "subset_index": int(index)})
    return samples, selected


def make_validation(graphdef):
    """构造不求梯度、不更新参数的固定噪声验证函数。"""
    @jax.jit
    def evaluate(params, observation, actions, rng):
        model = nnx.merge(graphdef, params)
        return jnp.mean(model.compute_loss(rng, observation, actions, train=False))
    return evaluate


# 四、训练、周期验证和检查点保存。
def main():
    """运行有限训练步数，记录初始及周期验证损失，保存中间和最终模型。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--exp-name", default="lora_001")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-episode", type=int, help="仅用指定训练轨迹做过拟合试验")
    args = parser.parse_args()
    if args.steps < 2 or min(args.save_every, args.eval_every) < 1:
        raise ValueError("训练步数至少2，保存与验证间隔必须为正数")
    if not args.exp_name or any(char in args.exp_name for char in "/\\"):
        raise ValueError("实验名称必须为单个目录名")
    cfg = build_config(args)
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(cfg.checkpoint_dir)
    if args.dry_run:
        print(cfg)
        return
    if jax.default_backend() != "gpu":
        raise RuntimeError("训练需要JAX GPU")

    output = ROOT / "logs/a2_pi0" / f"lora_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")])
    (output / "config.txt").write_text(repr(cfg) + "\n")
    logging.info("实验=%s，日志=%s", cfg.exp_name, output)
    loader = OpenArmLoader(cfg, split="train", shuffle=True,
        episode_ids=None if args.train_episode is None else [args.train_episode])
    iterator = iter(loader)
    samples, selected = validation_samples(cfg)
    (output / "validation_samples.json").write_text(json.dumps(selected, indent=2) + "\n")
    logging.info("训练帧=%d，验证抽样=%d", loader.num_frames, len(samples))
    params = load_parameters(cfg)
    abstract = nnx.eval_shape(cfg.model.create, jax.random.key(cfg.seed))
    graphdef = nnx.graphdef(abstract)
    tx = optimizer_lib.create_optimizer(cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None)
    opt_state = tx.init(params.filter(cfg.trainable_filter))
    train_step = make_train_step(cfg, graphdef, tx)
    evaluate = make_validation(graphdef)
    rng = jax.random.key(cfg.seed + 1)
    validation_records = []

    def validate(step):
        """对固定12帧计算相同随机目标的验证损失并即时落盘。"""
        losses = [float(evaluate(params, *sample, jax.random.key(10000 + i)))
                  for i, sample in enumerate(samples)]
        if not np.isfinite(losses).all():
            raise RuntimeError("验证损失非有限")
        record = {"step": step, "mean_flow_loss": float(np.mean(losses)), "sample_losses": losses}
        validation_records.append(record)
        with (output / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        logging.info("validation step=%d mean_flow_loss=%.6f", step, record["mean_flow_loss"])

    validate(0)
    for step in range(1, cfg.num_train_steps + 1):
        batch = jax.device_put(next(iterator))
        rng, key = jax.random.split(rng)
        began = time.perf_counter()
        params, opt_state, metrics = train_step(params, opt_state, key, *batch)
        metrics = {name: float(value) for name, value in jax.device_get(metrics).items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"step={step}训练数值异常：{metrics}")
        record = {"step": step, **metrics, "seconds": time.perf_counter() - began}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if step == 1 or step % cfg.log_interval == 0:
            logging.info("step=%d loss=%.6f update_norm=%.8f seconds=%.3f", step, metrics["loss"], metrics["update_norm"], record["seconds"])
        if step % args.eval_every == 0 or step == cfg.num_train_steps:
            validate(step)
        if step % args.save_every == 0 or step == cfg.num_train_steps:
            path = save_checkpoint(cfg, params, opt_state, loader.data_config(), step=step)
            logging.info("已保存检查点：%s", path)

    summary = {"experiment": cfg.exp_name, "steps": cfg.num_train_steps, "train_episode_ids": loader.episode_ids,
        "checkpoint": str(cfg.checkpoint_dir / str(cfg.num_train_steps)),
        "initial_validation_flow_loss": validation_records[0]["mean_flow_loss"],
        "final_validation_flow_loss": validation_records[-1]["mean_flow_loss"],
        "validation_samples": selected, "task_success_verified": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("PASS：训练、固定验证抽样及检查点保存完成")


# 五、命令行入口。
if __name__ == "__main__":
    main()
