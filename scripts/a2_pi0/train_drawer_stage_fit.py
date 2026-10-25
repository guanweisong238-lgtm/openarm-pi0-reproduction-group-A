"""从无损检查点续训，按阶段加权强化接近帧，用动作与空间误差验收。

只更新 LoRA：图像增强、冻结规则和动作表示保持不变；支持维度选择与关节加权。
每500步记录分阶段flow损失、三噪声采样动作MAE、正运动学夹持点位置/姿态误差，
并按"连续两次检查无改善即停"给出续训建议。
所有指标取自同一条训练轨迹，不是验证集，也不构成任务成功证据。
本入口不覆盖已有检查点，也不自动宣称过拟合完成。
"""
from drawer_training_config import ROOT, get_drawer_config
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# 阶段边界与 evaluate_drawer_overfit.py 的离线分段完全一致，便于同口径对比。
PHASES = (("approach", 0, 150), ("grasp", 150, 192), ("pull", 192, 282),
          ("open_hold", 282, 312), ("push", 312, 402), ("closed_hold", 402, 462))
GROUPS = {"approach": ("approach",), "grasp": ("grasp",),
          "rest": ("pull", "open_hold", "push", "closed_hold")}
MIXTURE = {"approach": 0.5, "grasp": 0.2, "rest": 0.3}
# 早期接近帧在诊断中误差最大（第0帧右臂0.18 rad），单列出来避免被阶段均值掩盖。
EARLY_APPROACH = (0, 1, 2, 4, 8)
ANCHORS_PER_PHASE = 4
NOISE_SEEDS = (42, 7, 99)
GRIP_OFFSET = np.array([0.0, 0.0, -0.16])
# 目标门槛：接近阶段右臂MAE与夹持点位置误差，用于筛选而不是宣布成功。
TARGET_APPROACH_ARM_MAE = 0.01
TARGET_POSITION_MM = 2.0
# 姿态只作诊断参考，不参与停止判定：接近段当前约8°，其余阶段1.2–1.7°。
TARGET_ATTITUDE_DEG = 3.0
STALL_TOLERANCE = 0.005  # 改善不足0.5%视为本次未改善
REGRESSION_TOLERANCE = 0.20  # 任一关键指标恶化超过此比例视为回退，即使另一项改善
# 训练损失的维度选择。all32 为上游口径；right8 只算右臂7维+右夹爪，
# 把左臂（物理静止）与补齐16维从梯度中移除。改变后不能与 all32 实验直接比较。
LOSS_DIMS = {"all32": None, "all16": slice(0, 16), "right8": slice(0, 8)}


def objective_weights(loss_dims, joint_weighting="uniform"):
    """均值归一化保持总权重一致；仅改变选中维度之间的相对权重。"""
    if loss_dims not in LOSS_DIMS or joint_weighting not in ("uniform", "key147"):
        raise ValueError("未知损失维度或关节加权方案")
    count = {"all32": 32, "all16": 16, "right8": 8}[loss_dims]
    weights = np.ones(count, dtype=np.float32)
    if joint_weighting == "key147":
        if loss_dims != "right8":
            raise ValueError("key147 仅适用于 --loss-dims right8")
        weights[[0, 3, 6]] = 1.5
    return weights / weights.mean()


def reduce_objective(errors, weights):
    """errors 的末维已按 loss_dims 选取；训练与诊断共用此归约。"""
    return (errors * weights).mean()


def phase_frames():
    return {name: np.arange(start, stop) for name, start, stop in PHASES}


def group_frames():
    frames = phase_frames()
    return {group: np.concatenate([frames[name] for name in names])
            for group, names in GROUPS.items()}


def anchor_frames():
    """分阶段固定检查帧：每阶段均匀取样，approach另并入早期帧。"""
    anchors = {name: np.unique(np.linspace(start, stop - 1, ANCHORS_PER_PHASE, dtype=int))
               for name, start, stop in PHASES}
    anchors["approach"] = np.unique(np.concatenate([EARLY_APPROACH, anchors["approach"]]))
    return anchors


def sampling_summary(mixture, steps):
    """按帧统计期望更新次数，用于核对加权是否真的落在难帧上。"""
    frames = phase_frames()
    rows, total = [], sum(len(value) for value in frames.values())
    for group, names in GROUPS.items():
        pool = np.concatenate([frames[name] for name in names])
        for name in names:
            rows.append({"group": group, "phase": name, "frames": int(len(frames[name])),
                         "expected_updates_per_frame": mixture[group] / len(pool) * steps})
    return {"mixture": mixture, "steps": steps, "frames": total,
            "uniform_updates_per_frame": steps / total, "per_phase": rows}


def make_sampler(seed, mixture, pools):
    """每步先按组概率抽阶段组，再在组内均匀抽帧；同一随机种子可复现。"""
    groups = list(mixture)
    probabilities = [mixture[name] for name in groups]
    if min(probabilities) < 0 or abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("阶段采样比例必须非负且和为1")
    rng = np.random.default_rng(seed)

    def sample():
        group = groups[int(rng.choice(len(groups), p=probabilities))]
        pool = pools[group]
        return int(pool[int(rng.integers(len(pool)))]), group

    return sample


def fk_probe():
    """复现诊断脚本的夹持点定义：右腕控制site沿局部Z偏移16 cm。"""
    import mujoco
    from drawer_simulation import DrawerSimulation
    environment = DrawerSimulation()
    model, scratch = environment.model, mujoco.MjData(environment.model)
    site = model.site("right_ee_control_point").id
    qpos_ids = environment.qpos_ids

    def probe(base_qpos, action):
        scratch.qpos[:] = base_qpos
        scratch.qpos[qpos_ids] = action
        mujoco.mj_kinematics(model, scratch)
        rotation = scratch.site_xmat[site].reshape(3, 3).copy()
        return scratch.site_xpos[site] + rotation @ GRIP_OFFSET, rotation

    return environment, probe


def rotation_degrees(first, second):
    cosine = (np.trace(first.T @ second) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def improved(now, before):
    """改善不足 STALL_TOLERANCE 视为本次未改善。"""
    return now < before * (1.0 - STALL_TOLERANCE)


def group_value(metrics, group, key):
    """采样组级均值：approach/grasp 单段，rest 由拉出、推回和保持四段合成。"""
    return float(np.mean([metrics[name][key] for name in GROUPS[group]]))


def judge(action, spatial, baseline, previous, stalls):
    """比较本次与上次检查，返回（本次是否停滞、连续停滞次数、判定记录）。

    action/spatial 按阶段名索引，baseline/previous 是带 action、spatial 的记录。
    任一关键指标恶化超过 REGRESSION_TOLERANCE 即记为停滞，避免一项严重退化
    被另一项的微小改善掩盖（本轮 3500→4000 就是这样漏判的）。
    """
    arm_now = action["approach"]["right_arm_mae_rad"]
    arm_before = previous["action"]["approach"]["right_arm_mae_rad"]
    position_now = spatial["approach"]["position_error_mm_first_action_mean"]
    position_before = previous["spatial"]["approach"]["position_error_mm_first_action_mean"]
    regressed = (arm_now > arm_before * (1.0 + REGRESSION_TOLERANCE)
                 or position_now > position_before * (1.0 + REGRESSION_TOLERANCE))
    stalled_now = regressed or not (improved(arm_now, arm_before)
                                    or improved(position_now, position_before))
    stalls = stalls + 1 if stalled_now else 0
    judgement = {
        "previous_check": previous["cumulative_step"],
        "stalled_this_check": stalled_now, "consecutive_stalls": stalls,
        "regressed_this_check": regressed,
        "recommend_stop": stalls >= 2,
        "approach_arm_mae_target_met": action["approach"]["right_arm_mae_rad"] < TARGET_APPROACH_ARM_MAE,
        "approach_position_target_met": (spatial["approach"]["position_error_mm_first_action_worst_frame"]
                                         < TARGET_POSITION_MM),
        "approach_attitude_target_met": (spatial["approach"]["attitude_error_deg_first_action_mean"]
                                         < TARGET_ATTITUDE_DEG),
        "all_noise_seeds_improved": all(
            improved(action["approach"]["by_seed"][seed]["right_arm_mae_rad"],
                     previous["action"]["approach"]["by_seed"][seed]["right_arm_mae_rad"])
            for seed in action["approach"]["by_seed"]),
        "rest_arm_mae_relative_to_baseline": (group_value(action, "rest", "right_arm_mae_rad")
                                              / (group_value(baseline["action"], "rest",
                                                             "right_arm_mae_rad") + 1e-12)),
    }
    return stalled_now, stalls, judgement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="续训起点，例如 lossless 实验的 2000 目录")
    parser.add_argument("--exp-name", default="stage_fit_lossless_002")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--check-interval", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=0,
                        help="按累计步数每隔多少步保存检查点；0 表示每次检查都保存")
    parser.add_argument("--order-seed", type=int, default=123)
    parser.add_argument("--loss-dims", choices=tuple(LOSS_DIMS), default="all32",
                        help="训练损失的动作维度；right8 只算右臂7维+右夹爪")
    parser.add_argument("--joint-weighting", choices=("uniform", "key147"), default="uniform",
                        help="key147：right8 内 joint1/4/7 相对权重1.5，其余含夹爪为1；均值归一化")
    parser.add_argument("--check-sampling", action="store_true",
                        help="只打印阶段划分与期望更新次数，不加载模型和数据")
    parser.add_argument("--dry-run", action="store_true",
                        help="构造数据与诊断帧后退出，不创建训练目录")
    args = parser.parse_args()
    weights = objective_weights(args.loss_dims, args.joint_weighting)
    if args.steps < 1 or args.check_interval < 1 or args.lr <= 0 or args.save_every < 0:
        raise ValueError("步数、检查间隔或学习率不合法")
    if any(x in args.exp_name for x in ("/", "\\")) or args.exp_name in (".", ".."):
        raise ValueError("实验名称不合法")
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(f"起点检查点缺少 params：{args.checkpoint}")

    anchors = anchor_frames()
    summary = sampling_summary(MIXTURE, args.steps)
    summary["anchor_frames"] = {name: value.tolist() for name, value in anchors.items()}
    summary["joint_weighting"] = args.joint_weighting
    summary["normalized_objective_weights"] = weights.tolist()
    if args.check_sampling:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    import flax.nnx as nnx
    import jax
    import optax
    from openpi.models import model as model_lib
    from drawer_lossless_dataset import transformed_dataset
    from local_data_loader import collate_numpy
    from drawer_training_objective import flow_squared_errors
    from train_smoke import save_checkpoint

    cfg = get_drawer_config(args.exp_name, args.steps, lossless=True)
    dataset, data_config = transformed_dataset(cfg)
    protocol = json.loads((args.checkpoint / "input_protocol.json").read_text())
    if protocol["image_source"] != "original_npz_uint8_rgb":
        raise ValueError("续训起点不是无损图像检查点")

    batches = []
    for index in range(len(dataset)):
        value = collate_numpy([dataset[index]])
        actions = value.pop("actions")
        batches.append((model_lib.Observation.from_dict(value), actions))
    print(f"缓存完成：{len(batches)}帧", flush=True)

    environment, probe = fk_probe()
    with np.load(environment.episode_root / "episode.npz", allow_pickle=False) as episode:
        teacher_qpos = episode["qpos"].copy()
        teacher_action = episode["action"].astype(np.float64).copy()
    if args.dry_run:
        # 用示教动作自身做一次FK自检：夹持段夹持点均值应与此前诊断的0.505087 m一致。
        try:
            tips = np.array([probe(teacher_qpos[frame], teacher_action[frame])[0]
                             for frame in range(150, 192)])
            grasp_z = float(tips[:, 2].mean())
            if abs(grasp_z - 0.505087) > 2e-3:
                raise RuntimeError(f"FK夹持点与既有诊断不一致：{grasp_z:.6f} m")
            detail = {name: len(value) for name, value in anchors.items()}
            print(f"PASS：数据与诊断帧构造通过；anchor={detail}；"
                  f"夹持段夹持点Z={grasp_z:.6f} m", flush=True)
        finally:
            environment.close()
        return
    if jax.default_backend() != "gpu":
        environment.close()
        raise RuntimeError("续训要求 JAX GPU")

    params = nnx.state(cfg.model.load(model_lib.restore_params(args.checkpoint / "params")))
    graphdef = nnx.graphdef(nnx.eval_shape(cfg.model.create, jax.random.key(cfg.seed)))
    tx = cfg.optimizer.create(args.lr)
    opt_state = tx.init(params.filter(cfg.trainable_filter))
    stats = data_config.norm_stats["actions"]
    mean = np.asarray(stats.mean, dtype=np.float32)
    scale = np.asarray(stats.std, dtype=np.float32) + 1e-6
    noises = {seed: np.random.default_rng(seed).standard_normal((1, 16, 32)).astype(np.float32)
              for seed in NOISE_SEEDS}
    pools = group_frames()
    dims = LOSS_DIMS[args.loss_dims]

    def update(current, optimizer, obs, actions, key):
        graph = nnx.merge(graphdef, current)
        graph.train()

        def loss_fn(module):
            # dims 由 --loss-dims 决定；图像增强保持原设置。
            errors = flow_squared_errors(module, key, obs, actions, augment=True, dims=dims)
            return reduce_objective(errors, weights)

        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, cfg.trainable_filter))(graph)
        trainable = current.filter(cfg.trainable_filter)
        updates, optimizer = tx.update(grads, optimizer, trainable)
        nnx.update(graph, optax.apply_updates(trainable, updates))
        return nnx.state(graph), optimizer, loss, optax.global_norm(grads)

    # jax.jit 的 fun 是必需位置参数，donate_argnums 只能用关键字后置包装。
    update = jax.jit(update, donate_argnums=(0, 1))

    def fit_loss(current, obs, actions, key):
        return flow_squared_errors(nnx.merge(graphdef, current), key, obs, actions).mean((0, 1))

    fit_loss = jax.jit(fit_loss)

    def sample(current, obs, noise):
        return nnx.merge(graphdef, current).sample_actions(jax.random.key(0), obs,
                                                           noise=noise, num_steps=10)

    sample = jax.jit(sample)

    anchor_batches = {name: [batches[int(frame)] for frame in frames]
                      for name, frames in anchors.items()}

    def evaluate(current):
        """分阶段flow损失、三噪声动作MAE与FK空间误差；全部为训练轨迹拟合。"""
        flow, action, spatial = {}, {}, {}
        for name, batch_list in anchor_batches.items():
            values = np.array([np.asarray(fit_loss(current, *batch, jax.random.key(12000 + int(frame))))
                               for frame, batch in zip(anchors[name], batch_list)])
            # objective 与训练使用同一维度和权重；其余分组指标保持未加权口径。
            selected = values if dims is None else values[..., dims]
            objective = float(reduce_objective(selected, weights))
            flow[name] = {"all32": float(values.mean()), "right_arm": float(values[:, :7].mean()),
                          "right_gripper": float(values[:, 7].mean()),
                          "padding16": float(values[:, 16:].mean()), "objective": objective}
            per_seed, reference = {}, None
            for seed, noise in noises.items():
                frame_errors, frame_predictions, frame_targets = [], [], []
                for frame, (obs, actions) in zip(anchors[name], batch_list):
                    prediction = np.asarray(sample(current, obs, noise))[0, :, :16]
                    target = np.asarray(actions)[0, :, :16]
                    valid = min(cfg.model.action_horizon, len(batches) - int(frame))
                    frame_errors.append(np.abs(prediction[:valid] - target[:valid]) * scale)
                    frame_predictions.append(prediction)
                    frame_targets.append(target)
                combined = np.concatenate(frame_errors)
                per_seed[str(seed)] = {"right_arm_mae_rad": float(combined[:, :7].mean()),
                                       "right_gripper_mae_rad": float(combined[:, 7].mean())}
                if seed == NOISE_SEEDS[0]:
                    reference = (frame_errors, frame_predictions, frame_targets)
            action[name] = {
                "right_arm_mae_rad": float(np.mean([v["right_arm_mae_rad"] for v in per_seed.values()])),
                "right_gripper_mae_rad": float(np.mean([v["right_gripper_mae_rad"] for v in per_seed.values()])),
                "right_arm_mae_worst_frame_rad": float(max(err[:, :7].mean() for err in reference[0])),
                "by_seed": per_seed}
            first_positions, first_attitudes = [], []
            worst_positions, worst_attitudes = [], []
            for frame, errors, prediction, target in zip(anchors[name], *reference):
                # 首动作对应闭环实际执行的第一步，与既有诊断同口径；最差步只用于暴露尾部偏差。
                worst = int(np.argmax(errors[:, :7].mean(1)))
                for step, positions, attitudes in ((0, first_positions, first_attitudes),
                                                   (worst, worst_positions, worst_attitudes)):
                    predicted_tip, predicted_rotation = probe(
                        teacher_qpos[int(frame)], prediction[step] * scale + mean)
                    target_tip, target_rotation = probe(
                        teacher_qpos[int(frame)], target[step] * scale + mean)
                    positions.append(float(np.linalg.norm(predicted_tip - target_tip) * 1000))
                    attitudes.append(rotation_degrees(predicted_rotation, target_rotation))
            spatial[name] = {
                "position_error_mm_first_action_mean": float(np.mean(first_positions)),
                "position_error_mm_first_action_worst_frame": float(np.max(first_positions)),
                "position_error_mm_worst_step_mean": float(np.mean(worst_positions)),
                "attitude_error_deg_first_action_mean": float(np.mean(first_attitudes)),
                "attitude_error_deg_worst_step_mean": float(np.mean(worst_attitudes)),
                "per_frame_position_mm_first_action": first_positions}
        return flow, action, spatial

    try:
        output = ROOT / "logs/a2_pi0" / f"drawer_stage_fit_{datetime.now():%Y%m%d-%H%M%S-%f}"
        output.mkdir(parents=True, exist_ok=True)
        base_step = int(args.checkpoint.name) if args.checkpoint.name.isdigit() else 0
        print(f"RUN_DIRECTORY={output}", flush=True)

        flow, action, spatial = evaluate(params)
        baseline = {"step": 0, "cumulative_step": base_step, "flow_loss_by_phase": flow,
                    "action": action, "spatial": spatial,
                    "scope": "same-training-trajectory fit; not validation, not task success"}
        records = [baseline]
        with (output / "stage_fit.jsonl").open("a") as stream:
            stream.write(json.dumps(baseline) + "\n")
        print(f"BASELINE approach_arm_mae={action['approach']['right_arm_mae_rad']:.6f} "
              f"approach_pos_mm={spatial['approach']['position_error_mm_first_action_mean']:.3f}", flush=True)

        stalls, began = 0, time.perf_counter()
        best = {"cumulative_step": base_step,
                "approach_arm_mae_rad": baseline["action"]["approach"]["right_arm_mae_rad"],
                "approach_position_mm": baseline["spatial"]["approach"]["position_error_mm_first_action_mean"],
                "approach_attitude_deg": baseline["spatial"]["approach"]["attitude_error_deg_first_action_mean"]}
        for step in range(1, args.steps + 1):
            index, group = make_sampler(args.order_seed + step, MIXTURE, pools)()
            obs, actions = jax.device_put(batches[index])
            params, opt_state, loss, grad = update(params, opt_state, obs, actions,
                                                  jax.random.key(args.order_seed * 1000 + step))
            loss, grad = float(loss), float(grad)
            if not np.isfinite([loss, grad]).all() or grad == 0:
                raise RuntimeError(f"第{step}步梯度或损失异常：loss={loss} grad={grad}")
            if step == 1 or step % 100 == 0:
                print(f"step={step} group={group} loss={loss:.6f} grad_norm={grad:.4f} "
                      f"elapsed={time.perf_counter()-began:.0f}s", flush=True)
            if step % args.check_interval != 0 and step != args.steps:
                continue
            cumulative = base_step + step
            flow, action, spatial = evaluate(params)
            previous = records[-1]
            _, stalls, judgement = judge(action, spatial, baseline, previous, stalls)
            record = {"step": step, "cumulative_step": cumulative, "flow_loss_by_phase": flow,
                      "action": action, "spatial": spatial, "judgement": judgement,
                      "scope": "same-training-trajectory fit; not validation, not task success"}
            records.append(record)
            if action["approach"]["right_arm_mae_rad"] < best["approach_arm_mae_rad"]:
                best = {"cumulative_step": cumulative,
                        "approach_arm_mae_rad": action["approach"]["right_arm_mae_rad"],
                        "approach_position_mm": spatial["approach"]["position_error_mm_first_action_mean"],
                        "approach_attitude_deg": spatial["approach"]["attitude_error_deg_first_action_mean"]}
            with (output / "stage_fit.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"CHECK step={cumulative} approach_arm_mae="
                  f"{action['approach']['right_arm_mae_rad']:.6f} approach_pos_mm="
                  f"{spatial['approach']['position_error_mm_first_action_mean']:.3f} stalls={stalls}", flush=True)
            # 诊断每 check-interval 步一次，保存按 save-every 稀疏执行，避免 8 个检查点撑爆磁盘。
            if args.save_every <= 0 or cumulative % args.save_every == 0 or step == args.steps:
                checkpoint = save_checkpoint(cfg, params, opt_state, data_config, step=cumulative)
                (checkpoint / "input_protocol.json").write_text(json.dumps(dict(
                    protocol, resumed_from=str(args.checkpoint.resolve()), stage_mixture=MIXTURE,
                    lr=args.lr, newly_trained_steps=step, loss_dims=args.loss_dims,
                    joint_weighting=args.joint_weighting,
                    normalized_objective_weights=weights.tolist(),
                    sampling_seed=args.order_seed), indent=2) + "\n")
            if judgement["recommend_stop"]:
                print(f"STOP：连续{stalls}次检查无改善，停止增加步数并转入对照", flush=True)
                break
    finally:
        environment.close()

    first, last = records[0], records[-1]
    result = {"experiment": args.exp_name, "resumed_from": str(args.checkpoint.resolve()),
              "newly_trained_steps": last["step"], "base_step": base_step,
              "lr": args.lr, "mixture": MIXTURE, "order_seed": args.order_seed,
              "loss_dims": args.loss_dims,
              "joint_weighting": args.joint_weighting,
              "normalized_objective_weights": weights.tolist(),
              "check_interval": args.check_interval, "save_every": args.save_every,
              "sampling": summary, "checks": len(records) - 1,
              "approach_arm_mae_rad": {"before": first["action"]["approach"]["right_arm_mae_rad"],
                                       "after": last["action"]["approach"]["right_arm_mae_rad"]},
              "approach_position_mm": {"before": first["spatial"]["approach"]["position_error_mm_first_action_mean"],
                                       "after": last["spatial"]["approach"]["position_error_mm_first_action_mean"]},
              "rest_arm_mae_rad": {"before": group_value(first["action"], "rest", "right_arm_mae_rad"),
                                   "after": group_value(last["action"], "rest", "right_arm_mae_rad")},
              "final_judgement": last["judgement"], "best_approach_check": best,
              "independent_validation": False,
              "closed_loop_tested": False, "overfit_success_verified": False,
              "acceptance_note": "动作与空间误差达标不等于抓握成功，最终以同条件闭环为准",
              "source_episode_sha256": hashlib.sha256(
                  (ROOT / "data/drawer_trial/contact_roundtrip_rgb_001/episode.npz").read_bytes()).hexdigest()}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS：阶段强化拟合完成，检查点={cfg.checkpoint_dir}；记录={output}", flush=True)


if __name__ == "__main__":
    main()
