"""重载抽屉检查点，评估单轨迹动作重建，再执行纯模型闭环。

离线指标来自训练轨迹；闭环每4步重观测，策略只接收实际状态、图像和文本。
不执行专家、不使用示教动作或中间状态纠正轨迹。仿真在推理期间暂停。
"""
from drawer_training_config import ROOT, DATA_ROOT, get_drawer_config
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def offline(policy, output, noise, lossless=False):
    if lossless:
        from drawer_lossless_dataset import LosslessDrawerDataset
        dataset = LosslessDrawerDataset()
    else:
        from local_data_loader import EpisodeSubsetDataset
        dataset = EpisodeSubsetDataset(repo_id="local/openarm_drawer_trial", root=DATA_ROOT,
            episodes=[0], video_backend="pyav", delta_timestamps={"action": [i / 30 for i in range(16)]})
    total_abs, total_sq, count = np.zeros(16), np.zeros(16), 0
    predictions, targets, masks, records = [], [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        observation = {key: np.asarray(sample[key]) for key in ("observation.state",
            "observation.images.ceiling", "observation.images.wrist_left", "observation.images.wrist_right")}
        observation["prompt"] = sample["task"]
        began = time.perf_counter()
        predicted = np.asarray(policy.infer(observation, noise=noise)["actions"])
        seconds = time.perf_counter()-began
        if predicted.shape != (16, 16) or not np.isfinite(predicted).all():
            raise ValueError("离线预测形状或数值异常")
        target = np.asarray(sample["action"])
        valid = ~np.asarray(sample["action_is_pad"], dtype=bool)
        error = (predicted-target)[valid].astype(np.float64)
        total_abs += np.abs(error).sum(0)
        total_sq += (error**2).sum(0)
        count += len(error)
        predictions.append(predicted); targets.append(target); masks.append(valid)
        record = {"frame": index, "mae": float(np.abs(error).mean()),
                  "right_arm_mae": float(np.abs(error[:, :7]).mean()),
                  "right_gripper_mae": float(np.abs(error[:, 7]).mean()), "inference_seconds": seconds}
        records.append(record)
        if index % 30 == 0:
            print(f"offline frame={index}/{len(dataset)} MAE={record['mae']:.6f}", flush=True)
    per_joint = total_abs/count
    phases = {}
    for name, start, stop in (("approach",0,150),("grasp",150,192),("pull",192,282),
                              ("open_hold",282,312),("push",312,402),("closed_hold",402,462)):
        error = (np.asarray(predictions[start:stop])-np.asarray(targets[start:stop]))[np.asarray(masks[start:stop])]
        phases[name] = {"mae": float(np.abs(error).mean()), "right_arm_mae": float(np.abs(error[:,:7]).mean()),
                        "right_gripper_mae": float(np.abs(error[:,7]).mean())}
    result = {"frames": len(dataset), "valid_action_steps": count, "mae_all": float(per_joint.mean()),
        "mae_right_arm": float(per_joint[:7].mean()), "mae_right_gripper": float(per_joint[7]),
        "mae_per_joint": per_joint.tolist(), "rmse_per_joint": np.sqrt(total_sq/count).tolist(),
        "phase_metrics": phases, "independent_validation": False, "samples": records,
        "image_source": "original_npz_uint8_rgb" if lossless else "lerobot_lossy_video"}
    np.savez_compressed(output / "offline_predictions.npz", prediction=np.asarray(predictions),
                        target=np.asarray(targets), valid=np.asarray(masks))
    write_json(output / "offline.json", result)
    print(f"OFFLINE_COMPLETE MAE={result['mae_all']:.6f} right_arm={result['mae_right_arm']:.6f}", flush=True)
    return result


def closed_loop(policy, output, noise, seconds, chunk, reference_initial=None):
    import cv2
    import mujoco
    from drawer_simulation import DrawerSimulation
    from contact_drawer_controller import finger_handle_contacts
    # 该导入只用于接触计数，不实例化或执行专家。
    environment = DrawerSimulation()
    writer = cv2.VideoWriter(str(output / "closed_loop.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            30/chunk, (960, 240))
    if not writer.isOpened():
        environment.close()
        raise RuntimeError("闭环视频无法创建")
    rows, inference_times = [], []
    opened = False
    open_streak = closed_streak = bilateral_streak = 0
    grasped = success = False
    predictions = []
    initial_reproduction_error = None
    try:
        maximum_steps = round(seconds*30)
        while environment.steps < maximum_steps:
            observation = environment.observation()
            began = time.perf_counter()
            block = np.asarray(policy.infer(observation, noise=noise)["actions"])
            if environment.steps == 0 and reference_initial is not None:
                initial_reproduction_error = float(np.max(np.abs(block-reference_initial)))
                if initial_reproduction_error > 1e-6:
                    raise RuntimeError(f"Initial lossless offline/live action mismatch: {initial_reproduction_error}")
            elapsed = time.perf_counter()-began
            inference_times.append(elapsed)
            if block.shape != (16,16) or not np.isfinite(block).all():
                raise ValueError("模型闭环输出异常")
            predictions.append(block)
            frame = cv2.cvtColor(np.concatenate([observation[f"observation.images.{name}"]
                for name in ("ceiling","wrist_left","wrist_right")], axis=1), cv2.COLOR_RGB2BGR)
            opening = -float(environment.data.qpos[environment.drawer_qpos])
            cv2.putText(frame, f"MODEL | t={environment.data.time:.2f}s | drawer={opening*1000:.1f}mm",
                        (8, 230), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 1)
            writer.write(frame)
            for action in block[:min(chunk,maximum_steps-environment.steps)]:
                before = environment.data.qpos.copy()
                applied = environment.step(action)
                opening = -float(environment.data.qpos[environment.drawer_qpos])
                contacts, forces = finger_handle_contacts(environment.model, environment.data)
                bilateral_streak = bilateral_streak+1 if np.all(contacts>0) and np.all(forces>.01) else 0
                grasped = grasped or bilateral_streak>=5
                open_streak = open_streak+1 if opening>=.08 else 0
                opened = opened or open_streak>=30
                closed_streak = closed_streak+1 if opened and abs(opening)<=.005 else 0
                success = bool(opened and closed_streak>=30 and grasped)
                rows.append({"time": float(environment.data.time), "opening": opening,
                    "qpos_before": before, "qpos": environment.data.qpos.copy(),
                    "state": environment.data.qpos[environment.qpos_ids].copy(), "requested": action.copy(),
                    "applied": applied, "contacts": contacts, "forces": forces,
                    "weld_active": environment.data.eq_active.copy()})
                if success:
                    break
            if len(inference_times)%10==0 or success:
                print(f"closed_loop steps={environment.steps} open_mm={opening*1000:.1f} opened={opened} grasped={grasped} success={success}",flush=True)
                write_json(output / "closed_loop_progress.json", {"steps": environment.steps,
                    "opening_m": opening, "opened": opened, "grasped": grasped, "success": success})
            if success:
                break
    finally:
        writer.release()
        environment.close()
    result = {"steps": len(rows), "simulation_seconds": float(environment.data.time),
        "peak_open_m": float(max(row["opening"] for row in rows)), "final_open_m": float(rows[-1]["opening"]),
        "opened_8cm_held_1s": bool(opened), "final_closed_5mm_held_1s_after_opening": closed_streak>=30,
        "bilateral_grasp_observed": bool(grasped), "success_open_then_close": success,
        "active_grasp_weld_frames": int(sum(bool(np.any(row["weld_active"][environment.weld_mask])) for row in rows)),
        "clipped_control_steps": int(sum(bool(np.any(row["requested"]!=row["applied"])) for row in rows)),
        "inference_calls": len(inference_times), "mean_inference_seconds": float(np.mean(inference_times)),
        "mean_warm_inference_seconds": float(np.mean(inference_times[1:])) if len(inference_times)>1 else None,
        "simulation_paused_during_inference": True, "action_chunk_executed": chunk,
        "teacher_actions_executed": 0, "intermediate_states_injected": 0,
        "policy_observation_keys": ["observation.state","observation.images.ceiling","observation.images.wrist_left",
                                    "observation.images.wrist_right","prompt"],
        "initialization": "original episode initial integration state only", "independent_validation": False,
        "lossless_initial_offline_live_action_max_error": initial_reproduction_error}
    np.savez_compressed(output / "closed_loop.npz", **{key:np.asarray([row[key] for row in rows]) for key in rows[0]},
                        action_blocks=np.asarray(predictions), inference_seconds=np.asarray(inference_times))
    write_json(output / "closed_loop.json",result)
    print(f"CLOSED_LOOP_COMPLETE {json.dumps(result)}",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=24)
    parser.add_argument("--chunk",type=int,default=4)
    parser.add_argument("--noise-seeds",default="42",
                        help="逗号分隔的固定采样噪声种子；多于一个时每个种子单独成子目录")
    parser.add_argument("--skip-offline",action="store_true")
    args=parser.parse_args()
    seeds=[int(value) for value in args.noise_seeds.split(",") if value.strip()]
    if args.seconds<=0 or args.chunk<1 or args.chunk>16:
        raise ValueError("闭环时长或动作块长度不合法")
    if not seeds or any(seed<0 for seed in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError("噪声种子必须是不重复的非负整数")
    output=ROOT/"logs/a2_pi0"/f"drawer_eval_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True,exist_ok=False)
    print(f"EVALUATION_DIRECTORY={output}",flush=True)
    import sys
    import jax
    if jax.default_backend() != "gpu":
        raise RuntimeError("本次模型评估要求JAX GPU")
    sys.path[:0]=[str(ROOT/".cache/drawer-deps"),str(ROOT/".cache/openarm-control")]
    from openpi.policies import policy_config
    protocol_path=args.checkpoint/"input_protocol.json"
    protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    lossless=protocol.get("image_source")=="original_npz_uint8_rgb"
    if lossless:
        import hashlib
        from drawer_training_config import EPISODE_ROOT
        if hashlib.sha256((EPISODE_ROOT/"episode.npz").read_bytes()).hexdigest()!=protocol["source_episode_sha256"]:
            raise ValueError("Lossless source episode differs from training")
    policy=policy_config.create_trained_policy(get_drawer_config(lossless=lossless), args.checkpoint,
                                               sample_kwargs={"num_steps":10})
    result={"checkpoint":str(args.checkpoint.resolve()),"independent_validation":False,
            "jax_devices":[str(device) for device in jax.devices()],"seeds":seeds,
            "image_source":"original_npz_uint8_rgb" if lossless else "lerobot_lossy_video"}
    runs={}
    for seed in seeds:
        # 同一检查点、同一初始状态，只换采样噪声，用于区分拟合改善与噪声波动。
        noise=np.random.default_rng(seed).standard_normal((16,32)).astype(np.float32)
        target=output if len(seeds)==1 else output/f"seed_{seed}"
        target.mkdir(parents=True,exist_ok=True)
        run={"seed":seed}
        print(f"SEED_START {seed} dir={target}",flush=True)
        if not args.skip_offline:
            run["offline"]=offline(policy,target,noise,lossless=lossless)
        reference_initial=None
        if lossless and not args.skip_offline:
            with np.load(target/"offline_predictions.npz",allow_pickle=False) as predictions:
                reference_initial=predictions["prediction"][0].copy()
        run["closed_loop"]=closed_loop(policy,target,noise,args.seconds,args.chunk,
                                      reference_initial=reference_initial)
        runs[str(seed)]=run
        write_json(output/"summary.json",dict(result,runs=runs))
    result["runs"]=runs
    if len(seeds)==1:
        # 保留单种子的顶层字段，既有收尾脚本仍可直接读取。
        result["seed"]=seeds[0]
        result["offline"]=runs[str(seeds[0])].get("offline")
        result["closed_loop"]=runs[str(seeds[0])]["closed_loop"]
    if len(seeds)>1:
        keys=("peak_open_m","final_open_m","opened_8cm_held_1s","bilateral_grasp_observed",
              "success_open_then_close","clipped_control_steps")
        result["across_seeds"]={key:{seed:runs[seed]["closed_loop"][key] for seed in runs} for key in keys}
    write_json(output/"summary.json",result)
    print(f"PASS：抽屉模型评估完成，记录={output}",flush=True)


if __name__=="__main__":
    main()
