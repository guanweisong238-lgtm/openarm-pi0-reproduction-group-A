# 本文件负责：重新加载 LoRA 试训练产物，验证保存后的模型可以独立推理。
# 使用第24条验证轨迹的首帧作为输入，不参与训练或归一化统计计算。
# 通过官方策略加载接口读取模型及 checkpoint 内的归一化统计。
# 检查动作形状、有限值及固定噪声下的重复性；不评估抓取成功率。

# 一、先设置项目环境，再导入模型相关模块
from training_config import DATA_ROOT, ROOT, get_train_config

from datetime import datetime
import json
import logging
import time

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.policies import policy_config


# 二、本次验收对象：已完成的10步试训练 checkpoint
CHECKPOINT = (
    ROOT / "checkpoints/pi0_openarm_lora/smoke_001/10"
)
CAMERAS = ("ceiling", "wrist_left", "wrist_right")


def load_validation_observation():
    """读取验证轨迹24的首帧，仅返回推理所需字段。

    不把 action 标签传给策略，避免将单步标签误当成训练动作序列。
    图像保留 LeRobot 输出表示，由 OpenArmInputs 统一转换。
    """
    dataset = LeRobotDataset(
        repo_id="local/openarm_pickup",
        root=DATA_ROOT,
        episodes=[24],
        video_backend="pyav",
    )
    sample = dataset[0]
    assert int(sample["episode_index"]) == 24

    observation = {
        "observation.state": np.asarray(sample["observation.state"]),
        "prompt": sample["task"],
    }
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        observation[key] = np.asarray(sample[key])
    return observation


# 三、重新加载与推理验收
def main():
    """在新进程加载模型，两次使用相同观测和噪声进行推理。"""
    cfg = get_train_config()
    if not (CHECKPOINT / "params").is_dir():
        raise FileNotFoundError(f"checkpoint不存在：{CHECKPOINT}")

    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    output = ROOT / "logs/a2_pi0" / f"reload_{run_id}"
    output.mkdir(parents=True)

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "run.log"),
        ],
    )

    observation = load_validation_observation()
    logging.info("验证输入：episode=24，frame=0")
    logging.info("加载训练后模型：%s", CHECKPOINT)

    # 官方接口从 checkpoint/assets 中读取实际保存的归一化统计。
    policy = policy_config.create_trained_policy(
        cfg,
        CHECKPOINT,
        sample_kwargs={"num_steps": 10},
    )

    # 固定噪声，以便重复检查；这不等于正式多样本验证。
    noise = np.random.default_rng(42).standard_normal(
        (cfg.model.action_horizon, cfg.model.action_dim)
    ).astype(np.float32)

    outputs = []
    timings = []
    for index in range(2):
        start = time.perf_counter()
        result = policy.infer(observation, noise=noise)
        actions = np.asarray(result["actions"])
        elapsed = time.perf_counter() - start

        assert actions.shape == (16, 16), actions.shape
        assert np.isfinite(actions).all(), "动作存在NaN或无穷值"

        outputs.append(actions)
        timings.append(elapsed)
        logging.info(
            "推理%d：shape=%s，耗时=%.3fs",
            index + 1, actions.shape, elapsed,
        )

    np.testing.assert_allclose(
        outputs[0], outputs[1], rtol=1e-5, atol=1e-6
    )

    # 四、保存验收证据，不覆盖 checkpoint
    np.save(output / "actions.npy", outputs[-1])
    summary = {
        "checkpoint": str(CHECKPOINT),
        "source_episode": 24,
        "source_frame": 0,
        "action_shape": list(outputs[-1].shape),
        "finite": True,
        "repeatability_passed": True,
        "first_call_seconds_including_compilation": timings[0],
        "second_call_seconds": timings[1],
        "limitation": "仅验证重载推理，不代表抓取成功或仿真兼容",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    logging.info("验收记录：%s", output)
    logging.info("PASS：训练后checkpoint重载及推理检查通过")


# 五、命令行入口
if __name__ == "__main__":
    main()