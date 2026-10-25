# 本文件负责：验收 A1 新增的 A2 模型接口，不加载或训练策略模型。
# 检查左右臂映射、非法动作拒绝、限幅反馈、300个30Hz周期的时间误差及复位。
# 默认采集同一时刻的三路RGB样张，结果写入 outputs/a2_interface/。
# 验收成功仅证明接口可运行，不代表相机已标定、公开示教可复现或抓取成功。

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

# 一、无窗口验收优先使用 EGL；调用方可通过环境变量显式选择其他后端。
# 必须在导入 mujoco 前设置；--no-render 模式不创建任何图形上下文。
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

from simulation.env import ROOT
from simulation.policy_env import CAMERAS, POLICY_ACTUATORS, OpenArmPolicyEnv, PolicyInterfaceConfig


def verify_numeric_interface(env):
    """验证真实模型上的顺序映射、输入拒绝、限幅、时间调度和复位。

    使用保持 home 的位置目标推进300周期，避免把未经标定的示教当成可执行轨迹。
    返回可序列化结果；任何检查失败立即抛出异常，不打印成功结论。
    """
    # 二、唯一编号探针只用于重排检查，不下发给机器人。
    probe = np.arange(16, dtype=np.float64)
    np.testing.assert_array_equal(env.to_policy(env.to_native(probe)), probe)
    for index, name in enumerate(POLICY_ACTUATORS):
        native_index = env.env.actuator_names.index(name)
        assert env.to_native(probe)[native_index] == index

    home = env.reset(render=False)
    native_home = env.env.get_observation()
    np.testing.assert_allclose(
        home["observation.state"], env.to_policy(native_home["joint_pos"]), atol=1e-6
    )
    target = env.to_policy(native_home["ctrl"])
    for invalid in (np.zeros(32), np.full(16, np.nan), np.full(16, np.inf)):
        before = env.env.data.time
        before_ctrl = env.env.data.ctrl.copy()
        try:
            env.step(invalid, render=False)
        except ValueError:
            pass
        else:
            raise AssertionError("非法动作未被拒绝")
        assert env.env.data.time == before
        np.testing.assert_array_equal(env.env.data.ctrl, before_ctrl)

    # 三、右夹爪设为明显越界目标，确认已裁剪且诊断字段与真实控制量一致。
    clipped_target = target.copy()
    clipped_target[7] = -2.0
    clipped = env.step(clipped_target, render=False)
    assert clipped["control.clipped_mask"][7]
    np.testing.assert_allclose(
        clipped["control.applied"], env.to_policy(env.env.data.ctrl)
    )
    assert np.all(env.env.data.ctrl >= env.lower)
    assert np.all(env.env.data.ctrl <= env.upper)

    # 四、重置后逐周期验证时间误差上界，而不仅看最终累计时间。
    env.reset(render=False)
    maximum_error = 0.0
    substeps = set()
    for index in range(300):
        observation = env.step(target, render=False)
        assert np.isfinite(observation["observation.state"]).all()
        expected = (index + 1) / env.config.control_hz
        error = abs(observation["timestamp"] - expected)
        assert error <= env.physics_dt / 2 + 1e-9
        maximum_error = max(maximum_error, error)
        substeps.add(observation["timing.substeps"])
    final_time = observation["timestamp"]
    assert np.isclose(final_time, 300 / env.config.control_hz, atol=env.physics_dt / 2 + 1e-9)
    reset = env.reset(render=False)
    assert reset["timestamp"] == reset["nominal_timestamp"] == 0
    np.testing.assert_array_equal(reset["observation.state"], home["observation.state"])
    return {
        "cycles": 300,
        "nominal_control_hz": env.config.control_hz,
        "physics_dt": env.physics_dt,
        "substep_counts": sorted(substeps),
        "final_time_seconds": final_time,
        "max_time_error_seconds": maximum_error,
        "native_indices_in_policy_order": env.policy_indices.tolist(),
        "mapping_invalid_action_clipping_reset": "passed",
    }


def verify_rendering(env, output):
    """获取同一时刻三路RGB，保存样张及跨进程观测快照，验证不推进物理。

    NPZ 只保存数值数组和字符串，可在A2独立环境中用 allow_pickle=False 读取。
    """
    before = env.env.data.time
    observation = env.get_observation()
    assert env.env.data.time == before == observation["timestamp"]
    for key in CAMERAS:
        image = observation[key]
        assert image.shape == (env.config.height, env.config.width, 3)
        assert image.dtype == np.uint8
        assert image.max() > image.min(), f"相机可能输出常量图像：{key}"
        Image.fromarray(image).save(output / f"{key.rsplit('.', 1)[-1]}.png")
    np.savez_compressed(output / "policy_observation.npz", **observation)
    return {key: list(observation[key].shape) for key in CAMERAS}


def main():
    """读取配置并执行验收，支持仅检查数值接口；产物不纳入Git。"""
    parser = argparse.ArgumentParser(description="验收OpenArm的A2三相机/30Hz接口")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/a2_interface.json")
    parser.add_argument("--no-render", action="store_true", help="跳过图像渲染，仅检查数值接口")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    config = PolicyInterfaceConfig.from_file(args.config)
    if args.width is not None:
        config = replace(config, width=args.width)
    if args.height is not None:
        config = replace(config, height=args.height)
    output = ROOT / "outputs/a2_interface"
    output.mkdir(parents=True, exist_ok=True)
    # 不沿用历史结果：每次报告明确标明本次是否真正执行了渲染检查。
    with OpenArmPolicyEnv(config) as env:
        result = verify_numeric_interface(env)
        result["images"] = None if args.no_render else verify_rendering(env, output)
        result["camera_and_dataset_calibration_verified"] = False
        result["task_success_verified"] = False
    name = "numeric_validation.json" if args.no_render else "validation.json"
    (output / name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("PASS：A2模型接口验收通过" + ("（本次未检查渲染）" if args.no_render else ""))


# 五、脚本入口；异常不吞掉，避免将失败验收报告为成功。
if __name__ == "__main__":
    main()
