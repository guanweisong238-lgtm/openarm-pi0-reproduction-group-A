"""阶段续训的判定逻辑回归测试：键名、停滞累计与验收门槛。

只验证纯函数，不加载模型、不训练、不需要GPU。
运行：.venv/bin/python tests/test_drawer_stage_judge.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/a2_pi0"))
import train_drawer_stage_fit as stage

NAMES = [name for name, _, _ in stage.PHASES]


def action_block(arm, gripper=0.01):
    """按阶段名构造 action 指标；键必须与 evaluate 产出一致。"""
    return {name: {"right_arm_mae_rad": arm, "right_gripper_mae_rad": gripper,
                   "right_arm_mae_worst_frame_rad": arm * 2,
                   "by_seed": {str(seed): {"right_arm_mae_rad": arm, "right_gripper_mae_rad": gripper}
                               for seed in stage.NOISE_SEEDS}}
            for name in NAMES}


def spatial_block(position, attitude=1.0):
    """按阶段名构造空间指标；键必须与 evaluate 产出一致。"""
    return {name: {"position_error_mm_first_action_mean": position,
                   "position_error_mm_first_action_worst_frame": position * 1.5,
                   "position_error_mm_worst_step_mean": position * 10,
                   "attitude_error_deg_first_action_mean": attitude,
                   "attitude_error_deg_worst_step_mean": attitude * 5,
                   "per_frame_position_mm_first_action": [position]}
            for name in NAMES}


def record(step, arm, position):
    return {"cumulative_step": step, "action": action_block(arm), "spatial": spatial_block(position)}


def main():
    # 采样组均值：rest 由拉出、推回和保持四段合成。
    assert abs(stage.group_value(action_block(0.04), "rest", "right_arm_mae_rad") - 0.04) < 1e-12
    assert abs(stage.group_value(action_block(0.0), "grasp", "right_gripper_mae_rad") - 0.01) < 1e-12
    print("PASS group_value")

    baseline = record(2000, 0.06, 6.0)

    # 明显改善：不记停滞，且记录起点与组级相对值。
    stalled, stalls, verdict = stage.judge(action_block(0.05), spatial_block(5.0), baseline, baseline, 1)
    assert not stalled and stalls == 0, (stalled, stalls)
    assert verdict["previous_check"] == 2000
    assert abs(verdict["rest_arm_mae_relative_to_baseline"] - 0.05 / 0.06) < 1e-9
    assert verdict["all_noise_seeds_improved"] is True
    assert verdict["recommend_stop"] is False
    print("PASS improved")

    # 改善不足 0.5%：记一次停滞。
    previous = record(2500, 0.0500, 5.0)
    stalled, stalls, verdict = stage.judge(action_block(0.0499), spatial_block(4.999),
                                           baseline, previous, 0)
    assert stalled and stalls == 1, (stalled, stalls)
    print("PASS stall detected")

    # 连续两次停滞：建议停止增加步数。
    stalled, stalls, verdict = stage.judge(action_block(0.0498), spatial_block(4.998),
                                           baseline, previous, 1)
    assert stalled and stalls == 2 and verdict["recommend_stop"] is True, (stalls, verdict)
    print("PASS consecutive stall stops")

    # 动作未改善但位置改善：只要一项改善就不算停滞。
    stalled, stalls, verdict = stage.judge(action_block(0.0499), spatial_block(4.0),
                                           baseline, previous, 1)
    assert not stalled and stalls == 0, (stalled, stalls)
    print("PASS either-metric improvement")

    # 本轮实测复现：3500→4000 右臂恶化53%、位置仅改善1.9%，必须判为停滞，
    # 不能被"位置略好"掩盖（这正是第一次跑漏判的情形）。
    stalled, stalls, verdict = stage.judge(action_block(0.0608), spatial_block(9.075),
                                          baseline, record(3500, 0.0397, 9.255), 0)
    assert stalled and stalls == 1, (stalled, stalls)
    assert verdict["regressed_this_check"] is True
    print("PASS regression not masked by small gain")

    # 位置在 20% 以内变差、右臂明显改善 -> 不算回退。
    stalled, stalls, verdict = stage.judge(action_block(0.02), spatial_block(10.0),
                                          baseline, record(3500, 0.0397, 9.0), 0)
    assert not stalled and stalls == 0, (stalled, stalls)
    assert verdict["regressed_this_check"] is False
    print("PASS small position loss tolerated")

    # 门槛：右臂 0.009 rad 达标；最差帧位置 2.4 mm 未达标；姿态 1.0° 达标。
    _, _, verdict = stage.judge(action_block(0.009), spatial_block(1.6),
                                baseline, record(2500, 0.05, 5.0), 0)
    assert verdict["approach_arm_mae_target_met"] is True
    assert verdict["approach_position_target_met"] is False, verdict
    assert verdict["approach_attitude_target_met"] is True
    # 姿态 8°（当前接近段实际水平）不达标，且不影响 recommend_stop。
    _, _, verdict = stage.judge(action_block(0.05), spatial_block(5.0, attitude=8.0),
                                baseline, record(2500, 0.05, 5.0), 0)
    assert verdict["approach_attitude_target_met"] is False
    assert verdict["recommend_stop"] is False
    print("PASS thresholds")
    print("PASS：阶段续训判定逻辑全部通过")


if __name__ == "__main__":
    main()
