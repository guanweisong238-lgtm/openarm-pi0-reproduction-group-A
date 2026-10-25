"""损失维度掩码的归属测试：确认 right8 取的是右臂7维+右夹爪，不含左臂与补齐维。

只验证切片语义与 objective 计算，不加载模型、不需要 GPU。
运行：.venv/bin/python tests/test_drawer_loss_dims.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/a2_pi0"))
import train_drawer_stage_fit as stage


def main():
    # 三个区间给不同量级，任何串位都会让断言失败。
    error = np.zeros((1, 16, 32), dtype=np.float64)
    error[..., :8] = 1.0        # 右臂7维 + 右夹爪
    error[..., 8:16] = 100.0    # 左臂7维 + 左夹爪
    error[..., 16:] = 50.0      # 补齐16维

    right8 = stage.LOSS_DIMS["right8"]
    assert right8 == slice(0, 8), right8
    selected = error[..., right8]
    assert selected.shape == (1, 16, 8), selected.shape
    assert np.allclose(selected.mean(), 1.0), "right8 混入了左臂或补齐维"
    print("PASS right8 excludes left arm and padding")

    assert stage.LOSS_DIMS["all16"] == slice(0, 16)
    assert np.isclose(error[..., stage.LOSS_DIMS["all16"]].mean(), (8 * 1.0 + 8 * 100.0) / 16)
    assert stage.LOSS_DIMS["all32"] is None
    print("PASS all16 and all32")

    # 诊断侧 objective：与 update 里 .mean() 的口径一致。
    values = error.mean(0)  # 每帧每维的平均，(16, 32)
    assert np.isclose(values.mean(), (8 * 1.0 + 8 * 100.0 + 16 * 50.0) / 32)
    assert np.isclose(values[..., right8].mean(), 1.0)
    assert np.isclose(values[..., stage.LOSS_DIMS["all16"]].mean(), (8 * 1.0 + 8 * 100.0) / 16)
    print("PASS objective matches training mean")
    # 默认配方保持原均值；加权只强调 joint1/4/7，夹爪不串位、不剔除。
    for name, count in (("all32", 32), ("all16", 16), ("right8", 8)):
        weights = stage.objective_weights(name)
        sample = np.arange(2 * count, dtype=np.float32).reshape(2, count)
        np.testing.assert_allclose(stage.reduce_objective(sample, weights), sample.mean())
    weights = stage.objective_weights("right8", "key147")
    np.testing.assert_allclose(weights.sum(), 8)
    np.testing.assert_allclose(weights[[0, 3, 6]] / weights[7], 1.5)
    np.testing.assert_allclose(weights[[1, 2, 4, 5, 7]], weights[7])
    try:
        stage.objective_weights("all32", "key147")
    except ValueError:
        pass
    else:
        raise AssertionError("key147 应拒绝 all32")
    print("PASS normalized joint weights and default compatibility")

    # 真正经过 JAX jit/autodiff，检查被强调维度的梯度相对比例。
    import jax
    import jax.numpy as jnp
    def objective(prediction):
        return stage.reduce_objective(prediction ** 2, weights)
    gradient = np.asarray(jax.jit(jax.grad(objective))(jnp.ones((1, 16, 8))))
    np.testing.assert_allclose(gradient[0, 0, [0, 3, 6]] / gradient[0, 0, 7], 1.5)
    # 诊断先平均帧/时间，与训练直接归约的加权目标一致。
    full_error = np.arange(3 * 16 * 32, dtype=np.float32).reshape(3, 16, 32)
    np.testing.assert_allclose(stage.reduce_objective(full_error[..., :8], weights),
                               stage.reduce_objective(full_error.mean((0, 1))[:8], weights), rtol=1e-6)
    print("PASS JIT gradient weighting and diagnostic objective agreement")
    print("PASS：损失维度掩码归属正确")


if __name__ == "__main__":
    main()
