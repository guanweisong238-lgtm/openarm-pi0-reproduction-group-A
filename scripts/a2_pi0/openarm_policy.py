import dataclasses

import numpy as np

from openpi import transforms


CAMERA_MAPPING = {
    "base_0_rgb": "observation.images.ceiling",
    "left_wrist_0_rgb": "observation.images.wrist_left",
    "right_wrist_0_rgb": "observation.images.wrist_right",
}


def parse_image(value):
    image = np.asarray(value)

    if image.ndim != 3:
        raise ValueError(f"图像应为三维数组，实际为 {image.shape}")

    # LeRobot 返回 CHW；推理接口也允许输入 HWC。
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"图像应包含三个通道：{image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        if (
            not np.isfinite(image).all()
            or image.min() < 0
            or image.max() > 1
        ):
            raise ValueError("浮点图像应位于 [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"不支持的图像类型：{image.dtype}")

    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    def __call__(self, data):
        state = np.asarray(data["observation.state"], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("状态必须是数值有限的 16 维向量")

        prompt = data.get("prompt", data.get("task"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("需要通过 prompt 或 task 提供任务指令")

        result = {
            "state": state,
            "image": {
                target: parse_image(data[source])
                for target, source in CAMERA_MAPPING.items()
            },
            "image_mask": {
                target: np.True_ for target in CAMERA_MAPPING
            },
            "prompt": prompt,
        }

        # 训练时提供动作，推理时不需要。
        if "action" in data:
            actions = np.asarray(data["action"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[-1] != 16
                or not np.isfinite(actions).all()
            ):
                raise ValueError("动作必须为数值有限的 (预测长度, 16) 数组")
            result["actions"] = actions

        return result


@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 16:
            raise ValueError(f"模型动作形状异常：{actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("模型动作包含非有限值")
        return {"actions": actions[:, :16]}