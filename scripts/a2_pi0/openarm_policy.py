# 本文件负责：定义 OpenArm 数据与 OpenPI 模型之间的输入、输出格式转换。
# 输入端处理 16 维状态、任务指令和三路 RGB 图像，训练时同时传递动作序列。
# 保留右臂在前、左臂在后的原始顺序；输出端将模型动作裁剪回 16 维。
# 归一化、32 维补齐、仿真左右臂重排及控制限幅由其他模块负责。

import dataclasses

import numpy as np

from openpi import transforms


# 一、相机语义映射
# 键是模型的标准图像槽位，值是 LeRobot 数据字段；槽位的左右含义不随关节顺序变化。
CAMERA_MAPPING = {
    "base_0_rgb": "observation.images.ceiling",
    "left_wrist_0_rgb": "observation.images.wrist_left",
    "right_wrist_0_rgb": "observation.images.wrist_right",
}


# 二、图像表示统一
def parse_image(value):
    """将单路 RGB 图像转换为连续存储的 HWC uint8 数组。

    参数：value 为可转换成 NumPy 数组的图像，支持 CHW 或 HWC 排列。
    浮点输入必须已缩放到 [0, 1]；uint8 输入按 [0, 255] 保留。
    返回：HWC、三通道、uint8 图像；尺寸缩放由后续模型变换完成。
    异常：维度、通道、数值范围或数据类型不符合约定时抛出 ValueError。
    """
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
        # 先舍入再转整数，避免直接截断引入系统性偏小误差。
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"不支持的图像类型：{image.dtype}")

    return np.ascontiguousarray(image)


# 三、训练和推理共用的输入适配
@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    """将数据集或机器人调用方的字段映射到 OpenPI 预处理协议。

    状态、动作的前八维属于右臂，后八维属于左臂；每臂均为七关节加夹爪。
    本类不判断位置单位、关节零点或物理限位，不将原始位置改为增量。
    """

    def __call__(self, data):
        """返回未归一化的模型输入字典，训练输入另包含 actions。

        必需字段为 observation.state、CAMERA_MAPPING 中的三路图像，以及
        prompt 或 task 字符串；可选 action 必须为 (预测长度, 16)。
        缺少必需键时抛出 KeyError，形状或数值不合法时抛出 ValueError。
        """
        state = np.asarray(data["observation.state"], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("状态必须是数值有限的 16 维向量")

        # 推理方通常传 prompt；LeRobot 读取器通常提供 task。
        prompt = data.get("prompt", data.get("task"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("需要通过 prompt 或 task 提供任务指令")

        result = {
            "state": state,
            "image": {
                target: parse_image(data[source])
                for target, source in CAMERA_MAPPING.items()
            },
            # 当前协议要求三路图像全部存在，因此都标为有效，不将缺图静默补零。
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


# 四、模型输出恢复为机器人动作维度
@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    """移除模型补齐的动作维度，保持数据集原有的右臂优先顺序。

    在完整策略流水线中应放在反归一化之后；本类自身不负责反归一化。
    输出不是可直接下发给 A1 仿真的控制量，桥接层仍需核对顺序、标定和限幅。
    """

    def __call__(self, data):
        """接收含二维 actions 的字典，返回 (预测长度, 16) 动作字典。

        输入至少包含 16 个动作维度，且所有输入动作数值必须有限。
        """
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 16:
            raise ValueError(f"模型动作形状异常：{actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("模型动作包含非有限值")
        return {"actions": actions[:, :16]}
