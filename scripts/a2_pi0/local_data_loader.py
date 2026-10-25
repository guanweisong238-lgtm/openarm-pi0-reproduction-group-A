# 本文件负责：按指定的训练/验证划分读取本地 OpenArm 数据，生成模型可用的批次。
# 串联输入适配、归一化、图像缩放、指令分词和维度补齐，返回 Observation 与未来动作序列。
# 数据读完后循环加载；直接运行时仅检查第一个训练批次的形状和数值有效性。
# 本文件不加载预训练模型、不计算梯度，也不保存训练权重。

# 先加载配置，设置项目缓存路径。
from training_config import DATA_ROOT, get_train_config
from pathlib import Path

import json

import jax
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi import transforms
from openpi.models import model as model_lib


# 一、旧版LeRobot子集兼容：动作边界索引按选中轨迹的局部编号查询。
class EpisodeSubsetDataset(LeRobotDataset):
    """保持原始轨迹编号用于视频路径，仅修正未来动作边界的数组索引。"""

    def _get_query_indices(self, idx, ep_idx):
        """将原始轨迹编号映射到子集位置，沿用上游末端填充与掩码逻辑。"""
        local_episode = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
        return super()._get_query_indices(idx, local_episode)


# 二、逐样本变换包装
class TransformedDataset(torch.utils.data.Dataset):
    """包装 LeRobot 数据集，在访问每一帧时应用配置指定的输入变换。

    保留底层按轨迹读取未来动作的行为，只负责将样本转换成模型输入字段。
    """

    def __init__(self, dataset, input_transforms):
        """接收可索引数据集和有序变换列表，组合成单个可调用变换。"""
        self.dataset = dataset
        self.transform = transforms.compose(input_transforms)

    def __len__(self):
        """返回选中轨迹的帧数，不是轨迹数或批次数。"""
        return len(self.dataset)

    def __getitem__(self, index):
        """读取一帧及其未来动作，再做输入适配和预处理。

        底层图像解码或变换异常不在此吞掉，便于定位到具体错误样本。
        """
        return self.transform(self.dataset[index])


# 二、批次组装
def collate_numpy(samples):
    """沿新增批次维堆叠样本树的每个叶节点，返回同结构的 NumPy 数组树。

    输入适配应已移除无关元数据，分词器应已把任务文本转换成数值 Token。
    同一批次的字典结构和数组尺寸必须一致，否则树映射或堆叠会报错。
    """
    return jax.tree.map(
        lambda *values: np.stack(values),
        *samples,
    )


# 三、本地目录与轨迹划分加载器
class OpenArmLoader:
    """显式选择本地训练或验证轨迹，并循环产生模型批次。

    元数据中的 splits 仅声明区间，真正的数据隔离通过 episodes 参数实现。
    本加载器返回主机侧 NumPy 数组；训练入口负责设备放置及梯度更新。
    """

    def __init__(self, cfg, split="train", shuffle=True, episode_ids=None, data_root=None):
        """根据训练配置构造加载器。

        参数：cfg 包含模型预测长度、批次大小、变换和随机种子；split 为
        info.json 中的划分名；shuffle 控制每轮是否打乱帧的采样顺序。
        data_root 可显式选择其它本地数据，默认仍为原抓方块目录；配置中的统计须匹配。
        注意：drop_last=True 会丢弃每轮不足一个批次的尾部，验证时亦遵循此规则。
        异常：划分不存在、数据解码失败或无法组成完整批次时向调用方报错。
        """
        # 1. 将左闭右开的轨迹区间转换为显式编号，防止验证数据进入训练。
        self.data_root = Path(data_root) if data_root is not None else DATA_ROOT
        info = json.loads((self.data_root / "meta/info.json").read_text())
        start, stop = map(int, info["splits"][split].split(":"))
        allowed = list(range(start, stop))
        # 单轨迹试验仅允许从指定划分内部选择，防止训练混入验证轨迹。
        self.episode_ids = allowed if episode_ids is None else list(episode_ids)
        if not self.episode_ids or len(set(self.episode_ids)) != len(self.episode_ids) or any(ep not in allowed for ep in self.episode_ids):
            raise ValueError("episode_ids必须是当前划分内非空、无重复的轨迹编号")
        self._data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

        # 2. 按原数据 fps 读取未来动作；尾部不足由 LeRobot 在本轨迹末端填充。
        # shuffle 只改变当前观测帧的采样顺序，不会打乱动作块内部的时间顺序。
        raw_dataset = EpisodeSubsetDataset(
            repo_id=self._data_config.repo_id,
            root=self.data_root,
            episodes=self.episode_ids,
            delta_timestamps={
                "action": [
                    i / info["fps"]
                    for i in range(cfg.model.action_horizon)
                ]
            },
            video_backend="pyav",
        )

        # 3. 变换顺序是协议的一部分：先适配16维原值，再归一化，最后补齐至32维。
        # 如果先补齐再使用16维统计，容易造成维度不匹配或填充维的含义变化。
        data = self._data_config
        dataset = TransformedDataset(
            raw_dataset,
            [
                *data.repack_transforms.inputs,
                *data.data_transforms.inputs,
                transforms.Normalize(
                    data.norm_stats,
                    use_quantiles=data.use_quantile_norm,
                ),
                *data.model_transforms.inputs,
            ],
        )
        self.num_frames = len(dataset)

        # 4. 固定打乱顺序的随机种子；后续各轮仍会推进生成器状态。
        # 当前 num_workers=0 便于调试；图像最终仍按模型约定使用 NumPy 排列。
        generator = torch.Generator().manual_seed(cfg.seed)
        self._loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            drop_last=True,
            collate_fn=collate_numpy,
            generator=generator,
        )
        if len(self._loader) == 0:
            raise ValueError("数据不足以形成一个完整批次")

    def data_config(self):
        """返回实际使用的数据配置，供训练入口保存统计和数据资产信息。"""
        return self._data_config

    def __iter__(self):
        """无限返回 (Observation, actions)，由调用方的训练步数决定停止时机。

        actions 的形状为 (批次大小, 预测长度, 32)。Observation.from_dict
        将标准字段封装成模型观测，并把 uint8 图像转换到模型约定的浮点范围。
        验证程序应限制批次数，不能对该无限迭代器直接遍历到结束。
        """
        while True:
            for batch in self._loader:
                actions = batch.pop("actions")
                observation = model_lib.Observation.from_dict(batch)
                yield observation, actions


# 四、单批次验收入口
# 使用固定顺序读取首批，只验证数据通路；不证明模型训练成功或仿真接口兼容。
if __name__ == "__main__":
    cfg = get_train_config()
    loader = OpenArmLoader(cfg, split="train", shuffle=False)

    assert loader.episode_ids == list(range(24))
    assert loader.num_frames == 4548

    observation, actions = next(iter(loader))

    print("训练轨迹:", loader.episode_ids)
    print("训练帧数:", loader.num_frames)
    print("状态:", observation.state.shape)
    print("动作:", actions.shape)
    print("指令Token:", observation.tokenized_prompt.shape)

    assert observation.state.shape == (1, 32)
    assert actions.shape == (1, 16, 32)

    for name, image in observation.images.items():
        print("图像:", name, image.shape, image.dtype)
        assert image.shape == (1, 224, 224, 3)

    # 同时覆盖图像、掩码、指令 Token、状态及动作的有限值检查。
    arrays = jax.tree.leaves([observation.to_dict(), actions])
    assert all(np.isfinite(value).all() for value in arrays)

    print("PASS：训练批次读取、预处理和有限值检查通过")
