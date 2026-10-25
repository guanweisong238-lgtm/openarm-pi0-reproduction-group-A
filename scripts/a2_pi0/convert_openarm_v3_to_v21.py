# 本文件负责：将本项目的 OpenArm LeRobot v3.0 数据转换为当前读取器支持的 v2.1 格式。
# 按轨迹和视频时间偏移读取原始数据，保留 16 维状态、动作顺序及训练/验证划分。
# 选择顶部、左腕和右腕三路图像，重新编码视频，并在独立输出目录保存转换记录。
# 支持单条轨迹试转换或全量转换；不覆盖原始数据，不进行关节标定或动作单位换算。

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# 一、保留的相机字段
# 原始数据的两路头部图像仍保留在源目录，但不纳入当前三视角训练配置。
CAMERAS = [
    "observation.images.ceiling",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
]


def load_rows(directory):
    """按文件路径顺序读取目录中的全部 Parquet 行，返回字典列表。

    directory 为数据或轨迹元数据目录；文件解析异常直接向上传播。
    本项目数据量较小，采用整表载入；此方法不适合直接处理超大数据集。
    """
    rows = []
    for path in sorted(directory.rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def main():
    """转换指定数量的轨迹，重新编码视频并写出 v2.1 元数据。

    --source 指向已检查的 OpenArm v3.0 数据；--output 必须是不存在的目录。
    --limit 默认为 1，值为 0 时转换全部轨迹。失败后可能留下不完整输出，
    本脚本不会自动续传或清理该目录，不能仅凭目录存在就判断转换成功。
    """
    # 二、命令行参数与输出目录保护
    # 默认相对路径按运行时工作目录解析，应从小组项目根目录启动。
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/openarm_pickup")
    parser.add_argument(
        "--output",
        default="data/processed/openarm_pickup_smoke",
    )
    parser.add_argument(
        "--limit", type=int, default=1,
        help="转换轨迹数；0 表示全部",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    if output.exists():
        raise FileExistsError(
            f"输出目录已存在，请指定新的 --output：{output}"
        )
    if args.limit < 0:
        raise ValueError("--limit 不能小于 0")

    # 三、读取源数据布局、任务文本及轨迹分组
    # v3 的多个轨迹可以共享一个数据文件和一个视频，因此不能按文件数计轨迹数。
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("该脚本针对已检查的 v3.0 数据")

    fps = info["fps"]
    episodes = sorted(
        load_rows(source / "meta/episodes"),
        key=lambda row: row["episode_index"],
    )
    if args.limit:
        episodes = episodes[:args.limit]

    rows_by_episode = {}
    for row in load_rows(source / "data"):
        rows_by_episode.setdefault(row["episode_index"], []).append(row)

    tasks = {
        row["task_index"]: row["task"]
        for row in pq.read_table(source / "meta/tasks.parquet").to_pylist()
    }

    # 四、定义输出字段和轨迹编号映射
    # 重新生成 v2.1 元数据，不复制 v3 的存储路径和统计信息。
    # 让当前固定版本的 LeRobot 写入特征描述，避免新版 List 类型与旧读取器不兼容。
    features = {}
    for key in ["observation.state", "action", *CAMERAS]:
        feature = info["features"][key]
        features[key] = {
            "dtype": feature["dtype"],
            "shape": tuple(feature["shape"]),
            "names": feature.get("names"),
        }

    # 保留原始 train/val 归属，转换前检查划分。
    source_split = {}
    for split, interval in info["splits"].items():
        start, stop = map(int, interval.split(":"))
        for eid in range(start, stop):
            if eid in source_split:
                raise ValueError(f"轨迹 {eid} 出现在多个划分中")
            source_split[eid] = split

    mapping = []
    for new_id, ep in enumerate(episodes):
        old_id = ep["episode_index"]
        if old_id not in source_split:
            raise ValueError(f"轨迹 {old_id} 没有数据划分")
        mapping.append({
            "episode_index": new_id,
            "source_episode_index": old_id,
            "split": source_split[old_id],
        })

    # 五、创建 v2.1 写入器
    # local/openarm_pickup 只是本地标识；这里不上传数据，也不访问同名远程仓库。
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id="local/openarm_pickup",
        root=output,
        fps=fps,
        robot_type=info["robot_type"],
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )

    # 六、按轨迹对齐数值帧与三路视频，再交给 LeRobot 保存
    for ep in episodes:
        eid = ep["episode_index"]
        rows = sorted(
            rows_by_episode[eid],
            key=lambda row: row["frame_index"],
        )
        if len(rows) != ep["length"]:
            raise ValueError(f"轨迹 {eid} 长度不匹配")

        # 此数据已检查为从 0 开始的固定频率轨迹。
        for index, row in enumerate(rows):
            if row["frame_index"] != index:
                raise ValueError(f"轨迹 {eid} 帧编号异常")
            if not np.isclose(
                row["timestamp"], index / fps, atol=1e-5, rtol=0
            ):
                raise ValueError(f"轨迹 {eid} 时间戳异常")

        captures = {}
        try:
            for key in CAMERAS:
                prefix = f"videos/{key}"
                path = source / info["video_path"].format(
                    video_key=key,
                    chunk_index=ep[f"{prefix}/chunk_index"],
                    file_index=ep[f"{prefix}/file_index"],
                )
                cap = cv2.VideoCapture(str(path))
                captures[key] = cap
                if not cap.isOpened():
                    raise RuntimeError(f"无法打开视频：{path}")
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not np.isclose(video_fps, fps):
                    raise ValueError(f"视频帧率不匹配：{path}")

                # from_timestamp 是本轨迹在共享视频内的起始时间，不是轨迹内时间。
                # 验证其落在帧边界后只定位一次，再顺序读帧，避免逐帧随机 seek。
                start = ep[f"{prefix}/from_timestamp"]
                first_frame = round(start * video_fps)
                if not np.isclose(
                    first_frame / video_fps, start, atol=1e-5
                ):
                    raise ValueError(f"视频起点不在帧边界：{path}")
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame):
                    raise RuntimeError(f"视频定位失败：{path}")

            # add_frame 按固定 fps 生成输出时间戳；前面的检查保证与源时间对应。
            # 状态和动作维度不重排，不使用 ALOHA 的关节翻转或夹爪换算。
            for row in rows:
                frame = {
                    "observation.state": np.asarray(
                        row["observation.state"], dtype=np.float32
                    ),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "task": tasks[row["task_index"]],
                }
                for key, cap in captures.items():
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"视频解码失败：episode={eid}, "
                            f"frame={row['frame_index']}, camera={key}"
                        )
                    # OpenCV 解码为 BGR，而训练图像协议要求 RGB。
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if image.shape != features[key]["shape"]:
                        raise ValueError(f"图像尺寸不匹配：{key}")
                    frame[key] = image
                dataset.add_frame(frame)

            # 每条轨迹单独保存并编码；旧版读取器据此正确处理未来动作的轨迹边界。
            dataset.save_episode()
            print(f"已转换原始轨迹 {eid}，共 {len(rows)} 帧", flush=True)
        finally:
            # 即使某一路读取失败，也释放已打开的视频句柄。
            for cap in captures.values():
                cap.release()

    # 七、保存训练/验证划分和转换来源
    # 将原始划分映射到输出轨迹编号。
    # 元数据声明划分不等于训练时自动筛选，加载器仍须显式传入 episodes。
    target_info_path = output / "meta/info.json"
    target_info = json.loads(target_info_path.read_text())
    splits = {}
    for split in info["splits"]:
        ids = [
            item["episode_index"]
            for item in mapping if item["split"] == split
        ]
        if ids:
            if ids != list(range(ids[0], ids[-1] + 1)):
                raise ValueError(f"输出划分 {split} 不是连续区间")
            splits[split] = f"{ids[0]}:{ids[-1] + 1}"
    target_info["splits"] = splits
    target_info_path.write_text(
        json.dumps(target_info, indent=2) + "\n"
    )
    (output / "conversion.json").write_text(
        json.dumps({
            "source": str(source),
            "source_format": info["codebase_version"],
            "selected_cameras": CAMERAS,
            "episode_mapping": mapping,
        }, indent=2) + "\n"
    )
    print(f"转换完成：{output}")


# 八、命令行入口
if __name__ == "__main__":
    main()
