import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


CAMERAS = [
    "observation.images.ceiling",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
]


def load_rows(directory):
    rows = []
    for path in sorted(directory.rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def main():
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

    # 重新生成 v2.1 元数据，不复制 v3 的存储路径和统计信息。
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

                start = ep[f"{prefix}/from_timestamp"]
                first_frame = round(start * video_fps)
                if not np.isclose(
                    first_frame / video_fps, start, atol=1e-5
                ):
                    raise ValueError(f"视频起点不在帧边界：{path}")
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame):
                    raise RuntimeError(f"视频定位失败：{path}")

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
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if image.shape != features[key]["shape"]:
                        raise ValueError(f"图像尺寸不匹配：{key}")
                    frame[key] = image
                dataset.add_frame(frame)

            dataset.save_episode()
            print(f"已转换原始轨迹 {eid}，共 {len(rows)} 帧", flush=True)
        finally:
            for cap in captures.values():
                cap.release()

    # 将原始划分映射到输出轨迹编号。
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


if __name__ == "__main__":
    main()