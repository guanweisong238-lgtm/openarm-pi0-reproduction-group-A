# 本文件负责：下载官方 pi0_base 基础权重，为 OpenArm LoRA 训练提供初始化参数。
# 使用匿名访问获取官方文件列表，通过 HTTPS 下载，并逐文件校验大小和 MD5。
# 已校验文件直接复用；缺失或损坏文件重新下载，校验通过后才替换正式文件。
# 权重、下载锁和日志均保存在本项目中；本脚本不加载模型、不启动训练。

import base64
from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import time
from urllib.parse import quote

# 一、项目路径与公开权重来源
ROOT = Path(__file__).resolve().parents[2]
PREFIX = "openpi-assets/checkpoints/pi0_base"
TARGET = ROOT / "weights/openpi" / PREFIX
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"

import filelock
import fsspec
import requests


# 二、本地完整性校验
def verify_file(path, info):
    """检查本地文件大小和 MD5 是否与官方对象元数据一致。

    参数：path 为本地路径；info 包含官方 size 和 md5Hash。
    返回：文件存在且两项校验均通过时返回 True，否则返回 False。
    官方未提供 MD5 时直接报错，避免把未校验文件误判为成功。
    """
    if not info.get("md5Hash"):
        raise RuntimeError(f"官方对象缺少 MD5：{info['name']}")
    if not path.is_file() or path.stat().st_size != info["size"]:
        return False

    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)

    actual = base64.b64encode(digest.digest()).decode()
    return actual == info["md5Hash"]


# 三、单文件下载与失败重试
def download_file(session, name, info, destination):
    """下载指定版本的公开对象，校验后原子替换正式文件。

    下载写入 .download-part 临时文件，失败最多重试三次。
    每次重试重新下载当前文件；已完成的其他文件不会重复下载。
    网络、写入或完整性错误最终向调用方传播。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".download-part")
    url = "https://storage.googleapis.com/" + quote(name, safe="/")

    for attempt in range(1, 4):
        try:
            logging.info(
                "下载 %s：%.1f MiB，第 %d/3 次尝试",
                destination.name, info["size"] / 2**20, attempt,
            )
            received = 0
            next_report = 256 * 1024 * 1024

            # generation 固定本次下载的对象版本，确保与所查校验值对应。
            with session.get(
                url,
                params={"generation": info["generation"]},
                stream=True,
                timeout=(15, 120),
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    # 保留服务端存储的原始字节，避免自动解压影响校验。
                    for block in response.raw.stream(
                        8 * 1024 * 1024, decode_content=False
                    ):
                        output.write(block)
                        received += len(block)
                        if received >= next_report:
                            logging.info(
                                "%s：%.1f / %.1f MiB",
                                destination.name,
                                received / 2**20,
                                info["size"] / 2**20,
                            )
                            next_report = received + 256 * 1024 * 1024

            if not verify_file(partial, info):
                raise RuntimeError(f"文件大小或 MD5 不匹配：{name}")

            partial.replace(destination)
            logging.info("下载并校验通过：%s", name)
            return
        except Exception:
            if attempt == 3:
                raise
            logging.exception("本次下载失败，准备重试")
            time.sleep(1)


# 四、完整 checkpoint 下载流程
def main():
    """检查官方全部文件，补齐权重，并输出完整性验收结果。

    重复执行会校验并复用已有文件。只有所有官方文件通过大小和 MD5
    检查后才打印 PASS；此结果不代替后续模型参数加载验证。
    """
    log_dir = ROOT / "logs/a2_pi0"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    log_path = log_dir / f"download_pi0_base_{run_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    logging.info("权重目录：%s", TARGET)
    logging.info("日志：%s", log_path)

    # 下载和检查共用互斥锁，避免两个进程同时修改相同文件。
    with filelock.FileLock(str(TARGET) + ".lock"):
        fs = fsspec.filesystem(
            "gs", token="anon", requests_timeout=30
        )
        entries = fs.find(PREFIX, detail=True)
        objects = [
            (name, info)
            for name, info in sorted(entries.items())
            if info["type"] == "file" and not name.endswith("/")
        ]
        if not objects:
            raise RuntimeError("官方文件列表为空")

        total = sum(info["size"] for _, info in objects)
        logging.info(
            "官方文件数：%d，总大小：%.2f GiB",
            len(objects), total / 2**30,
        )

        with requests.Session() as session:
            for name, info in objects:
                relative = name.removeprefix(PREFIX + "/")
                destination = TARGET / relative
                if verify_file(destination, info):
                    logging.info("已有文件校验通过：%s", relative)
                else:
                    download_file(session, name, info, destination)

    logging.info("PASS：pi0_base 全部文件大小和 MD5 校验通过")


# 五、命令行入口：异常会让进程以失败状态退出。
if __name__ == "__main__":
    main()