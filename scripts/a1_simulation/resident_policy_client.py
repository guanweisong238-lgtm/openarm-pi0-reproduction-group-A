# 本文件负责：管理A2常驻推理进程，为A1闭环提供动作请求接口。
# 首次调用时启动模型进程并等待预热，后续复用同一进程和模型。
# 等待期间只刷新窗口，不推进物理；异常或结束时负责清理子进程。
# 文件通信不引入OpenPI依赖，不合并A1和A2的虚拟环境。

import json
from pathlib import Path
import subprocess
import time

import numpy as np

from simulation.env import ROOT


# 一、常驻进程生命周期及文件请求协议。
class ResidentPolicyClient:
    """在独立A2虚拟环境中维护一个推理进程。"""

    def __init__(self, checkpoint=None):
        """保存可选检查点路径，延迟到首个观测就绪后启动模型进程。"""
        self.checkpoint = Path(checkpoint).resolve() if checkpoint is not None else (
            ROOT.parent / "openarm-pi0-reproduction-group-A"
            / "checkpoints/pi0_openarm_lora/smoke_001/10"
        )
        self.process = None
        self.log = None
        self.workspace = None
        self.counter = 0
        self.last_inference_seconds = None
        self.last_request_seconds = None

    def wait_for(self, path, viewer):
        """等待响应并刷新窗口，检查子进程退出及错误报告。"""
        while True:
            if not viewer.is_running():
                raise InterruptedError("窗口已关闭")

            fatal = self.workspace / "fatal.json"
            if fatal.exists():
                raise RuntimeError(fatal.read_text())

            if path.exists():
                return json.loads(path.read_text())

            if self.process.poll() is not None:
                raise RuntimeError(
                    f"A2推理进程已退出，请查看："
                    f"{self.workspace / 'worker.log'}"
                )

            viewer.sync()
            time.sleep(0.02)

    def infer(self, snapshot, log_path, viewer):
        """发送新观测，返回本轮原始预测及结果目录。"""
        if self.process is None:
            # 每次闭环运行使用独立通信目录，避免混入历史请求。
            self.workspace = log_path.parent.parent / "resident"
            self.workspace.mkdir(parents=True, exist_ok=False)

            a2_root = ROOT.parent / "openarm-pi0-reproduction-group-A"
            self.log = (self.workspace / "worker.log").open("w")
            self.process = subprocess.Popen(
                [
                    str(a2_root / ".venv/bin/python"),
                    str(
                        a2_root
                        / "scripts/a2_pi0/resident_inference.py"
                    ),
                    "--workspace", str(self.workspace),
                    "--warmup", str(snapshot),
                    "--checkpoint", str(self.checkpoint),
                ],
                cwd=a2_root,
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )

            print("首次加载模型并预热，请稍等……", flush=True)
            ready = self.wait_for(self.workspace / "ready.json", viewer)
            if Path(ready["checkpoint"]).resolve() != self.checkpoint:
                raise RuntimeError("常驻服务加载的检查点与请求不一致")
            print("已加载检查点:", self.checkpoint, flush=True)
            print("模型已常驻，后续复用当前进程。", flush=True)

        directory = log_path.parent
        # 请求耗时包含文件通信和读取，模型耗时由A2单独测量；均不含首次预热。
        began = time.perf_counter()
        number = self.counter
        self.counter += 1

        # 先写完请求，再以原子替换方式发布。
        request = self.workspace / f"request_{number:04d}.json"
        temporary = request.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "observation": str(snapshot.resolve()),
            "output": str(directory.resolve()),
        }))
        temporary.replace(request)

        response = self.wait_for(
            self.workspace / f"response_{number:04d}.json",
            viewer,
        )
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "推理失败"))
        if Path(response["observation_path"]).resolve() != snapshot.resolve():
            raise RuntimeError("推理响应对应的观测来源不一致")

        actions = np.load(
            directory / "actions.npy", allow_pickle=False
        )
        if actions.shape != (16, 16) or not np.isfinite(actions).all():
            raise ValueError("模型输出异常")

        elapsed = response["inference_seconds"]
        self.last_inference_seconds = elapsed
        self.last_request_seconds = time.perf_counter() - began
        print(
            f"本轮模型推理耗时：{elapsed:.3f}s，"
            f"请求总耗时：{self.last_request_seconds:.3f}s",
            flush=True,
        )
        return actions, directory

    def close(self):
        """请求正常结束；超时后终止进程并关闭日志。"""
        if self.process is not None and self.process.poll() is None:
            (self.workspace / "stop").touch()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

        if self.log is not None:
            self.log.close()
