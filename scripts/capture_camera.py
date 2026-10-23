from pathlib import Path
import mujoco
from PIL import Image
from simulation.env import OpenArmEnv

ROOT = Path(__file__).resolve().parents[1]
output_dir = ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 实例化我们在 Day 4 封装好的环境
env = OpenArmEnv()
obs = env.reset()

# 2. 我们在 Day 2 查出的两台手腕相机名称
camera_names = ("camera_wrist_left", "camera_wrist_right")

print("📸 正在调用 MuJoCo 离屏渲染器拍照...")
# 创建 320x240 的渲染器
with mujoco.Renderer(env.model, width=320, height=240) as renderer:
    for cam_name in camera_names:
        # 检查相机是否存在
        if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) < 0:
            raise ValueError(f"模型中找不到相机: {cam_name}")

        # 更新场景并提取 RGB 图像 (高度 240, 宽度 320, 通道 3)
        renderer.update_scene(env.data, camera=cam_name)
        img_arr = renderer.render().copy()

        # 保存为 png 图片
        save_path = output_dir / f"{cam_name}.png"
        Image.fromarray(img_arr).save(save_path)
        print(f"  ✅ 成功保存: {save_path} (尺寸: {img_arr.shape})")

print("🎉 相机拍照测试完成！请到 outputs/ 文件夹查看样张。")
