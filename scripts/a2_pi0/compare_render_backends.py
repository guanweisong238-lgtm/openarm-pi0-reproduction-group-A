"""对比 `glfw` 与 `egl` 两个渲染后端在同一初态下的像素差异。

为什么需要这个工具：
本项目的策略是确定性端点策略，对输入高度敏感（见 `optimization_log.md` 第 3.2 节），
渲染后端一旦不同，动作就会分叉，进而改变成败判定。2026-11-01 首次批量自检即因此
得到与历史证据相反的结果。因此"渲染后端"必须作为固定实验条件记录，跨批次比较前要核对。

用法：
    python compare_render_backends.py                    # 用默认起点（ID12）
    python compare_render_backends.py --episode-id 6      # 换起点
    python compare_render_backends.py --steps 4           # 额外比较推进 4 步后的观测

注意：两个后端必须在不同进程中渲染，因为 `MUJOCO_GL` 是进程级环境变量。
本脚本会自动起两个子进程完成渲染，然后逐像素比较。
"""
import os
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ('ceiling', 'wrist_left', 'wrist_right')
DEFAULT_EPISODE = 'data/processed/openarm_drawer_success_v003/episodes/episode_012'


def render(backend, output, episode_path, extra_steps):
    """子进程入口：按指定后端渲染并存成 npy。"""
    os.environ['MUJOCO_GL'] = backend
    os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'mesa')
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
    os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    sys.path[:0] = [str(ROOT / 'scripts/a2_pi0'), str(ROOT / '.cache/drawer-deps'),
                    str(ROOT / '.cache/openarm-control')]
    import numpy as np
    import mujoco
    from drawer_simulation import DrawerSimulation
    env = DrawerSimulation(ROOT / episode_path)
    try:
        Path(output).mkdir(parents=True, exist_ok=True)
        observation = env.observation()
        for name in CAMERAS:
            np.save(Path(output) / f'{name}.npy', observation[f'observation.images.{name}'])
        if extra_steps:
            for _ in range(extra_steps):
                env.step(observation['observation.state'])
            after = env.observation()
            for name in CAMERAS:
                np.save(Path(output) / f'{name}_step{extra_steps}.npy', after[f'observation.images.{name}'])
        print(json.dumps({'backend': backend, 'mujoco': mujoco.__version__, 'status': 'ok'}))
    finally:
        env.close()


def compare(glfw_dir, egl_dir, extra_steps):
    import numpy as np
    keys = list(CAMERAS) + ([f'{name}_step{extra_steps}' for name in CAMERAS] if extra_steps else [])
    rows = []
    for key in keys:
        glfw = np.load(Path(glfw_dir) / f'{key}.npy').astype(np.int16)
        egl = np.load(Path(egl_dir) / f'{key}.npy').astype(np.int16)
        difference = np.abs(glfw - egl)
        pixels = difference.shape[0] * difference.shape[1]
        touched = int((difference.max(axis=2) > 0).sum())
        rows.append(dict(camera=key, max_abs_delta=int(difference.max()),
                         mean_abs_delta=float(difference.mean()),
                         changed_values=int((difference > 0).sum()), total_values=int(difference.size),
                         pixels_touched=touched, total_pixels=pixels,
                         pixels_touched_ratio=touched / pixels))
    return rows


def main():
    if '--render' in sys.argv:
        index = sys.argv.index('--render')
        render(sys.argv[index + 1], sys.argv[index + 2], sys.argv[index + 3], int(sys.argv[index + 4]))
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-id', type=int, default=12, help='示教清单中的起点编号')
    parser.add_argument('--episode-path', type=Path, help='直接指定 episode 目录，优先于 --episode-id')
    parser.add_argument('--steps', type=int, default=4, help='额外比较推进若干控制步后的观测；0 表示不比较')
    parser.add_argument('--output', type=Path, help='npy 落盘目录，默认用临时目录')
    args = parser.parse_args()

    if args.episode_path:
        episode_path = args.episode_path
    else:
        manifest = json.loads((ROOT / 'data/processed/openarm_drawer_success_v003/manifest.json').read_text())
        row = next((r for r in manifest['episodes'] if r['episode_id'] == args.episode_id), None)
        if row is None:
            raise SystemExit(f'起点 {args.episode_id} 不存在于示教清单')
        episode_path = row['path']

    base = args.output or Path(tempfile.mkdtemp(prefix='render-compare-'))
    print(f'episode={episode_path}')
    for backend in ('glfw', 'egl'):
        result = subprocess.run([sys.executable, __file__, '--render', backend,
                                 str(base / backend), str(episode_path), str(args.steps)],
                                capture_output=True, text=True)
        if result.returncode:
            print(f'[{backend}] 渲染失败：{result.stdout}{result.stderr}')
            raise SystemExit(1)
        print(f'[{backend}] 渲染完成')

    rows = compare(base / 'glfw', base / 'egl', args.steps)
    print(f'\n{"camera":<22}{"max_delta":>10}{"mean_delta":>13}{"changed_values":>18}{"pixels_touched":>18}')
    for row in rows:
        print(f'{row["camera"]:<22}{row["max_abs_delta"]:>10}{row["mean_abs_delta"]:>13.6f}'
              f'{row["changed_values"]:>10}/{row["total_values"]:<7}{row["pixels_touched"]:>10}/{row["total_pixels"]:<7}')
    verdict = 'IDENTICAL' if all(row['max_abs_delta'] == 0 for row in rows) else 'DIFFERENT'
    print(f'\nVERDICT {verdict}')
    print(f'npy 保留在 {base}')
    print('说明：本结果只对该机器与该驱动组合成立。后端不一致时，批量结果不得与历史批次并列比较。')


if __name__ == '__main__':
    main()
