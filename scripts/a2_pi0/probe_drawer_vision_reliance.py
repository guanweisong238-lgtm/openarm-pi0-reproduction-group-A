"""Measure how much the frozen drawer policy actually uses vision.

Four probes on one frozen checkpoint, all read-only (no training, no state injection):

  1. drawer shifted +-20mm along the handle-length axis (Y)
  2. drawer shifted +-20mm along the pull axis (X)
  3. image ablation: all-black images / images from a different start
  4. proprioception ruler: one right-arm joint moved by 0.01 rad

The robot state is held fixed inside each probe so that only one channel changes.
Writes a JSON summary; never writes into data/ or logs/.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/a2_pi0"))
os.environ.setdefault("MUJOCO_GL", "glfw")

import jax  # noqa: E402
import mujoco  # noqa: E402

from drawer_simulation import DrawerSimulation  # noqa: E402
from drawer_start_fit_config import get_policy_config  # noqa: E402

IMG_KEYS = ("observation.images.ceiling", "observation.images.wrist_left",
            "observation.images.wrist_right")
PROP_KEYS = ("observation.state", "observation.velocity")


def ee_jacobian(model, data):
    """3x7 linear map: right-arm joint deltas -> end-effector translation (m)."""
    site = model.site("right_ee_control_point").id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    cols = [model.jnt_dofadr[model.joint(f"openarm_right_joint{i}").id] for i in range(1, 8)]
    return jacp[:, cols]


class Probe:
    def __init__(self, args):
        self.args = args
        self.checkpoint = Path(args.checkpoint)
        if jax.default_backend() != "gpu":
            raise RuntimeError("本探测需要 JAX GPU 后端")
        protocol = json.loads((self.checkpoint / "input_protocol.json").read_text())
        from drawer_policy_loader import create_checkpoint_precision_policy
        self.policy = create_checkpoint_precision_policy(
            get_policy_config(protocol), self.checkpoint,
            sample_kwargs={"num_steps": int(protocol.get("sampling_steps", 1))})
        self.noise = np.random.default_rng(
            int(protocol["fixed_noise_seed"])).standard_normal(
            (int(protocol.get("action_horizon", 4)), 32)).astype(np.float32)
        manifest = json.loads((ROOT / args.manifest).read_text())
        self.cases = {e["episode_id"]: e for e in manifest["episodes"]}
        self.run = ROOT / args.run
        if args.workdir:
            self.work = Path(args.workdir)
        else:
            self.work = Path(os.environ.get("TMPDIR", "/tmp")) / "drawer_vision_probe"
        self.work.mkdir(parents=True, exist_ok=True)

    def trial_dir(self, episode):
        for name in sorted(os.listdir(self.run)):
            if name.endswith("_episode_%d" % episode):
                return self.run / name
        raise KeyError(episode)

    def qpos_at(self, episode, step):
        with np.load(self.trial_dir(episode) / "closed_loop.npz") as raw:
            return raw["qpos"][step - 1]

    def env_for(self, episode, offset=None):
        """Load the simulator, optionally shifting the drawer along one axis."""
        if not offset:
            return DrawerSimulation(ROOT / self.cases[episode]["path"])
        axis, metres = offset
        src = ROOT / self.cases[episode]["path"]
        dst = self.work / ("ep_%d_%s%+03.0fmm" % (episode, axis, metres * 1000))
        if dst.is_dir():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        payload = json.loads((dst / "report.json").read_text())
        overrides = payload.setdefault("scene_overrides", {})
        key = "drawer_offset_%s_m" % axis
        overrides[key] = float(overrides.get(key, 0.0)) + metres
        (dst / "report.json").write_text(json.dumps(payload))
        return DrawerSimulation(dst)

    def infer(self, obs):
        return np.asarray(self.policy.infer(obs, noise=self.noise)["actions"])

    def run_offset_probe(self, label, axis, offsets_mm, steps):
        rows = []
        for step in steps:
            q = self.qpos_at(self.args.episode, step)
            for dy in offsets_mm:
                env = self.env_for(self.args.episode, (axis, dy / 1000.0))
                try:
                    env.data.qpos[:] = q
                    mujoco.mj_forward(env.model, env.data)
                    jac = ee_jacobian(env.model, env.data)
                    block = self.infer(env.observation())
                    rows.append(dict(probe=label, step=step, offset_mm=dy,
                                     ee_delta_mm=((jac @ block[0, :7]) * 1000).tolist()))
                finally:
                    env.close()
        return rows

    def run_vision_ablation(self, steps):
        rows = []
        for step in steps:
            base_env = self.env_for(self.args.episode)
            try:
                base_env.data.qpos[:] = self.qpos_at(self.args.episode, step)
                mujoco.mj_forward(base_env.model, base_env.data)
                jac = ee_jacobian(base_env.model, base_env.data)
                obs_a = base_env.observation()
            finally:
                base_env.close()
            other_env = self.env_for(self.args.other_episode)
            try:
                other_env.data.qpos[:] = self.qpos_at(self.args.other_episode, step)
                mujoco.mj_forward(other_env.model, other_env.data)
                obs_other = other_env.observation()
            finally:
                other_env.close()

            obs_black = dict(obs_a)
            for k in IMG_KEYS:
                obs_black[k] = np.zeros_like(obs_a[k])
            obs_swap = dict(obs_other)
            for k in PROP_KEYS:
                obs_swap[k] = obs_a[k]

            acts = {}
            for name, obs in (("normal", obs_a), ("black", obs_black), ("other_start", obs_swap)):
                block = self.infer(obs)
                acts[name] = block
                rows.append(dict(probe="vision_ablation", step=step, variant=name,
                                 first_ee_delta_mm=((jac @ block[0, :7]) * 1000).tolist()))
            for name in ("black", "other_start"):
                delta = acts[name] - acts["normal"]
                steps_mm = np.linalg.norm((jac @ delta[:, :7].T) * 1000, axis=0)
                rows.append(dict(probe="vision_ablation_diff", step=step, variant=name,
                                 max_action_rad=float(np.abs(delta).max()),
                                 first_ee_mm=float(steps_mm[0]),
                                 max_ee_mm=float(steps_mm.max())))
        return rows

    def run_proprio_ruler(self, step, dq):
        env = self.env_for(self.args.episode)
        rows = []
        try:
            jid = env.model.joint("openarm_right_joint1")
            adr = jid.qposadr[0]
            q = self.qpos_at(self.args.episode, step)
            acts = {}
            for d in (0.0, dq):
                env.data.qpos[:] = q
                env.data.qpos[adr] += d
                mujoco.mj_forward(env.model, env.data)
                jac = ee_jacobian(env.model, env.data)
                block = self.infer(env.observation())
                acts[d] = (block, jac)
                self_move = np.linalg.norm(
                    jac @ np.array([d if i == 0 else 0.0 for i in range(7)])) * 1000
                rows.append(dict(probe="proprio", step=step, dq_rad=d,
                                 arm_moved_mm=float(self_move),
                                 ee_delta_mm=((jac @ block[0, :7]) * 1000).tolist()))
            delta = acts[dq][0] - acts[0.0][0]
            jac = acts[0.0][1]
            steps_mm = np.linalg.norm((jac @ delta[:, :7].T) * 1000, axis=0)
            rows.append(dict(probe="proprio_diff", step=step, dq_rad=dq,
                             max_action_rad=float(np.abs(delta).max()),
                             first_ee_mm=float(steps_mm[0]), max_ee_mm=float(steps_mm.max())))
        finally:
            env.close()
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episode", type=int, default=400)
    ap.add_argument("--other-episode", type=int, default=401)
    ap.add_argument("--run", default="logs/a2_pi0/grasp_improvement_20261102/baseline020_final100")
    ap.add_argument("--manifest",
                    default="data/processed/openarm_drawer_eval_v004_final/manifest.json")
    ap.add_argument("--steps", type=int, nargs="+", default=[40, 68])
    ap.add_argument("--offset-mm", type=float, nargs="+", default=[-20.0, 0.0, 20.0])
    ap.add_argument("--dq", type=float, default=0.01)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    probe = Probe(args)
    rows = []
    rows += probe.run_offset_probe("offset_y", "y", args.offset_mm, args.steps)
    rows += probe.run_offset_probe("offset_x", "x", args.offset_mm, args.steps)
    rows += probe.run_vision_ablation(args.steps)
    rows += probe.run_proprio_ruler(args.steps[-1], args.dq)

    payload = dict(checkpoint=str(args.checkpoint), episode=args.episode,
                   other_episode=args.other_episode, steps=args.steps,
                   offset_mm=args.offset_mm, dq_rad=args.dq, records=rows)
    Path(args.output).write_text(json.dumps(payload, indent=1) + "\n")
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
