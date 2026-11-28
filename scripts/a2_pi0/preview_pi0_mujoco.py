#!/usr/bin/env python3
"""Preview the new four-camera π0 checkpoint in an approximate contact scene."""
import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio_ffmpeg
import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "assets/real_replay_scene"
RUN_NAME = "real_20261128_202032_pi0_lora_1000_v001"
sys.path.insert(0, str(Path(__file__).resolve().parent / "deploy"))
from policy_server import PolicyService, RealFourCameraDataset

VIEWS = ("global", "chest", "wrist_left", "wrist_right")
SIDES = ("right", "left")
GRIP = [7, 15]
SIM_RATE = 30
PLAN_STEPS = 20
SEED = 1002

FIXTURES = """
<body name="table" pos="0.31 0 0.313">
  <geom name="table_top" type="box" size="0.39 0.43 0.023" rgba="0.12 0.12 0.14 1" friction="1 0.005 0.0001"/>
</body>
<body name="cabinet" pos="0.502 0.015 0.336">
  <geom name="cabinet_base" type="box" size="0.139 0.166 0.036" pos="0 0 0.036" rgba="0.82 0.84 0.86 1"/>
  <geom name="cabinet_top" type="box" size="0.139 0.166 0.008" pos="0 0 0.218" rgba="0.86 0.88 0.90 1"/>
  <geom name="cabinet_side_left" type="box" size="0.139 0.006 0.071" pos="0 0.160 0.143" rgba="0.83 0.85 0.87 1"/>
  <geom name="cabinet_side_right" type="box" size="0.139 0.006 0.071" pos="0 -0.160 0.143" rgba="0.83 0.85 0.87 1"/>
  <geom name="cabinet_back" type="box" size="0.007 0.154 0.071" pos="0.132 0 0.143" rgba="0.84 0.86 0.88 1"/>
  <body name="drawer" pos="-0.004 0 0.086">
    <joint name="drawer_slide" type="slide" axis="-1 0 0" limited="true" range="0 0.135" damping="1" frictionloss="0.15"/>
    <geom name="tray_floor" type="box" size="0.121 0.148 0.006" mass="0.26" rgba="0.94 0.94 0.95 1"/>
    <geom name="tray_front" type="box" size="0.007 0.148 0.063" pos="-0.126 0 0.061" mass="0.14" rgba="0.94 0.94 0.95 1"/>
    <geom name="tray_back" type="box" size="0.006 0.148 0.055" pos="0.124 0 0.055" mass="0.08" rgba="0.94 0.94 0.95 1"/>
    <geom name="tray_left" type="box" size="0.122 0.006 0.055" pos="0 0.145 0.055" mass="0.08" rgba="0.94 0.94 0.95 1"/>
    <geom name="tray_right" type="box" size="0.122 0.006 0.055" pos="0 -0.145 0.055" mass="0.08" rgba="0.94 0.94 0.95 1"/>
    <geom name="drawer_handle" type="cylinder" fromto="-0.172 -0.053 0.054 -0.172 0.073 0.054" size="0.009" mass="0.02" rgba="0.03 0.53 0.86 1"/>
    <geom name="handle_stem_left" type="cylinder" fromto="-0.132 0.064 0.054 -0.172 0.064 0.054" size="0.008" mass="0.012" rgba="0.03 0.53 0.86 1"/>
    <geom name="handle_stem_right" type="cylinder" fromto="-0.132 -0.043 0.054 -0.172 -0.043 0.054" size="0.008" mass="0.012" rgba="0.03 0.53 0.86 1"/>
  </body>
</body>
<body name="cube" pos="0.343 -0.244 0.359">
  <freejoint name="cube_free"/>
  <geom name="cube_geom" type="box" size="0.022 0.022 0.022" mass="0.030" friction="1 0.01 0.001" rgba="0.91 0.15 0.19 1"/>
</body>
"""


def camera_pose(position, look, up):
    z = np.asarray(position, float) - look
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return x, np.cross(z, x), z


def add_camera(world, name, position, look, up, fovy):
    x, y, _ = camera_pose(position, look, up)
    ET.SubElement(world, "camera", {
        "name": name, "pos": " ".join(map(str, position)),
        "xyaxes": " ".join(map(str, np.r_[x, y])), "fovy": str(fovy),
    })


def build_model(reference_fit=False):
    xml = ET.parse(SCENE / "empty_recovery_scene.xml").getroot()
    xml.find("./asset/model").set("file", str(SCENE / "openarm_pedestal.xml"))
    xml.find("./visual/global").attrib.update(offwidth="960", offheight="540")
    xml.find("./visual/quality").set("shadowsize", "2048")
    world = xml.find("worldbody")
    world.extend(ET.fromstring(f"<worldbody>{FIXTURES}</worldbody>"))
    if reference_fit:
        # Fit this preview to the episode 13 demonstration. Keep the original scene
        # as the default so the earlier rollout remains reproducible.
        cabinet = world.find("body[@name='cabinet']")
        cabinet.set("pos", "0.502 0.065 0.336")
        drawer = cabinet.find("body[@name='drawer']")
        drawer.find("geom[@name='tray_front']").set("pos", "-0.166 0 0.061")
        drawer.find("geom[@name='tray_floor']").attrib.update(
            size="0.141 0.148 0.006", pos="-0.02 0 0"
        )
        drawer.find("geom[@name='tray_left']").attrib.update(
            size="0.142 0.006 0.055", pos="-0.02 0.145 0.055"
        )
        drawer.find("geom[@name='tray_right']").attrib.update(
            size="0.142 0.006 0.055", pos="-0.02 -0.145 0.055"
        )
    add_camera(world, "global", [0.25, -0.18, 1.40], [.31, 0, .40], [1, 0, 0], 48)
    add_camera(world, "chest", [.04, 0, .61], [.42, .012, .428], [0, 0, 1], 64)
    add_camera(world, "overview", [.37, -1.03, 1.02], [.29, 0, .54], [0, 0, 1], 45)
    return mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


class Preview:
    def __init__(self, initial, reference_fit=False):
        m = self.model = build_model(reference_fit)
        arm_rate, gripper_rate = (1.83, 4.31) if reference_fit else (.30, .50)
        self.rate = np.tile([arm_rate] * 7 + [gripper_rate], 2)
        d = self.data = mj.MjData(m)
        names = [f"{side}_{kind}" for side in SIDES
                 for kind in [*(f"joint{i}" for i in range(1, 8)), "finger_joint1"]]
        joints = [m.joint("openarm_" + name).id for name in names]
        self.qpos = m.jnt_qposadr[joints]
        self.controls = np.array(
            [m.actuator(name.replace("finger_joint1", "finger1") + "_ctrl").id for name in names])
        self.low = np.maximum(m.jnt_range[joints, 0], m.actuator_ctrlrange[self.controls, 0])
        self.high = np.minimum(m.jnt_range[joints, 1], m.actuator_ctrlrange[self.controls, 1])
        self.command = np.clip(initial, self.low, self.high)
        self.initial_projection = float(np.max(np.abs(self.command - initial)))
        d.qpos[self.qpos] = self.command
        for i, side in enumerate(SIDES):
            d.joint(f"openarm_{side}_finger_joint2").qpos = self.command[GRIP[i]]
        d.ctrl[self.controls] = self.command
        self.cube_geom = m.geom("cube_geom").id
        self.tray_floor = m.geom("tray_floor").id
        front = m.geom("tray_front")
        back = m.geom("tray_back")
        left = m.geom("tray_left")
        right = m.geom("tray_right")
        self.tray_low = np.array([front.pos[0] + front.size[0], right.pos[1] + right.size[1]])
        self.tray_high = np.array([back.pos[0] - back.size[0], left.pos[1] - left.size[1]])
        mj.mj_forward(m, d)
        self.initial_max_penetration_mm = max(
            (max(0., -contact.dist) * 1000 for contact in d.contact), default=0.
        )
        if reference_fit and self.initial_max_penetration_mm > 1:
            raise ValueError("Initial overlap exceeds 1 mm in the reference-fit scene")
        self.camera_renderer = mj.Renderer(m, 216, 384)
        self.video_renderer = mj.Renderer(m, 432, 768)
        self.applied_clipping = np.zeros(16, dtype=int)
        self.frame = 0

    def observe(self):
        return self.data.qpos[self.qpos].copy()

    def images(self):
        images = {}
        for view in VIEWS:
            if view.startswith("wrist_"):
                side = view.split("_")[1]
                hand = self.data.body(f"openarm_{side}_ee_base_link")
                mat = hand.xmat.reshape(3, 3)
                sign = 1 if side == "left" else -1
                eye = hand.xpos + mat @ np.array([.071, sign * .050, -.092])
                look = hand.xpos + mat @ np.array([-.014, 0, -.186])
                _, up, z = camera_pose(eye, look, mat[:, 0])
                self.camera_renderer.update_scene(self.data, camera="chest")
                for camera in self.camera_renderer.scene.camera:
                    camera.pos[:] = eye
                    camera.forward[:] = -z
                    camera.up[:] = up
            else:
                self.camera_renderer.update_scene(self.data, camera=view)
            images[view] = self.camera_renderer.render().copy()
        return images

    def step(self, target):
        clipped = np.clip(target, self.low, self.high)
        self.applied_clipping += np.abs(clipped - target) > 1e-6
        self.command += np.clip(clipped - self.command, -self.rate / SIM_RATE, self.rate / SIM_RATE)
        self.data.ctrl[self.controls] = self.command
        self.frame += 1
        while self.data.time < self.frame / SIM_RATE - 1e-9:
            mj.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("nonfinite MuJoCo dynamics")

    def objects(self):
        d = self.data
        drawer = d.body("drawer")
        cube = d.body("cube")
        transform = drawer.xmat.reshape(3, 3).T
        cube_local = transform @ (cube.xpos - drawer.xpos)
        extent = np.abs(transform @ cube.xmat.reshape(3, 3)) @ self.model.geom_size[self.cube_geom]
        supported = any(
            {contact.geom1, contact.geom2} == {self.cube_geom, self.tray_floor} and contact.dist <= 0
            for contact in d.contact
        )
        inside = (supported
                  and np.all(cube_local[:2] - extent[:2] >= self.tray_low - .002)
                  and np.all(cube_local[:2] + extent[:2] <= self.tray_high + .002))
        return float(d.joint("drawer_slide").qpos[0]), cube.xpos.copy(), bool(inside)

    def close(self):
        self.camera_renderer.close()
        self.video_renderer.close()

    def composite(self, images, status, title):
        self.video_renderer.update_scene(self.data, camera="overview")
        panel = np.full((720, 1280, 3), (20, 24, 32), dtype=np.uint8)
        panel[38:470, 0:768] = self.video_renderer.render()
        self.text(panel, title, 16, 25, .68)
        for i, view in enumerate(VIEWS):
            x = (0, 384, 816, 816)[i]
            y = (491, 491, 52, 295)[i]
            panel[y:y+216, x:x+384] = images[view]
            self.text(panel, view.upper().replace("_", " "), x+8, y+22, .53)
        self.text(panel, status, 797, 570, .62)
        self.text(panel, "Simulated images and feedback only", 797, 605, .56)
        self.text(panel, "Approximate layout and viewpoints", 797, 630, .56)
        return np.ascontiguousarray(panel)

    @staticmethod
    def text(image, text, x, y, scale):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (4, 7, 10), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (244, 248, 252), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=13)
    parser.add_argument("--run", default=RUN_NAME)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--reference-fit", action="store_true", help="Use the simulation-only scene fit for episode 13")
    parser.add_argument("--reference-demo", action="store_true", help="Replay recorded IK targets instead of loading π0")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 1:
        parser.error("--seconds must be a finite duration of at least one second")
    args.output.mkdir(parents=True, exist_ok=False)
    run = ROOT / "results" / args.run
    checkpoint = ROOT / "checkpoints" / args.run / f"step_{args.step:04d}"
    config = json.loads((run / "config.json").read_text())
    data = RealFourCameraDataset("validation", config["candidate"])
    index = next(i for i, entry in enumerate(data.entries) if entry["episode_id"] == args.episode)
    sample_file = ROOT / data.entries[index]["samples_path"]
    with np.load(sample_file, allow_pickle=False) as episode:
        sim = Preview(episode["state"][0], args.reference_fit)
        recorded = episode["actions"][:, 0] if args.reference_demo else None
    if args.reference_demo:
        title = "OpenArm V2  |  RECORDED DEMONSTRATION  |  MUJOCO"
    else:
        title = f"OpenArm V2  |  PI0 STEP {args.step}  |  MUJOCO CLOSED LOOP"
        service = PolicyService(checkpoint, run)
        print(f"π0 warmed up in {service.warmup():.2f} s", flush=True)
    movie = args.output / "mujoco_rollout.mp4"
    writer = imageio_ffmpeg.write_frames(str(movie), (1280, 720), fps=15,
                                        codec="libx264", pix_fmt_out="yuv420p",
                                        output_params=["-movflags", "+faststart"])
    writer.send(None)
    records = []
    predictions = []
    initial_slide, initial_cube, _ = sim.objects()
    try:
        for step in range(round(args.seconds * SIM_RATE)):
            images = sim.images() if step % 2 == 0 else None
            state = sim.observe()
            action_index = step % PLAN_STEPS
            if recorded is None and action_index == 0:
                actions, inference_ms = service.infer(images, state, seed=SEED)
                predictions.append({"step": step, "inference_ms": inference_ms})
                print(f"plan {len(predictions)} at {step / SIM_RATE:.2f}s  "
                      f"drawer={sim.objects()[0]*1000:.1f}mm", flush=True)
            slide, cube, inside = sim.objects()
            if images is not None:
                status = f"t {step / SIM_RATE:5.2f}s     drawer {max(0., slide)*1000:5.1f} mm"
                writer.send(sim.composite(images, status, title))
            target = recorded[min(step, len(recorded) - 1)] if recorded is not None else actions[action_index]
            records.append((float(sim.data.time), state, target.copy(), slide, cube, inside))
            sim.step(target)

        slide, cube, inside = sim.objects()
        end = sim.composite(sim.images(), f"END      drawer {max(0., slide)*1000:5.1f} mm", title)
        for _ in range(15):
            writer.send(end)
        cv2.imwrite(str(args.output / "final.jpg"), cv2.cvtColor(end, cv2.COLOR_RGB2BGR))
    finally:
        writer.close()
        sim.close()

    time_s, feedback, target, drawer, cubes, in_drawer = map(np.asarray, zip(*records))
    np.savez_compressed(args.output / "rollout.npz", time_s=time_s, feedback_rad=feedback,
                        model_target_rad=target, drawer_m=drawer, cube_xyz_m=cubes,
                        cube_center_inside_tray=in_drawer)
    report = {
        "checkpoint": None if args.reference_demo else str(checkpoint.relative_to(ROOT)),
        "episode_for_initial_joints": args.episode,
        "simulated_seconds": float(sim.data.time),
        "plans": predictions,
        "seed": SEED,
        "initial_joint_projection_max_rad": sim.initial_projection,
        "initial_max_penetration_mm": sim.initial_max_penetration_mm,
        "applied_target_clips_per_axis": sim.applied_clipping.tolist(),
        "drawer_max_open_mm": float((max(max(drawer), slide) - initial_slide) * 1000),
        "drawer_final_open_mm": float(slide * 1000),
        "cube_max_lift_mm": float((max(max(cubes[:, 2]), cube[2]) - initial_cube[2]) * 1000),
        "cube_final_displacement_mm": float(np.linalg.norm(cube - initial_cube) * 1000),
        "cube_final_inside_drawer": inside,
        "cube_inside_drawer_for_final_half_second": bool(inside and np.all(in_drawer[-15:])),
        "scene": "episode13_reference_fit" if args.reference_fit else "original_preview",
        "simulation": "MuJoCo position actuators and contacts; approximate object layout and camera poses.",
        "input": "Recorded pre-driver IK targets" if args.reference_demo else
                 "Simulated images and feedback, initialized at the first measured joint sample",
        "sim_arm_and_gripper_slew_rad_s": sim.rate[:8].tolist(),
        "scheduling": "30 Hz, with replanning every 20 steps without inference delay in model mode.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "plans"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
