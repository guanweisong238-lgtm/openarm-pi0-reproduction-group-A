from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "openarm_mujoco/v2/pedestal/pedestal.xml"
model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if home_id < 0:
    raise ValueError("Missing home keyframe")
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

print("nq, nv, nu, ncam:", model.nq, model.nv, model.nu, model.ncam)
print("physics_dt:", model.opt.timestep)
print("keyframe_ctrl_raw:", data.ctrl.tolist())
print("home_qpos:", data.qpos.tolist())
for jid in range(model.njnt):
    print("joint", jid, model.joint(jid).name,
          "type", int(model.jnt_type[jid]),
          "qpos_addr", int(model.jnt_qposadr[jid]),
          "qvel_addr", int(model.jnt_dofadr[jid]),
          "range", model.jnt_range[jid].tolist())
for aid in range(model.nu):
    print("actuator", aid, model.actuator(aid).name,
          "transmission", int(model.actuator_trntype[aid]),
          "target_ids", model.actuator_trnid[aid].tolist(),
          "limited", bool(model.actuator_ctrllimited[aid]),
          "ctrlrange", model.actuator_ctrlrange[aid].tolist())
for cid in range(model.ncam):
    print("camera", cid, model.camera(cid).name)
