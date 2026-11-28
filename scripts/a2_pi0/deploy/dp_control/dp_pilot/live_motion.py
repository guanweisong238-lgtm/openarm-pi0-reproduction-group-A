"""Standalone motion panel. Launching this file never enables the arms."""
import argparse
import fcntl
from collections import deque
import json
from pathlib import Path
from queue import Empty
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk

from dp_paths import CAMERA_DIR, HERE, SDK_PY, SOURCE_LOCAL
from live_verify import SIDES, LiveSession
from pi0_observations import PLAN_AGE_NS, Pi0Policy, predict_live
from task_reference import RELEASE
from motion_arm import DEMO_ID, DEMO_FRAME, LEFT_J5_PREDICTION_BOUND, accept_target
from trial_recording import TrialRecording
from gravity_profile import DEFAULT_PERCENT, description as gravity_description


class MotionSession(LiveSession):
    def __init__(self, gravity_percent=DEFAULT_PERCENT):
        super().__init__()
        self.gravity_percent = gravity_percent
        self.links = {}
        self.phase = "idle"
        self.checked, self.ready = set(), set()
        self.home_ready = set()
        self.enabled = set()
        self.limits = {}
        self.since = 0.
        self.first_prediction = False
        self.permit_after = None
        self.camera = None
        self.tmp = None
        self.stop_log = None
        self.motion_requested = False
        self.disabled = set()
        self.jobs = []
        self.recent_targets = deque(maxlen=12)
        self.recorder = None

    def acquire(self):
        d = Path.home()/".cache/openarm-dp-field"
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.ownership = (d/"verify.lock").open("a")
        fcntl.flock(self.ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.stop_log = d/f"motion-stop-{time.time_ns()}.jsonl"
        self.phase = "loading"
        self.since = time.monotonic()

    def launch_arms(self):
        for side in SIDES:
            parent, child = socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
            self.links[side] = parent
            with child:
                self.workers[side] = subprocess.Popen(
                    [str(SDK_PY),"-B",str(HERE/"motion_arm.py"),"--side",side,"--fd",str(child.fileno()),
                     "--gravity-percent",str(self.gravity_percent)],
                    pass_fds=(child.fileno(),),stdin=subprocess.DEVNULL)
            parent.setblocking(False)
        self.phase, self.since = "checking", time.monotonic()

    def authorize_home(self):
        if (self.phase != "inspected" or self.locked.is_set()
                or self.checked != set(SIDES)):
            return False
        self.check_inspection_feedback()
        self.phase, self.since = "loading", time.monotonic()
        return True

    def background(self, fn, *args):
        job = threading.Thread(target=fn, args=args, daemon=True)
        self.jobs.append(job)
        job.start()

    def reset_blocker(self):
        if not self.locked.is_set(): return "请先停止本次会话"
        if any(p.poll() is None for p in self.workers.values()) or any(
                link.fileno() >= 0 for link in self.links.values()):
            return "等待双臂进程退出及停机记录收齐"
        if self.motion_requested and self.disabled != set(SIDES):
            return "双臂失能未确认，保持锁定，需现场处置"
        if any(t.is_alive() for t in self.jobs) or (self.camera and self.camera.poll() is None):
            return "等待旧任务和相机退出"
        if self.camera and self.camera.returncode != 0:
            for port in ("1","2.2","2.3","2.4"):
                for interface in (0,1):
                    driver=Path(f"/sys/bus/usb/devices/3-{port}:1.{interface}/driver")
                    if not driver.exists() or driver.resolve().name!="uvcvideo":
                        return "相机未正常退出，等待 UVC 接口恢复"
        return None

    def new_idle_session(self):
        if self.reset_blocker() is not None: return None
        if self.tmp: self.tmp.cleanup()
        if hasattr(self,"ownership"): self.ownership.close()
        return MotionSession(self.gravity_percent)

    def check_inspection_feedback(self):
        now = time.time_ns()
        with self.guard:
            for side in SIDES:
                samples = self.samples[side]
                if not samples or not 0 <= now-samples[-1][0] < 250_000_000:
                    raise RuntimeError(f"{side} 未使能反馈过期")

    def send(self, side, **msg):
        self.links[side].send(json.dumps(msg,allow_nan=False).encode())

    def stop(self, reason):
        if self.locked.is_set():
            return
        phase, stamp = self.phase, time.time_ns()
        self.stop_reason = reason
        self.locked.set()
        self.phase = "locked"
        for side, link in self.links.items():
            try:
                self.send(side,kind="stop")
            except OSError:
                pass
            try:
                link.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        if self.recorder: self.recorder.finish(stamp,reason)
        self.record(stamp, "pc", {"kind":"stop", "phase":phase, "reason":reason,
                                  "trial_directory":str(self.recorder.path) if self.recorder else None,
                                  "demo_episode":DEMO_ID, "demo_frame":DEMO_FRAME, "left_j5_prediction_bound_rad":LEFT_J5_PREDICTION_BOUND,
                                  "recent_dispatched_targets":list(self.recent_targets)})
        if self.camera is not None and self.camera.poll() is None:
            self.camera.terminate()

    def record(self, stamp, side, message):
        if self.stop_log is None: return
        try:
            with self.stop_log.open("a", encoding="utf-8") as log:
                print(json.dumps({"time_ns":stamp, "side":side, **message},
                                 ensure_ascii=False, allow_nan=False),file=log)
        except (OSError, ValueError, TypeError) as exc:
            print(f"无法保存停机记录：{exc}",file=sys.stderr)

    def begin(self):
        if self.phase != "ready" or self.locked.is_set() or self.ready != set(SIDES):
            return False
        self.permit_after = time.time_ns()
        if self.recorder and not self.recorder.start(self.permit_after):
            self.stop("无法启动本次实验记录")
            return False
        self.phase = "running"
        return True

    def move_to_demo(self):
        if self.phase != "home_ready" or self.home_ready != set(SIDES) or self.locked.is_set():
            return False
        self.phase, self.since = "approach", time.monotonic()
        try:
            for side in SIDES:
                self.send(side,kind="task_start")
        except OSError as exc:
            self.stop(f"示范起点请求发送失败：{exc}")
            return False
        return True

    def accept(self, anchor, pred, feedback, observation_ns):
        if self.locked.is_set(): return
        if not 0 <= time.time_ns()-anchor < 100_000_000:
            raise RuntimeError(f"策略首步已过期：年龄 {(time.time_ns()-anchor)/1e6:.1f} ms")
        if not 0 <= anchor - observation_ns <= PLAN_AGE_NS:
            raise RuntimeError("π0 动作块时间戳异常或过期")
        self.first_prediction = True
        if self.phase != "running" or observation_ns <= self.permit_after: return
        now = time.time_ns()
        if self.recorder:
            self.recorder.put(dict(kind="prediction",anchor_ns=anchor,observation_ns=observation_ns,q_target=pred,q_obs=feedback))
        for i, side in enumerate(SIDES):
            sl = slice(8*i,8*i+8)
            try:
                accept_target(pred[sl],feedback[sl],self.limits[side],anchor,now,
                              side=side)
            except ValueError as exc:
                raise ValueError(f"{side}: {exc}") from exc
        for i, side in enumerate(SIDES):
            self.send(side,kind="target",stamp=anchor,q=pred[i*8:(i+1)*8])
        self.recent_targets.append({"anchor_ns":anchor, "observation_ns":observation_ns, "q_target":list(pred), "q_obs":list(feedback)})
        if self.recorder: self.recorder.put(dict(kind="sent_to_workers",anchor_ns=anchor))

    def capture_observation(self, anchor, frames, q, arms):
        r = self.recorder
        if not r: return
        waypoint=self.phase=="resync" and not r.waypoint_saved
        if not waypoint and (r.start_ns is None or anchor<=r.start_ns): return
        row=dict(kind="observation",anchor_ns=anchor,q_obs=q.tolist(),
                 feedback_ns={s:t for s,(t,_) in arms.items()},
                 camera_ns={f.metadata["camera"]:f.metadata["timestamp_ns"] for f in frames})
        if waypoint: r.waypoint(row,tuple(frames))
        else: r.put(row,tuple(frames))

    def inputs(self):
        for side, link in self.links.items():
            if link.fileno() < 0: continue
            for _ in range(64):
                try: packet = link.recv(8192)
                except BlockingIOError: break
                if not packet:
                    link.close()
                    if not self.locked.is_set(): self.stop(f"{side} 控制进程断开")
                    break
                m = json.loads(packet)
                k = m["kind"]
                if k=="left_j5_trace":
                    self.record(time.time_ns(),side,m)
                if k in ("fault","unconfirmed","disabled"):
                    self.stop(side+": "+m.get("reason",k))
                    # Record both sides, even if the peer's message latched the UI first.
                    self.record(time.time_ns(), side, m)
                if k in ("feedback","checked"):
                    with self.guard:
                        self.samples[side].append((m["stamp"],m["q"]))
                    if "command" in m and self.recorder:
                        self.recorder.put(dict(kind="driver",side=side,anchor_ns=m["anchor_ns"],
                              feedback_ns=m["stamp"],dispatch_ns=m["dispatch_ns"],
                              q_command=m["command"],q_obs=m["q"],
                              scale=m.get("scale"),tau_ff_nm=m.get("tau_ff_nm")))
                if k=="checked":
                    import numpy as np
                    self.limits[side]=np.asarray(m["limits"],dtype=float)
                    self.checked.add(side)
                    if not self.locked.is_set():
                        yield side, "8轴新回包均为未使能"
                if k=="enabled" and not self.locked.is_set(): yield side, "8轴已确认使能"
                if k=="ready": self.ready.add(side)
                if k=="home_ready": self.home_ready.add(side)
                if k=="enabled": self.enabled.add(side)
                if k in ("disabled","unconfirmed"):
                    if k=="disabled": self.disabled.add(side)
                    yield side, ("8轴回报未使能（非停稳证明）" if k=="disabled" else "未确认："+m["reason"])

    def advance(self):
        if self.locked.is_set(): return
        for side in self.links:
            if side in self.checked:
                self.send(side,kind="ping",mono=time.monotonic())
        elapsed = time.monotonic()-self.since
        if self.phase=="checking" and self.checked==set(SIDES):
            self.phase="inspected"
        if self.phase=="inspected":
            self.check_inspection_feedback()
        elif self.phase=="preflight" and self.checked==set(SIDES) and self.first_prediction:
            self.motion_requested=True
            for side in SIDES: self.send(side,kind="home",confirm="START")
            self.phase,self.since="enabling",time.monotonic()
        elif self.phase=="enabling" and self.enabled==set(SIDES):
            for side in SIDES: self.send(side,kind="release_home")
            self.phase,self.since="homing",time.monotonic()
        elif self.phase=="homing" and self.home_ready==set(SIDES):
            self.phase="home_ready"
        elif self.phase in ("checking","loading","preflight","enabling","homing","approach","resync") and elapsed > {"checking":8,"loading":60,"preflight":8,"enabling":3,"homing":10,"approach":10,"resync":3}[self.phase]:
            raise TimeoutError("准备/回位超时")


def load_motion_model():
    return Pi0Policy()


def launch(gravity_percent=DEFAULT_PERCENT):
    # CUDA graph capture can hold the GIL longer than the arm heartbeat lease,
    # even on a background thread. Finish it before the GUI can create workers.
    print("正在预热本地策略，此时不打开机械臂或相机……", flush=True)
    model_error = None
    try:
        motion_model = load_motion_model()
    except Exception as exc:
        motion_model = None
        model_error = f"模型预热失败，只能查状态：{exc}"
        print(model_error, file=sys.stderr, flush=True)
    camdir = CAMERA_DIR
    sys.path.insert(0,str(camdir))
    from bridge import Client
    from contextlib import closing

    s = MotionSession(gravity_percent=gravity_percent)
    root=tk.Tk()
    root.title("OpenArm V2 · 真机运动")
    status=tk.StringVar(value=model_error or "待机：策略预热完成，没有打开 CAN，不会自动运动。")
    arms={side:tk.StringVar(value=side+"：未连接") for side in SIDES}
    record_enabled=tk.BooleanVar(value=True)
    tk.Label(root,text=f"真机运动 · OpenArm V2 · {RELEASE.arch}",font=("",18,"bold")).pack(pady=7)
    tk.Label(root,text=f"原 PD · J1–J7 首条位置指令即使用 {gravity_percent}% 模型重力（限幅）").pack()
    tk.Label(root,text=f"左 J5 原始预测偏差上限 {LEFT_J5_PREDICTION_BOUND:.2f} rad；电机指令限速不变。").pack()
    tk.Label(root,text="①只查询。点击②才授权使能和移动；复位不会回位。\n保持工作区净空，实体急停可及。失能可能导致下落。").pack()
    tk.Label(root,textvariable=status,wraplength=540).pack()
    record_box=tk.Checkbutton(root,text="④起记录拉抽屉实验（四路画面＋关节，保存至本仓库 logs）",
                              variable=record_enabled)
    record_box.pack(anchor="w")
    recording_status=tk.StringVar(value="")
    tk.Label(root,textvariable=recording_status,wraplength=620).pack(anchor="w")
    for a in arms.values(): tk.Label(root,textvariable=a).pack(anchor="w")
    def stop(reason="操作员停止"):
        s.stop(reason)
        status.set("已锁定："+str(s.stop_reason)+"；检查两臂状态。")
        for button in (check, home, demo, begin): button.configure(state="disabled")

    def wait_camera(path):
        try:
            until=time.monotonic()+35
            while not s.locked.is_set():
                if s.camera is not None and s.camera.poll() is not None:
                    raise RuntimeError("相机服务已退出，请检查设备授权和终端日志")
                if path.is_socket():
                    with closing(Client(path)) as stream:
                        h=stream.health()
                    if h.get("error") or h.get("mock") is not False: raise ValueError("真实相机服务异常")
                    if h.get("ready") and h.get("basic_image_ready"): break
                if time.monotonic()>until: raise TimeoutError("相机未就绪")
                time.sleep(.08)
            if s.locked.is_set(): return
            s.events.put(("camera_ready",path))
        except Exception as exc:
            s.events.put(("fault",str(exc)))

    def inspect_arms():
        if s.phase != "idle": return
        try:
            s.acquire()
            record_box.configure(state="disabled")
            if record_enabled.get():
                trial_meta = {
                    "stop_log":str(s.stop_log), "demo_episode":DEMO_ID, "demo_frame":DEMO_FRAME,
                    "checkpoint":str(RELEASE.checkpoint),
                    "arm_gravity":gravity_description(s.gravity_percent)}
                s.recorder=TrialRecording.prepare(s.stop_log.stem, trial_meta)
                s.jobs.append(s.recorder.thread)
                recording_status.set("记录已准备："+s.recorder.path.name)
            s.launch_arms()
            check.configure(state="disabled")
        except Exception as exc: stop(str(exc))

    def camera_inventory(path):
        try:
            subprocess.run([sys.executable,"-B","-c",
                "import sys;sys.path.insert(0,sys.argv[1]);from check_four_cameras import inventory;inventory()",
                str(SOURCE_LOCAL/"camera_bandwidth")],check=True,timeout=5)
            s.events.put(("launch_camera",path))
        except Exception as exc:
            s.events.put(("fault",str(exc)))

    def prepare():
        try:
            if motion_model is None:
                status.set(model_error); return
            if not s.authorize_home():
                status.set("需两侧未使能反馈检查通过");return
            path=Path.home()/".cache/openarm-dp-field/camera.sock"
            if not path.is_socket():
                if not Path("/usr/local/lib/openarm-dp/uvc_pair_probe").is_file():
                    raise RuntimeError("未安装设备授权：请先按原 DP 工作台安装相机设备授权")
                s.tmp=tempfile.TemporaryDirectory(prefix="dp-motion-camera-")
                path=Path(s.tmp.name)/"camera.sock"
                s.background(camera_inventory,path)
            else:
                s.background(wait_camera,path)
            home.configure(state="disabled")
        except Exception as exc: stop(str(exc))

    check=tk.Button(root,text="① 查双臂状态（不运动）",command=inspect_arms)
    check.pack(fill="x",padx=14,pady=4)
    home=tk.Button(root,text="② 确认使能并移动到官方启动位",command=prepare,state="disabled")
    home.pack(fill="x",padx=14,pady=4)
    # Resolve the session on click: reset replaces s but keeps these Tk widgets.
    demo=tk.Button(root,text=f"③ 已核对路径，移动到示范起点（示范 {DEMO_ID} / 帧 {DEMO_FRAME}）",command=lambda: s.move_to_demo(),state="disabled")
    demo.pack(fill="x",padx=14,pady=4)
    begin=tk.Button(root,text="④ 已核对任务场景，开始 π0",command=lambda: s.begin(),state="disabled")
    begin.pack(fill="x",padx=14,pady=4)
    tk.Button(root,text="双臂停止并锁定   [Esc]",bg="#aa2025",fg="white",font=("",18,"bold"),
              command=stop).pack(fill="x",padx=14,pady=6)
    def policy_thread(model,path,once):
        class PreflightComplete(Exception):
            pass
        def measured(ts,p,q,observation_ns):
            s.events.put(("target",(ts,p.tolist(),q.tolist(),observation_ns)))
            if once:
                raise PreflightComplete
        try:
            predict_live(s,path,model,measured,on_observation=s.capture_observation)
        except PreflightComplete:
            pass
        except Exception as exc:
            s.events.put(("fault",f"实时观测/策略推理：{type(exc).__name__}: {exc}"))
    model_path=None
    preflight_started=False
    def reset():
        nonlocal s,model_path,preflight_started
        if closing_window: return
        new=s.new_idle_session()
        if new is None: return
        s=new
        model_path,preflight_started=None,False
        for side in SIDES: arms[side].set(side+"：未连接")
        status.set(model_error or "新会话待机：复位未启动设备，请点①重新检查")
        check.configure(state="normal")
        record_box.configure(state="normal"); recording_status.set("")
        reset_button.configure(state="disabled")
    reset_button=tk.Button(root,text="复位工作台（不运动）",state="disabled", command=reset)
    reset_button.pack(fill="x",padx=14,pady=4)
    def watch_camera(path):
        try:
            with closing(Client(path)) as stream:
                stream.socket.settimeout(.25)
                while not s.locked.wait(.08):
                    h=stream.health()
                    if not h.get("ready") or not h.get("basic_image_ready") or h.get("error"):
                        raise RuntimeError("相机失去有效输入")
        except Exception as exc:
            s.events.put(("fault",f"相机监测：{type(exc).__name__}: {exc}"))
    def tick():
        nonlocal model_path,preflight_started
        try:
            for side,text in s.inputs(): arms[side].set(side+"："+text)
            if not s.locked.is_set():
                while True:
                    try: k,v=s.events.get_nowait()
                    except Empty: break
                    if k=="fault": raise RuntimeError(v)
                    if k=="launch_camera":
                        s.camera=subprocess.Popen([sys.executable,"-B",str(camdir/"service.py"),
                              "--allow-cameras","--socket",str(v),"--max-seconds","180","--privilege","workbench",
                              "--motion-workbench"])
                        s.background(wait_camera,v)
                    if k=="camera_ready":
                        model_path=(motion_model,v)
                        s.background(watch_camera,v)
                        s.phase,s.since="preflight",time.monotonic()
                    if k=="target":
                        s.accept(*v)
                        if s.phase=="resync": s.phase="ready"
                if model_path is not None and s.checked==set(SIDES) and not preflight_started:
                    preflight_started=True
                    s.background(policy_thread,*model_path,True)
                if s.phase=="approach" and s.ready==set(SIDES):
                    s.phase,s.since="resync",time.monotonic()
                    s.background(policy_thread,*model_path,False)
                s.advance()
                if s.phase=="enabling":
                    for side in SIDES:
                        if side not in s.enabled: arms[side].set(side+"：等待使能确认")
        except Exception as exc: stop(str(exc))
        if s.locked.is_set():
            stop(str(s.stop_reason))
            blocker=s.reset_blocker()
            status.set(f"已锁定：{s.stop_reason}；{blocker or '现场确认安全后可复位工作台'}")
            reset_button.configure(state="disabled" if blocker or closing_window else "normal")
        elif s.phase!="idle":
            status.set({"checking":"查询双臂八轴状态，不使能",
                        "inspected":model_error or "双臂八轴均回报未使能。点②才会使能并移动。",
                        "loading":"准备真实相机","preflight":"等待双臂反馈与实时预测",
                        "enabling":"正在确认双臂使能，尚未发送回位目标",
                        "homing":"等待官方位反馈：已到位 " + ("、".join(sorted(s.home_ready)) or "无") + "；两侧到位才开放③",
                        "home_ready":"双臂已到官方启动位，③可用；确认路径净空再点击",
                        "approach":f"正在限速移动到 episode {DEMO_ID} 示范起点","resync":"已到示范起点，重新对齐观测",
                        "ready":"已到示范起点；核对抽屉、方块和双臂后才点④",
                        "running":"π0 正在控制双臂"}[s.phase])
            home.configure(state="normal" if s.phase=="inspected" and motion_model is not None else "disabled")
            demo.configure(state="normal" if s.phase=="home_ready" else "disabled")
            begin.configure(state="normal" if s.phase=="ready" else "disabled")
        if s.recorder:
            r=s.recorder
            if r.error: recording_status.set("实验记录不完整："+r.error)
            elif r.dropped: recording_status.set("实验记录有丢样："+str(dict(r.dropped)))
            elif r.start_ns is not None:
                recording_status.set(("记录中：" if not r.closed.is_set() else
                                     "正在保存：" if r.thread.is_alive() else "已保存：")+str(r.path))
        root.after(15,tick)
    closing_window=False
    def close():
        nonlocal closing_window
        stop("窗口关闭")
        if closing_window: return
        closing_window=True
        def wait():
            if any(p.poll() is None for p in s.workers.values()) or (s.camera and s.camera.poll() is None) or (s.recorder and s.recorder.thread.is_alive()):
                root.after(100,wait)
            else:
                if s.tmp: s.tmp.cleanup()
                root.destroy()
        wait()
    root.bind("<Escape>",lambda e:stop())
    root.protocol("WM_DELETE_WINDOW",close)
    try:
        tick();root.mainloop()
    finally:
        if motion_model is not None:
            motion_model.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--trial-left-j5",action="store_true",help="兼容旧命令；左 J5 默认已为 0.50 rad")
    parser.add_argument("--no-gravity",action="store_true",help="保留原 PD，零前馈对照")
    args=parser.parse_args()
    launch(0 if args.no_gravity else DEFAULT_PERCENT)
