"""Motion tests use local socket pairs and fake motor drivers only."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import threading
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import openarm_driver

from live_motion import MotionSession, launch
from live_arm_probe import MotorStatusMismatch
from task_reference import DEMO_ID, DEMO_FRAME, start_row
from motion_arm import accept_target, await_enable, ArrivalWindow, Control, policy_step, run, task_start_pose


class GuardTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("DISPLAY"), "requires Tk display")
    def test_actual_buttons_after_in_place_reset(self):
        import gc
        import tkinter as tk
        # Finalize Tcl-related cycles here, not in a later simulated motor thread.
        self.addCleanup(gc.collect)
        root=tk.Tk()
        root.withdraw()
        old, new = MotionSession(), MotionSession()
        new.send=Mock()
        def next_tick():
            pulse=tk.BooleanVar(master=root)
            root.after(40,lambda:pulse.set(True))
            root.wait_variable(pulse)
        def exercise():
            buttons=[w for w in root.winfo_children() if isinstance(w,tk.Button)]
            def click(prefix):
                b=next(w for w in buttons if w.cget("text").startswith(prefix))
                self.assertNotEqual(str(b.cget("state")),"disabled")
                b.invoke()
            click("双臂停止"); next_tick(); click("复位")
            self.assertTrue(old.locked.is_set())
            new.phase="home_ready"; new.home_ready.update(("left","right"))
            next_tick(); click("③")
            self.assertEqual(new.phase,"approach")
            self.assertEqual([c.kwargs for c in new.send.call_args_list],[{"kind":"task_start"}]*2)
            new.phase="ready"; new.ready.update(("left","right"))
            next_tick(); click("④"); self.assertEqual(new.phase,"running")
        try:
            with patch("live_motion.tk.Tk",return_value=root),patch.object(root,"mainloop",exercise), \
                 patch("live_motion.load_motion_model",return_value=Mock()), \
                 patch("live_motion.MotionSession",side_effect=[old,new]), \
                 patch("live_motion.subprocess.Popen",side_effect=AssertionError) as proc:
                launch()
                proc.assert_not_called()
        finally:
            root.destroy()

    def test_default_left_j5_prediction_envelope(self):
        for i, delta, passes in [(4,.4127,True),(4,.50,True),(4,-.50,True),
                                 (4,.5001,False),(12,.4127,False),(3,.4127,False)]:
            with self.subTest(i=i,delta=delta):
                s=MotionSession(); s.phase="running"; s.permit_after=0; s.send=Mock()
                s.limits={side:np.tile([-2.,2.],(8,1)) for side in ("left","right")}
                q=[0.]*16; q[i]=delta
                now=time.time_ns()
                if passes:
                    s.accept(now,q,[0.]*16,now)
                    self.assertEqual(s.send.call_count,2)
                else:
                    with self.assertRaises(ValueError): s.accept(now,q,[0.]*16,now)
                    s.send.assert_not_called()

    def test_small_reversals_softened_without_overshoot_or_faster_large_moves(self):
        zero = np.zeros(8); last = zero.copy()
        for target in (.03, -.03, .03, -.03, .8, -.8, 0):
            target = zero + target
            actual = policy_step(last, target, .02)
            raw_step = np.clip(target-last, -np.array([.30]*7+[.50])*.02,
                               np.array([.30]*7+[.50])*.02)
            self.assertTrue((np.abs(actual-last) <= np.abs(raw_step)+1e-12).all())
            self.assertTrue(((actual-last)*(target-last) >= 0).all())
            if np.max(np.abs(target)) == .03:
                self.assertTrue((np.abs(actual-last) < np.abs(raw_step)).all())
            last = actual
        np.testing.assert_array_equal(policy_step(last, last, .02), last)

    def test_pc_stop_saves_first_cause_after_notifying_both_arms(self):
        s=MotionSession(); s.phase="running"; s.permit_after=0
        s.links={side:Mock() for side in ("left","right")}
        s.limits={side:np.tile([-2.,2.],(8,1)) for side in s.links}
        with tempfile.TemporaryDirectory() as d:
            s.stop_log=Path(d)/"stop.jsonl"
            s.accept(time.time_ns(), [.01]*16, [0.]*16, time.time_ns()-100_000)
            with self.assertRaisesRegex(ValueError,"right: .*J4") as exc:
                s.accept(time.time_ns(), [0.]*11+[.59]+[0.]*4, [0.]*16, time.time_ns()-100_000)
            for link in s.links.values(): self.assertEqual(link.send.call_count,1)
            s.stop(str(exc.exception)); s.stop("later")
            entries=[json.loads(line) for line in s.stop_log.read_text().splitlines()]
            self.assertEqual(len(entries),1)
            self.assertEqual((entries[0]["side"],entries[0]["phase"]),("pc","running"))
            self.assertIn("0.5900",entries[0]["reason"])
            self.assertEqual(len(entries[0]["recent_dispatched_targets"]),1)
            for link in s.links.values():
                self.assertEqual(json.loads(link.send.call_args.args[0]),{"kind":"stop"})

    def test_model_preheated_before_session_or_workers_exist(self):
        events = []
        class PanelReady(Exception): pass
        def session(**kwargs):
            events.append("session")
            raise PanelReady
        with patch("live_motion.load_motion_model", side_effect=lambda:events.append("preheat")), \
             patch("live_motion.MotionSession", side_effect=session), \
             patch("live_motion.subprocess.Popen") as worker:
            with self.assertRaises(PanelReady): launch()
        self.assertEqual(events, ["preheat", "session"])
        worker.assert_not_called()

    def test_inspection_requires_manual_home_and_fresh_disabled_feedback(self):
        s = MotionSession(); s.send = Mock()
        s.phase, s.since = "checking", time.monotonic()
        s.first_prediction = True  # Even an available prediction cannot authorize motion.
        for side in ("left", "right"):
            self.assertFalse(s.authorize_home())
            s.checked.add(side)
            s.samples[side].append((time.time_ns(), [0.]*8))
            s.advance()
        for _ in range(3): s.advance()
        s.send.assert_not_called()
        self.assertEqual(s.phase, "inspected")
        self.assertFalse(s.begin())
        s.samples["right"].append((time.time_ns()-300_000_000, [0.]*8))
        with self.assertRaisesRegex(RuntimeError, "right"): s.authorize_home()
        s.samples["right"].append((time.time_ns(), [0.]*8))
        self.assertTrue(s.authorize_home())
        self.assertFalse(s.authorize_home())
        self.assertEqual(s.phase, "loading")
        s.send.assert_not_called()
        s.stop("red")
        self.assertFalse(s.authorize_home())

    def test_manual_reset_is_fresh_and_does_not_revive_old_work(self):
        s = MotionSession()
        s.motion_requested=True; s.stop("故障")
        s.disabled.add("left")
        self.assertIsNone(s.new_idle_session())
        s.disabled.add("right")
        job=Mock(); job.is_alive.return_value=True; s.jobs.append(job)
        self.assertIsNone(s.new_idle_session())
        job.is_alive.return_value=False
        s.events.put(("target","old"))
        n=s.new_idle_session()
        self.assertTrue(s.locked.is_set())
        self.assertEqual(n.phase,"idle")
        self.assertTrue(n.events.empty())
        self.assertFalse(n.authorize_home())
        self.assertFalse(n.links or n.workers)

    def test_settling_only_extends_bounded_measurable_progress(self):
        pose=np.zeros(8)
        def sample(window, t, moving, stuck=0.):
            q=pose.copy(); q[:2]=[moving,stuck]; window.check(t,q,pose)
        w=ArrivalWindow(0,2)
        for i in range(80):
            t=i*.05
            sample(w,t,.20-.025*t)
        with self.assertRaisesRegex(RuntimeError,"超时"):
            sample(w,4.01,.099)
        w=ArrivalWindow(0,2)
        for i in range(41): sample(w,i*.05,.20-i*.001,.12)
        with self.assertRaisesRegex(RuntimeError,"未持续趋近"):
            sample(w,2.05,.158,.12)


    def test_enable_transition_and_faults(self):
        pending = MotorStatusMismatch(1, [1]*7+[0])
        read = Mock(side_effect=[pending, {"q":[0]*8}])
        self.assertEqual(await_enable(read,lambda:None)["q"],[0]*8)
        for err in (MotorStatusMismatch(1,[1]*7+[9]), ValueError("missing fresh replies")):
            read = Mock(side_effect=err)
            with self.assertRaises(ValueError): await_enable(read,lambda:None)
            read.assert_called_once()
        with self.assertRaises(MotorStatusMismatch):
            await_enable(Mock(side_effect=[pending,MotorStatusMismatch(1,[0]*8)]),lambda:None)
        clock = Mock(side_effect=[0., .16])
        with self.assertRaisesRegex(TimeoutError,"夹爪"):
            await_enable(Mock(side_effect=pending),lambda:None,clock=clock)
        with self.assertRaisesRegex(RuntimeError,"stop"):
            await_enable(Mock(side_effect=pending),Mock(side_effect=[None,RuntimeError("stop")]))

    def test_only_both_enabled_release_homing(self):
        s = MotionSession(); s.send = Mock()
        s.phase = "preflight"; s.checked=set(("left","right")); s.first_prediction=True
        s.advance()
        self.assertEqual(s.phase,"enabling")
        s.enabled.add("left")
        s.send.reset_mock(); s.advance(); s.send.assert_not_called()
        s.enabled.add("right"); s.advance()
        self.assertEqual(s.phase,"homing")
        self.assertEqual([c.kwargs for c in s.send.call_args_list],[{"kind":"release_home"}]*2)

    def test_start_requires_both_ready_and_a_post_click_prediction(self):
        s=MotionSession()
        s.send=Mock()
        self.assertFalse(s.begin())
        s.phase="ready"
        s.ready.add("left")
        self.assertFalse(s.begin())
        s.ready.add("right")
        self.assertTrue(s.begin())
        s.accept(time.time_ns(),[0]*16,[0]*16,s.permit_after)
        s.send.assert_not_called()
        ts=time.time_ns()
        s.limits={k:np.tile([-2.,2.],(8,1)) for k in ("left","right")}
        p=[.05]*8+[.1]*8
        with self.assertRaises(ValueError): s.accept(ts,[.05]*8+[1.]*8,[0]*16,ts)
        s.send.assert_not_called()
        s.accept(ts,p,[0]*16,ts)
        self.assertEqual([c.kwargs["q"] for c in s.send.call_args_list],[p[:8],p[8:]])
        s.stop("red")
        s.accept(time.time_ns(),p,[0]*16,ts)
        self.assertEqual(s.send.call_count,2)
        self.assertFalse(s.begin())

    def test_demo_pose_matches_both_halves_of_training_observation(self):
        cfg = openarm_driver.Config("openarm_pedestal")
        self.assertEqual((DEMO_ID,DEMO_FRAME),(0,0))
        row = start_row()
        for i, side in enumerate(("left","right")):
            q = task_start_pose(side, cfg.get_joint_limits(side+"_arm"))
            np.testing.assert_array_equal(q, row[8*i:8*i+8])
        with self.assertRaisesRegex(ValueError, "限位"):
            task_start_pose("right", np.zeros((8,2)))

    def test_manual_demo_move_after_both_official_poses(self):
        s = MotionSession(); s.send = Mock()
        s.phase, s.since = "homing", time.monotonic()
        s.home_ready.add("right"); s.advance()
        self.assertFalse(s.move_to_demo())
        s.home_ready.add("left"); s.advance()
        s.send.assert_not_called()
        self.assertFalse(s.begin())
        self.assertTrue(s.move_to_demo())
        self.assertEqual([c.kwargs for c in s.send.call_args_list], [{"kind":"task_start"}]*2)
        self.assertFalse(s.move_to_demo())
        self.assertFalse(s.begin())

    def test_34_degree_jump_and_expired_sample_rejected(self):
        q=np.zeros(8); limits=np.tile([-2.,2.],(8,1)); t=time.time_ns()
        for x, age in (([0,0,0,.589,0,0,0,0],0),(q,100_000_000),([np.nan]*8,0),([3]*8,0)):
            with self.assertRaises(ValueError): accept_target(x,q,limits,t-age,t)

    def test_disconnected_or_stalled_parent(self):
        a,b=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
        b.close()
        with a:
            with self.assertRaises(RuntimeError): Control(a).check()
        a,b=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
        with a,b:
            c=Control(a); c.last_ping-=2
            with self.assertRaisesRegex(RuntimeError,"heartbeat"): c.check()

    def driver_rehearsal(self, stop_in_home=False, stop_before_release=False, stop_in_approach=False,
                         side="left", stalled=False, trial=False, gravity_percent=100):
        moved,disabled,enabled=[],[],[]
        feedforward = []
        start_events = []
        cfg = openarm_driver.Config("openarm_pedestal")
        preset=np.asarray(cfg.get_start_config()["moves"][-1]["position"][side+"_arm"])
        demo=task_start_pose(side,cfg.get_joint_limits(side+"_arm"))
        prediction=demo+.01
        if trial: prediction[4]=demo[4]+.4127
        class Driver(openarm_driver.SingleArmDriver):
            def __init__(self, side, cfg):
                self.openarm=Mock(); self.joint_offsets=np.zeros(8)
                self.openarm.enable_all.side_effect=lambda:(enabled.append(True),start_events.append("enable"))
                self.openarm.refresh_all.side_effect=lambda:start_events.append("refresh")
                self.last_command=preset.copy()
                self.latest_state={"qpos":preset.copy()}
            def set_latest_state(self, **_): start_events.append("receive")
            def move_to_start_position(self):
                for _ in range(12):
                    self.send_position(preset)
                    time.sleep(.005)
                return True
            def send_position(self,q):
                feedforward.append(self.openarm.arm.pending[2].copy())
                self.last_command=np.array(q)
                self.last_command_dispatch_timestamp_ns=time.time_ns()
                moved.append(np.array(q))
                return True
            def stop(self): disabled.append(True)
        transition = [MotorStatusMismatch(1,[1]*7+[0])]
        def sample(*args,**kw):
            if len(args)==4 and args[3]==1 and transition:
                self.assertEqual(kw, {"pending":"baseline"})
                start_events.append("verify")
                raise transition.pop()
            return {"q":(moved[-1] if moved and not stalled else preset).tolist(),"stamp":time.time_ns()}
        a,b=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
        with ExitStack() as stack, a:
            stack.enter_context(patch.object(openarm_driver,"SingleArmDriver",Driver))
            stack.enter_context(patch("motion_arm.open_probe",return_value=(Mock(),np.zeros(8))))
            def mark(_):
                start_events.append("mark")
                return "baseline"
            stack.enter_context(patch("motion_arm.before_refresh",side_effect=mark))
            stack.enter_context(patch("motion_arm.fresh",side_effect=sample))
            worker=threading.Thread(target=run,args=(side,b,gravity_percent),daemon=True); worker.start()
            a.settimeout(.5)
            def send(**m): a.send(json.dumps(m).encode())
            issued=ready=stop_sent=trace_seen=False
            deadline=time.monotonic()+5
            try:
                while time.monotonic()<deadline:
                    if not stop_sent: send(kind="ping",mono=time.monotonic())
                    packet=a.recv(8192)
                    if not packet: break
                    m=json.loads(packet)
                    if m["kind"]=="checked":
                        self.assertEqual(moved,[])
                        send(kind="home",confirm="START"); issued=True
                    if m["kind"]=="enabled":
                        self.assertFalse(moved)
                        if stop_before_release:
                            send(kind="stop"); stop_sent=True
                        else:
                            send(kind="release_home")
                    if m["kind"]=="home_ready":
                        np.testing.assert_allclose(moved, np.tile(preset, (len(moved), 1)))
                        send(kind="task_start")
                    if m["kind"]=="ready":
                        np.testing.assert_allclose(moved[-1], demo, atol=.001)
                        ready=True
                        send(kind="target",q=prediction.tolist(),stamp=time.time_ns())
                    if issued and moved and not stop_sent and (stop_in_home or
                            (stop_in_approach and np.any(moved[-1]!=preset)) or
                            (ready and np.any(moved[-1]>demo+.001))):
                        send(kind="stop"); stop_sent=True
                    if m["kind"]=="fault" and stalled:
                        self.assertIn("跟踪偏差过大",m["reason"])
                        self.assertIn("实测=",m["reason"])
                        stop_sent=True  # No further writes while receiving the disable result.
                    if m["kind"]=="disabled" and not trial: break
                    if m["kind"]=="left_j5_trace":
                        trace_seen=True
                        self.assertEqual(disabled,[True])
                        row=m["rows"][0]
                        self.assertEqual(row[3], prediction[4])
                        self.assertLess(abs(row[4]-row[5]), .013)
                        self.assertIsInstance(row[2],int)
                        break
            finally:
                a.shutdown(socket.SHUT_WR); worker.join(1)
            self.assertFalse(worker.is_alive())
            self.assertEqual(trace_seen,trial)
            self.assertTrue(stop_sent)
            self.assertEqual(ready,not (stop_in_home or stop_before_release or stop_in_approach or stalled))
            self.assertEqual(enabled,[True])
            self.assertEqual(start_events,["enable","receive","mark","refresh","receive","verify"])
            self.assertEqual(disabled,[True])
            if moved:
                from vr_gravity import Gravity, CAP
                g = Gravity(side)
                observed = [preset, *(moved[:-1] if not stalled else [preset]*(len(moved)-1))]
                expected = [np.clip(gravity_percent/100*g.torque(q), -CAP, CAP) for q in observed]
                np.testing.assert_allclose(feedforward, expected)
            if stop_before_release:
                self.assertFalse(moved)
            if ready:
                self.assertTrue(any(np.any(v!=preset) for v in moved))
                if not trial: self.assertLessEqual(np.max(np.abs(moved[-1]-demo)),.010001)

    def test_official_path_and_stop_during_execution(self): self.driver_rehearsal()
    def test_stop_during_preset(self): self.driver_rehearsal(True)
    def test_stop_before_dual_arm_release(self): self.driver_rehearsal(stop_before_release=True)
    def test_stop_en_route_to_demonstration(self): self.driver_rehearsal(stop_in_approach=True)
    def test_right_demo_handoff(self): self.driver_rehearsal(side="right")
    def test_zero_gravity_control(self): self.driver_rehearsal(stop_in_home=True, gravity_percent=0)
    def test_default_left_j5_does_not_dispatch_prediction_as_single_step(self):
        self.driver_rehearsal(trial=True)
    def test_right_stalled_approach_rejects_handoff(self): self.driver_rehearsal(side="right",stalled=True)
