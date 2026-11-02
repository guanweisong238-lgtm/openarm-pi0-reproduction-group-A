"""Run the next authorized repair curriculum and strict checkpoint screening without interactive prompts."""
import json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
STATE=ROOT/'results/drawer_repair_20261101/round020.json'


def save(state):
    STATE.write_text(json.dumps(state,indent=2)+'\n')


def main():
    state=dict(status='waiting_for_precision_controls',target_achieved=False,
        training_data='data/drawer_feedback_queries/v005_randomized_rehearsal',
        experiment='feedback_endpoint_020_randomized_rehearsal',candidates=[])
    save(state)
    prior=ROOT/'logs/a2_pi0/repair_screen_016_native/report.json'
    while True:
        if prior.exists():
            report=json.loads(prior.read_text())
            if report['target_achieved']:
                state.update(status='target_achieved_by_precision_control',target_achieved=True,accepted=report);save(state);return
            if any(not r['passed'] for r in report['stages']):break
        time.sleep(5)
    checkpoint_root=ROOT/'checkpoints/pi0_openarm_drawer_endpoint_v001'/state['experiment']
    if not (checkpoint_root/'6000/input_protocol.json').exists():
        state['status']='training';save(state);print('ROUND020_TRAIN_START',flush=True)
        cmd=[sys.executable,str(ROOT/'scripts/a2_pi0/train_drawer_feedback_endpoint.py'),
             '--data',str(ROOT/state['training_data']),
             '--checkpoint',str(ROOT/'checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_016/2000'),
             '--exp-name',state['experiment'],'--steps','6000','--lr','1e-5','--eval-every','500',
             '--endpoint-path','cached','--cartesian-weight','.5','--gripper-weight','4','--save-every','2000']
        subprocess.run(cmd,cwd=ROOT,check=True)
    state['status']='screening';save(state)
    for step in [2000,4000,6000]:
        checkpoint=checkpoint_root/str(step);out=ROOT/f'logs/a2_pi0/repair_screen_020_{step}'
        subprocess.run([sys.executable,str(ROOT/'scripts/a2_pi0/run_drawer_repair_gates.py'),
                        '--checkpoint',str(checkpoint),'--weight-precision','checkpoint','--output',str(out)],cwd=ROOT,check=True)
        report=json.loads((out/'report.json').read_text());state['candidates'].append(report);save(state)
        if report['target_achieved']:
            state.update(status='target_achieved',target_achieved=True,accepted=report);save(state)
            (STATE.parent/'accepted_model.json').write_text(json.dumps(report,indent=2)+'\n')
            print('REPAIR_TARGET_ACHIEVED',str(checkpoint),flush=True);return
    state['status']='round_finished_target_not_yet_achieved';save(state)
    print('ROUND020_FINISHED_TARGET_NOT_YET_ACHIEVED',flush=True)


if __name__=='__main__':main()
