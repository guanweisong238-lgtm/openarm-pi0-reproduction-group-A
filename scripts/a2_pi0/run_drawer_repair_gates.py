"""Screen one immutable checkpoint on training starts, then require a complete valid 33-start run."""
import argparse,hashlib,json,subprocess,sys,os
from pathlib import Path
from drawer_success_training_config import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weight-precision',choices=('bfloat16','checkpoint'),default='checkpoint')
    args=p.parse_args();args.checkpoint=args.checkpoint.resolve();args.output.mkdir(parents=True,exist_ok=True)
    protocol_hash=hashlib.sha256((args.checkpoint/'input_protocol.json').read_bytes()).hexdigest()
    report=dict(checkpoint=str(args.checkpoint),checkpoint_protocol_sha256=protocol_hash,
                target_achieved=False,stages=[],protocol='docs/drawer-repair-eval-protocol.md')
    for name,ids in [('gate_a',[12,10]),('gate_b',[9,11]),('full_33',list(range(13))+list(range(100,120)))]:
        out=args.output/name;index=out/'index.json'
        print('GATE_START',name,str(args.checkpoint),flush=True)
        if not index.exists():
            cmd=[sys.executable,str(ROOT/'scripts/a2_pi0/run_drawer_eval_isolated.py'),
                 '--checkpoint',str(args.checkpoint),'--weight-precision',args.weight_precision,'--output',str(out),'--episode-ids',*map(str,ids)]
            with (args.output/f'{name}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,DRAWER_EVAL_GL='glfw',MUJOCO_GL='glfw'))
        rows=json.loads(index.read_text());cfg=json.loads((out/'run_config.json').read_text())
        if (len(rows)!=len(ids) or {r['episode_id'] for r in rows}!=set(ids)
            or cfg['checkpoint_protocol_sha256']!=protocol_hash or Path(cfg['checkpoint']).resolve()!=args.checkpoint):
            raise RuntimeError('Incomplete or mismatched checkpoint evidence')
        expected=dict(seconds=24.,task='open',seed=42,action_chunk=4,gripper_mode='continuous',repeat=1,step_limit=720,gl_backend='glfw',weight_precision=args.weight_precision)
        if cfg['run']!=expected:raise RuntimeError('Evaluation runtime differs from frozen repair protocol')
        passed=all(r['success'] and r['evidence_valid'] for r in rows)
        result=dict(name=name,trials=len(rows),success=sum(r['success'] and r['evidence_valid'] for r in rows),passed=passed,path=str(out))
        report['stages'].append(result)
        print('GATE_DONE',json.dumps(result),flush=True)
        if name=='full_33':
            subprocess.run([sys.executable,str(ROOT/'scripts/a2_pi0/summarize_drawer_eval.py'),'--run',str(out)],check=True)
            report['target_achieved']=passed
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not passed:break
    print('CANDIDATE_DONE',json.dumps(report),flush=True)


if __name__=='__main__':main()
