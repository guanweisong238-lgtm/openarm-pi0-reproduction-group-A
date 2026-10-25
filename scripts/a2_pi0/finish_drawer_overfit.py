"""等待本次训练完成后，自动重载检查点执行离线和闭环验证，发布小型结果。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def publish(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--training-run",type=Path,required=True)
    parser.add_argument("--training-pid",type=int,required=True)
    args=parser.parse_args()
    output=ROOT/"logs/a2_pi0/drawer_overfit_experiment_001"
    output.mkdir(parents=True,exist_ok=False)
    state={"status":"waiting_for_training","training_run":str(args.training_run),
           "pid":os.getpid(),"training_pid":args.training_pid,"started_at":datetime.now().astimezone().isoformat()}
    def status(**values):
        state.update(values,updated_at=datetime.now().astimezone().isoformat())
        publish(output/"status.json",state)
    try:
        status()
        while not (args.training_run/"summary.json").exists():
            os.kill(args.training_pid,0)
            time.sleep(10)
        training=json.loads((args.training_run/"summary.json").read_text())
        checkpoint=Path(training["checkpoint"])
        if not (checkpoint/"params").is_dir():
            raise FileNotFoundError(checkpoint)
        status(status="evaluation",checkpoint=str(checkpoint))
        print("开始重载检查点并评估：",checkpoint,flush=True)
        with (output/"evaluation.log").open("w") as stream:
            subprocess.run([sys.executable,"scripts/a2_pi0/evaluate_drawer_overfit.py",
                            "--checkpoint",str(checkpoint)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        lines=(output/"evaluation.log").read_text().splitlines()
        directories=[line.split("=",1)[1] for line in lines if line.startswith("EVALUATION_DIRECTORY=")]
        if len(directories)!=1:
            raise RuntimeError("无法识别评估目录")
        evaluation_directory=Path(directories[0])
        evaluation=json.loads((evaluation_directory/"summary.json").read_text())
        fit=[json.loads(line) for line in (args.training_run/"fit_diagnostics.jsonl").read_text().splitlines()]
        result={"experiment":"overfit_roundtrip_001","date":"2026-10-24","training":training,
            "training_run":str(args.training_run),"evaluation_directory":str(evaluation_directory),
            "fit_curve":[{"step":r["step"],"train_flow_loss":r["train_flow_loss"]} for r in fit],
            "offline":{k:v for k,v in evaluation["offline"].items() if k!="samples"},
            "closed_loop":evaluation["closed_loop"],"independent_validation":False,
            "task_success_verified":evaluation["closed_loop"]["success_open_then_close"]}
        target=ROOT/"results/a2_drawer_overfit_001.json"
        publish(target,result)
        status(status="complete",result=str(target),evaluation_directory=str(evaluation_directory))
        print("PASS：过拟合与闭环验证完成，结果=",target,flush=True)
    except Exception as error:
        status(status="failed",error=repr(error))
        raise


if __name__=="__main__":
    main()
