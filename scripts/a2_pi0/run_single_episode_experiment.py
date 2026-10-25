# 本文件负责：顺序运行episode 0的5000步训练、动作趋势检查和整轨迹离线评估。
# 各阶段使用独立子进程，前一阶段失败即停止，避免误用未完成的检查点。
# 模型从pi0_base重新训练，不是恢复1000步实验；旧检查点不覆盖。
# 即时保存状态、命令和日志；只做离线验收，不自动启动仿真或判定抓取成功。

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
NAME = 'overfit_ep0_5000_001'
RUN = ROOT / 'logs/a2_pi0' / NAME
CHECKPOINT = ROOT / 'checkpoints/pi0_openarm_lora' / NAME / '5000'


# 一、可靠记录运行状态，便于中途查询。
def publish(value):
    """通过临时文件替换状态文件，避免读到半个JSON。"""
    temporary = RUN / 'status.tmp'
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(RUN / 'status.json')


def run_stage(stage, arguments, state):
    """同步运行一个阶段，收集完整日志，非零退出码阻止后续阶段。"""
    command = [sys.executable, *arguments]
    state.update(stage=stage, updated_at=datetime.now().astimezone().isoformat())
    state['commands'][stage] = command
    publish(state)
    print('开始阶段:', stage, flush=True)
    with (RUN / f'{stage}.log').open('w') as output:
        subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
    print('完成阶段:', stage, flush=True)


def result_directory(log, marker):
    """从已完成阶段的PASS行读取真实产物目录。"""
    lines = [line for line in log.read_text().splitlines() if line.startswith('PASS') and marker in line]
    if len(lines) != 1:
        raise RuntimeError(f'无法识别阶段产物：{log}')
    return Path(lines[0].rsplit(marker, 1)[1].strip())


# 二、完整实验流水线。
def main():
    """训练并执行固定条件验收，失败时保留错误与已完成产物。"""
    if (ROOT / 'checkpoints/pi0_openarm_lora' / NAME).exists():
        raise FileExistsError('实验检查点已存在，禁止覆盖')
    RUN.mkdir(parents=True, exist_ok=False)
    state = {'status':'running', 'pid':os.getpid(), 'experiment':NAME,
        'started_at':datetime.now().astimezone().isoformat(), 'commands':{},
        'initialization':'pi0_base，重新训练', 'training_episode':0,
        'steps':5000, 'simulator_actions_executed':0}
    try:
        run_stage('training', ['scripts/a2_pi0/train_lora.py', '--train-episode','0',
            '--steps','5000','--exp-name',NAME,'--save-every','2500','--eval-every','500'], state)
        run_stage('trends', ['scripts/a2_pi0/check_action_trends.py',
            '--checkpoint',str(CHECKPOINT)], state)
        run_stage('episode_eval', ['scripts/a2_pi0/evaluate_actions.py',
            '--checkpoint',str(CHECKPOINT),'--split','train','--episode','0',
            '--frames-per-episode','0'], state)
        trends_dir = result_directory(RUN/'trends.log','结果目录:')
        eval_dir = result_directory(RUN/'episode_eval.log','记录=')
        trends = json.loads((trends_dir/'summary.json').read_text())
        evaluation = json.loads((eval_dir/'summary.json').read_text())
        comparison = {'experiment':NAME,'checkpoint':str(CHECKPOINT),
            'episode0_mae_1000_step_baseline':0.0288699,
            'episode0_mae_5000_steps':evaluation['mae_all'],
            'trends_report':str(trends_dir/'summary.json'),
            'episode_evaluation_report':str(eval_dir/'summary.json'),
            'first_frame_motion':[{'seed':r['seed'],
                'right3_predicted_delta':r['predicted_delta'][2],
                'right3_true_delta':r['target_delta'][2],
                'active_displacement_ratio':r['active_displacement_ratio']}
                for r in trends['records'] if r['frame']==0],
            'task_success_verified':False}
        target = ROOT/'results'/f'{NAME}_evaluation.json'
        target.write_text(json.dumps(comparison,indent=2,ensure_ascii=False)+'\n')
        state.update(status='complete',stage='complete',comparison_report=str(target))
    except Exception as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        state['updated_at']=datetime.now().astimezone().isoformat()
        publish(state)


# 三、命令行入口。
if __name__ == '__main__':
    main()
