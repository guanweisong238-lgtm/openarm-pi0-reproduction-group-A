# 本文件负责：诊断真实示教抓取阶段的夹爪闭合预测及闭环观测差异。
# 固定5000步单轨迹模型和噪声，比较原始示教、仿真以及只替换部分输入的结果。
# 混合输入仅用于敏感性分析，不代表真实可执行状态；本脚本不执行仿真动作。
from training_config import ROOT, DATA_ROOT, get_train_config
from local_data_loader import EpisodeSubsetDataset
from infer_sim_snapshot import load_snapshot
import json
from datetime import datetime
import numpy as np
from openpi.policies import policy_config


# 一、提取不包含动作标签的观测。
def make_observation(sample):
    """从示教样本提取三路图像、实际状态和任务指令。"""
    return {'observation.state':np.asarray(sample['observation.state']),
        'prompt':sample['task'],**{f'observation.images.{c}':np.asarray(sample[f'observation.images.{c}'])
        for c in ['ceiling','wrist_left','wrist_right']}}


def main():
    """分别检查闭合起点及输入替换实验，保存夹爪完整16步输出。"""
    run=ROOT.parent/'openarm-a1/outputs/a2_interface/closed_loop_20261024-085628-930828'
    checkpoint=ROOT/'checkpoints/pi0_openarm_lora/overfit_ep0_5000_001/5000'
    dataset=EpisodeSubsetDataset(repo_id='local/openarm_pickup',root=DATA_ROOT,episodes=[0],video_backend='pyav',delta_timestamps={'action':[i/30 for i in range(16)]})
    policy=policy_config.create_trained_policy(get_train_config(),checkpoint,sample_kwargs={'num_steps':10})
    noise=np.random.default_rng(42).standard_normal((16,32)).astype(np.float32)
    records=[]

    # 二、统一评估：不使用夹爪闭合阈值作为任务成功判定。
    def evaluate(frame,label,obs,target):
        """记录右夹爪当前状态、16步预测和示教动作目标。"""
        predicted=np.asarray(policy.infer(obs,noise=noise)['actions'])
        record={'frame':frame,'case':label,'right_gripper_state':float(obs['observation.state'][7]),
            'predicted_gripper':predicted[:,7].tolist(),'target_gripper':target[:,7].tolist(),
            'prediction_first4_mean':float(predicted[:4,7].mean()),'target_first4_mean':float(target[:4,7].mean()),
            'prediction_last':float(predicted[-1,7])}
        records.append(record)
        print(frame,label,'state=',round(record['right_gripper_state'],4),'pred_first4=',round(record['prediction_first4_mean'],4),'target=',round(record['target_first4_mean'],4),flush=True)

    # 三、使用完全真实的示教输入，检查模型是否学会闭合过渡。
    for frame in [90,96,100,104,108,110,112,116,120,150]:
        sample=dataset[frame]
        evaluate(frame,'demo',make_observation(sample),np.asarray(sample['action']))

    # 四、在动作过渡处替换输入，区分图像、整体状态和夹爪状态的影响。
    for frame in [112,120]:
        sample=dataset[frame];demo=make_observation(sample);target=np.asarray(sample['action'])
        sim,_,_=load_snapshot(run/f'cycle_{frame//4:03d}/observation.npz')
        evaluate(frame,'sim',sim,target)
        evaluate(frame,'demo_images_sim_state',{**demo,'observation.state':sim['observation.state']},target)
        evaluate(frame,'sim_images_demo_state',{**sim,'observation.state':demo['observation.state']},target)
        state=demo['observation.state'].copy();state[7]=sim['observation.state'][7]
        evaluate(frame,'demo_except_gripper_state',{**demo,'observation.state':state},target)
        state=sim['observation.state'].copy();state[7]=demo['observation.state'][7]
        evaluate(frame,'sim_except_gripper_state',{**sim,'observation.state':state},target)
    output=ROOT/'logs/a2_pi0'/f'gripper_transition_{datetime.now():%Y%m%d-%H%M%S-%f}'
    output.mkdir(parents=True)
    report={'checkpoint':str(checkpoint),'source_run':str(run),'records':records,'simulation_actions_executed':0,
        'limitation':'混合输入是离线敏感性实验，不能直接据此认定因果或任务成功'}
    (output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('PASS:',output,flush=True)


# 五、命令行入口。
if __name__=='__main__':
    main()
