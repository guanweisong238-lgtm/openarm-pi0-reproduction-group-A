# 本文件负责：用原始示教观测检查模型完整动作块的方向、幅度及多噪声稳定性。
# 不把动作标签传给策略，不更新参数，不连接仿真或执行动作。
# 对每个起点输出16维误差、首尾变化与有效运动维度的方向符合率。
# 方向比较仅用于真实变化超过0.03rad的维度；不能替代完整轨迹或闭环成功评估。

# 一、先配置项目环境，再导入推理和数据依赖。
from training_config import ROOT, DATA_ROOT, get_train_config
from local_data_loader import EpisodeSubsetDataset

import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from openpi.policies import policy_config


# 二、动作趋势指标。
def compare_trend(prediction, target):
    """对有效动作计算误差和首尾位移，剔除几乎静止维度的方向评价。"""
    predicted_delta = prediction[-1] - prediction[0]
    target_delta = target[-1] - target[0]
    active = np.abs(target_delta) > 0.03
    return {
        'mae': float(np.abs(prediction-target).mean()),
        'first4_mae': float(np.abs(prediction[:4]-target[:4]).mean()),
        'predicted_delta': predicted_delta.tolist(),
        'target_delta': target_delta.tolist(),
        'active_dimensions': np.flatnonzero(active).tolist(),
        'direction_agreement': float(np.mean(np.sign(predicted_delta[active]) == np.sign(target_delta[active]))) if active.any() else None,
        'active_displacement_ratio': float(np.abs(predicted_delta[active]).sum()/np.abs(target_delta[active]).sum()) if active.any() else None,
        'right3_prediction': prediction[:,2].tolist(),
        'right3_target': target[:,2].tolist(),
    }


# 三、读取示教观测与重复推理。
def main():
    """在指定训练轨迹的多个起点检查动作趋势，并保存数值证据。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/pi0_openarm_lora/overfit_ep0_001/1000')
    parser.add_argument('--episode',type=int,default=0)
    parser.add_argument('--frames',type=int,nargs='+',default=[0,4,8,16,24,36,60,90,120,150,180,210,240])
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,0,1])
    args=parser.parse_args()
    info=json.loads((DATA_ROOT/'meta/info.json').read_text())
    dataset=EpisodeSubsetDataset(repo_id='local/openarm_pickup',root=DATA_ROOT,episodes=[args.episode],video_backend='pyav',delta_timestamps={'action':[i/info['fps'] for i in range(16)]})
    if any(i<0 or i>=len(dataset) for i in args.frames):
        raise ValueError('选中帧超出轨迹范围')
    policy=policy_config.create_trained_policy(get_train_config(),args.checkpoint,sample_kwargs={'num_steps':10})
    records=[]; predictions=[]; targets=[]
    for index in args.frames:
        sample=dataset[index]
        observation={'observation.state':np.asarray(sample['observation.state']),'prompt':sample['task']}
        for camera in ['ceiling','wrist_left','wrist_right']:
            key=f'observation.images.{camera}'; observation[key]=np.asarray(sample[key])
        valid=~np.asarray(sample['action_is_pad'],dtype=bool)
        target=np.asarray(sample['action'])[valid]
        for seed in args.seeds:
            noise=np.random.default_rng(seed).standard_normal((16,32)).astype(np.float32)
            prediction=np.asarray(policy.infer(observation,noise=noise)['actions'])
            if prediction.shape!=(16,16) or not np.isfinite(prediction).all():
                raise ValueError('模型输出形状或数值异常')
            metrics=compare_trend(prediction[valid],target)
            record={'frame':index,'seed':seed,'valid_steps':int(valid.sum()),**metrics}
            records.append(record); predictions.append(prediction); targets.append(np.asarray(sample['action']))
            print(f'frame={index} seed={seed} MAE={metrics["mae"]:.5f} ratio={metrics["active_displacement_ratio"]}',flush=True)
    # 四、记录原始动作和带来源的报告，禁止把抽样结论当作成功率。
    output=ROOT/'logs/a2_pi0'/f'action_trends_{datetime.now():%Y%m%d-%H%M%S-%f}'
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'predictions.npz',predictions=predictions,targets=targets)
    report={'checkpoint':str(args.checkpoint.resolve()),'episode':args.episode,'frames':args.frames,'seeds':args.seeds,'direction_threshold_rad':0.03,'records':records,'task_success_verified':False}
    (output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('PASS：示教输入动作趋势检查完成，结果目录:',output,flush=True)


# 五、命令行入口。
if __name__=='__main__':
    main()
