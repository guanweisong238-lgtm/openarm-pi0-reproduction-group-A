"""低维留一轨迹诊断：位置 vs 位置+当前速度，不是Pi0任务成功证据。"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from drawer_success_training_config import ROOT
from drawer_simulation import DrawerSimulation


def kernel(a,b):
    distance=np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0)
    return np.exp(-distance/a.shape[1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();env=DrawerSimulation();dofs=env.velocity_ids;velocity=[]
    with np.load(env.episode_root/'episode.npz') as d:
        state=d['state'].copy();action=d['action'].copy()
        for a in action:velocity.append(env.data.qvel[dofs].copy());env.step(a)
    env.close();indices=np.r_[np.arange(60),np.arange(75,90)]
    sets=[dict(name='canonical12',q=state[indices,:7],v=np.asarray(velocity)[indices,:7],
        y=action[indices,:7]-state[indices,:7])]
    for row in json.loads((ROOT/'data/drawer_recovery/v002/manifest.json').read_text())['episodes']:
        with np.load(ROOT/row['path']/'episode.npz') as d:
            sets.append(dict(name=row['episode_id'],q=d['state'][indices,:7].copy(),
                v=d['qvel'][indices][:,dofs[:7]].copy(),y=d['action'][indices,:7]-d['state'][indices,:7]))
    rows=[]
    for j,test in enumerate(sets):
        train=[s for i,s in enumerate(sets) if i!=j];row={'held_out':test['name']}
        for with_velocity in (False,True):
            feature=lambda s:np.concatenate([s['q'],s['v']],1) if with_velocity else s['q']
            x=np.concatenate([feature(s) for s in train]);y=np.concatenate([s['y'] for s in train])
            mean=x.mean(0);std=np.maximum(x.std(0),1e-6);x=(x-mean)/std
            weights=cho_solve(cho_factor(kernel(x,x)+np.eye(len(x))*.001),y)
            pred=kernel((feature(test)-mean)/std,x)@weights
            row['q_and_velocity' if with_velocity else 'q_only']=float(np.abs(pred-test['y']).mean())
        rows.append(row);print(row,flush=True)
    report=dict(scope='leave-one-correlated-trajectory-out kernel ridge; no images; not policy success or unique causal proof',
        hyperparameters=dict(kernel='RBF',gamma='1/feature_dimension',alpha=.001,
            standardization='training-fold mean/std, floor1e-6',eligible_source_frames=indices.tolist(),
            target='first absolute action minus current position, right7',metric='joint MAE radians'),rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
