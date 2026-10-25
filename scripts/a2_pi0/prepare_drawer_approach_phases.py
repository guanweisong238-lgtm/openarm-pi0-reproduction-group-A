"""独立重放13条示教，按整个把手组件的接触事件建立接近/抓握阶段；不修改源数据。"""
from drawer_success_training_config import ROOT,MANIFEST
import sys
sys.path[:0]=[str(ROOT/".cache/openarm-control"),str(ROOT/".cache/drawer-deps")]
from drawer_simulation import DrawerSimulation
from contact_drawer_controller import finger_handle_contacts
from pathlib import Path
import hashlib,json
import numpy as np

OUT=ROOT/'results/drawer_approach_phases_v001'


def main():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads(MANIFEST.read_text());rows=[]
    for row in manifest['episodes']:
        env=DrawerSimulation(ROOT/row['path']);counts=[];forces=[];opening=[];error=0.
        with np.load(ROOT/row['path']/'episode.npz') as e:
            actions=e['action'].copy();reference=e['qpos'].copy();terminal=e['terminal_qpos'].copy()
        try:
            for i,action in enumerate(actions):
                error=max(error,float(np.max(abs(env.data.qpos-reference[i]))))
                c,f=finger_handle_contacts(env.model,env.data,include_stems=True)
                counts.append(c);forces.append(f);opening.append(-float(env.data.qpos[env.drawer_qpos]))
                env.step(action)
            error=max(error,float(np.max(abs(env.data.qpos-terminal))))
        finally:env.close()
        if error>=1e-6:raise ValueError(f'重放不一致: ID{row["episode_id"]} {error}')
        counts=np.asarray(counts);forces=np.asarray(forces);opening=np.asarray(opening)
        both=np.all(counts>0,axis=1)&np.all(forces>.01,axis=1)
        stable=np.convolve(both.astype(int),np.ones(5,int),mode='valid')==5
        if not stable.any():raise ValueError('没有5帧稳定双侧组件接触')
        contact=int(np.flatnonzero(stable)[0]);pull_indices=np.flatnonzero((np.arange(len(both))>=contact)&(opening>=.002))
        if not len(pull_indices):raise ValueError('成功示教缺少2mm拉开事件')
        pull=int(pull_indices[0]);end=min(len(both),max(contact+5,min(contact+20,pull+1)))
        if contact<=9:raise ValueError('接触已在起步范围内，须单独处理')
        result=dict(row,first_stable_contact_frame=contact,first_open2mm_frame=pull,phase_end_exclusive=end,
            phase_ranges={'start':[0,9],'approach':[9,contact],'grasp':[contact,end]},
            replay_max_qpos_error=error,contact_scope='whole handle assembly, bar and stems',
            event_arrays=f'episode_{row["episode_id"]:03d}_events.npz')
        np.savez_compressed(OUT/result['event_arrays'],counts=counts,forces=forces,opening=opening,bilateral=both)
        rows.append(result);print('PHASE',row['episode_id'],row['split'],'contact',contact,'open2mm',pull,'end',end,'replay',error,flush=True)
        (OUT/'phases.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            definition='first bilateral contact sustained >=5 frames, force>.01N each; stop max20 contact frames or 2mm opening; preserve original horizon labels',episodes=rows),indent=2)+'\n')
    print('PASS all 13 original dynamics replayed; train/validation unchanged',flush=True)


if __name__=='__main__':main()
