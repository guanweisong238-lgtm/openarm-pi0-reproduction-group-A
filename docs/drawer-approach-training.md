# 接近与首次稳定夹持阶段训练

从011/450起点模型继续，独立训练入口train_drawer_approach.py，不改变旧全轨迹训练脚本或缓存配置。

## 阶段数据

13条原始示教均从完整积分初态独立重放，整个把手组件接触重新计算，所有逐帧qpos误差0。
first contact为右夹爪两指都接触横杆/支柱、各法向力>.01N、连续至少5帧的首次起点。
start=[0,9)，approach=[9,contact)，grasp=[contact,end)。
end=min(轨迹长,max(contact+5,min(contact+20,首次开2mm帧+1)))。
人工轨迹的首次夹持可能早于其后松开重抓和真正拉出；本轮只覆盖首次夹持。
阶段清单results/drawer_approach_phases_v001/phases.json，独立事件NPZ不改原数据。
训练10条prefix 3452帧，验证3条prefix 927帧；原始完整8632/2068帧仍完整保留。
未来16步标签保留原始动作，窗口可自然延伸出采样阶段，不裁剪成伪造保持动作。

## 训练安排

均匀选10条train轨迹，再按start/approach/grasp=.30/.35/.35抽帧；validation只用于评估。
1200更新，lr3e-6、warmup100、cosine到3e-7，fresh optimizer，每300步评估、600步保存。
右7残差+独立train-only统计，左标准化状态屏蔽，LoRA+三IO共50,087,968参数FP32更新。
关闭图像增强，low_mix时间采样，首4步时间权重4，后12步1，全32维动作目标。
cache仍使用原绝对动作版本，在运行时转换残差；严禁把残差数据写进绝对缓存。
每100步本地Matplotlib更新loss，配置和统计SHA归档；固定检查73 train/20 validation观测×3种子。
分列各阶段指标，并保留ID12首9帧最坏位置/姿态，观察起步遗忘。

## 闭环验收（训练后执行）

先从两个train起点ID12和ID8、seed42实际运行到各自phase_end/30秒；使用chunk4与resample噪声。
若初次稳定夹持可实现，再扩大种子/验证起点；未通过则保留失败，不宣称完整任务成功。
局部通过条件：完成预定时长，无专家动作/中间状态注入，右夹爪两指持续至少5帧接触把手组件，
各法向力>.01N。这只验证首次稳定双侧接触，牢固抓取仍需后续拉动检验。
不按同一时间点的路径偏差判定接触成功，轨迹可有不同速度；同时记录到原接触夹持点的最小/末端距离。
未训练完整拉开、保持和关闭；scope必须标为phase diagnostic。
评分器assess_drawer_approach_rollout.py已验证原示教接触能检出、单侧接触被拒绝、专家重放不计为策略成功。

原生训练命令：

```bash
.venv/bin/python scripts/a2_pi0/train_drawer_approach.py \
  --exp-name approach13_delta_012 --steps 1200 --lr 3e-6 \
  --eval-every 300 --save-every 600
```
