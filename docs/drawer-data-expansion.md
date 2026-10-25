# 扩充脚本专家示教（2026-10-24）

用户决定先扩充数据，后续单独检验数据覆盖是否改善策略；不再以单轨迹精度达标为扩充前提。
保留原训练数据、归一化统计和检查点，禁止把扩充本身视为已证明数据量是失败主因。

## 来源和语义

复用固定版本 `14ea59faad6706b9add5d7f8d386e43239a6360f` 的上游模型、IK等接口，
使用本仓库修正的接触控制器，不更新上游、不启用抓取weld。
这些是脚本专家生成的仿真示教，不是人工示教或上游直接提供的数据集。

## 第一批可行性试采

输出位于 `data/drawer_expansion/`，数值版 `pilot_numeric/`，RGB版 `pilot_rgb/`。
默认保持柜体X偏移0.10m、14mm把手、30Hz、16维绝对动作、原三路相机和往返任务。

- 抓握俯仰12°、18°：数值试采均通过往返，峰值分别93.43、93.12mm。
  两条RGB采集也已完成，每条462帧，往返、无weld、独立回放均通过；
  RGB版state/action/qpos/drawer/contact与各自数值版逐元素一致。
  原轨迹加两条新RGB共1386帧，但暂未接入训练。三路图像及动作窗口审核
  见 `results/drawer_expansion_audit_20261024.json`。
- 柜体X偏移0.095m、俯仰15°：只拉开42.88mm，失败，不纳入成功数据。
- 初始右臂扰动±0.03rad、原15°：旧对称采样seed101超限被拒绝，102失败，103成功。
- ready的joint6接近上限，新增采样改为扰动窗口与关节/执行器限位交集内均匀采样，
  避免裁剪边界堆积。边界附近分布不对称；实际delta写入每条report。
- 新采样seed104/105/106失败，峰值68.04/58.59/58.82mm；107/108成功，
  峰值93.22/93.25mm。旧103与新采样种子不可按同一算法复现，应以保存的实际初态为准。

采集入口新增 `--initial-right-jitter` 和 `--seed`。该seed仅控制初始条件，
不是π0推理起点噪声seed。扰动仅在reset时修改qpos和初始ctrl，轨迹内不注入中间状态，
标签仍为真正执行的控制目标。默认幅度0保持历史采集行为。

12°/18°增加运动策略多样性，但起始观测相同可能对应不同后续动作，不能据此断言
提升纠偏能力。优先在固定15°专家策略下改变可观测初始状态；之后再扩大柜体位置等范围。
实测原15°与新12°首帧state完全相同，但前16步动作最大差0.02050rad，
说明仅变专家姿态会引入同起点的多种动作，不能作为初始状态覆盖增加的证据。

审核命令（输出路径不得已存在）：

```bash
.venv/bin/python scripts/a2_pi0/audit_drawer_expansion.py \
  --output results/drawer_expansion_audit_new.json
```

## 质量门槛

每条必须真实双侧接触抓握、无weld、打开≥80mm连续1秒、最终关闭误差≤5mm连续1秒，
无损RGB/state/action逐帧对齐且有限，独立动作回放误差<1e-6。必须检查接触记录，
不能仅使用专家返回True。失败尝试保留诊断，不能混标为成功。
先数值筛选、后RGB；同参数数值/RGB须再次比较动作和物理状态，图像需非黑且非恒定。

## 下一批RGB采集命令

107作为训练候选，108作为整条保留的验证候选；这是小规模流程试验，不能据此声称泛化。
实际RGB数据产出后再次验收，拒绝覆盖已有目录。

```bash
for task_seed in 107 108; do
  LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/python scripts/a2_pi0/collect_drawer_trial.py \
    --mode contact --expert contact --drawer-offset-x 0.10 --pull-distance 0.10 \
    --return-drawer --wrist-view side --contact-pitch 15 \
    --initial-right-jitter 0.03 --seed "$task_seed" \
    --output "data/drawer_expansion/pilot_rgb/initial_seed${task_seed}"
done
```

扩大到更多轨迹前先验证小批次。训练/验证按整条episode及初始条件分组，禁止随机拆相邻帧。
当前 `drawer_lossless_dataset.py` 和阶段训练入口仍绑定原单轨迹；新数据不会自动被训练读取。
后续需增加多episode索引，确保未来16步动作不跨episode，按episode读取实际阶段边界，
仅用训练轨迹重算独立归一化统计，使用新资产ID/实验名，并做独立初始条件的闭环评估。
不覆盖旧统计，旧检查点必须继续配原统计；换统计后直接续训会改变模型输入/输出含义。

## 第二批：减少同质化（2026-10-24）

用户要求继续扩充，并明确关注同质化。初始微扰107/108相对原轨迹的接近段动作差
仅0.001686/0.003084rad，抓握段动作相同，不能仅靠此类样本增加数量。
本批新增 `--approach-offset X Y Z` 与 `--drawer-offset-y`，默认值保持原行为。
控制器仍用IK及真实接触执行，模型不读取路点或柜体真值。
回放和 `DrawerSimulation` 已支持恢复柜体Y偏移，策略环境整条回放误差0。

数值筛选位于 `data/drawer_expansion/diverse_numeric/`：

| 变化 | 数值往返验收 | 峰值开度mm |
|---|---|---:|
| 接近路点Z由55降到15mm | 未通过 | 80.19（未满足完整保持/往返标准） |
| 接近路点Z由55升到95mm | 未通过 | 58.28 |
| 接近路点Y=-30mm | 通过 | 93.31 |
| 接近路点Y=+30mm | 通过 | 92.99 |
| 柜体Y=-5mm | 未通过 | 46.33 |
| 柜体Y=+5mm | 通过 | 93.09 |
| 柜体Y=-10mm | 未通过 | 46.33 |
| 柜体Y=+10mm | 未通过 | 45.02 |

筛选成功不等于专家对整个随机化范围稳定；失败也保留记录。
左右侧向路径相对原轨迹，接近段真实夹持点平均距离分别21.95/17.09mm，
初始微扰108仅1.51mm。此指标是按时间对齐的轨迹差异，不是任务误差或泛化证明。
细节见 `results/drawer_diversity_numeric_20261024.json`。

批次参数与预先确定的划分：`configs/drawer-expansion-batch002.json`。
两条侧向接近进入train，柜体+5mm和初始微扰108进入validation。
原15°及12°/18°同起点变体全部进入train，不跨split。
验证集仍是较窄范围的局部验证，不能称广泛泛化。

```bash
# 输出目录必须不存在；每条RGB完成后对照数值版检查动作和物理轨迹逐元素一致。
.venv/bin/python scripts/a2_pi0/collect_drawer_expansion_batch.py \
  --config configs/drawer-expansion-batch002.json --workers 2

# 所有RGB完成后构建清单；拒绝失败、重复动作或同初始条件跨集合。
.venv/bin/python scripts/a2_pi0/build_drawer_expansion_manifest.py \
  --config configs/drawer-expansion-batch002.json \
  --output data/drawer_expansion/dataset_v002.json
```

`drawer_multi_episode_dataset.py` 提供按清单的无损多轨迹读取，按所属episode做尾部填充，
不跨轨迹拼未来动作。训练入口尚未接入此读取器，原统计与原数据仍不变。

第二批RGB已全部完成：两条侧向接近、柜体+5mm、初始微扰108，共4条。
全部通过真实接触/无weld/往返/独立回放和图像数据审核；state/action/qpos/drawer及
双侧接触计数、接触力与数值版逐元素一致。批次报告：
`data/drawer_expansion/diverse_rgb/batch_report.json`。
最终清单 `data/drawer_expansion/dataset_v002.json` 共7条、3234帧：train 5条2310帧，
validation 2条924帧。全扩充审核为 `results/drawer_expansion_audit_batch002_20261024.json`，
其中numeric_only仅是试采，rejected均不进入清单。
读取器已在实际数据上验证轨迹首尾、16步窗口边界、尾部填充及OpenArmInputs适配；
检查记录：`results/drawer_multi_episode_checks_20261024.json`。
这仍是同一场景附近的局部扩充，不包含验收成功的偏低/偏高恢复示教，不宣称广泛泛化。
