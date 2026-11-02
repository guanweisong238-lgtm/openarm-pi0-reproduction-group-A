# 扩范围抓取修复与独立验收（2026-11-02）

已完成清理、失败诊断、独立训练数据采集、训练/开发集筛选和最终新100起点评估。旧模型、原始数据、历史失败与optimization_log原文保留。

开发集选定权重：`checkpoints/pi0_openarm_drawer_endpoint_v001/feedback_endpoint_020_randomized_rehearsal/2000`。最终100个起点在训练前锁定，不参与选权重。

| 模型 | 原范围 | 扩大范围 | 合计 |
| --- | ---: | ---: | ---: |
| baseline020 | 66/70 | 19/30 | 85/100 |

两种模型若都执行，使用完全相同的新起点，独立动作重放全部有效。若开发集没有合格改进，则保留020并只执行一份最终测试，不虚报新模型改进。
最终测试结果无论好坏均保留，不再据此挑选其他权重。near/expanded定义与上一轮新100起点相同；合计仅代表预定70/30混合，同模拟器结果不代表真机。

详见protocol.json、training_data_audit.json、candidates.json、frozen_final_models.json、final_comparison.json与各独立重放报告。
