联合候选D的完整答案Accuracy为 **89.6667%（2690/3000）**。D−A：-0.6667 pp，95%CI [-1.2671, -0.0673]；D−B：-0.3333 pp，95%CI [-0.7705, +0.0997]。交互(D−B)−(C−A)：-0.4000 pp，95%CI [-0.8760, +0.0670]。交互CI跨0，尚不能确认负向联合效应，也不能证明等价或无交互。

| 条件 | 写入层 | 答案Attention | Correct / 3000 | Accuracy |
|---|---|---|---:|---:|
| A | L0–L27 | 正常 | 2710 | 90.3333% |
| B | L0–L19 | 正常 | 2700 | 90.0000% |
| C | L0–L27 | L2–L5 o_proj输出置零 | 2712 | 90.4000% |
| D | L0–L19 | L2–L5 o_proj输出置零 | 2690 | 89.6667% |

所有条件P−1与真实Decode全部28层均禁止直接读原始Image Key。Image Query原生。只改变R→Image权限和目标答案行o_proj输出；Norm、MLP、Residual、视觉Hidden和Dense KV保留。输出置零不等于物理跳算。

| 对比（负值为损失） | ΔACC pp | 普通95%CI | 原对→错 | 原错→对 | 首Token翻转 | 完整文本改变 |
|---|---:|---|---:|---:|---:|---:|
| B-A | -0.3333 | [-0.7947, +0.1315] | 30 | 20 | 95 | 111 |
| C-A | +0.0667 | [-0.3680, +0.5018] | 21 | 23 | 248 | 377 |
| D-B | -0.3333 | [-0.7705, +0.0997] | 27 | 17 | 253 | 378 |
| D-A | -0.6667 | [-1.2671, -0.0673] | 51 | 31 | 297 | 425 |

六任务完整结果（各500题）：

| Task | A | B | C | D | D−A pp | D−B pp |
|---|---:|---:|---:|---:|---:|---:|
| ChooseAttr | 85.00% | 84.00% | 85.20% | 85.40% | +0.40 | +1.40 |
| ChooseCat | 89.80% | 90.40% | 89.60% | 89.40% | -0.40 | -1.00 |
| ChooseRel | 90.00% | 90.40% | 90.40% | 90.80% | +0.80 | +0.40 |
| CompareAttr | 88.80% | 86.60% | 88.40% | 84.60% | -4.20 | -2.00 |
| LogicalObj | 96.60% | 96.60% | 96.40% | 96.40% | -0.20 | -0.20 |
| QueryAttr | 91.80% | 92.00% | 92.40% | 91.40% | -0.40 | -0.60 |

CompareAttr、QueryAttr及其他任务的全部配对CI和损害计数见task_paired_comparisons.csv。所有原对变错案例含问题、Gold、完整答案、Token IDs和图片SHA，完整列于harm_cases.csv，没有筛选或隐藏。

| 条件 | EOS | 截断 | 空输出 | 重复三元组 |
|---|---:|---:|---:|---:|
| A | 3000 | 0 | 0 | 0 |
| B | 3000 | 0 | 0 | 0 |
| C | 3000 | 0 | 0 | 0 |
| D | 3000 | 0 | 0 | 0 |

A/B复用：True；全部历史逐题数据已与原CSV、Checkpoint SHA、Prompt、Gold、输入绑定和独立评分核验。最短/典型/最长GPU回放见smoke_summary.json。新增正式生成6000次，最终四条件共12000条记录。
模型OpenGVLab/InternVL3-8B @853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager，Torch2.6.0+cu124、Transformers4.49.0、B1/Greedy/正常EOS/max_new_tokens32/logits_to_keep1、256视觉Token；GPU GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e。

使用20,000次Image SHA整簇配对Bootstrap，问题加权；四项差值与一项交互为未作多重比较校正的普通95%CI，任务子组也为探索性。−1pp只作参考。CI跨0不证明等价；L20为看过原结果后提出的候选，当前样本不构成独立确认的非劣效证据，也不支持物理跳算、KV压缩或速度结论。

持久化结果：/home/featurize/work/ATE-internvl3-svh-joint-3000/vlm_research/results/internvl3_svh_joint_l20_skip2to5_3000；独立工作树：/home/featurize/work/ATE-internvl3-svh-joint-3000；分支codex/internvl3-svh-joint-l20-skip2to5-3000。完整预测per_question.csv，全部Checkpoint在checkpoints/，恢复方式见recovery.json。历史实验未覆盖，本轮未推送GitHub、未创建PR、未运行其他Benchmark。

任务损失检查（D−A，D−B）：CompareAttr -4.20 pp / -2.00 pp；ChooseCat -0.40 pp / -1.00 pp；QueryAttr -0.40 pp / -0.60 pp；LogicalObj -0.20 pp / -0.20 pp；ChooseAttr +0.40 pp / +1.40 pp；ChooseRel +0.80 pp / +0.40 pp。子组差异不能由总体平均替代。

最终验收：12000条逐题记录、全部6000次新生成、全部3000个检查点和独立评分已通过；PNG至少300DPI、PDF为矢量，图形已目视检查。GPU无计算进程，工作进程与监督进程已结束；服务器未关闭。

重点任务CompareAttr：D−A -4.20 pp，普通95%CI [-6.51, -2.01]；D−B -2.00 pp，普通95%CI [-3.41, -0.80]。所有子组CI均未作多重比较校正。

重点任务QueryAttr：D−A -0.40 pp，普通95%CI [-1.80, +0.99]；D−B -0.60 pp，普通95%CI [-1.62, +0.40]。所有子组CI均未作多重比较校正。

图中灰色竖线为0pp，橙色点线为−1pp参考线；完整英文图注见figure_caption_en.md。

−1pp参考：D−A点估计为 -0.6667 pp，位于参考线之上；普通95%CI下界为 -1.2671 pp。此为当前样本上的探索性质量对照，不构成独立非劣效验收。
本轮源码仅本地Commit 5b0b362ba7410d2f7671c75c32c68793ab853ba1；本轮结果未推送GitHub。此前Figure 3与L20扩展数据已独立上传GitHub Commit 1cdcf6b2a0b6b5606b00789961904043849cbbdc。
加载前审计兼容性修正记录位于engineering_attempts/attempt01_loader_context/：首次模型在inference_mode下加载，参数版本计数审计报错，未开始Smoke或正式生成；改回历史no_grad加载方式后，Native/No-op严格一致且A/B全部有效Logits回放匹配。未放宽门槛，未改变干预算法。
