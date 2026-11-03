# Semantic-k STOP两项微调：GQA360 Pilot

**新条件316/360，Accuracy=87.778%；原S为319/360（88.611%）。** 相对S净-3题，ΔACC=-0.833pp，图片簇配对95%CI [-2.222, +0.278]pp。纠正1题、损害4题。

## 两项规则与样本冻结

只把to加入STOP、can移出STOP；在独立groups_variant.py中保留原groups.py全部其他字节。16项CPU测试包含原12项及4项新增：STOP差集精确、其余函数与正则/种子逐字相同、合成完整词组可行性导致k=4→0及0→4。所有360题均重新运行原token_groups/select，完整natural输出与历史冻结记录完全一致；新版本随后按原算法自然得到新k，不做事后等预算修正。

样本逐字复用commit 74afff563d255f23b0859f19def2de64accff109的同一360题，六任务各60、360张唯一图片。原Prompt IDs、词组结构、正则词边界、词组顺序、图像与输入Hash均不变；to/can保留在Prompt中，唯一改变是Prefill R直接读Image Key的权限。没有正确性筛选或重新抽样。

94题Writer改变、266题不变；266题均不含to/can，审计后精确复用原S。含to的92题、含can的3题，交集1题。94题全部发生k变化，2题还改变了其他词的最终选择。原平均k=7.972222，新平均k=7.530556；平均R预算25.953%→24.691%（-1.262pp）。这是词表与预算共同改变的干预，不是固定k下to/can的独立位置贡献。

## 主要统计与六任务

主比较仅为新条件−原S，按Image SHA整簇配对Bootstrap20000次、seed20261104，Question等权，报告探索性95%CI；All/Q和子组为辅助描述。360题为内部Pilot，历史图片排除与任务均衡约束仍存在。Changed子集按输入权限变化定义，不按答案筛选；其效果不能外推为总体样本效果。

| 条件 | 正确/360 | ACC | 平均k | 平均k/R |
|---|---:|---:|---:|---:|
| All-Writer | 327/360 | 90.833% | 29.903 | 100.000% |
| Semantic-k | 319/360 | 88.611% | 7.972 | 25.953% |
| Question-Only | 324/360 | 90.000% | 12.903 | 42.199% |
| Semantic-STOP-to-KEEP-can | 316/360 | 87.778% | 7.531 | 24.691% |

| 子集 | n | 原S正确 | 新正确 | 纠正 / 损害 |
|---|---:|---:|---:|---:|
| writers_changed | 94 | 82 | 79 | 1/4 |
| writers_unchanged | 266 | 237 | 237 | 0/0 |
| to_related | 92 | 80 | 77 | 1/4 |
| to_only | 91 | 79 | 76 | 1/4 |
| can_related | 3 | 3 | 3 | 0/0 |
| can_only | 2 | 2 | 2 | 0/0 |
| to_and_can | 1 | 1 | 1 | 0/0 |
| other_selection_changes | 2 | 2 | 2 | 0/0 |

子集可重叠，尤其to_related和can_related共享1题；不能把两组净变化相加。六任务所有条件ACC见task_accuracy.csv；各任务新−S及Changed子集的配对CI见paired_comparisons.csv。

## to规则与完整损害案例

to相关92题共纠正1、损害4；不含can的91题共纠正1、损害4。这些是对选择规则整体干预的结果，同时包含k与可行预算变化，不能解释为单个to位置的单位因果贡献。全部to题、实际移除位置、答案和正负评分见to_cases.csv；token_changes.csv另列to/can实际位置、候选预算、最终新增/移除位置和其他词连带变化。

| 损害ID | 任务 | 完整问题 | Gold | 原S | 新条件 | old k→new k |
|---|---|---|---|---|---|
| 0251920 | CompareAttr | What is common to the sidewalk and the sock? | color | Color | Material | 8→7 |
| 121023683 | CompareAttr | What is common to the sweater and the frame? | color | Color | Theater | 7→6 |
| 14209478 | ChooseRel | Is the car to the left or to the right of the man that is wearing a shirt? | left | Left | Right | 13→11 |
| 18464271 | ChooseAttr | Does the door appear to be blue or white? | blue | Blue | White | 6→5 |

## can：全部实际例子

词性来自既有冻结spaCy解析，仅作描述，不安装资源、不按词性重新决定can权限，也不人工依答案修正规则。下表包含全部3题，因此不会只展示修复成功例。

| ID | 完整问题 | 既有POS/依存 | Gold | 原S | 新条件 | k变化 | 实际can新增位置 |
|---|---|---|---|---|---|---|
| 12642596 | On which side of the image is the garbage can? | AUX ccomp →is | right | Right | Right | 5→6 | [592] |
| 17484369 | Is the soda can to the right or to the left of the man on the right? | AUX attr →Is | left | Left | Left | 11→10 | [587] |
| 18699253 | On which side of the photo is the garbage can? | AUX ccomp →is | right | Right | Right | 5→6 | [592] |

**解析歧义审计：上述3题的can实际均指容器名词（garbage can / soda can），既有解析器却全部标为AUX，这是解析标注错误，不能把这3题解释为助动词对照。** 该人工语法说明只用于描述，未修改STOP、Token集合、Prompt或任何生成。3道名词can题答案全部不变；本360题没有实际助动词can例子，不能评价助动词开放的效果，也未追加样本。can_cases.csv保留原POS及描述性用法两列。

## 预算连带变化：不能单独归因

全部94个Changed题的新旧k均不同；word_group_changes.csv保存每个改变词组的原/新core和选中标记。下列2题在to/can以外还有接收位置改变，完整答案与所有新增/移除位置均列出。

- **04527888**：Is the metal bowl to the right or to the left of the cooking utensil which is to the right of the cooking pot? Gold=right；S=Right（1），新=Right（1）；k 21→17；其他新增=[]，其他移除=[611]。
- **17736766**：Is the plastic bottle to the right or to the left of her? Gold=left；S=Left（1），新=Left（1）；k 10→9；其他新增=[597]，其他移除=[]。

## 是否值得扩展到3000题

本批没有净Accuracy收益，应报告负面/零净收益结果，不建议据此直接扩展到3000题。预算变小或个别案例修复也不能替代完整答案净质量评估。本轮未追加任何STOP搜索、单独to/can新条件或大规模生成。

## 执行与可验证交付

独立分支codex/gqa360-semantic-stop-to-keep-can。sample_manifest SHA256=`621d8325b62fd32ff5a1beb215dd72dfae6dea8041bb3fec126e9ea4c53f0486`；sample_plan SHA256=`d3a4ad0b98b1077bd857895d4529e940ef4cf446d4a6c99dd2464fdff3c86b4d`。GPU冻结执行commit `50ba5dd6bc3d5c6e32e95ca9f5f6343e970cfd64`。模型revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；FP16/Eager/Transformers4.49.0，Greedy32，logits_to_keep=1；P−1/真实Decode在32层禁读Image Key，MLP/Residual/QKV/完整KV保持原生，每条件独立真实Cache。

使用UUID `GPU-b08489af-0a8a-eb48-55ba-8346733af417` 与 `GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6`，各47条实际新增，分片按图像SHA且科学题互不重叠。每卡三道固定真实Smoke通过Native/No-op全部有效Logits严格torch.equal、原All/S答案Token和首Token完整向量严格复现、独立KV、Hook正常/异常清理与权重/输入不变；再对自身一个真正Changed的新Writer输入执行实际Mask与32层MLP Smoke，两卡全部通过后才放行正式采集。

94条新增、266条新条件精确复用，另有1080条All/S/Q复用，共1440个标签、360题完整覆盖，无缺失。GPU采集墙钟104.4秒（含加载/Smoke/正式处理，不含CPU准备和统计）。输出异常标签数0。全部原始JSON答案/Token、紧凑NPZ审计和SHA索引已保存。新主审计NPZ总计398597字节，只含Mask审计、Token IDs和元数据；Logits仅保存哈希，未归档首Token或逐步完整向量。Smoke严格比较的完整向量仅在内存使用。不能据哈希声称可从本包重算完整Logit分布。

磁盘/data不足以做完整检出，本轮从一开始就在空间充足的/data1创建稀疏独立Worktree并冻结磁盘预检；共享历史源码只读、按SHA验证。首次CPU冻结后仅把Smoke输入改为每卡真正Changed题，并重新CPU冻结；两次360题Writer计划及token_changes哈希完全相同，旧CPU冻结保存在attempts，最终冻结前无GPU启动。

GPU释放：{"all_own_children_exited": true, "selected_devices_no_compute_processes": true, "selected_uuids": ["GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6", "GPU-b08489af-0a8a-eb48-55ba-8346733af417"]}。模型/依赖/评分器完整SHA复核通过，全部新主生成Mask禁止概率0、非目标精确，权重版本/输入/同题投影视觉特征保持不变，32MLP和独立KV/Hook检查通过。原STOP选择器、历史结果及main未修改；不声明加速或物理剪枝。执行顺序：run freeze→提交冻结→run driver→CPU analyze。本轮停止。

## 交付复核与存储说明

独立CPU交付复核从逐题CSV重新计数，并独立重算20,000次配对Bootstrap，与主统计一致；所有Checkpoint、Token预算、任务汇总与冻结计划一致，详见delivery_audit.json。

CSV、JSON答案及原始Checkpoint提交Git；紧凑NPZ只保存在服务器`/data1/likai/ATE-resources/experiments/gqa360_semantic_stop_to_keep_can/`，路径及SHA见checkpoint_index.csv与smoke_audit_index.csv，不声称GitHub包含这些NPZ。

初版CPU统计中的can解释曾机械沿用错误的AUX标注，现已改为明确区分原解析和实际容器名词用法；初版派生文件保存在attempts/analysis_before_can_POS_audit。该修订只更正描述，所有Token计划、GPU输出、Accuracy及预算统计均未变，也没有追加GPU运行。
