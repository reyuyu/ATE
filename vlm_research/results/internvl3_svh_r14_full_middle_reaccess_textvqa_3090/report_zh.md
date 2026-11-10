# InternVL3-SVH-R14全部允许中层回看：TextVQA Image-only

**主结果为全新B组128题：R14 Soft Accuracy=7.031%，R6=7.812%，R0=8.516%，Native=58.516%。**
R14−R0=-1.484pp（普通95%CI [-4.766, +1.406]pp），R14−R6=-0.781pp（[-2.344, +0.000]pp）。追回同题Native−R0差距的-2.969%，仍距Native 51.484pp。

状态：EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER；历史NUMERICAL_EQUIVALENCE_FAILED保持不变。R14仅开放L6–L19，不包含L0–L1，L2–L5答案Attention仍跳算，L20前仍删除视觉Hidden，L20–L27仍无原始视觉Hidden。它绝非28层全部视觉通路恢复。

**现有L6–L19全部回看不足以恢复文字识别能力。** 本结论针对该冻结SVH-R V2、Image-only和本轮样本；不自动扩层、调Prompt或扩大样本。

## A组：历史96题追补对照

| method | N | soft_accuracy_pct | meta_count | answering_prefix_count | mean_KV_live_MiB | mean_KV_capacity_MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Native | 96 | 58.646 | 0 | 0 | 18.411 | 18.411 |
| R0 | 96 | 6.354 | 43 | 43 | 3.906 | 3.906 |
| R6 | 96 | 10.104 | 36 | 36 | 6.852 | 6.852 |
| R14 | 96 | 9.062 | 37 | 37 | 10.841 | 10.841 |

| base | candidate | difference_pp | CI95_low_pp | CI95_high_pp | improved | declined | unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | R6 | 3.750 | 0.104 | 8.229 | 5 | 2 | 89 |
| R0 | R14 | 2.708 | -1.667 | 7.396 | 5 | 3 | 88 |
| R6 | R14 | -1.042 | -3.125 | 0.000 | 0 | 1 | 95 |
| R14 | Native | 49.583 | 38.854 | 59.896 | 56 | 5 | 35 |

A组是上轮已用于判断R6的Holdout，不能重新包装为R14与R6完全独立确认。Native/R0/R6共288条原记录严格匹配后复用，只新增96次R14；输入Tensor、Prompt、图像、Gold、模型Revision、32Token上限、Scorer和冻结核心前向SHA全部核验，不修改父结果。

## B组：独立128题主验证

| method | N | soft_accuracy_pct | meta_count | answering_prefix_count | mean_KV_live_MiB | mean_KV_capacity_MiB |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 128 | 8.516 | 39 | 39 | 3.786 | 3.786 |
| R6 | 128 | 7.812 | 31 | 31 | 6.773 | 6.773 |
| R14 | 128 | 7.031 | 31 | 31 | 10.747 | 10.747 |
| Native | 128 | 58.516 | 0 | 0 | 18.347 | 18.347 |

| base | candidate | difference_pp | CI95_low_pp | CI95_high_pp | improved | declined | unchanged | zero_to_positive | positive_to_zero | not_full_to_full | full_to_not_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | R6 | -0.703 | -3.516 | 1.797 | 3 | 3 | 122 | 3 | 1 | 0 | 3 |
| R0 | R14 | -1.484 | -4.766 | 1.406 | 3 | 4 | 121 | 3 | 2 | 0 | 4 |
| R6 | R14 | -0.781 | -2.344 | 0.000 | 0 | 1 | 127 | 0 | 1 | 0 | 1 |
| R14 | Native | 51.484 | 43.281 | 59.766 | 75 | 3 | 50 | 72 | 1 | 61 | 2 |

从剩余TextVQA原始问题总体按固定Seed与QuestionID/ImageSHA哈希排名，按顺序接受首道未选图片的问题，直至128题；未使用类别配额、Gold、历史预测或元回答选题。与上一轮Discovery96/Holdout96/Smoke8全部200张图像零交集，一图一题。在任何R14 GPU推理前已冻结完整清单SHA：e1630f459f7285aeeb8ac111cfe19568e2728bf081bf17906745fbe8a92192f0。问题文字分类仅在抽样后作描述性分析；原总体剩余分布与B组实际分布见sample_manifest.json。
B组R0/R6/R14共384次均为新生成；Native为该128题逐题严格绑定的历史预测，并非旧样本均值。

## 同题恢复比例与不确定性

| group | R14_minus_R0_pp | R14_minus_R6_pp | Native_minus_R14_pp | Native_minus_R0_pp | recovered_gap_pct |
| --- | --- | --- | --- | --- | --- |
| A | 2.708 | -1.042 | 49.583 | 52.292 | 5.179 |
| B | -1.484 | -0.781 | 51.484 | 50.000 | -2.969 |

恢复比例=(R14−R0)/(Native−R0)，同组计算且不截断到0–100%；分母零时记UNDEFINED。它是描述性点估计。所有差值CI均按图片SHA整簇配对Bootstrap5000次，固定Seed=20261112，普通未校正95%CI；有限样本及多重比较不能提供全量非劣效或机制确认。区间跨零不证明等价。Soft Score提高/下降/不变与0↔正分、非满分↔满分分别报告，不能只数答案变化。

## 元回答：消除是否带来真实文字得分

| phase | base | candidate | meta_before | meta_after | meta_resolved | meta_new | meta_resolved_and_score_improved | meta_to_scored_answer | zero_to_positive | positive_to_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_historical_followup96 | R0 | R14 | 43 | 37 | 10 | 4 | 2 | 2 | 5 | 3 |
| A_historical_followup96 | R6 | R14 | 36 | 37 | 0 | 1 | 0 | 0 | 0 | 1 |
| B_independent128 | R0 | R14 | 39 | 31 | 12 | 4 | 2 | 2 | 3 | 2 |
| B_independent128 | R6 | R14 | 31 | 31 | 2 | 2 | 0 | 0 | 0 | 1 |

元回答检测沿用冻结规则，覆盖answering前缀、answering does not require、answer not in image/text、cannot determine等完整文本模式。各条件全部元回答、重复、空白和截断均原样纳入官方十答案软评分，不删除或修正。meta_response_analysis.csv列出数量，meta_transition_summary.csv区分消除/新增及真正获得非零软分；逐题原始文本、十答案Gold与实际分数变化见per_question_soft_changes.csv。元回答变少本身不证明OCR质量恢复。

## 文字问题类型

| method | type | N | soft_accuracy_pct | difference_vs_R0_pp | improved | declined | meta_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native | signs | 6 | 50.000 | 50.000 | 3 | 0 | 0 |
| Native | products | 21 | 68.095 | 65.238 | 15 | 0 | 0 |
| Native | titles | 4 | 50.000 | 50.000 | 2 | 0 | 0 |
| Native | numbers | 23 | 56.087 | 38.696 | 10 | 1 | 0 |
| Native | short_words | 12 | 65.833 | 65.833 | 9 | 0 | 0 |
| Native | multiword | 4 | 25.000 | 25.000 | 1 | 0 | 0 |
| Native | other | 58 | 58.276 | 47.414 | 34 | 4 | 0 |
| R0 | signs | 6 | 0.000 | 0.000 | 0 | 0 | 3 |
| R0 | products | 21 | 2.857 | 0.000 | 0 | 0 | 9 |
| R0 | titles | 4 | 0.000 | 0.000 | 0 | 0 | 1 |
| R0 | numbers | 23 | 17.391 | 0.000 | 0 | 0 | 2 |
| R0 | short_words | 12 | 0.000 | 0.000 | 0 | 0 | 3 |
| R0 | multiword | 4 | 0.000 | 0.000 | 0 | 0 | 2 |
| R0 | other | 58 | 10.862 | 0.000 | 0 | 0 | 19 |
| R6 | signs | 6 | 0.000 | 0.000 | 0 | 0 | 3 |
| R6 | products | 21 | 4.286 | 1.429 | 1 | 0 | 7 |
| R6 | titles | 4 | 0.000 | 0.000 | 0 | 0 | 1 |
| R6 | numbers | 23 | 10.000 | -7.391 | 0 | 2 | 0 |
| R6 | short_words | 12 | 0.000 | 0.000 | 0 | 0 | 4 |
| R6 | multiword | 4 | 0.000 | 0.000 | 0 | 0 | 2 |
| R6 | other | 58 | 11.724 | 0.862 | 2 | 1 | 14 |
| R14 | signs | 6 | 0.000 | 0.000 | 0 | 0 | 3 |
| R14 | products | 21 | 4.286 | 1.429 | 1 | 0 | 7 |
| R14 | titles | 4 | 0.000 | 0.000 | 0 | 0 | 0 |
| R14 | numbers | 23 | 10.000 | -7.391 | 0 | 2 | 0 |
| R14 | short_words | 12 | 0.000 | 0.000 | 0 | 0 | 3 |
| R14 | multiword | 4 | 0.000 | 0.000 | 0 | 0 | 2 |
| R14 | other | 58 | 10.000 | -0.862 | 2 | 2 | 16 |

Signs/Products/Titles/Numbers/Short words/Multiword/Other分类来自问题文字启发式，沿用父实验，未用于B组选题。小类别仅作描述，全部损害与恢复记录保留。

## 实际质量EOS缓存与可选速度

| group | method | N | mean_quality_EOS_KV_live_MiB | mean_quality_EOS_KV_capacity_MiB | increment_vs_R0_MiB | increment_vs_R6_MiB | mean_actual_output_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Native | 96 | 18.411 | 18.411 | 14.505 | 11.559 | 4.219 |
| A | R0 | 96 | 3.906 | 3.906 | 0.000 | -2.946 | 6.885 |
| A | R6 | 96 | 6.852 | 6.852 | 2.946 | 0.000 | 5.740 |
| A | R14 | 96 | 10.841 | 10.841 | 6.935 | 3.989 | 5.500 |
| B | Native | 128 | 18.347 | 18.347 | 14.561 | 11.575 | 4.398 |
| B | R0 | 128 | 3.786 | 3.786 | 0.000 | -2.987 | 5.672 |
| B | R6 | 128 | 6.773 | 6.773 | 2.987 | 0.000 | 5.391 |
| B | R14 | 128 | 10.747 | 10.747 | 6.962 | 3.975 | 4.852 |

以上是正常EOS质量生成末端的真实持久KV Live/底层Storage Capacity，不是理论预算、显存峰值或固定32缓存。输出长度不同会影响均值；R14恰有14层保留原始256视觉KV，R6为6层、R0为0层，L2–L5持久缓存为空。逐层K/V形状、BF16 dtype、物理长度和真实存储字节已经独立核验。

正式测速状态：**NOT_RUN**。GPU前冻结门槛为B组R14对R0、R6的普通配对95%CI下界均>0，未满足则停止扩展。没有使用旧TextVQA/GQA时间或理论KV填充未执行数据。

正式E2E/TTFT/TPOT/KV@32均为NOT_RUN。只报告本轮实际质量EOS KV，不生成虚构性能行。

## Smoke、复用与数值豁免

6题Dense-R14/Physical-R14 Smoke：完整Token一致6/6，标准化答案一致6/6；Native/No-op有效Logits严格torch.equal，首个Native与历史Token一致。相同生成前缀Logits最大MaxAbs=2.437500、步均MeanAbs=0.226501，仅记录，不用旧0.125门槛阻止本轮探索，也不宣称数值等价。最短/最长输入完整结构审计验证真实Hidden删除、跳算QKV/Attention、全部14视觉KV、非K无持久原始图像KV、Mask禁用概率0、原始逻辑RoPE、Norm/MLP/Residual和独立Cache；无NaN/Inf或Hook泄漏。

冻结V2核心未改变，不导入C1/C2修复、不做Kernel或新层搜索。历史BF16形状相关误差限制纯机制归因、跨实现外推和数值等价结论。所有实际Prompt均重新核对原图/原问题/原生模板和固定指令：
Answer with the text visible in the image, using a word or short phrase. Do not explain.
没有OCR Tokens、识别文字、Caption、reference text、Hint、Gold或Rationale进入模型。目录名字不作为输入证据。

完成480次新增正式质量（A96+B384），另复用416条质量参考记录；正式测速0次，模型加载1次。独立官方Answer Processor/十答案leave-one-out复算逐条一致，完整Checkpoint、Token、日志和SHA保留。GPU释放=True。没有运行GQA、其他Benchmark或更改父实验/main，不自动扩样、扩层或关闭服务器。

SSD：/home/featurize/ate-runs/internvl3-svh-r14-full-middle-reaccess-textvqa-3090
持久结果：/home/featurize/work/ATE-internvl3-svh-r14-full-middle-reaccess-textvqa-3090/vlm_research/results/internvl3_svh_r14_full_middle_reaccess_textvqa_3090/
分支：codex/internvl3-svh-r14-full-middle-reaccess-textvqa-3090

研究范围判断：若R14未明显优于R6且仍远低于Native，当前证据不支持把继续扩大回看范围当作已证实方案；任何新增通路应另立协议，本轮不执行。
