# InternVL3-SVH-R TextVQA Image-only快速搜索

**状态：CONFIRMED_EXPLORATORY_TEXTVQA_RECOVERY。** 本轮为EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER，历史NUMERICAL_EQUIVALENCE_FAILED保持原样，不宣称Dense/Physical数值等价。

Discovery冻结质量候选 **R6**（K=[14, 15, 16, 17, 18, 19]），低成本候选 **R6**（K=[14, 15, 16, 17, 18, 19]）。
独立Holdout96：Native 58.646%，R0 6.354%，R6 10.104%，R6 10.104%。
是否恢复取决于以下同题软分差和图像簇区间，不能以回答更正常代替质量收益。不得将96题结果当作完整5000题成绩。

## Image-only审计与抽样

全部5000题均重新构造原生Chat Template、原始问题及冻结指令，逐题比较完整Prompt、Token IDs和256图像Token位置。指令严格为：
Answer with the text visible in the image, using a word or short phrase. Do not explain.

数据目录含ocr，实际模型输入只有原图448×448单图块像素、问题、模板和该指令；没有OCR Tokens、sidecar、reference text、Caption、Hint、Gold或Rationale。200张选中图片（正式192、独立Smoke8）经过原图SHA、官方Processor和BF16像素Tensor绑定核验。image_only_input_audit.json、population_prompt_audit.csv和sample_manifest.json提供可审计证据。

先冻结sampling_freeze.json，再读取历史Native的绑定元数据。Discovery96与Holdout96按Image SHA划分，每题不同图片；Smoke另有8张不重叠图片。问题文字启发式七类，不使用Gold长度、预测或旧正确性选题。两集合实际分层均为：signs16/products16/titles12/numbers16/short_words8/multiword16/other12。该分层改变了总体问题分布，不能与历史全5000题均值直接比较。

## 阶段A：48题七候选

| method | N | soft_accuracy_pct | meta_count | answering_prefix_count | mean_KV_capacity_MiB |
| --- | --- | --- | --- | --- | --- |
| R0 | 48 | 10.833 | 20 | 19 | 3.882 |
| R2 | 48 | 9.583 | 16 | 16 | 4.866 |
| R4-M | 48 | 12.917 | 17 | 17 | 5.828 |
| R4-L | 48 | 9.583 | 14 | 14 | 5.847 |
| R6 | 48 | 17.083 | 15 | 15 | 6.828 |
| R8 | 48 | 15.625 | 15 | 15 | 7.826 |
| R10 | 48 | 15.625 | 15 | 15 | 8.846 |
| Native | 48 | 63.750 | 0 | 0 | 18.366 |

按GPU前冻结规则：Soft Accuracy优先，再按元回答率和实际KV成本打破平局，仅前三个回看候选和R0补齐另外48题。不按48题结果宣称显著性。

## Discovery96与冻结选择

| method | N | soft_accuracy_pct | meta_count | answering_prefix_count | mean_KV_capacity_MiB |
| --- | --- | --- | --- | --- | --- |
| R0 | 96 | 9.792 | 39 | 38 | 3.872 |
| R6 | 96 | 12.604 | 30 | 29 | 6.821 |
| R8 | 96 | 11.875 | 30 | 29 | 7.826 |
| R10 | 96 | 11.875 | 28 | 28 | 8.834 |
| Native | 96 | 58.021 | 0 | 0 | 18.382 |

best_quality为Soft Accuracy最高；best_efficiency允许距离最高2.0pp，优先较少层和较低实测KV；R0亦可成为低成本候选。最终选择SHA：c56d638789c2065dd4bcde3176edba6e1b3f1d65725c7e59f1cb6432ddb3df05。Holdout预测正文/Token/评分在此之前没有读取；只用于事前绑定核验的历史元数据不参与选层。

## Holdout96：真实质量与配对不确定性

| method | N | soft_accuracy_pct | meta_count | answering_prefix_count | mean_KV_live_MiB | mean_KV_capacity_MiB |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 96 | 6.354 | 43 | 43 | 3.906 | 3.906 |
| R6 | 96 | 10.104 | 36 | 36 | 6.852 | 6.852 |
| Native | 96 | 58.646 | 0 | 0 | 18.411 | 18.411 |

| candidate | difference_pp | CI95_low_pp | CI95_high_pp | improved | declined | unchanged | meta_resolved | meta_resolved_and_score_improved | meta_to_scored_answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R6 | 3.750 | 0.104 | 7.917 | 5 | 2 | 89 | 10 | 2 | 2 |

Bootstrap采用Image SHA整簇配对5000次，普通未校正95%CI。样本每图一题仍按图像簇执行。搜索与多子类带来选择和多重比较，CI跨零不证明等价，不宣称全量非劣效。无Holdout后重选层。若两个候选都没有改善，则为NO_CONFIRMED_TEXTVQA_RECOVERY；若点估计正向但区间未排除零，明确属于未确认趋势，也不自动测速。

## 哪些文字问题恢复、元回答是否转成有效文字答案

| method | question_text_type | N | soft_accuracy_pct | difference_vs_R0_pp | improved | declined | meta_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | signs | 16 | 12.500 | 0.000 | 0 | 0 | 11 |
| R0 | products | 16 | 0.000 | 0.000 | 0 | 0 | 11 |
| R0 | titles | 12 | 10.833 | 0.000 | 0 | 0 | 1 |
| R0 | numbers | 16 | 0.000 | 0.000 | 0 | 0 | 2 |
| R0 | short_words | 8 | 15.000 | 0.000 | 0 | 0 | 3 |
| R0 | multiword | 16 | 0.000 | 0.000 | 0 | 0 | 7 |
| R0 | other | 12 | 13.333 | 0.000 | 0 | 0 | 8 |
| R6 | signs | 16 | 12.500 | 0.000 | 0 | 0 | 10 |
| R6 | products | 16 | 6.250 | 6.250 | 1 | 0 | 7 |
| R6 | titles | 12 | 15.833 | 5.000 | 1 | 1 | 2 |
| R6 | numbers | 16 | 1.875 | 1.875 | 1 | 0 | 2 |
| R6 | short_words | 8 | 11.250 | -3.750 | 0 | 1 | 3 |
| R6 | multiword | 16 | 12.500 | 12.500 | 2 | 0 | 4 |
| R6 | other | 12 | 13.333 | 0.000 | 0 | 0 | 8 |
| Native | signs | 16 | 67.500 | 55.000 | 10 | 1 | 0 |
| Native | products | 16 | 75.000 | 75.000 | 12 | 0 | 0 |
| Native | titles | 12 | 58.333 | 47.500 | 7 | 2 | 0 |
| Native | numbers | 16 | 44.375 | 44.375 | 9 | 0 | 0 |
| Native | short_words | 8 | 56.250 | 41.250 | 4 | 0 | 0 |
| Native | multiword | 16 | 66.250 | 66.250 | 11 | 0 | 0 |
| Native | other | 12 | 35.833 | 22.500 | 5 | 1 | 0 |

全部原始元回答、空输出、重复和截断均原样纳入十答案官方软评分，没有删题、修改答案或更改Prompt。meta_response_analysis.csv报告answering前缀与更广元回答启发式；per_question_changes.csv逐题区分消除元回答、软分提高及实际获得非零软分，避免仅凭文本变正常宣称OCR能力恢复。元回答检测依据完整文本内容，不仅依赖长度/EOS。启发式分组和元回答标记均可能有歧义，正式Soft Accuracy不依赖这些标签。

## 相同四层预算

| phase | base | candidate | N | difference_pp | CI95_low_pp | CI95_high_pp |
| --- | --- | --- | --- | --- | --- | --- |
| StageA48_R4_equal_budget | R4-M | R4-L | 48 | -3.333 | -10.833 | 4.167 |

StageA48是两个四层候选共同预设比较；若二者均入围，另报告共同Discovery96。未入Holdout的条件不补测、不冒充独立确认。

## KV成本与有条件速度

| method | N | mean_actual_quality_EOS_KV_live_MiB | mean_actual_quality_EOS_KV_capacity_MiB | KV_increment_vs_R0_MiB | mean_generated_tokens |
| --- | --- | --- | --- | --- | --- |
| R0 | 96 | 3.906 | 3.906 | 0.000 | 6.885 |
| R6 | 96 | 6.852 | 6.852 | 2.946 | 5.740 |
| Native | 96 | 18.411 | 18.411 | 14.505 | 4.219 |

这里是正常EOS质量生成结束时真实持久KV，不是固定32KV，不是理论容量。各方法输出长度不同，成本同时受答案长度影响。每层K/V形状、BF16 dtype、物理长度与底层存储字节均独立重算，L2–L5持久KV为零，只有K层持久保留256原始视觉KV，非K层仅文本KV。临时零值Key/Value scratch不等于持久原始视觉KV，也不声称消除了全部被屏蔽Key计算。

测速状态：**COMPLETE**。GPU前冻结门槛为预选候选相对R0软分为正且图像簇95%CI下界大于零；候选优先次序预设为best_quality、best_efficiency，不按Holdout重新排名。

| method | requests | E2E_ms | TTFT_ms | TPOT_ms | KV_live_MiB | KV_capacity_MiB | peak_allocated_MiB | peak_reserved_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native | 36 | 1115.7428545266157 | 145.16987363984097 | 31.308805835057246 | 19.919921875 | 19.919921875 | 15286.63232421875 | 15444.0 |
| R0 | 36 | 1093.0702740265083 | 124.74011955540239 | 31.23645659584212 | 5.07421875 | 5.07421875 | 15286.63330078125 | 15444.0 |
| R6 | 36 | 1065.4014544185582 | 124.53205797242441 | 30.35062569181077 | 8.07421875 | 8.07421875 | 15286.63330078125 | 15444.0 |

实测R6相对Native E2E加速1.047倍，KV@32相对R0增加3.000MiB。12张冻结不同图片、每条件2预热及每图3正式重复；同步墙钟包含Vision、Projector、Prefill、31真实Decode和必要清理，排除磁盘、Tokenizer、预处理、H2D和写盘；旋转顺序，不复用质量阶段图像特征进行测速。

## Smoke、数值豁免与执行范围

Native/No-op有效Logits严格torch.equal通过，8张独立Smoke题Native完整Token与历史一致，192正式题历史Native全部完成Prompt/图像/BF16像素Tensor/Tokenizer/模型Revision/32上限绑定后复用。没有复用历史Physical-D，新R0全部由冻结V2本轮生成。

Smoke只运行R0/R4-M/R8对应Dense-R/Physical-R；最短/最长输入记录完整结构审计，其余只保留有效Logits摘要。L20真实退出256视觉Hidden、L2–L5答案Attention真实跳算、原始逻辑RoPE、KV权限、非目标连接、Norm/MLP/Residual、禁用概率0和Cache独立均通过。无NaN/Inf或Hook泄漏。全部质量审计关闭重型Hook，无28层Hidden或逐步全词表Logits复制。

- R0 Smoke：{'N': 8, 'full_tokens_equal': 8, 'normalized_answer_equal': 8, 'physical_meta': 4, 'dense_meta': 4}；相同生成前缀最大Logits MaxAbs=5.500000，步均MeanAbs=0.237650。
- R4-M Smoke：{'N': 8, 'full_tokens_equal': 7, 'normalized_answer_equal': 8, 'physical_meta': 4, 'dense_meta': 4}；相同生成前缀最大Logits MaxAbs=5.000000，步均MeanAbs=0.222326。
- R8 Smoke：{'N': 8, 'full_tokens_equal': 8, 'normalized_answer_equal': 8, 'physical_meta': 4, 'dense_meta': 4}；相同生成前缀最大Logits MaxAbs=3.000000，步均MeanAbs=0.224303。

历史数值FAILED保持原样，V2物理删行引起BF16 GEMM形状误差，本轮不使用C1/C2，也不做Kernel搜索。Dense/Physical的质量差异不能全部归因于视觉回看机制，不能宣称数值等价；结果只是对这一冻结实现的探索性工程观察。是否值得更大独立TextVQA验证应以Holdout趋势、区间、元回答真实恢复及KV/速度成本共同判断，不能自动扩大。

实际正式新增质量720次（上限816），历史Native正式参考192题、新Native正式质量0次，正式测速108请求；模型加载1次。没有运行GQA、OCRBench或其他Benchmark，没有训练、改Prompt、删异常或扩候选。全部预测由官方Answer Processor与独立leave-one-out实现逐条复算一致，来源/输入/协议/Cache字节核验通过。GPU释放：True；不关闭服务器。

SSD：/home/featurize/ate-runs/internvl3-svh-r-fast-textvqa-imageonly-3090
持久结果：/home/featurize/work/ATE-internvl3-svh-r-fast-textvqa-imageonly-3090/vlm_research/results/internvl3_svh_r_fast_textvqa_imageonly_3090/
分支：codex/internvl3-svh-r-fast-textvqa-imageonly-3090

## 十个研究问题的直接结论

1. Discovery按冻结规则选中的最佳配置为R6，K=[14, 15, 16, 17, 18, 19]；best_efficiency=R6。
2. 独立Holdout相对R0软分+3.750pp，普通95%CI [+0.104, +7.917]pp，软分提高/下降/不变=5/2/89。
3. 元回答由43降到36；消除10条，其中仅2题伴随软分提高，不能只凭元回答下降声称恢复识字。
4. 按软分总和的净恢复贡献排序：multiword +2.000软分单位（提高2、下降0题）, products +1.000软分单位（提高1、下降0题）, titles +0.600软分单位（提高1、下降1题）, numbers +0.300软分单位（提高1、下降0题）, signs +0.000软分单位（提高0、下降0题）, other +0.000软分单位（提高0、下降0题）, short_words -0.300软分单位（提高0、下降1题）。各类仅8–16题，属于描述性观察。
5. 四层中心/后段的共同StageA48结果与配对CI见上表；不能据粗筛或仅元回答率判断位置优劣，两者未进入独立Holdout。
6. 本轮显示独立正向收益的候选使用6层；2/4层只在48题粗筛，未证明6层是必要下限，也未独立验证所有更小预算。
7. 真实质量EOS KV增量见kv_cost.csv，固定32实际KV及E2E仅使用本轮TextVQA测速，不能混用长度或GQA数据。
8. 真实E2E收益及倍数见速度表，包含临时scratch开销，没有用理论Token节省代替实测。
9. 数值豁免保留历史FAILED，限制纯机制归因和数值等价结论；正向软分只说明当前冻结实现的小样本观察。
10. 与Native仍相差48.542pp，区间下界接近零；仅值得有限独立扩样确认，不宜据此直接宣称OCR退化已解决或自动跑全量5000题。
