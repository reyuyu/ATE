# InternVL3 SVH-R Late9 七项质量评测

状态：**PARTIAL_5_OF_7_FULL_BENCHMARKS**。本次仅CPU离线补算，不运行推理、不使用GPU、不下载或替换图片。

| Benchmark | Native | 原SVH | Late9 | Late9−SVH | 覆盖率 | 状态 |
|---|---:|---:|---:|---:|---|---|
| GQA (%) | 62.1561 | 53.1086 | 61.4327 | +8.3241 pp | 12578/12578 | FULL_COMPLETE |
| ScienceQA-IMG (%) | 97.5211 | 97.4715 | 97.4219 | -0.0496 pp | 2017/2017 | FULL_COMPLETE |
| TextVQA (%) | 66.3420 | 10.5980 | 56.0040 | +45.4060 pp | 5000/5000 | FULL_COMPLETE |
| POPE (%) | 90.4444 | 90.2222 | 90.4444 | +0.2222 pp | 9000/9000 | FULL_COMPLETE |
| MMBench Circular (%) | 83.6735 | 83.4184 | 83.5884 | +0.1701 pp | 4377/4377旋转记录，1176/1176组 | FULL_COMPLETE |
| MME共同有效子集 (points) | 2321.4703 | 2314.8546 | 2300.9363 | -13.9183 points | 共同有效2372/2374题、1186个完整双题组 | PARTIAL |
| OCRBench已测原始积分 | 2/3 | 0/3 | 2/3 | +2个正确答案 | 共同有效3/1000题 | PARTIAL |

MME为14个子任务分别计算Acc(0–100)+Acc+(0–100)后求和，不是混合所有题后的准确率。本表MME不是完整2374题官方成绩，OCRBench不是完整0–1000分，也不将3题比例外推。五项完整成绩保持原值。机器可读总表：`seven_benchmark_scores.csv`。

## MME14子任务共同有效样本成绩

每格顺序为 **Acc / Acc+ / Score**，前三项以百分数或对应积分表示。三方法严格使用相同题目和完整双题组；所有14任务均有有效双题组。

| 子任务 | 共同有效/计划题数 | 双题组 | Native | 原SVH | Late9 |
|---|---:|---:|---:|---:|---:|
| existence | 60/60 | 30 | 98.3333 / 96.6667 / 195.0000 | 98.3333 / 96.6667 / 195.0000 | 98.3333 / 96.6667 / 195.0000 |
| count | 60/60 | 30 | 85.0000 / 70.0000 / 155.0000 | 85.0000 / 70.0000 / 155.0000 | 85.0000 / 70.0000 / 155.0000 |
| position | 60/60 | 30 | 88.3333 / 80.0000 / 168.3333 | 88.3333 / 80.0000 / 168.3333 | 88.3333 / 80.0000 / 168.3333 |
| color | 60/60 | 30 | 91.6667 / 83.3333 / 175.0000 | 93.3333 / 86.6667 / 180.0000 | 91.6667 / 83.3333 / 175.0000 |
| posters | 294/294 | 147 | 91.1565 / 82.9932 / 174.1497 | 91.4966 / 83.6735 / 175.1701 | 91.1565 / 82.9932 / 174.1497 |
| celebrity | 340/340 | 170 | 85.2941 / 70.5882 / 155.8824 | 85.0000 / 70.0000 / 155.0000 | 85.0000 / 70.0000 / 155.0000 |
| scene | 400/400 | 200 | 86.5000 / 74.0000 / 160.5000 | 86.7500 / 74.5000 / 161.2500 | 86.7500 / 74.5000 / 161.2500 |
| landmark | 398/400 | 199 | 92.7136 / 85.4271 / 178.1407 | 92.4623 / 84.9246 / 177.3869 | 91.9598 / 84.4221 / 176.3819 |
| artwork | 400/400 | 200 | 84.2500 / 72.0000 / 156.2500 | 83.5000 / 71.0000 / 154.5000 | 83.7500 / 71.0000 / 154.7500 |
| OCR | 40/40 | 20 | 92.5000 / 85.0000 / 177.5000 | 92.5000 / 85.0000 / 177.5000 | 92.5000 / 85.0000 / 177.5000 |
| commonsense_reasoning | 140/140 | 70 | 81.4286 / 64.2857 / 145.7143 | 81.4286 / 64.2857 / 145.7143 | 80.7143 / 62.8571 / 143.5714 |
| numerical_calculation | 40/40 | 20 | 72.5000 / 45.0000 / 117.5000 | 70.0000 / 45.0000 / 115.0000 | 72.5000 / 45.0000 / 117.5000 |
| text_translation | 40/40 | 20 | 100.0000 / 100.0000 / 200.0000 | 100.0000 / 100.0000 / 200.0000 | 100.0000 / 100.0000 / 200.0000 |
| code_reasoning | 40/40 | 20 | 87.5000 / 75.0000 / 162.5000 | 85.0000 / 70.0000 / 155.0000 | 82.5000 / 65.0000 / 147.5000 |
| 14任务共同子集部分总分 | 2372/2374 | 1186 | 2321.4703 | 2314.8546 | 2300.9363 |

Late9−Native = **-20.5340 MME points**；Late9−SVH = **-13.9183 MME points**。

三方法各自都成功保存2372题，题目集合完全相同，全部组成1186个完整双题组。因此`MME_AVAILABLE_PER_METHOD`视图与共同子集视图分母、结果相同，并非拿不同分母相减。各方法均无空答案、截断、非EOS结束或格式异常；没有单方法额外失败或无法评分的记录。错误答案仍正常计分。

缺失的`landmark/03d5e3bfc958be38.jpg:0`和`:1`属于同一双题组，三方法都没有该组预测。要求的原图SHA256为`ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a`，原件903786字节；本机同名图不是相同字节。本轮未插补、未换图、未自动记错。

完整数据：`mme_common_subset_14tasks.csv`、`mme_common_subset_totals.json`、`mme_available_per_method.csv`和`mme_missing_pair_audit.csv`。冻结官方`calculation.py`逐任务结果与独立有理数计算交叉验证，原生官方输出保存在`mme_official_scorer_output.json`。

历史另一轮完整2374题的Native约2321.58、SVH约2314.97仅为背景，不参加本表差值，不替换本轮实测分母。

## OCRBench实际已评分逐题表

以下文本均为原始保存输出；数学答案Gold的完整候选、原图SHA256、Token IDs、输入绑定和停止状态见CSV，没有改写评分规则。

| Question ID | 数据子类 / 来源 | 原始Gold | Native输出 (分) | 原SVH输出 (分) | Late9输出 (分) |
|---|---|---|---|---|---|
| 0053 | Irregular Text Recognition / IC15_1811 | `EXPERIENCE` | `Experience` (1) | `Give` (0) | `Experience` (1) |
| 0963 | Handwritten Mathematical Expression Recognition / HME100k | `y _ { 2 } = \frac { - 1 - \sqrt { 5 } } { 2 }`；`y _ 2 = \frac { - 1 - \sqrt 5 } { 2 }` | `y_{2}=\frac{-1-\sqrt{5}}{2}` (1) | `x=\frac{1 }{2}` (0) | `y_{2}=\frac{-1-\sqrt{5}}{2}` (1) |
| 0988 | Handwritten Mathematical Expression Recognition / HME100k | `N a H C O _ { 3 } ( 4 ) H _ { 2 } S O`；`N a H C O _3 ( 4 ) H _2 S O` | `NaHCO_{33}(4)H_{2}SO` (0) | `HCl` (0) | `NaHCO_{3}(4)` (0) |

冻结OCRBench v1官方评分循环与独立标量实现一致：**Native 2/3，原SVH 0/3，Late9 2/3；PARTIAL，共同有效3/1000题**。9条预测均成功评分、正常EOS，无空答、截断或异常。没有按1000/3外推，不能据此判断全量OCRBench性能。详见`ocrbench_partial_per_question.csv`和`ocrbench_partial_scores.json`。

## CPU补算与发布验收

保留基线Commit `65c4cac2f925ebefb073c8db03caab119fcc765a`。新增脚本`vlm_research/internvl3_svhr_partial_scores.py`只读两个SQLite，重新核验全部106041条预测的原始JSON SHA256、Question/Gold/Image/Prompt/Processor/协议和源码绑定，并复算评分。数据库前后文件SHA256不变，原始逐题导出、旧CSV、历史验证及推理源码不改动。

新增9项CPU回归测试通过，涵盖官方MME14任务求和、完整双题组、缺失任务不记零、OCR大小写/子串及数学空格规则。MME和OCRBench逐题分数与历史记录完全一致。全量TextVQA另发现108条浮点末位差异，最大为`2.220446049250313e-16`；解析答案完全相同，全部低于旧CPU报告器既有`1e-10`评分复核门槛。原始值和新值逐条保存在`score_recalculation_discrepancies.json`，不静默改写历史。使用已有系统CPU Python/依赖，未升级任何模型环境；失败开发尝试和环境选择见`score_recalculation_execution.md`。

旧`artifact_hashes.json`完整保存在`pre_recalculation_artifact_hashes.json`，旧`final_cpu_review.json`仍是原Commit的冻结验收。当前根SHA清单重新封存更新报告和新增补算文件；`score_recalculation_validation.json`记录本轮输入、源码、输出及数据库前后哈希。原`validation.json`和`protocol.json`不改写，整体仍为PARTIAL，`EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER`及历史`NUMERICAL_EQUIVALENCE_FAILED`继续保留。

本轮依用户新授权提交并推送原同名实验分支，不修改main、不创建PR、不重写历史。远端HEAD及报告/CSV/JSON内容SHA256在推送后独立复核。

| 方法 | 持久原始视觉KV层数 | 视觉Hidden存活Block数 | 固定32 Token E2E/TTFT/TPOT |
|---|---:|---:|---|
| 原SVH | 0（0 MiB视觉数据） | 20 | 尚未测量 |
| Late9 | 9（4.5 MiB视觉数据） | 28 | 尚未测量 |

视觉数据大小来自全部逐请求实际K/V形状：2×4 KV Heads×256行×128维×2字节，每保留层0.5 MiB。Native为28层/14 MiB，Late9较Native少9.5 MiB（67.86%），但较原SVH多4.5 MiB；这不是总KV、分配器峰值或实测速率。详见visual_kv_payload_audit.json。
效率测试须在七项质量全部完成后执行。不能用自然EOS生成耗时或历史GPU测速替代本轮固定32 Token测试；Late9保留全部28层视觉Hidden，没有L20 Prefill早退收益。

## 配对差异

Image SHA256整簇配对Bootstrap 20,000次，seed=20261110，普通未校正95%CI。保留原生评分单位，TextVQA为软分变化而非二值纠错。

| Benchmark | 比较 | N | 差值 | 95%CI | 得分提高/降低 |
|---|---|---:|---:|---|---|
| GQA (FULL_COMPLETE) | physical_d−native | 12578 | -9.0475 | [-10.0444, -8.0664] | 401/1539 |
| GQA (FULL_COMPLETE) | late9−native | 12578 | -0.7235 | [-1.0888, -0.3498] | 263/354 |
| GQA (FULL_COMPLETE) | late9−physical_d | 12578 | +8.3241 | [+7.3966, +9.2598] | 1389/342 |
| ScienceQA-IMG (FULL_COMPLETE) | physical_d−native | 2017 | -0.0496 | [-0.2538, +0.1506] | 2/3 |
| ScienceQA-IMG (FULL_COMPLETE) | late9−native | 2017 | -0.0992 | [-0.3005, +0.0981] | 1/3 |
| ScienceQA-IMG (FULL_COMPLETE) | late9−physical_d | 2017 | -0.0496 | [-0.2551, +0.1510] | 2/3 |
| TextVQA (FULL_COMPLETE) | physical_d−native | 5000 | -55.7440 | [-57.2574, -54.2066] | 182/3102 |
| TextVQA (FULL_COMPLETE) | late9−native | 5000 | -10.3380 | [-11.4821, -9.2042] | 281/854 |
| TextVQA (FULL_COMPLETE) | late9−physical_d | 5000 | +45.4060 | [+43.9178, +46.9024] | 2560/170 |
| POPE (FULL_COMPLETE) | physical_d−native | 9000 | -0.2222 | [-0.5444, +0.0778] | 35/55 |
| POPE (FULL_COMPLETE) | late9−native | 9000 | +0.0000 | [-0.2444, +0.2444] | 29/29 |
| POPE (FULL_COMPLETE) | late9−physical_d | 9000 | +0.2222 | [-0.0556, +0.5111] | 47/27 |
| MME (PARTIAL) | physical_d−native | 2372 | -6.6157 | [-32.1751, +16.3655] | 6/10 |
| MME (PARTIAL) | late9−native | 2372 | -20.5340 | [-48.6802, +2.9276] | 3/11 |
| MME (PARTIAL) | late9−physical_d | 2372 | -13.9183 | [-35.3600, +2.2887] | 3/7 |
| MMBench (FULL_COMPLETE) | physical_d−native | 4377 | -0.2551 | [-0.6745, +0.0856] | 5/18 |
| MMBench (FULL_COMPLETE) | late9−native | 4377 | -0.0850 | [-0.4939, +0.2584] | 9/16 |
| MMBench (FULL_COMPLETE) | late9−physical_d | 4377 | +0.1701 | [-0.1688, +0.5141] | 10/4 |
| OCRBench (PARTIAL) | physical_d−native | 3 | -66.6667 | [-100.0000, +0.0000] | 0/2 |
| OCRBench (PARTIAL) | late9−native | 3 | +0.0000 | [+0.0000, +0.0000] | 0/0 |
| OCRBench (PARTIAL) | late9−physical_d | 3 | +66.6667 | [+0.0000, +100.0000] | 2/0 |

## 未完整项目共同子集

以下不是正式全量成绩；MME单位为原始积分，OCR子集为实测正确题数而非0–1000总分。

| 项目/共同子集 | 题数 | Native | 原SVH | Late9 |
|---|---:|---:|---:|---:|
| MME Perception | 2112/2114 | 1695.7560 | 1699.1403 | 1692.3649 |
| MME Cognition | 260/260 | 625.7143 | 615.7143 | 608.5714 |
| OCRBench已测原始积分 | 3/1000 | 2/3 | 0/3 | 2/3 |

下方历史配对统计表中OCRBench的百分比差值仅是这3题的数学归一化，不能称为OCRBench正式Accuracy；主表只报告实测原始积分。

## 十项研究问题

1. GQA Query与Non-Query分别比较；均值大幅恢复不等于逐题或完全恢复Native。
- GQA Query（6805题）：Late9−native=-1.1462pp，95%CI [-1.8044,-0.4981]；Late9−physical_d=+15.3564pp，95%CI [+13.7171,+17.0222]。CI跨0不等于等价；不能把均值恢复视为全部Query逐题恢复。
- GQA Non-Query（5773题）：Late9−native=-0.2252pp，95%CI [-0.5658,+0.1217]；Late9−physical_d=+0.0346pp，95%CI [-0.3340,+0.4164]。CI跨0不等于等价；不能把均值恢复视为全部Query逐题恢复。
2. TextVQA全5000题：Native/原SVH/Late9=66.3420/10.5980/56.0040。Late9−原SVH=+45.4060pp，95%CI [+43.9178,+46.9024]；软分提高/降低=2560/170题，0分变正分=2506题。按总体软分差计算恢复原损失的81.45%，不是二值纠错率。
3. ScienceQA、POPE、MME、MMBench：不能用净分掩盖逐题新增损害。
- ScienceQA-IMG（FULL_COMPLETE）：差值-0.0992，95%CI [-0.3005,+0.0981]；题目正确变错/错误变对=3/1。MMBench翻转为旋转记录层面，正式主分为Circular组；MME差值为原始积分。
- POPE（FULL_COMPLETE）：差值+0.0000，95%CI [-0.2444,+0.2444]；题目正确变错/错误变对=29/29。MMBench翻转为旋转记录层面，正式主分为Circular组；MME差值为原始积分。
- MME（PARTIAL）：差值-20.5340，95%CI [-48.6802,+2.9276]；题目正确变错/错误变对=11/3。MMBench翻转为旋转记录层面，正式主分为Circular组；MME差值为原始积分。
- MMBench（FULL_COMPLETE）：差值-0.0850，95%CI [-0.4939,+0.2584]；题目正确变错/错误变对=16/9。MMBench翻转为旋转记录层面，正式主分为Circular组；MME差值为原始积分。
4. OCRBench仅3/1000题三方严格配对（PARTIAL），不能给出正式0–1000总分，也不能判断全量OCR改善。
5. Late9相对原SVH的完整配对差值与20,000次图像簇Bootstrap区间见上表；缺失项目只对应共同子集，不能外推全量。
6. 9层回看持久视觉KV为4.5 MiB：较Native节省9.5 MiB，较原SVH增加4.5 MiB；总KV还含文本，按真实Token长度另见kv_layout.csv.gz。
7. 相对原SVH，视觉Hidden多存活8个Decoder Block，即每图多2048个视觉Token×Block计算行；实际Prefill额外毫秒尚未独立测量，不能用行数代替时间。
8. 七项质量未全部完成，因此按任务顺序不启动固定32 Token效率阶段。没有本轮可报告的E2E/TTFT/TPOT加速比，不能宣称Late9比Native快。
9. GQA/TextVQA/OCRBench复用逐记录原始SHA验证通过的历史Native与SVH；ScienceQA/POPE/MME/MMBench本轮重新生成两基线。缺原图的题目不进入三方法共同结果，保留BLOCKED清单。
10. 全部Late9质量仍带BF16历史数值豁免标签。Smoke严格一致只对已测输入成立，物理/Dense未获全域数值等价证明；原单tile、Image-only及暂停策略差异均须保留，不能冒充其他论文排行榜协议。


## 协议与复用

原SVH输入协议来自6c294aae，续跑结果来自3297559f；Late9物理引擎来自1407275c。旧代码和历史结果未修改。当前入口仅适配本地路径、GPU UUID、分片、严格复用和检查点。
GQA、TextVQA、OCRBench历史JSON含全部28层KV布局，重建后逐条匹配发布的原始记录SHA256；再核对本地输入与BF16 pixels后才复用。其他四项缺少可核验的原始逐记录SHA，因此Native/SVH重新配对生成。详见historical_reuse_audit.json与逐题generation_origin。逐题导出明确区分generation_gpu_uuid与input_verification_gpu_uuid；复用预测仍来自历史GPU，不冒充本轮生成。
BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0，448×448单图块、256视觉Token。各Benchmark保持原Prompt与生成上限，TextVQA没有OCR辅助输入。
数值标签继续为EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER，不宣称数值等价。两GPU各七任务Native/No-op/官方Chat严格一致；Late9物理布局、权限、RoPE、Norm/MLP与Cache独立性通过。Smoke的Dense/Physical误差完整保留，不能推广成所有输入数值一致。

## 覆盖与限制

本机MME的一张landmark图与冻结原件SHA不同，涉及同图两题；OCRBench本机HF版本与历史官方ZIP版本的绝大多数图片字节不同。Google Drive原件下载连接超时。未换图、重编码或改SHA，缺失请求完整保留并标记BLOCKED。
本轮续跑规则在GPU前冻结：非结构性长输出、空答、解释、重复、自然截断均保留计分；NaN/Inf/OOM、哈希、映射或KV结构错误停止Worker。该规则与旧v1暂停策略不同，不改变输入、评分或算法。
Late9与原SVH的收益不能解释为整模型仅需九层视觉处理，也不代表已加速。Prompt其他位置仍有视觉信息，且固定1tile协议不是作者多tile排行榜复现。

## 逐题与审计

per_question.csv.gz保存完整输出、Token IDs、Gold、评分、异常与绑定；kv_layout.csv.gz保存真实逐层形状和Live/Capacity。所有预测均独立CPU重评分，并核验原始JSON校验和、样本、Processor及实际KV长度。原始SQLite留server_only、不上传大缓存或权重。
原65c4cac2提交阶段仅本地保存；本次CPU补算依用户后续明确授权推送同名分支，不修改main、不创建PR。

## 正式执行耗时与资源释放

- Worker0正式进程墙钟约1.920小时（包含加载、输入检查和复用；不是方法速度测量）。
- Worker1正式进程墙钟约1.933小时（包含加载、输入检查和复用；不是方法速度测量）。
本实验两Worker已退出，nvidia-smi计算进程列表中均已消失；进程退出前的残留allocator字节原样保存在release.json，不冒充提前归零。GPU释放审计保留当时其他GPU进程，不终止或修改它们。
