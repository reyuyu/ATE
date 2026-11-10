# stop_policy_v2 补完：原推理协议不变

父结果Commit：6c294aae67ddb2a5cb3706ded76dcf3edb26c859。本分支保留原protocol.json字节及全部原始暂停证据；v2不改变模型、Prompt、图片、评分器、原生EOS、生成上限或Physical-D。原四项全量结果按父Commit直接复用，不重新推理或重计分。

# InternVL3-SVH 七项 Benchmark 单图块质量评测

Physical-D 相比同卡 Native 的已完成全集差值：GQA -9.048 pp；ScienceQA-IMG -0.050 pp；TextVQA -55.744 pp；POPE -0.222 pp；MME Perception +3.388 raw_points；MME Cognition -10.000 raw_points；MMBench -0.255 Circular_pp。未完成项不推断得分或损失。

普通95%CI整体低于0的项目：GQA、TextVQA；区间包含0的项目：ScienceQA-IMG、POPE、MME、MME、MMBench。区间包含0不能解释为证明等价。固定单图块协议与作者多图块/完整推理配方不等同，单题异常保留、计分并继续；仅结构错误或预先冻结的系统性异常触发停止。

当前已完整完成 6/7 项。只有 COMPLETE 行为全集正式分数；其余不推断缺失分数。

| Benchmark | N | Native | Physical-D | Δ | 95% CI | 单位 / 状态 |
|---|---:|---:|---:|---:|---|---|
| GQA  | 12578 | 62.156 | 53.109 | -9.048 | [-10.044, -8.066] | pp / COMPLETE |
| ScienceQA-IMG  | 2017 | 97.521 | 97.471 | -0.050 | [-0.289, +0.151] | pp / COMPLETE |
| TextVQA  | 5000 | 66.342 | 10.598 | -55.744 | [-57.257, -54.207] | pp / COMPLETE |
| POPE  | 9000 | 90.444 | 90.222 | -0.222 | [-0.533, +0.078] | pp / COMPLETE |
| MME Perception | 2114 | 1695.865 | 1699.253 | +3.388 | [-14.110, +21.826] | raw_points / COMPLETE |
| MME Cognition | 260 | 625.714 | 615.714 | -10.000 | [-28.125, +0.000] | raw_points / COMPLETE |
| MMBench  | 4377 | 83.673 | 83.418 | -0.255 | [-0.673, +0.086] | Circular_pp / COMPLETE |
| OCRBench | — | — | — | — | — | READY_RESUME_V2 |

| POPE 子集 | 指标 | Native | Physical-D | Δpp | 95%CI |
|---|---|---:|---:|---:|---|
| micro_all | F1 | 90.131 | 89.834 | -0.297 | [-0.642, +0.032] |
| macro_3_strategies | F1 | 90.161 | 89.860 | -0.301 | [-0.645, +0.026] |
| random | Accuracy | 92.433 | 92.000 | -0.433 | [-0.733, -0.133] |
| random | F1 | 92.021 | 91.525 | -0.496 | [-0.837, -0.174] |
| popular | Accuracy | 90.767 | 90.633 | -0.133 | [-0.467, +0.200] |
| popular | F1 | 90.432 | 90.219 | -0.212 | [-0.585, +0.143] |
| adversarial | Accuracy | 88.133 | 88.033 | -0.100 | [-0.500, +0.300] |
| adversarial | F1 | 88.030 | 87.835 | -0.195 | [-0.613, +0.212] |

## 自然生成长度审计
各行N为该条件实际保存的生成数；暂停任务为子集诊断，不是全集。Token长度包含真实EOS Token。

| 任务 / 条件 | N | 均值 | P50 | P95 | 最大 | EOS% | 截断% | 解析失败% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GQA / native | 12578 | 2.32 | 2.0 | 4.0 | 15 | 100.00 | 0.00 | 0.00 |
| GQA / physical_d | 12578 | 2.28 | 2.0 | 3.0 | 14 | 100.00 | 0.00 | 0.00 |
| ScienceQA-IMG / native | 2017 | 2.00 | 2.0 | 2.0 | 2 | 100.00 | 0.00 | 0.00 |
| ScienceQA-IMG / physical_d | 2017 | 2.00 | 2.0 | 2.0 | 2 | 100.00 | 0.00 | 0.00 |
| TextVQA / native | 5000 | 4.19 | 3.0 | 9.0 | 32 | 99.84 | 0.16 | 0.00 |
| TextVQA / physical_d | 5000 | 6.12 | 4.0 | 14.0 | 32 | 99.98 | 0.02 | 0.00 |
| POPE / native | 9000 | 2.83 | 3.0 | 3.0 | 3 | 100.00 | 0.00 | 0.00 |
| POPE / physical_d | 9000 | 2.88 | 3.0 | 3.0 | 3 | 100.00 | 0.00 | 0.00 |
| MME / native | 2374 | 2.97 | 3.0 | 3.0 | 3 | 100.00 | 0.00 | 0.00 |
| MME / physical_d | 2374 | 2.98 | 3.0 | 3.0 | 3 | 100.00 | 0.00 | 0.00 |
| MMBench / native | 4377 | 2.00 | 2.0 | 2.0 | 2 | 100.00 | 0.00 | 0.00 |
| MMBench / physical_d | 4377 | 2.00 | 2.0 | 2.0 | 2 | 100.00 | 0.00 | 0.00 |
| OCRBench / native | 490 | 4.24 | 4.0 | 8.0 | 38 | 100.00 | 0.00 | 0.00 |
| OCRBench / physical_d | 490 | 3.29 | 2.0 | 8.0 | 59 | 100.00 | 0.00 | 0.00 |

## 输出异常
全部原始异常和评分见 output_anomalies.csv。以下仅列前20条诊断，不删除其原始预测。

- GQA / 201185759 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：answering does not require reading text in the image.
- GQA / 201322641 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：answering does not require reading the text in the image.
- GQA / 201548912 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：answering does not require reading the text in the image
- GQA / 201682270 / native: ["UNEXPECTED_ANSWER_FORM"]; 原文：answering does not require reading text in the image
- GQA / 201758426 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：answering does not require reading text in the image.
- GQA / 202241127 / native: ["UNEXPECTED_ANSWER_FORM"]; 原文：answering does not require reading text in the image
- GQA / 20340893 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：answering does not require reading text in the image.
- GQA / 20929402 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：answering does not require reading the text in the image.
- TextVQA / 34935 / native: ["ABNORMAL_STOP_OR_TRUNCATION"]; 原文：saggio intorno diverse opinioni di alcuni moderni politici sopra i delitti e le pene dell'avvocato francescantonio
- TextVQA / 35106 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：the answer is not in the text so it is unknown
- TextVQA / 35266 / native: ["REPETITION","ABNORMAL_STOP_OR_TRUNCATION"]; 原文：3.9% ALE 3.9% ALE 3.9% ALE 3.9% ALE 3.9%
- TextVQA / 35450 / native: ["ABNORMAL_STOP_OR_TRUNCATION"]; 原文：GAROPODE 1.00 HOT CHOCOLATE 1.50 FRESH FRUITS 0.50 DRINK
- TextVQA / 35554 / native: ["UNEXPECTED_ANSWER_FORM","ABNORMAL_STOP_OR_TRUNCATION"]; 原文："this is the first work of high heroic fantasy i've seen that has taken the work of tolkien, assimilated it totally and deeply and made something
- TextVQA / 36184 / physical_d: ["REPETITION","ABNORMAL_STOP_OR_TRUNCATION"]; 原文：j. j. j. j. j. j. j. j. j. j. j. j. j. j. j. j.
- TextVQA / 36329 / native: ["UNEXPECTED_ANSWER_FORM"]; 原文：i don't believe in love because i still don't have you!
- TextVQA / 36329 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：the answer is in the image
- TextVQA / 36339 / native: ["ABNORMAL_STOP_OR_TRUNCATION"]; 原文：devi lask, x-men, final crisis, batman, amazing spider-man, army of ex, captain america, xanador, secret war
- TextVQA / 36351 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：the answer is not in the image
- TextVQA / 36372 / native: ["ABNORMAL_STOP_OR_TRUNCATION"]; 原文：allALL WAY</ref><box>[[210, 7STOP</ref><box>[[62, 65, 979, 7BUSH
- TextVQA / 36810 / native: ["ABNORMAL_STOP_OR_TRUNCATION"]; 原文：User, Team, College, User: who: st, GET /user/{userId}, PUT /user/{userId}, DELETE /user/{userId}, POST /

主标签：ATE_SVH_1TILE_SCORER_ALIGNED。不是 InternVL 作者完整排行榜配置；固定 1 tile、256 视觉 Token，与多图块、CoT、作者 max_new_tokens 等设置存在差异。MMBench 为 EN Dev20230712 Circular，采用官方 exact_matching，不调用 LLM judge。TextVQA 为 image-only，保留全部十答案共识软评分。MME 保留已知历史图片 SHA 差异，报告 Perception/Cognition 原始积分。

Native 保留原生答案读图；Physical-D 同时包含答案禁图、L20视觉退出、L2–L5答案Attention跳算。因此配对差值是联合策略效果，不能归因于单一机制。数值状态保持 FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER，不宣称 NUMERICAL_PASS。Physical-D 核心来自 e288eee271b7aeb64e5df59783d3201bb2a6b313，SHA256 不变。Native/No-op/官方 Chat 严格一致及物理结构检查独立记录在各任务 smoke_validation.json。

任务安全上限：POPE 8；MME/GQA/ScienceQA/MMBench 16；TextVQA 32；OCRBench 128。Greedy、B1、正常 EOS、每次全新 Cache。v1预检、暂停和异常原文均保留在history/stop_policy_v1及父Commit。本轮stop_policy_v2仅改变继续执行策略：单题解释、重复、错误/解析失败或原上限截断均保留并计分，不据此停止全集；结构性错误、NaN/Inf/OOM、哈希不符和冻结系统性监测规则仍硬停止。

95%CI 为固定 seed=20261110、20,000 次 Image SHA256 整簇配对 Bootstrap 的普通区间，未作多重比较校正，不证明等价。OCRBench 使用固定1000题尺度的簇重采样均值；MME 每次重采样重新计算各类 acc+acc_plus；Circular 按完整发布题组聚合。不同原生单位不平均。

每个任务的逐题完整答案、Token IDs、Gold、输入绑定和分数见对应 per_question.csv.gz；原始 SQLite 检查点保留在服务器 SSD，并经哈希校验同步持久化目录。图像读取时校验实际 SHA，未重新下载模型或图片。

本轮不执行固定32 Token效率测试。自然生成耗时或长度不能解释为物理加速。

| 任务 | 当前状态 / 错误 |
|---|---|
| GQA | GENERATION_COMPLETE:  |
| ScienceQA-IMG | GENERATION_COMPLETE:  |
| TextVQA | GENERATION_COMPLETE:  |
| POPE | GENERATION_COMPLETE:  |
| MME | GENERATION_COMPLETE:  |
| MMBench | GENERATION_COMPLETE:  |
| OCRBench | READY_RESUME_V2:  |

GPU UUID：GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e

服务器持久化目录：/home/featurize/work/ATE-internvl3-svh-stop-policy-v2-3090/vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090

GitHub 分支：https://github.com/reyuyu/ATE/tree/codex/internvl3-svh-stop-policy-v2-three-benchmarks-3090


## V2恢复与异常保留

- GQA：COMPLETE；复用原预测 1544 条，新生成 23612 条。完整原始JSON字节对照见prediction_provenance_v2.csv.gz。
- TextVQA：COMPLETE；复用原预测 716 条，新生成 9284 条。完整原始JSON字节对照见prediction_provenance_v2.csv.gz。
- OCRBench：PAUSED_OUTPUT_FORMAL；复用原预测 待分析 条，新生成 待分析 条。完整原始JSON字节对照见prediction_provenance_v2.csv.gz。

原父版本暂停样本（GQA无关解释、TextVQA长书名截断、OCRBench重复标题）均原样保留且纳入本轮评分，没有重生成、跳过或修正。TextVQA始终为32 Token；任何未来64/128 Token版本须独立完整运行，不能混入本轮。
OUTPUT_ANOMALY完整清单见各任务及根目录output_anomalies_v2.csv.gz：包括解释、重复、截断、零分/解析失败及部分TextVQA软评分。错误与部分分数只用于质量审计，不用于停止判断，也不改变官方评分。
系统性监测按100题/方法的非重叠窗口：连续两个窗口空输出≥10%、重复≥10%、其他行为格式异常≥25%或截断≥50%时暂停；连续10题空输出/重复也暂停。阈值在新增GPU生成前冻结，不使用Gold或Accuracy来决定停止、不自动调整。逐窗口及长度P50/P95/最大值见各任务output_monitor_v2.json。
MME完整Benchmark为2374条；表中Perception为2114条、Cognition为260条，分别按官方原始积分计算。MMBench为4377条已发布旋转记录、1176个Circular组。原四项汇总与逐题产物的SHA256保持父版本一致。
数值状态仍是FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER，Native/No-op及新结构性错误仍为硬门槛。所有比较仍是单图块ATE协议，不能冒充作者完整多图块排行榜结果。
