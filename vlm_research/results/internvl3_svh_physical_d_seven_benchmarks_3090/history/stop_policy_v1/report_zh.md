# InternVL3-SVH 七项 Benchmark 单图块质量评测

Physical-D 相比同卡 Native 的已完成全集差值：ScienceQA-IMG -0.050 pp；POPE -0.222 pp；MME Perception +3.388 raw_points；MME Cognition -10.000 raw_points；MMBench -0.255 Circular_pp。未完成项不推断得分或损失。

普通95%CI整体低于0的项目：暂无；区间包含0的项目：ScienceQA-IMG、POPE、MME、MME、MMBench。区间包含0不能解释为证明等价。固定单图块协议与作者多图块/完整推理配方不等同，异常输出项按门槛暂停并保留原文。

当前已完整完成 4/7 项。只有 COMPLETE 行为全集正式分数；其余不推断缺失分数。

| Benchmark | N | Native | Physical-D | Δ | 95% CI | 单位 / 状态 |
|---|---:|---:|---:|---:|---|---|
| GQA | — | — | — | — | — | PAUSED_OUTPUT_FORMAL |
| ScienceQA-IMG  | 2017 | 97.521 | 97.471 | -0.050 | [-0.289, +0.151] | pp / COMPLETE |
| TextVQA | — | — | — | — | — | PAUSED_OUTPUT_FORMAL |
| POPE  | 9000 | 90.444 | 90.222 | -0.222 | [-0.533, +0.078] | pp / COMPLETE |
| MME Perception | 2374 | 1695.865 | 1699.253 | +3.388 | [-14.110, +21.826] | raw_points / COMPLETE |
| MME Cognition | 2374 | 625.714 | 615.714 | -10.000 | [-28.125, +0.000] | raw_points / COMPLETE |
| MMBench  | 4377 | 83.673 | 83.418 | -0.255 | [-0.673, +0.086] | Circular_pp / COMPLETE |
| OCRBench | — | — | — | — | — | PAUSED_OUTPUT_FORMAL |

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
| GQA / native | 772 | 2.31 | 2.0 | 4.0 | 5 | 100.00 | 0.00 | 0.00 |
| GQA / physical_d | 772 | 2.27 | 2.0 | 3.0 | 13 | 100.00 | 0.00 | 0.00 |
| ScienceQA-IMG / native | 2017 | 2.00 | 2.0 | 2.0 | 2 | 100.00 | 0.00 | 0.00 |
| ScienceQA-IMG / physical_d | 2017 | 2.00 | 2.0 | 2.0 | 2 | 100.00 | 0.00 | 0.00 |
| TextVQA / native | 358 | 4.19 | 3.0 | 8.1 | 32 | 99.72 | 0.28 | 0.00 |
| TextVQA / physical_d | 358 | 6.42 | 4.0 | 14.0 | 21 | 100.00 | 0.00 | 0.00 |
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
- TextVQA / 34935 / native: ["ABNORMAL_STOP_OR_TRUNCATION"]; 原文：saggio intorno diverse opinioni di alcuni moderni politici sopra i delitti e le pene dell'avvocato francescantonio
- OCRBench / 0480 / physical_d: ["REPETITION"]; 原文：2016 Calendar - 2016 Calendar - 2016 Wall Calendar - 2016 Monthly Wall Calendar - 2016 Calendar - 2016 Calendar - 2016 Calendar - 2016 Wall Calendar

主标签：ATE_SVH_1TILE_SCORER_ALIGNED。不是 InternVL 作者完整排行榜配置；固定 1 tile、256 视觉 Token，与多图块、CoT、作者 max_new_tokens 等设置存在差异。MMBench 为 EN Dev20230712 Circular，采用官方 exact_matching，不调用 LLM judge。TextVQA 为 image-only，保留全部十答案共识软评分。MME 保留已知历史图片 SHA 差异，报告 Perception/Cognition 原始积分。

Native 保留原生答案读图；Physical-D 同时包含答案禁图、L20视觉退出、L2–L5答案Attention跳算。因此配对差值是联合策略效果，不能归因于单一机制。数值状态保持 FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER，不宣称 NUMERICAL_PASS。Physical-D 核心来自 e288eee271b7aeb64e5df59783d3201bb2a6b313，SHA256 不变。Native/No-op/官方 Chat 严格一致及物理结构检查独立记录在各任务 smoke_validation.json。

任务安全上限：POPE 8；MME/GQA/ScienceQA/MMBench 16；TextVQA 32；OCRBench 128。Greedy、B1、正常 EOS、每次全新 Cache。预检及正式异常记录完整保留；未通过长度门槛的任务暂停，不将子集成绩写成全集成绩。

95%CI 为固定 seed=20261110、20,000 次 Image SHA256 整簇配对 Bootstrap 的普通区间，未作多重比较校正，不证明等价。OCRBench 使用固定1000题尺度的簇重采样均值；MME 每次重采样重新计算各类 acc+acc_plus；Circular 按完整发布题组聚合。不同原生单位不平均。

每个任务的逐题完整答案、Token IDs、Gold、输入绑定和分数见对应 per_question.csv.gz；原始 SQLite 检查点保留在服务器 SSD，并经哈希校验同步持久化目录。图像读取时校验实际 SHA，未重新下载模型或图片。

本轮不执行固定32 Token效率测试。自然生成耗时或长度不能解释为物理加速。

| 任务 | 当前状态 / 错误 |
|---|---|
| GQA | PAUSED_OUTPUT_FORMAL:  |
| ScienceQA-IMG | GENERATION_COMPLETE:  |
| TextVQA | PAUSED_OUTPUT_FORMAL:  |
| POPE | GENERATION_COMPLETE:  |
| MME | GENERATION_COMPLETE:  |
| MMBench | GENERATION_COMPLETE:  |
| OCRBench | PAUSED_OUTPUT_FORMAL:  |

GPU UUID：GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e

服务器持久化目录：/home/featurize/work/ATE-internvl3-svh-seven-benchmarks-3090/vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090

GitHub 分支：https://github.com/reyuyu/ATE/tree/codex/internvl3-svh-physical-d-seven-benchmarks-3090

## 最终独立审计与暂停解释

全量完成 4/7 项：ScienceQA-IMG、POPE、MME、MMBench。GQA 保存 772 对，TextVQA 保存 358 对，OCRBench 保存 490 对；这三个子集不报告全量分数、差值或置信区间。

- GQA Q201185759：Physical-D 输出无关元解释，13 Token 后正常 EOS；按冻结短答案规则暂停。
- TextVQA Q34935：Native 转录较长意大利文书名，32 Token 达上限且无 EOS。可能是合理的真实长书名，但仍触发本轮预先冻结的保守规则；这不是 Physical-D 长输出或退化的证据，未事后提高上限。
- OCRBench Q0480：Native 正确输出 The Botanical Garden 2012；Physical-D 输出反复的 2016 Calendar 词组，59 Token 后正常 EOS，未达128上限。按重复规则暂停，完整原文与Token保留。输入/Prompt相同、EOS正确、Native/No-op和结构Smoke通过、最终逐层KV审计通过；重复行为的内部原因尚未确定，未据此放宽门槛或修改机制。

预检样本为冻结全集成员。输入、协议、源码绑定一致的预检预测在全集中复用一次，不重复计分；Smoke生成不计入Benchmark分数。各暂停项的保存数也包含已完成预检成员。

MME Cognition下降10原始分：代码推理−7.5、数值计算−2.5；计数、位置、MME OCR的官方分数保持不变，颜色+5。其余子任务和损害案例均保留，不以总体平均掩盖。未完成TextVQA/OCRBench，故不能给出这两项总体OCR能力损失结论。POPE Random Accuracy的普通95%CI低于0，属于多重描述性比较，未作校正。

最终CPU独立审计覆盖全部保存预测：冻结Question/Gold/Prompt/Image SHA/输入ID Tensor字节哈希、逐方法输入绑定、真实EOS/上限、完整题数及28层实际KV布局/Live字节均通过；九个冻结运行文件SHA256未变。官方数据/Scorer对齐、单图块口径与原严格数值Waiver仍须共同引用，不宣称作者排行榜或NUMERICAL_PASS。

GPU最终释放检查见 gpu_release.json：指定3090无计算进程、无本轮Worker/Controller；不关闭服务器。全部原始检查点与日志留在服务器持久化工作盘，轻量源码/预测/报告通过独立分支同步；实际Commit以最终publication.json和GitHub分支头为准。
