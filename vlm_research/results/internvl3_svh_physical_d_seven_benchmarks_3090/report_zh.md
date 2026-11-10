# InternVL3-SVH 七项 Benchmark 单图块质量评测

Physical-D 相比同卡 Native 的已完成全集差值：ScienceQA-IMG -0.050 pp。未完成项不推断得分或损失。

普通95%CI整体低于0的项目：暂无；区间包含0的项目：ScienceQA-IMG。区间包含0不能解释为证明等价。固定单图块协议与作者多图块/完整推理配方不等同，异常输出项按门槛暂停并保留原文。

当前已完整完成 1/7 项。只有 COMPLETE 行为全集正式分数；其余不推断缺失分数。

| Benchmark | N | Native | Physical-D | Δ | 95% CI | 单位 / 状态 |
|---|---:|---:|---:|---:|---|---|
| GQA | — | — | — | — | — | PAUSED_OUTPUT_FORMAL |
| ScienceQA-IMG  | 2017 | 97.521 | 97.471 | -0.050 | [-0.289, +0.151] | pp / COMPLETE |
| TextVQA | — | — | — | — | — | READY_PREFLIGHT |
| POPE | — | — | — | — | — | READY_PREFLIGHT |
| MME | — | — | — | — | — | READY_PREFLIGHT |
| MMBench | — | — | — | — | — | READY_PREFLIGHT |
| OCRBench | — | — | — | — | — | READY_PREFLIGHT |

## 自然生成长度审计
各行N为该条件实际保存的生成数；暂停任务为子集诊断，不是全集。Token长度包含真实EOS Token。

| 任务 / 条件 | N | 均值 | P50 | P95 | 最大 | EOS% | 截断% | 解析失败% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GQA / native | 772 | 2.31 | 2.0 | 4.0 | 5 | 100.00 | 0.00 | 0.00 |
| GQA / physical_d | 772 | 2.27 | 2.0 | 3.0 | 13 | 100.00 | 0.00 | 0.00 |
| ScienceQA-IMG / native | 2017 | 2.00 | 2.0 | 2.0 | 2 | 100.00 | 0.00 | 0.00 |
| ScienceQA-IMG / physical_d | 2017 | 2.00 | 2.0 | 2.0 | 2 | 100.00 | 0.00 | 0.00 |

## 输出异常
全部原始异常和评分见 output_anomalies.csv。以下仅列前20条诊断，不删除其原始预测。

- GQA / 201185759 / physical_d: ["UNEXPECTED_ANSWER_FORM"]; 原文：answering does not require reading text in the image.

主标签：ATE_SVH_1TILE_SCORER_ALIGNED。不是 InternVL 作者完整排行榜配置；固定 1 tile、256 视觉 Token，与多图块、CoT、作者 max_new_tokens 等设置存在差异。MMBench 为 EN Dev20230712 Circular，采用官方 exact_matching，不调用 LLM judge。TextVQA 为 image-only，保留全部十答案共识软评分。MME 保留已知历史图片 SHA 差异，报告 Perception/Cognition 原始积分。

数值状态保持 FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER，不宣称 NUMERICAL_PASS。Physical-D 核心来自 e288eee271b7aeb64e5df59783d3201bb2a6b313，SHA256 不变。Native/No-op/官方 Chat 严格一致及物理结构检查独立记录在各任务 smoke_validation.json。

任务安全上限：POPE 8；MME/GQA/ScienceQA/MMBench 16；TextVQA 32；OCRBench 128。Greedy、B1、正常 EOS、每次全新 Cache。预检及正式异常记录完整保留；未通过长度门槛的任务暂停，不将子集成绩写成全集成绩。

95%CI 为固定 seed=20261110、20,000 次 Image SHA256 整簇配对 Bootstrap 的普通区间，未作多重比较校正，不证明等价。OCRBench 使用固定1000题尺度的簇重采样均值；MME 每次重采样重新计算各类 acc+acc_plus；Circular 按完整发布题组聚合。不同原生单位不平均。

每个任务的逐题完整答案、Token IDs、Gold、输入绑定和分数见对应 per_question.csv.gz；原始 SQLite 检查点保留在服务器 SSD，并经哈希校验同步持久化目录。图像读取时校验实际 SHA，未重新下载模型或图片。

本轮不执行固定32 Token效率测试。自然生成耗时或长度不能解释为物理加速。

| 任务 | 当前状态 / 错误 |
|---|---|
| GQA | PAUSED_OUTPUT_FORMAL:  |
| ScienceQA-IMG | GENERATION_COMPLETE:  |
| TextVQA | READY_PREFLIGHT:  |
| POPE | READY_PREFLIGHT:  |
| MME | READY_PREFLIGHT:  |
| MMBench | READY_PREFLIGHT:  |
| OCRBench | READY_PREFLIGHT:  |

GPU UUID：GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e

服务器持久化目录：/home/featurize/work/ATE-internvl3-svh-seven-benchmarks-3090/vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090

GitHub 分支：https://github.com/reyuyu/ATE/tree/codex/internvl3-svh-physical-d-seven-benchmarks-3090
