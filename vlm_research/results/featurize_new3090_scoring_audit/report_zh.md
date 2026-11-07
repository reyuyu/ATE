# HF Native 15 条 Smoke 独立 CPU 评分审计

**结论：PASS / COMPLETE（审计任务）**。15/15 已存预测的解析与逐题分数正确，未发现影响 C3 的输入、Gold 或评分问题。
本报告全部比例与分数仅为固定 Smoke 样本的核账，不是正式 Benchmark 成绩。

- 原始预测：`/home/featurize/work/ATE/vlm_research/results/featurize_new3090_gpu_smoke/smoke_results.json`，SHA256 `41160b6e0d08597cac83c13457a1dc2dd6d78209f53a88a084372ebe7eb5481c`；审计前后不变。
- 独立方法：阅读冻结规则后对实际裸单词/字母输出逐项人工逻辑复核；TextVQA 用十次留一法的精确分数独立计算；未调用原 Benchmark 评分函数来充当独立复核。
- ID、原问题、Options、Gold、输入视图、Prompt 同时对照恢复输入/独立 Gold、冻结 Manifest 和原始数据源；15 组生成 Token ID 均可解码回原存文本，输入 Token Hash 亦匹配。
- 全程 CPU；CUDA_VISIBLE_DEVICES 为空，cuda_initialized=false；未加载模型权重、未重新生成、未下载数据、未修改原 Smoke 或冻结源码。

| Benchmark | Question ID | 原预测 | 原分数 | 独立分数 | 核对 |
|---|---|---|---:|---:|---|
| MME | OCR/0001.jpg:0 | No | 0.0 | 0.0 | 一致 |
| MME | OCR/0001.jpg:1 | No | 1.0 | 1.0 | 一致 |
| ScienceQA-IMG | 10001 | A | 0.0 | 0.0 | 一致 |
| ScienceQA-IMG | 10015 | A | 1.0 | 1.0 | 一致 |
| POPE | 10000001 | No | 0.0 | 0.0 | 一致 |
| POPE | 20000001 | No | 0.0 | 0.0 | 一致 |
| POPE | 1 | No | 0.0 | 0.0 | 一致 |
| TextVQA | 34602 | Dakota digital | 0.9 | 0.9 | 一致 |
| TextVQA | 34603 | Drupalcon | 0.0 | 0.0 | 一致 |
| MMBench | 1 | D | 1.0 | 1.0 | 一致 |
| MMBench | 1000001 | A | 1.0 | 1.0 | 一致 |
| MMBench | 2000001 | B | 1.0 | 1.0 | 一致 |
| MMBench | 3000001 | C | 1.0 | 1.0 | 一致 |
| GQA | 201030233 | No | 0.0 | 0.0 | 一致 |
| GQA | 201030272 | Yes | 1.0 | 1.0 | 一致 |

TextVQA 34602：dakota digital 在十答案中出现 3 次。排除匹配答案的 3 次各得 2/3，排除其余答案的 7 次各得 1，故 (3×2/3+7×1)/10=0.9。34603：drupalcon 与九个 copenhagen、一个 thursday 都不匹配，得 0。两题软评分均值 0.45，仅为核账结果，不能替换成二值准确率。
MME：OCR/0001.jpg 两题分别得 0、1；Accuracy=50%，Accuracy+=0%，该抽取 OCR 小样本的 ACC+ACC+=50。原 Smoke 未声称全量 MME Total，做法正确；其他 13 个任务缺失，完整 /2800 Total 保持 NOT_VALIDATED，不补零、不外推。
MMBench：1、1000001、2000001、3000001 构成完整 Circular 组；Gold/预测均依次为 D、A、B、C，对应同一正确选项内容。四轮行级得分均 1，因此组级 all-rotation hit=1，原汇总正确。
ScienceQA：10001 的 A 对索引 0，但 Gold=2（C），得 0；10015 的 A 对 Gold=0，得 1。10015 的 jeans/metal 题干看似异常，但原官方 problems.json 与论文 CQM-A 附件均确实如此，并非恢复映射错误。
POPE：三条不同策略的完整单词 No 按官方首句否定词规则解析为 no，Gold 均 yes，三条均 0。输入来自 8,910 论文视图，同时按 official_question_id 与三个策略原始 JSON 对齐。
GQA：No→no 与 yes 不同，得 0；Yes→yes 与 yes 相同，得 1。输出没有尾句号或多余空白，遵循冻结 rstrip(".").lower() 规则。
GQA 哈希修复审计：旧/新独立入口 input_for 的 AST 差异仅为 raw UTF-8 SHA 改成原 gqa_common.digest(text)；两个实际 Prompt 的 JSON digest 均等于冻结值，原 score_one AST 不变。所有已冻结 inference/scorer 文件 SHA 与 Smoke 元数据一致。其他入口改动仅为经用户允许的补跑与证据记录，没有改变数据或评分语义。
技术结论：本次 15 条没有需要离线修正的分数，也没有需要重跑 GPU 的评分原因。先前入口哈希编码错误已修复；无需修改冻结评分器。正式五 Benchmark 应继续使用原 paper prepare 的 pad-to-square，通用恢复桥接器不可替代该预处理。

细节见 per_question_audit.csv / per_question_audit.json；机器可读门禁见 audit_summary.json。

同步补记（2026-11-08）：按用户要求，此前本地验收证据随 C3-V2 结果分支补同步至 GitHub；原始生成与得分未修改。
