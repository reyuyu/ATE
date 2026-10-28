# Full / B4 / C2 跨数据集配对评测

全部 2620 题已完成，三个条件逐题同机配对。

B4：L0–L1 原 Prompt P−1→真实 Question 阻断；C2：L2–L4 原 P−1 MLP 输出置零。
生成仅在首次 Prefill 干预，随后使用修改后前缀的原生 KV Cache；Decode 不重新施加干预。
模型、Prompt、冻结样本和评分器均沿用 Phase 4.0。没有调参或重新采样。

| 单元 | 题数 | Full 生成评分 | B4 | C2 |
|---|---:|---:|---:|---:|
| gqa_chooseattr | 240 | 92.08% | 92.08% | 93.33% |
| gqa_logicalobj | 240 | 97.08% | 97.08% | 96.67% |
| gqa_queryattr | 240 | 84.58% | 84.58% | 85.42% |
| pope_adversarial | 300 | 83.33% | 83.33% | 83.67% |
| pope_popular | 300 | 88.67% | 88.67% | 89.00% |
| pope_random | 300 | 88.33% | 88.33% | 88.67% |
| scienceqa_img | 500 | 65.60% | 65.60% | 66.40% |
| textvqa | 500 | 49.68% | 49.28% | 49.08% |

## 解读边界

- GQA 三任务为作者 correct_question 子集，生成分数是 literal exact match，不冒称完整官方 GQA 结果。
- POPE 使用真实生成的 Accuracy/Precision/Recall/F1/Yes ratio；候选排序另列，不能混用。
- ScienceQA 使用选项字母生成评分，保留全部原选项；TextVQA 为无 OCR 输入的十答案软评分。
- 所有区间按实际图片 SHA256 聚类配对 Bootstrap，固定 10,000 次。多题同图的 McNemar 仅作题级描述。
- 多数据集、多条件的区间与 p 值未作多重检验修正，不凭单个名义显著结果宣称普遍机制。
- 开放任务不构造错误候选或 Margin；封闭任务保存完整候选全词表 LogP，平局算错。
- 准确率、Margin、绝对 Gold LogP 分别解读；MLP 阻断收益不等于证明增强视觉信息。
- 可选官方 GQA Balanced 仍阻塞，不计入本批，也没有替换数据。

详见 `aggregate.json`、`per_sample.csv`、逐题原始 Logits 和各 Worker 审计。

执行代码：`6cf638879454c32b73aa2892958352c3799fa306`。
