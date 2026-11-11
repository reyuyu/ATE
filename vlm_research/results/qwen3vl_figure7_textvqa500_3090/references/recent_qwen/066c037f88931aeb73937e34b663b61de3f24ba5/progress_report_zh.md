# Qwen3-VL-SVH-R 扩大视觉回看：阶段结果

状态：**PARTIAL，Discovery / Validation 完成，Held-out 因 OOM 停止**。
更新时间：2026-11-12，Asia/Shanghai。本报告不是完整实验结束报告。

## 当前结论

扩大答案视觉回看可明显恢复 TextVQA，但恢复主要损失需要很宽的层集合。
Validation 上，原 SVH 为 62.97%，L0 与 L7–L35 联合回看为 80.32%，
同题 Native 为 81.51%。不能把这个结果称为小预算恢复、非劣效或独立外部验证。
自然 EOS 的小规模真实测速没有显示加速，不能用固定长度结果替代它。

## 冻结语义

- Qwen3-VL-8B-Instruct，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16/eager。
- P−1 与每个真实 Decode 使用相同回看层集合；不是 P−1-only 或 Decode-only 主实验。
- R 写入仍为 L0–L21；主视觉 Hidden 仍在 L22 前退出。
- L1–L6 答案 Attention 跳算保持不变。因此 Wide30 不等于 Native。
- 晚层 Image KV 由独立因果视觉前缀路径产生，不复制其他层 KV。
- 功能搜索保留全部有效层视觉 KV，不以该路径报告物理 KV 节省。
- Physical 测速只存选定层视觉 KV，计入 sidecar 和数值对齐开销。
- 科学范围为方案 B / ATE 内部统一 HF 协议，不宣称作者官方复现。

## 完成范围

| 阶段 | 条件 | 每条件记录 | 记录总数 | 状态 |
|---|---:|---:|---:|---|
| Discovery | 41，含 K0 | 585 | 23,985 | 完成 |
| Validation | 6 | 1,321 | 7,926 | 检查点完成；7,914 条有效评分 |
| Held-out Unified | 4 | 计划 1,641 | 2,637 / 6,564 | PARTIAL / OOM 停止 |
| Held-out Physical、阶段消融 | 尚未完成 | — | — | 不报告最终分数 |

Validation 的 TextVQA 计划 500 题，498 题有效，436 个图片簇。
`36759`、`36747` 为已有历史 OOM；六条件都保留 BLOCKED 记录，不补零、不换题，
所以共 12 条阻塞记录。其余 Validation 样本为 GQA Query 250、NonQuery 100、
ScienceQA 100、MMBench 371 条旋转记录 / 100 个完整 Circular 组。

## TextVQA 完整阶段结果

| 条件 | 回看层 | K | Discovery 250题 | Validation 498有效题 |
|---|---|---:|---:|---:|
| Native | 原生 | 36 | 80.64% | 81.51% |
| SVH / K0 | 无 | 0 | 59.80% | 62.97% |
| Prior4 | L21/L23/L24/L26 | 4 | 68.12% | 68.90% |
| Wide21 | L11–L31 | 21 | 74.20% | 75.86% |
| Wide25 | L7–L31 | 25 | 74.48% | 76.61% |
| Wide29 | L7–L35 | 29 | 80.52% | 80.10% |
| Wide30 | L0 与 L7–L35 | 30 | 80.44% | 80.32% |

20,000 次图片 SHA 整簇配对 Bootstrap，题目等权，普通探索性 95% CI：

| Validation 比较 | 差值 pp | 95% CI | 软评分下降 / 上升题数 |
|---|---:|---|---:|
| Wide30 − SVH | +17.35 | [13.85, 21.00] | 19 / 110 |
| Wide30 − Native | −1.18 | [−2.69, 0.26] | 见逐题结果 |
| Wide29 − SVH | +17.13 | [13.58, 20.79] | 17 / 110 |
| Wide29 − Native | −1.41 | [−2.98, 0.10] | 见逐题结果 |

软评分升降不是二值纠错数。CI 跨零不代表无损，也没有达到可宣称 −1pp
非劣效的证据。该队列存在历史重复探索，不是从未使用过的外部测试集。

## 其他 Validation 任务

| 任务 | Native | SVH | Wide29 | Wide30 |
|---|---:|---:|---:|---:|
| GQA Query，250题 | 42.00% | 27.20% | 42.40% | 42.40% |
| GQA NonQuery，100题 | 83.00% | 81.00% | 83.00% | 82.00% |
| ScienceQA P2，100题 | 90.00% | 90.00% | 90.00% | 90.00% |
| MMBench P2，100个完整Circular组 | 88.00% | 88.00% | 88.00% | 88.00% |

全部条件、普通配对区间、EOS、截断、损害和恢复数见
`Validation_Unified_summary.csv`。POPE / MME 的 Held-out 尚不完整，不补造最终成绩。

## 已冻结候选与真实性能

读取 Held-out 前，`final_selection.json` 已冻结：
BestQuality = Wide30；BestEfficiency = Wide29；MinK = Prior4。
BestEfficiency 仅指本轮合格候选中的最低固定32 Token实测E2E，不代表自然EOS最快，
MinK 也不代表已经证明四层足以恢复主要损失。

测速为预冻结 Validation 的 10 个输入（每核心任务2个），同题同卡，
每模式3次预热、5次交错正式重复；每次冷独立KV、重算Vision。
计时从已准备好的Processor输入开始，包含Vision、语言Prefill、Decode和请求调用开销，
不包含磁盘读图 / Processor。固定Token测试不用于计算答案Accuracy。

| 条件 | 自然EOS E2E ms | 相对Native倍数 | 固定32 E2E ms | 相对Native倍数 |
|---|---:|---:|---:|---:|
| Native | 336.76 | 1.000× | 1,848.12 | 1.000× |
| SVH | 316.44 | 1.064× | 1,903.63 | 0.971× |
| Prior4 | 349.55 | 0.963× | 1,880.92 | 0.983× |
| Wide29 | 384.84 | 0.875× | 1,724.33 | 1.072× |
| Wide30 | 392.90 | 0.857× | 1,736.63 | 1.064× |

自然EOS平均输出长度分别为 Native 3.2、SVH 2.8、Wide29 / Wide30 3.0 Token。
Wide29 / Wide30 分别有9个输入与Native完整序列一致；该同序列子集的
Native/Candidate E2E比分别为0.830×、0.811×，同样不支持自然EOS加速。
宽回看增加Prefill / sidecar成本，短答案Decode节省不能保证抵消这部分开销。
这10个输入只是小规模工程测速，不可冒充六项完整Benchmark效率结论。

自然EOS平均Live KV：Native 52.44 MiB、SVH 9.98 MiB、Wide29 42.55 MiB、
Wide30 43.68 MiB。宽回看消耗了大部分原来的KV收益；不能从KV较小推断E2E较快。
逐输入Prefill、Vision、TPOT、KV容量和Peak VRAM见快照性能CSV。
硬件TFLOPS、DRAM带宽和本次独立执行FLOPs计数未测量，不填估造数据。

## 验证及失败记录

- 13项CPU回归通过；两GPU新路径Smoke通过。
- 两卡选层物理KV对照各9例，最大有效Logits误差均为0，Token一致，非目标Prefill一致。
  这是 `PASS_SELECTED_STORAGE_ONLY`，不等于原始Dense算法的全部数值验收通过。
- 历史严格数值状态继续为 `FAIL_RETAINED`；不改写旧失败。
- Held-out 在 `MME:posters/tt1974419.jpg:0` 停止。视觉编码器Eager Attention的
  FP32 Softmax申请约2.03 GiB失败，进程占用约23.21 GiB，仅余约374.94 MiB。
  日志中的GPU0是CUDA_VISIBLE_DEVICES下的逻辑编号，实际为物理GPU1。
- 不缩图、不改精度、不换后端、不删失败题；未自动重试或放宽门槛。
- 原始异常和栈保存在 `failure0.json`、`wider_pipeline_failure.json` 与Worker日志。
- 两个本任务Worker已退出；GPU1/3无本任务计算上下文。GPU0/2外部进程未触碰。

## GitHub快照范围与复核

本次同步包括完整Discovery / Validation逐题CSV、原始压缩记录、配对历史基线、
全部测速重复、协议、固定清单、源码、Smoke、失败和GPU释放记录。
Held-out只导出明确标记PARTIAL的原始记录与覆盖数量，不提供完整Held-out结论。
不提交活跃数据库、模型权重、原始图片或大特征数组。

`completed_snapshot/snapshot_audit.json` 保存每个导出文件SHA256及字节数。
每条压缩记录保留原Checkpoint ID与Payload SHA256；导出过程逐条验证哈希、
重复、计划数量、图片SHA及可用的Prompt绑定。

CPU复核：`CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m unittest tests wider_tests`。
快照导出：`CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python export_completed.py`。
后者只读读取Checkpoint，不加载模型，不修改实验源码和已有评分。

下一步应先诊断最坏MME输入的显存路径，再决定如何恢复未完成验证。
本次没有因OOM把部分Held-out改称完成，也不以Validation结果替代最终验证。
