> 同步说明：本报告现随独立证据分支公开；实验结束时的“未 Push”状态保留于下方原文。完整预测、数据库与原始报告已在 Windows 完成 SHA256 备份。归档范围、压缩 Smoke 文件及当前发布说明见七项结果目录的 [README](../v2drop_prefix_svh_sevenbench_3090/README.md)。

# V²Drop-prefix + llava-SVH：固定 200 题准确率验证

**状态：COMPLETE**；Physical Smoke：PASS；融合准确率生成 200/200。Native、V²Drop、SVH 的 600 条预测全部复用历史存档。

| Method | GQA100 | TextVQA100 Paper-OCR |
|---|---:|---:|
| Native Full | 59.0% | 54.4% |
| V²Drop | 61.0% | 53.4% |
| llava-SVH | 55.0% | 45.0% |
| V²Drop-prefix + SVH | 50.0% | 46.7% |

本次固定样本不支持“融合没有额外准确率损失”的结论：GQA 相对 SVH 下降 5.0 pp，TextVQA 上升 1.7 pp。数值与结构检查通过说明实现满足已验收语义；不意味着准确率等价。没有为改善结果调整层集合、Prompt 或样本。

GQA 为官方 exact-match；TextVQA 为十答案 VQA Soft Accuracy，保留空输出和软分，不转成普通二值准确率。此表仅为冻结 GQA100 + TextVQA100 的探索性检查，不是完整 Benchmark 成绩。

实际 GPU/释放状态：NVIDIA GeForce RTX 3090, GPU-ca74e8a3-413d-9ddb-76ac-3dc7a1a9a292, 610.57.04, 1 MiB, 24123 MiB；计算进程：无。模型加载 1 次；FP16 / Eager / torch 2.6.0+cu124 / Transformers 4.49.0；模型 Revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。
冻结研究基线 Git Commit：`e281754ef17b3accc09d6938b5b9f020048b2aec`。新融合执行器为独立目录中的未提交代码，具体内容 SHA256 记录于 protocol.json；没有改冻结核心、GitHub main 或 Push。

协议绑定：严格复用原始 200 行 input_manifest.jsonl.gz（SHA256 a2fc346eecb1dc023e3762d921d3682e324c2bcb7ca3b88ef0becab6ea295e2d），不是 efficiency_keys 的 30 行。Question ID、Gold、图像 SHA256、Prompt、Tokenizer、实际 FP16 Pixel Values 与输入 tensor 哈希均核对；GQA 上限 32、TextVQA 上限 128，Greedy、EOS=2。
历史 SVH：从完整 GQA 官方对齐 results.sqlite 与 TextVQA Paper-OCR predictions.jsonl.gz 提取同一 100+100 题。已独立重算 600 条存档分数。TextVQA 历史像素字段为转 FP16 前的 FP32 哈希，两种精度阶段均经 CPU 逐题验证，不是输入差异；历史 SVH TextVQA 的 2 条空输出原样保留。

| Benchmark | Fusion − 对照 | Δ (pp) | 配对 95% CI (pp) | 分数改善 / 恶化 |
|---|---|---:|---|---:|
| GQA | llava-SVH | -5.0 | [-10.28, +0.00] | 1 / 6 |
| GQA | Native | -9.0 | [-17.65, -0.93] | 5 / 14 |
| GQA | V2Drop | -11.0 | [-19.35, -3.09] | 3 / 14 |
| TextVQA | llava-SVH | +1.7 | [+0.00, +4.30] | 3 / 0 |
| TextVQA | Native | -7.7 | [-15.10, -0.30] | 5 / 14 |
| TextVQA | V2Drop | -6.7 | [-13.70, -0.20] | 4 / 11 |

CI：按同一批题的 Image SHA256 簇成对重采样 10,000 次，seed=20261109，百分位 95% CI。样本很小，区间覆盖零不能解释为等价或非劣效。与完整 V²Drop 的差值仅作描述，因为剪枝预算不同。
GQA 对 SVH：正确→错误 6 题；错误→正确 1 题。
TextVQA 对 SVH：软分提高 3 题、降低 0 题；满分→零分 0 题、零分→满分 1 题。另保存任意正分→零分/零分→正分计数，不把软评分强行二值化。

| Method | Benchmark | 生成/复用条数 | 空输出 | 截断 |
|---|---|---:|---:|---:|
| Native | GQA | 100 | 0 | 0 |
| Native | TextVQA | 100 | 0 | 0 |
| V2Drop | GQA | 100 | 0 | 0 |
| V2Drop | TextVQA | 100 | 0 | 0 |
| llava-SVH | GQA | 100 | 0 | 0 |
| llava-SVH | TextVQA | 100 | 2 | 0 |
| V2Drop-prefix+SVH | GQA | 100 | 0 | 0 |
| V2Drop-prefix+SVH | TextVQA | 100 | 2 | 0 |

4 图真实 GPU Smoke 全通过：576→288→144→0；L3/L17 实际 Hidden 行删除、原始文本位置保持，RoPE 精确取原始位置；R 在允许层对剩余视觉 Key 的实际 Attention 概率为正；P−1/Decode 视觉访问为零；L2–L5 答案 Q/K/V/O 实际跳算，Norm/MLP/Residual 正常；持久原始视觉 KV 为零。关闭剪枝与冻结 SVH 的 Hidden/Logits 逐位一致。
数值参考只用于 4 图各 3 Token Smoke，使用物理路径选出的同一视觉索引、完整形状 dense 参考来验证物理压缩；200 题使用真实 Physical 实现。原门禁 atol=0.1、rtol=0.003、Hidden relative RMSE≤0.01、cosine≤1e-4 均未修改；最大 RMSE=0.007122307，最大绝对 Logit 差=0.039062500，短序列输出相同。
官方选择算法：直接调用既有 hf_adapter.py 从固定官方 Commit 的 AST 提取的 L2 norm/top-k 选择器；实际图像起点动态传入，并与未改动固定 35 的原始选择块核对。融合只调用 L3/L17；不执行 L22。融合前层受到 SVH 的影响，因此不要求与独立 V²Drop 选择同一索引。

逐题原始输出、分数、绑定、选择索引保存在 predictions.sqlite / predictions.jsonl；每题事务与 progress.json 支持恢复，最终 all_methods_predictions.jsonl.gz 和 per_question_comparison.csv 保存四组逐题比较。没有重新生成 Native/V²Drop，也未补跑 SVH 正式样本；只为要求的 Smoke 做短 SVH/数值参考。
GPU 请求合计 216 次：200 次融合自然 EOS 准确率请求，以及 16 次三 Token Smoke 请求（每张图各 Physical / Dense 参考 / 关闭剪枝 / 冻结 SVH）；Smoke 不计入 200 题准确率。
没有运行 FLOPs、KV 容量效率、TPOT、延迟测试、额外数据集或层搜索。服务器未关闭。Windows 真实 SHA256 备份结果另见 windows_backup_receipt.json。
