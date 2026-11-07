# RTX 3090 原始 Physical C3-V2 正式评测

**状态：COMPLETE**；正式完成 22,678/22,678。

代码：`e281754ef17b3accc09d6938b5b9f020048b2aec`；模型：`llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；FP16 / Eager / Transformers 4.49.0。
实际环境：{'python': '3.11.8', 'torch': '2.6.0+cu124', 'cuda_runtime': '12.4', 'transformers': '4.49.0'}。模型加载 1 次；单一 GPU 推理进程。
物理正确性 Smoke：PASS。具体门禁、算子形状、位置与数值误差见 physical_smoke_summary.json；完整跟踪 physical_smoke.json 留存服务器。

| Benchmark | 完成/应完成 | 正式分数 |
|---|---:|---|
| MME | 2374/2374 | {"Perception": 1478.8232292917169, "Cognition": 348.92857142857144, "Total": 1827.7518007202882} |
| ScienceQA-IMG | 2017/2017 | 68.963808% (Accuracy) |
| POPE | 8910/8910 | 87.205387% (Accuracy) |
| TextVQA | 5000/5000 | 50.400000% (VQA Soft Accuracy) |
| MMBench | 4377/4377 行，1176/1176 组 | 63.010204% (Circular Accuracy) |

协议：五类输入逐字段绑定冻结 Paper Manifest 和独立 Gold；TextVQA 为 Paper-OCR，POPE 为 8,910 行（Random 2,910 / Popular 3,000 / Adversarial 3,000）。MME、POPE、TextVQA 生成上限为 128；ScienceQA、MMBench 为 1024；无新增 stop strings。
原始 C3：R 仅 L0–L19 可读图像；L20 前物理移除视觉 Hidden；P−1 与 Decode 无原始 Image Key；L2–L5 物理跳过答案 Attention；原 Norm/MLP/Residual 保留；使用未重新编号的 RoPE 与紧凑物理 KV。
Smoke 仅在五类各一题上检查实际 Physical C3，并对照同算法三 Token dense 数值参考；dense 不是正式评测实现，未重跑 HF Native 或视觉回看变体。门限沿用 atol=0.10、rtol=0.003、hidden RMSE≤0.01、cosine≤1e-4。
MME 保留 landmark/03d5e3bfc958be38.jpg 单图 SHA 差异及两题标记，未替换或删除图片；因此与旧全量 MME 不具备全部图片逐字节一致的条件。
Native 对照：无同模型、同完整 Paper 输入与后端的已完成 HF Native 对照；本轮只报告 C3 单条件指标，不计算配对差值。此前 15 条 Native Smoke 不能作为完整 Benchmark 基线。
检查点：predictions.sqlite 单题事务、FULL 同步与记录 SHA；progress.json 记录真实进度。恢复时先核对运行配置绑定和已完成记录，再跳过已完成键。最终 predictions.jsonl.gz 保存可追溯原始预测。
CPU Native 评分审计独立写入 ../featurize_new3090_scoring_audit/，仅 CPU，不重新生成 Native。
CPU 审计门禁：{'status': 'PASS', 'path': '/home/featurize/work/ATE/vlm_research/results/featurize_new3090_scoring_audit/audit_summary.json', 'sha256': '4ee1a0ada0c84cdef2deca0e36b16ff06ce1ab457245965d3b2c6f940ace251d'}。
GPU/峰值与释放：{'before': 'NVIDIA GeForce RTX 3090, GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e, 610.57.04, 24576 MiB, 266 MiB, 23859 MiB', 'peak_allocated_mib': 14427.97021484375, 'peak_reserved_mib': 15116.0}；{'allocated_mib': 32.0, 'reserved_mib': 32.0, 'process_exit_releases_context': True}。
未重新下载模型/数据、未全图 SHA 或重复 452 项 CPU 回归、未运行 GQA、其他层组合、性能 Profiling。

最终 CPU 核对：全部 22,678 条记录 SHA、配置绑定和题数验证通过；独立从已存逐题得分重算 14 项 MME 的 ACC/ACC+、全部 Circular 组以及其余三项均分，与冻结汇总一致。该项是独立聚合核账；另行的 Native 15 条审计独立验证实际答案解析与评分规则。

Native 评分审计：COMPLETE，15/15 原得分正确。TextVQA 34602 的十答案留一软分为 0.9；MME 单对图不能推导全量 Total；MMBench 四轮组正确。此前 GQA 哈希修复只影响独立入口的 SHA 编码，冻结数据、Prompt、推理与评分语义未变。

未发现本轮需要修改的冻结 Scorer、数据映射或 Prompt 问题；已保留 MME 单图版本差异。首次本轮入口的 Circular 组绑定检查已按冻结 Manifest 定义完整成员，未改数据或评分逻辑。

五项正式评测已完成，未启动其他实验。轻量证据按用户要求同步到 GitHub，完整预测与检查点留存服务器。GPU 退出后计算进程列表：空。

TextVQA 外部评分文件已额外核对：作者源码与冻结 vendor 的差异仅为本轮未调用的 best_answer 辅助方法；移除此方法后的整个模块 AST 一致，答案规范化与十答案软评分实现均未改变。两份文件 SHA 和作者 Commit 已记入 run_metadata.json。

POPE 三策略正式 Accuracy：Random 88.865979%（2,910 题），Popular 87.700000%（3,000 题），Adversarial 85.100000%（3,000 题）；合并 8,910 题为 87.205387%。

模型本轮仅加载一次；峰值分配显存约 14.09 GiB，峰值预留约 14.76 GiB。后置核对入口的 POPE 汇总字段引用错误已修复；未修改冻结 Scorer、原始预测或得分，修正后完整核对通过。本轮未发现仍需修复的阻塞性技术问题。
