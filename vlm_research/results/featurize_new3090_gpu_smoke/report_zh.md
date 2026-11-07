# 新 RTX 3090 HF LLaVA 六 Benchmark GPU Smoke

**最终结论：PASS**。仅验证最小生成与评分闭环，不是正式 Benchmark 成绩。

- 时间（北京时间）：2026-11-08T04:04:10.498095+08:00 至 2026-11-08T04:09:10.846975+08:00
- 代码：`e281754ef17b3accc09d6938b5b9f020048b2aec`；GQA 恢复源码按其冻结 SHA 验证。
- 模型：`llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；本地 `/home/featurize/data/ate-recovery/models/llava-1.5-7b-hf`。
- 软件：Python 3.11.8；PyTorch 2.6.0+cu124；CUDA runtime 12.4；Transformers 4.49.0；Pillow 11.1.0。
- 协议：全参数 FP16 / Eager；单卡、batch=1、greedy、原 EOS、独立 KV；无量化、卸载或 FlashAttention。
- GPU 检查：`NVIDIA GeForce RTX 3090, GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e, 610.57.04, 24576 MiB, 1 MiB, 24123 MiB`。
- GPU 模型加载：PASS；加载次数 2。
- 峰值显存（PyTorch allocated / reserved MiB）：14097.6708984375 / 14224.0；此数值为内存验收记录，非性能测试。

| Benchmark / 输入视图 | 样本 ID | 生成结果 | 评分链路 |
|---|---|---|---|
| MME / MME-paper | OCR/0001.jpg:0, OCR/0001.jpg:1 | OCR/0001.jpg:0: No；OCR/0001.jpg:1: No | PASS |
| ScienceQA-IMG / ScienceQA-IMG-paper | 10001, 10015 | 10001: A；10015: A | PASS |
| POPE / POPE-paper | 10000001, 20000001, 1 | 10000001: No；20000001: No；1: No | PASS |
| TextVQA / TextVQA-OCR | 34602, 34603 | 34602: Dakota digital；34603: Drupalcon | PASS |
| MMBench / MMBench-paper | 1, 1000001, 2000001, 3000001 | 1: D；1000001: A；2000001: B；3000001: C | PASS |
| GQA / GQA-testdev-balanced | 201030233, 201030272 | 201030233: No；201030272: Yes | PASS |

输入审计：选中样本的 Question ID、原问题、图片 SHA、Prompt、Options、Gold 均对照冻结 Manifest；TextVQA 保留 Paper-OCR 原文与全部十个人类答案；POPE 使用 8,910 行论文视图（Random 2,910 / Popular 3,000 / Adversarial 3,000），各选一题。
视觉证据：已完成生成的每题记录 576 个图片 Token、真实 vision tower / projector 调用，并核对 projector 的图像特征与送入 LLM 的对应 embedding 完全相等；记录 Prefill、逐 Token Decode 和 32 层 KV 长度。
五类论文输入调用原 paper prepare（pad-to-square）；GQA 调用冻结 vlm_common.prepare_inputs，核对完整展开 Token ID 与 rendered Prompt SHA。生成分别调用原 hf_generate(HF-Native) 与 GQA generate(Native)。
MME：同图完整双题，原 mme_score 仅将子任务枚举限制为本次抽取的任务，解析、配对和 ACC/ACC+ 公式未改；不计算全量 Total。此次样本不涉及 landmark/03d5e3bfc958be38.jpg 已知 SHA 差异；正式全量 MME 仍须记录该版本差异，不能声称全部图片逐字节复现旧实例。
MMBench：完整四轮 Circular 组，先调用冻结官方 exact matcher，再调用原 circular_rows；不调用外部 Judge。逐题评分器的实际执行状态以表格与 JSON 为准。
持久化与重评分验证：PASS: JSON written/read; all row scores and MME/Circular aggregate recomputed exactly。逐题原文、Gold、Options、Prompt、Token ID、调用证据和评分返回值见 smoke_results.json。

HF Native 正式质量实验的 GPU 推理与评分条件：具备本次六类冻结输入的 HF Native 端到端运行条件（限本次 Smoke 覆盖范围）。
正式运行仍需单独授权并登记本机运行冻结；旧多卡 UUID/绝对路径启动器不能直接照搬。此验收不覆盖 Author-Native、Physical C3 数值等价、长上下文极限、完整数据集吞吐或正式 Accuracy。
本轮未运行全量 Benchmark、CPU 全回归、全图 SHA、数据/模型下载、导出、Push 或后台 GPU 作业；数据包与其他 Worktree 未修改。

GQA 首次拦截原因：独立 Smoke 入口错误地用原始 UTF-8 SHA 对比冻结的 JSON digest。CPU 诊断确认两题的原 Prompt、全部展开 Token ID、Gold 与冻结记录一致；使用原 gqa_common.digest 后哈希匹配。错误属于本轮入口附加检查，不属于冻结模型、数据或评分器。
已保留首次失败证据，修正只涉及独立入口的哈希编码方式，未更改 Prompt、生成或评分协议。

执行记录：保留 1 次先前尝试；本次仅补跑未完成的 GQA 两题，前五类未重复生成。第二次模型加载已经用户明确允许。

GPU 释放：{'allocated_mib': 32.0, 'reserved_mib': 32.0, 'process_exit_releases_cuda_context': True}

最终只读核对：15/15 记录及六类状态 PASS；执行源码 SHA 与 Git Commit 均未改变，tracked Worktree 无改动；GPU 计算进程列表为空。共两次加载，分别生成 13 题与经用户批准补跑的 GQA 2 题，前五类未重跑。

正式入口集成注意：通用恢复桥接器 `ate_data.prepare_input` 直接调用 HF image processor；五类论文冻结协议要求 pad-to-square。本次五类实际调用原 `information_horizon_alignment.engine.prepare`，正式论文输入应沿用该入口。

同步补记（2026-11-08）：按用户要求，此前本地验收证据随 C3-V2 结果分支补同步至 GitHub；原始生成与得分未修改。
