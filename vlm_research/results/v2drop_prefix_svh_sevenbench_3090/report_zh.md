> 同步说明：本报告现随独立证据分支公开；实验结束时的“未 Push”状态保留于下方原文。完整预测、数据库与原始报告已在 Windows 完成 SHA256 备份。归档范围、压缩 Smoke 文件及当前发布说明见七项结果目录的 [README](../v2drop_prefix_svh_sevenbench_3090/README.md)。

# V²Drop-prefix + SVH：七项完整质量评测与 KV/TPOT

**状态：COMPLETE**。质量预测 36,256/36,256；固定长度效率请求 60/60。复用 MME 2,374 条和 GQA100/TextVQA100 共 2,574 条；仅生成其余缺失的融合结果。

| Method | KV (MiB) ↓ | TPOT (ms) ↓ | GQA ↑ | SQA ↑ | TextVQA ↑ | POPE ↑ | MME ↑ | MMB ↑ | OCRBench ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V²Drop-prefix + SVH | 97.5625 | 62.8795 | 56.1775 | 67.7739 | 49.8100 | 85.3648 | 1780.3954 | 62.3299 | 87.0000 |

GQA/SQA/TextVQA/POPE/MMB 单位为 %；TextVQA 为原十答案 VQA Soft Accuracy。MMB 为 1,176 个完整 Circular 组全对的比例。MME 为 Perception+Cognition 的 Total Score（不是 %）；OCRBench 为官方 v1 的 /1000 累加分。POPE 主列为冻结 8,910 题逐题 Accuracy，三策略及等权平均另存 POPE_strategies.csv / score_summary.json。
KV/TPOT 主列使用原 V²Drop 冻结的 30 个输入（GQA15 + TextVQA15）与固定 128 个输出 Token，另保留固定 32 Token 结果。每个长度预热 2 次；batch=1，单 GPU，同一计时方法。KV 按实际缓存 used_rows 和 dtype 计算 live bytes，排除预留容量；持久视觉 KV 为零。主表 TPOT=(墙钟 E2E−墙钟 TTFT)/(固定输出长度−1)，并单独保存 CUDA 逐 Decode 步 TPOT；两者均排除 Prefill。墙钟值包含真实执行、调度与 greedy/finite 检查。它不是七类数据所有请求的平均 TPOT；未混用旧 RTX 4090 计时数据。

| Benchmark | Completed / Expected | New | Reused | Empty | Truncated | Score |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 12578 / 12578 | 12478 | 100 | 0 | 1 | 56.1775 |
| ScienceQA-IMG | 2017 / 2017 | 2017 | 0 | 0 | 0 | 67.7739 |
| TextVQA | 5000 / 5000 | 4900 | 100 | 46 | 6 | 49.8100 |
| POPE | 8910 / 8910 | 8910 | 0 | 0 | 0 | 85.3648 |
| MME | 2374 / 2374 | 0 | 2374 | 0 | 0 | 1780.3954 |
| MMBench | 4377 / 4377 | 4377 | 0 | 0 | 0 | 62.3299 |
| OCRBench-v1 | 1000 / 1000 | 1000 | 0 | 126 | 14 | 87.0000 |

真实环境：NVIDIA GeForce RTX 3090, GPU-ca74e8a3-413d-9ddb-76ac-3dc7a1a9a292, 610.57.04, 23859 MiB；PyTorch 2.6.0+cu124、Transformers 4.49.0、FP16/Eager、Greedy、EOS=2、batch=1。模型 `llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；基线 Commit `e281754ef17b3accc09d6938b5b9f020048b2aec`。本轮新增源码及冻结核心的完整 SHA256 见 protocol.json；未修改 SVH/V²Drop 冻结核心、模型或旧结果。为在任何效率测量前对齐历史墙钟 TPOT，曾受控保存断点并重新加载一次；3,544 条已有预测完整保留，未重复生成。旧入口、受控中止记录及效率补充协议均保留。

算法只复用官方 V²Drop L3/L17 的 Hidden L2-change Top-k 选择，实际物理执行 576→288→144，L20 前按 SVH 退出全部视觉 Hidden；不执行完整 V²Drop 的 L22 剪枝。P−1/Decode 全层无原始视觉 KV，L2–L5 答案 Attention 物理跳算，Norm/MLP/Residual 保留，文本未裁剪/重编号。该融合条件与完整 V²Drop 的剪枝预算不同。
物理验收继承同 GPU、同执行器的四图严格数值门禁 PASS（门限不变）。本轮补充 SQA、POPE 三策略、完整 MMB Circular 组和 OCRBench 共 9 条真实输入的 Vision、物理形状、RoPE/位置、文本完整性、答案 Attention 跳算及缓存检查，均在正式预测中只计一次，详见 integration_smoke.json。
GQA 使用已审计的官方对齐 Vicuna-v1 完整 System+USER+Image+Question+Short Answer、RGB/CLIP mean square-pad/CLIP336、max_new_tokens=32；逐题实际 Token IDs 与 FP16 张量哈希和历史官方对齐 Manifest 一致。不能与早期 USER-only/crop 的 GQA 结果作同协议直接差值。其余六项逐题核对冻结 Prompt/Token IDs/FP32 像素哈希（随后转 FP16）、Gold 与图像标识。
生成上限：GQA32；MME/POPE/TextVQA/OCRBench128；SQA/MMB1024。质量生成遵循自然 EOS；固定长度只用于独立 KV/TPOT 测量。所有错误、空输出和截断原样保留。
评分重新读取持久化预测，调用冻结 Scorer；GQA/POPE/SQA/MME 逐题与 TextVQA 十答案软评分独立逻辑复核；MME Accuracy/Accuracy+ 与 MMB 完整 Circular 组独立计数；OCRBench 另执行固定官方源码 AST 的原始评分循环，确认逐题一致。MMB 沿用既有 OFFLINE_REPRODUCTION/VLMEvalKit exact matcher，未使用外部 LLM Judge。
MME 已知 landmark/03d5e3bfc958be38.jpg 字节版本差异涉及双题，两条均复用已完成正式结果，未替换或删题；旧/新 SHA 见 input_audit.json。图像完整性使用已恢复可信清单，未全量重新哈希或下载图片。
预测使用 SQLite PRIMARY KEY + FULL 同步逐题事务，并保存 protocol binding、样本摘要、QID/Gold、输入张量绑定、原始输出与 L3/L17 选择索引；progress.json 可恢复，原始完整结果另压缩保存。不生成其他方法、不搜索层或比例、不扩大数据视图。
GPU 状态：NVIDIA GeForce RTX 3090, GPU-ca74e8a3-413d-9ddb-76ac-3dc7a1a9a292, 610.57.04, 1 MiB, 24123 MiB；活动计算进程：无。服务器未关闭；未 GitHub Push。Windows 备份必须以真实 SFTP+SHA256 回执为准，见 windows_backup_receipt.json；无回执不得视作已备份。

MME Perception=1467.895358，Cognition=312.500000，Total=1780.395358。
