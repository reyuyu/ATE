# Qwen3-VL-SVH Physical 数值修复

## 结论

**固定 Smoke 门禁已真正通过，状态为 `PASS_FIXED_SMOKE_ONLY`。** 最短、中位、代表性长输入在修复后，SVH Dense 与 Physical 的全部有效 Logits（首答案及真实 EOS Decode）均逐位相等；独立新进程、无逐层探针复核同样通过。原始失败没有改判，历史源码没有修改。

这不是六项 Benchmark 的完成证明，也不是作者官方复现。采用方案 B：ATE 统一 HF；方案 A 继续 **BLOCKED**。本轮没有正式评测、测速、安装依赖或推送 GitHub。

## 冻结与资源

- 起点：`d89363f17a394de559f76366890551d3c802a1d1`。
- 分支：`codex/qwen3vl-svh-physical-numerical-repair`，独立工作树位于 `/data/lk/ATE-worktrees/qwen3vl-svh-physical-numerical-repair/`。
- 修复代码冻结：`78f9a98353`；诊断器屏蔽值记录修正后的独立复核执行冻结：`a07c91b3bb`。执行文件 SHA 在各次 `execution_protocol.json` 中。
- 模型 revision：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；Torch 2.6.0+cu124、Transformers 4.57.6、BF16/eager、batch1、`logits_to_keep=1`、Greedy、cap32、EOS `[151645,151643]`。
- 仅使用空闲物理 GPU3：`GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`。GPU0–2 上的进程未终止、未抢占；GPU3 已释放至 5 MiB，释放时没有本任务计算进程。
- 所有模型文件重新 SHA256 校验与历史清单一致，见 `model_hash_revalidation.json`。没有复制权重或更改共享环境。

## 先复现失败，不改变门槛

完整原始执行器来自起点的 `qwen3vl_c3v2_physical/engine.py`，本轮未编辑。历史三条输入来自 `fc3f10eabf48a30b05f4e927a00f50bcd20780ad`，Question ID、图片 SHA、Prompt、全部输入 Tensor SHA 均严格复核。

原门槛始终为 **Mean Abs ≤0.02、Max Abs ≤0.5、Top1 相同、完整生成 Token 相同、无 NaN/Inf**。Native/No-op 另行要求全部有效 Logits `torch.equal`。**没有要求 SVH 与 Native 相同**。

| 历史输入 | 总位置 / Image | 原版首步 Mean / Max | 修复后首步 Mean / Max | 修复后 EOS Mean / Max |
|---|---:|---:|---:|---:|
| 06905471 | 122 / 96 | 0.083481 / 0.445313 | 0 / 0 | 0 / 0 |
| 02497673 | 221 / 192 | 0.093913 / 0.765625 | 0 / 0 | 0 / 0 |
| 02287196 | 700 / 672 | 0.049649 / 0.625000 | 0 / 0 | 0 / 0 |

最短题的 Dense 完整向量与历史 `F_logits.pt` 逐位一致，原版首步 Mean 0.0834809989 精确复现最初失败量级。见 `historical_full_vector_reproduction.json`。不要将本轮 Original 与历史经过部分修复的 `C_before` 混为同一路径。

## 第一处分歧与修复

逐层捕获真实运算输出，而非替换前向。**本轮未修复 Original 最早分歧在 Prefill L1 的 Attention AV**：最短题 Max 0.00006103515625。物理跳过 P−1 后 Key 归约长度改变，细小 BF16 差异随后传播。原始位置、mRoPE、DeepStack 的对齐检查未发现结构错误。

历史部分修复已定位到中位题更晚的 L22 Q 投影分歧；原记录保留在父提交的 `historical_evidence/qwen3vl_c3v2_prefill_av_final/report_zh.md`，没有覆盖或重标 PASS。

新模块 `qwen3vl_svh_numerical_repair/engine.py` 只协调数值执行，不改 SVH 依赖图：

1. 复用历史 Early Dummy Zero Key 和 Decode Zero Key 修复：只添加临时、被屏蔽的零列；不投影或持久保存这些视觉 KV。
2. L22–L35 Prefill 的 Q/K/V/O、Gate/Up/Down 投影采用历史 cuBLASLt 桥接层，以 Dense 形状描述符选择相同算法，在**真实紧凑文本行**上执行。扩展到 Q/O/Down，不能只修 K/V/Gate/Up。
3. 后段 Attention 临时将零 K/V 列放回原始逻辑位置，全部新增 Image 列为 `-inf`；真实文本 K/V 与非目标 Mask 不改。Query 仍是紧凑文本 Query。

没有恢复 P−1/Decode 读图，没有改变 L1–L6 答案 Attention 跳算、L22 视觉 Hidden 退出、DeepStack、mRoPE、MLP 或权重。注意这里是 Qwen 冻结的 **L1–L6 / L22**，不是 LLaVA 的 L2–L5 / L20。

修复后，历史最短题 L22 AV 仍有 Max **7.45058e-9** 的瞬时次正规数差别，**该层输出已严格一致，后续没有传播，最终 Logits 为零误差**。不能声称所有内部元素都逐位一致。其他固定输入未检测到保留文本中间值分歧，详见 `first_divergences.json` 与逐层 CSV。

## 方案 B 与独立验证

方案 B 使用官方 Chat Template、image 后接题干与固定单词/短语后缀、无外部 OCR、min_pixels65536/max_pixels2097152，属于 **ATE 自定统一协议**。历史复现仍使用旧后缀，两类记录分开保存；没有为了通过而缩图。

| 方案 B 输入 | 总位置 / Image | 修复前两步最差 Mean / Max | 修复后两步最差 Mean / Max | 新进程复核 |
|---|---:|---:|---:|---|
| 06905471 | 123 / 96 | 0.123424 / 0.593750 | 0 / 0 | PASS |
| 02497673 | 222 / 192 | 0.162672 / 0.562500 | 0 / 0 | PASS |
| 02287196 | 701 / 672 | 0.118413 / 0.750000 | 0 / 0 | PASS |

- 6 项 CPU 测试通过，包括 Cache 存储独立、原位置映射、零列概率、异常清理和冻结门槛反例。
- 诊断阶段完成三条历史输入和三条方案 B 输入；独立新进程不使用逐层 Trace，再验证三条方案 B 输入。
- Native、No-op、HF `generate` 的完整有效 Logits 和 Token 严格一致，包含真实 EOS 预测。
- 保留文本的 36 层输出、全部真实文本 KV、mRoPE cos/sin、原始位置、DeepStack 特征及其视觉 Mask 均通过对应精确对齐检查。
- L22 后 Hidden 只剩 26/29/28（历史）或 27/30/29（方案 B）行文本；L1–L6 Decode QKV/O 不执行，全部 36 层 MLP/Norm/Residual 保留。
- 持久原始视觉 KV =0；紧凑 Cache 原地追加，不复制历史 KV。各层明细见 `per_layer_kv.csv`。
- 独立 CPU 程序不调用执行器的比较函数，用 NumPy FP64 对完整原始向量重算 30 条比较，原门槛不变。
- 所有步骤无 OOM、非有限激活、空答案或截断。这三个短答案都生成一个内容 Token 后 EOS；**并未验证长答案多步 Decode**。

首轮诊断器曾因将 `-inf` 与有限因果屏蔽值当普通激活相减而中断。这是记录器错误，原日志、原始向量及 `diagnose_failure.json` 保留；修正仅影响屏蔽值差异记录，不改变模型或 Logits 门槛。第二轮完整完成，独立复核通过。

## 计算代价与边界

本轮仍实际删除视觉 Hidden、减少后段 MLP/投影行数，持久 Cache 为纯文本。但临时零列恢复了完整 Key 归约长度，**QK/AV 会实际执行对零列的乘法**，不能将这些列当作省掉的 FLOPs。cuBLASLt 固定选择和临时分配也可能有额外开销。本轮没有 E2E 测速，不证明修复路径保留原加速倍数。

结论只覆盖 96、192、672 动态视觉 Token 的三个冻结输入，未验证方案 B 上限的全部形状、其他任务特殊 Token 或长 Decode。未经新预检，不能宣布六项评测就绪；原始 `NUMERICAL_EQUIVALENCE_FAILED` 记录继续保留，新增的是独立修复模块的 Smoke PASS。

## 后续仅提案，未启动

门禁已通过，可以提出每任务48题（共288题）的方案 B 预检，具体见 `preflight_48_proposal.json`。固定按输入哈希及视觉 Token 长度分层，保持 MME 配对、MMBench Circular 组；每任务取短/中/长和接近固定视觉预算上限的输入。先做 Native/No-op 与 Dense/Physical 有效 Logits、完整 Token、KV 生命周期校验，再自由生成和冻结评分。

新的形状若 cuBLASLt 选择失败、OOM 或数值超门槛，立即保留证据并停止对应预检，不减分辨率、不换精度、不放宽门槛。MMBench 仍只能称方案 B exact-matching/Circular 诊断，不冒充作者 Judge。

**本轮到此停止：未运行288题预检或六项全量，未推送。**

## 复核入口

```bash
cd /data/lk/ATE-worktrees/qwen3vl-svh-physical-numerical-repair
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_numerical_repair.tests -v
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_numerical_repair_analysis
```

`complete_logits_bf16_bits.npz` 与索引提供47份原始有效向量的无损 BF16 位编码，包括失败尝试。原始 `.pt` 保留在本地相应目录，未删除。`diagnosis_round2/*/Original_layerwise.csv` 和 `Physical_layerwise.csv` 提供逐层逐算子误差；`validation.json`、`runtime.json`、`token_kv_audit.csv`、执行快照及释放记录共同限定本次 PASS 范围。
