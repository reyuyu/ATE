# Qwen3-VL-SVH FAST-TRACK：7908投影修复与MME视觉显存修复

## 最终状态

**BLOCKED_NUMERICAL，未启动任何8题预检或六项全量生成。**

基于`d5ca88d7de604728eb1f0bcea3185f01d266e125`，方案B（ATE统一HF）继续使用；方案A仍BLOCKED。数值门槛没有变化：Native/No-op有效Logits严格相同；Dense/Physical逐步平均绝对误差<=0.02、最大绝对误差<=0.5，且Top1及完整Token序列一致。所有既有失败记录只读保留。

| 验证范围 | 本轮实际状态 |
|---|---|
| CPU回归 | 7项通过 |
| ScienceQA 7908，原视觉Eager | 修复后独立进程完整128生成步，误差0 |
| 此前13个已通过样本，原视觉Eager | 13/13完整生成回归PASS，误差0 |
| 新视觉Runtime独立算子及视觉27层 | 概率、AV、主视觉及三组DeepStack特征精确一致 |
| 新Runtime整模最坏输入Native/No-op | 3/3 PASS，最大峰值21126.295 MiB |
| 新Runtime原24例门禁 | 14 PASS / 1 FAIL / 9未运行 |
| 六项各8题预检 | 未启动 |
| 六项完整配对生成 | 未启动 |

本轮没有得到`FAST_TRACK_VALIDATED`；更不能写成`FULL_59_NUMERICAL_VALIDATED`。

## A：ScienceQA 7908的实际根因与修复

原始输入N=513、视觉Token460。L0输出、L1输入Hidden、RMSNorm输入及输出均精确相同；实际q_proj输入的前512行也相同。**首个真实分歧是L1 q_proj本身**：Dense对513行执行BF16投影，Physical为跳过P-1 Attention只投影512行。相同输入隔离重放精确复现两条旧输出，证明不是上游误差首次被L1观察到。

固定原始M对应的cuBLASLt描述符、实际只计算compact真实行后，q/k/v重放均精确一致。修复只适用于Prefill L1-L6 Q/K/V/O投影，仍不投影被跳过的P-1，不改变MLP、Norm、Residual、Mask、RoPE、DeepStack、位置或缓存。复用历史Lt桥接库，正式推理仍BF16，未增加FP32推理。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| L1 q_proj max abs | 0.00390625 | 0 |
| L1 q_proj mean abs | 0.0000949150563 | 0 |
| 首步全词表Logits mean abs | 0.146603972 | 0 |
| 首步全词表Logits max abs | 0.578125 | 0 |

新进程独立复测7908及原13例，Native/No-op严格一致，Dense/Physical全部有效Logits、Token、mRoPE、DeepStack与KV位置通过。7908的SVH自然生成到128上限、127次真实Decode，与Dense参考完整一致；此截断如实记录，不等同于高质量回答。统一新视觉Runtime后7908再次精确通过。

详细证据在`../qwen3vl_svh_sqa7908_repair/`，含完整状态、相同输入隔离、旧失败、修复后首步逐算子及独立全程验证。

## B：MME Native视觉OOM

原题`MME:landmark/0adb088e24d88123.jpg:1`：语言视觉Token2028，原视觉Patch8112，batch1。真实复现失败位于**视觉Attention第0层**，Q/K/V为`[1,16,8112,72]`，Softmax矩阵`[1,16,8112,8112]`，单个FP32缓冲约3.92 GiB。原Eager复现峰值Allocated 24,123,154,944字节，另一FP32缓冲申请失败；GPU启动前空闲，不是批处理、外部进程或缓存泄漏。

新Runtime明确命名为：**VISION_EAGER_ROW_SOFTMAX128_TEXT_EAGER**。

保持原始像素、动态Token预算、BF16权重、完整QK及AV矩阵乘法形状；仅将视觉FP32 Softmax临时矩阵按128个Query行计算，每行全部Key仍参与。未使用SDPA/FlashAttention；语言Attention仍Eager。此为新Runtime，Native、No-op、Dense、Physical全部统一应用，不能冒充旧Eager通过。

真实失败Q/K/V隔离重放时全部BF16概率与AV输出精确一致；随后对P99和最大MME在全部27层视觉网络验证主特征及三组DeepStack均精确一致。整模新进程独立Native/No-op自然EOS检验：

| 输入 | 视觉Token | Peak Allocated MiB |
|---|---:|---:|
| MME P99 | 2028 | 21044.970 |
| MME最大 | 2048 | 21126.295 |
| TextVQA最大36759 | 2028 | 21045.362 |

**原Eager仍为BLOCKED_MEMORY**。以上是数值与显存验收，不是性能Benchmark，也不宣称加速。新Runtime仍执行完整视觉QK/AV，语言的历史AV/QK补零额外运算也保留，见逐例审计；未把额外计算隐去。

## C：恢复24例后新发现的数值阻塞

同一冻结24例、分片、输入Token/像素及哈希均不变，在GPU1/3重新执行全部必要回归，而非把旧Runtime下PASS直接计入。14例严格PASS后，MME P99不再OOM，但语言Dense/Physical出现新数值失败：

| 步骤 | mean abs | max abs | Token相同 |
|---|---:|---:|---|
| Prefill | 0.070185587 | 0.9375 | 是 |
| Decode 1 | 0.087762952 | 0.5 | 是 |

Native/No-op仍严格相同；Physical结构验收通过，持久原视觉KV为0。**Token相同不能覆盖Logits超限，故自动关闭放行。**

逐算子诊断最早**观测到**L1 Attention probabilities存在差异（max=1.862645149e-9），但该层AV与最终Hidden仍相同；L2 probabilities差异增大（max=0.0006103515625），AV max=0.00048828125，L2最终Hidden首次不同（max=0.03125）。此前L0、L1 Norm及对齐后的Q/K/V投影、mRoPE、文本Key QK/Softmax均相同。

这里不能把L1 probabilities直接认定为最终根因：当前QK诊断只保存匹配文本Key，未排除图像Key QK分歧或Softmax形状/内核影响。**下一步需要针对MME该题捕获完整相同Q/K及完整Softmax输入，隔离重放早层QK/Softmax**，不应预设与旧晚层AV或7908投影问题相同。

失败保存在`numerical/MME_landmark_0adb088e24d88123.jpg_1/`：原始四路径结果、有效Logits、逐层及逐算子CSV、首个观测分歧、结构审计、失败日志。另一Worker完成当时正在执行的7908回归后读取STOP并退出；未继续其余9例，未启动任何质量预检。

## 复现与资源

- 独立主Worktree：`/data/lk/ATE-worktrees/qwen3vl-svh-sqa7908-projection-repair/`
- 分支：`codex/qwen3vl-svh-sqa7908-projection-repair`
- 独立显存Worktree：`/data/lk/ATE-worktrees/qwen3vl-svh-mme-vision-memory/`
- GPU1：`GPU-b08489af-0a8a-eb48-55ba-8346733af417`
- GPU3：`GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`
- 两卡任务进程全部退出，GPU0/2外部进程未触碰；最终实时状态见`gpu_release.json`。
- 测试源码快照：本地`21b3c9c379`及`protocol.json`完整SHA；模型Revision及原输入绑定均在协议内。
- `all24_status.csv`逐一列出未运行9例，`completion.json`明确标记BLOCKED_NUMERICAL。
- 全部工作位于/data；无模型、原图或大型数据下载，无main修改、无GitHub推送。

关键复现命令（在主Worktree、相同源快照执行；已有原子结果不可覆盖）：

```bash
env CUDA_VISIBLE_DEVICES=GPU-b08489af-0a8a-eb48-55ba-8346733af417 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_sqa7908_repair.verify verify
env CUDA_VISIBLE_DEVICES=GPU-b08489af-0a8a-eb48-55ba-8346733af417 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_fasttrack_runtime.run numerical --shard 0
env CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_fasttrack_runtime.run numerical --shard 1
```

以上是实际使用命令记录，不建议直接覆盖现有验证目录。修复后需新目录、新协议哈希和独立新进程，继续保留本次FAIL。

## 明确结论

ScienceQA 7908投影形状问题已修复并完成独立回归；MME视觉OOM已在新Runtime且不缩图的条件下解决。**整体Physical数值一致性仍未完全解决**：MME P99早层语言Attention存在新分歧，因此FAST-TRACK当前不可继续到质量评测，需先完成上述最小隔离诊断，不得以现有14例PASS宣称全24例通过。
