# MME Native视觉OOM：独立Runtime修复

## 状态

原始`VISION_EAGER_TEXT_EAGER`在冻结MME P99输入上仍为`BLOCKED_MEMORY`，没有改判旧协议PASS。

新增Runtime：`VISION_EAGER_ROW_SOFTMAX128_TEXT_EAGER`。保持像素、视觉Token、BF16、权重、QK/AV形状、位置和语言Eager不变，只将视觉FP32 Softmax临时矩阵按128个Query行处理。没有使用SDPA或FlashAttention，仍明确作为新的Runtime配置，需要Native/SVH统一应用并重新通过全体门禁。

## CPU及真实OOM定位

- 起点：`d5ca88d7de604728eb1f0bcea3185f01d266e125`；独立Worktree及`codex/qwen3vl-svh-mme-vision-memory`分支。
- 原失败：`MME:landmark/0adb088e24d88123.jpg:1`。
- 输入总长2062，语言视觉Token2028，原始ViT Patch为8112，Batch=1。
- 实际失败在视觉Attention第0层，Q/K/V均为`[1,16,8112,72]`。
- QK及Softmax矩阵为`[1,16,8112,8112]`，单个FP32矩阵4,211,490,816字节（约3.92 GiB）。旧Eager需要输入类型转换及Softmax输出等同时存活，申请另一缓冲时OOM。
- 本轮真实复现峰值Allocated为24,123,154,944字节；启动前GPU3无计算进程、显存5 MiB。不是其他任务占用、Batch错误或语言KV泄漏。
- 原始日志、CPU估算、当前层、实际形状、显存和Q/K/V捕获SHA均独立保存；没有覆盖旧失败。

## 数值证据

1. CPU三项测试通过：概率/AV输出一致、输入不变、异常恢复及文本路径不触碰。
2. 将真实失败Q/K/V在不加载语言模型的新进程中隔离重放。原完整Softmax与分行Softmax的全部BF16概率、AV输出均`torch.equal`；不是只比较Top1。
3. 将语言模型留在CPU、原视觉模型放GPU，执行完整27层原Eager视觉参考。对MME P99（2028视觉Token）和最大图（2048视觉Token），新旧Runtime主视觉输出及三组DeepStack特征均逐位一致。这是诊断放置方式，不作为整模推理性能结果。
4. 独立进程整模常驻GPU，原始输入和原生成上限，Native/No-op真实自然EOS生成逐步Logits及Token精确一致，无任何缩图。

| 冻结输入 | 语言视觉Token | 新Runtime Peak Allocated MiB | Native/No-op |
|---|---:|---:|---|
| MME P99 `0adb088e24d88123.jpg:1` | 2028 | 21044.970 | 精确一致 |
| MME max `01121b9b3c79afc4.jpg:1` | 2048 | 21126.295 | 精确一致 |
| TextVQA max `36759` | 2028 | 21045.362 | 精确一致 |

以上仅数值与显存验收，不是正式质量成绩，不宣称加速。

## 实现与限制

`vlm_research/qwen3vl_svh_vision_memory/runtime.py`只拦截`Qwen3VLVisionAttention`，语言Attention不变。QK仍完整BF16矩阵乘法及原缩放，AV仍完整原形状矩阵乘法。每行Softmax仍FP32，再转回BF16写入同尺寸概率缓冲；无稀疏化、模型精度变化或视觉预算变化。配置名中的128仅为内存分块行数，不是Token预算。

新路径使用独立Runtime名称和SHA。旧完整Eager的OOM仍保留；不能据此写成旧Eager六项官方协议复现成功。方案A继续BLOCKED，方案B后续必须统一该Runtime后重新验收。

## 文件与资源

- GPU3 UUID：`GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`。
- Worktree：`/data/lk/ATE-worktrees/qwen3vl-svh-mme-vision-memory/`。
- 实际Q/K/V：`/data/lk/ATE-resources/experiments/qwen3vl_svh_vision_memory/actual_failing_vision_qkv.pt`，SHA及大小见`capture.json`；不复制模型或图片。
- 证据：`cpu_diagnosis.json`、`oom_reproduction.json`、`isolated_softmax.json`、`vision_parity.json`、`memory_accept_completion.json`。
- 此独立显存进程均已退出。GPU0/2既有进程未触碰；最终统一报告另记录两张任务卡释放状态。

后续放行仍依赖ScienceQA数值修复、统一新Runtime下24例全部通过及六项各8题预检，不能由本显存测试单独解锁正式评测。
