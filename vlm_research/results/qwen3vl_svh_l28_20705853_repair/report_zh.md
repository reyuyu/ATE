# Qwen3-VL-SVH：20705853 Prefill 数值分歧修复

## 结论与范围

**本轮限定验收 PASS。** 题目 `20705853` 的 Dense 功能参考与修复后 Physical，在自然 EOS 停止的全部有效 Logits 上逐位一致；此前五个通过样本也全部通过原门禁。没有重新运行59个边界样本，没有启动六项48题预检或全量评测。因此不能据此将扩大验收或六项正式评测改判为 PASS。方案 B 仍是 ATE 统一 HF 协议，方案 A 作者官方协议仍为 BLOCKED。

基线：`dfd169ad206fabbd38de24a631108e716c661383`。独立分支：`codex/qwen3vl-svh-l28-20705853-repair`。原失败复现和逐算子记录冻结于 `a3e5890ac03d11ad69cb18946ce47b070c9840da`；实际修复及执行源码冻结于 `d84ac970fbbad82670e2ace32887dcf639110078`。未修改 main、历史实现或共享环境，未推送 GitHub。

## 固定输入与门禁

- Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16/eager，Transformers 4.57.6、Torch 2.6.0+cu124。
- 题目：`What makes up the desk?`，保留原 short-answer 后缀。图像 `n264887`，SHA256 `a5ccd2547d9b301ab5b5084804e3ce755721b3f152f4c87cedd097cc15c09238`。
- 实际 Prefill 长度122，动态视觉Token数96，真实文本行数26；未改Processor、像素、分辨率、Prompt、mRoPE或DeepStack。
- 原门禁保持：逐步全词表 Logits 的 mean absolute error ≤0.02、max absolute error ≤0.5，且 Top1、完整生成Token序列一致。Native/No-op 要求严格一致，不要求 SVH 与 Native 答案相同。
- 保留原 SVH：L22前视觉Hidden退出、L1–L6答案Attention跳算、答案禁图、文本持久KV及原始位置语义。未通过修改算法规避误差。

## 第一个真正分歧算子

逐项记录实际 forward 的层输入、Input Norm、Q/K/V投影与输入、Q/K Norm、mRoPE、Mask、QK、Softmax、AV、O投影、残差、MLP各投影及层输出。只按真实逻辑文本位置对齐；因果屏蔽值的 `-inf`/极小值比较按允许连接图审计，不把它们当作有效Logit。

| 位置 | 复现观察 | 最大绝对差 | 平均绝对差 |
|---|---|---:|---:|
| L0–L26已观测真实文本算子 | 一致 | 0 | 0 |
| **L27 AV** | **最早算术差异；未传播到该层输出** | 5.96046448e-08 | 5.59689068e-13 |
| L27层输出、L28输入 | 一致 | 0 | 0 |
| L28 Norm、Q/K/V、mRoPE、有效QK、Softmax | 一致 | 0 | 0 |
| **L28 AV** | 首次产生随后传播的差异 | 0.00048828125 | 4.72825290e-09 |
| L28层输出 | 差异经过O投影、残差和MLP传播 | 0.0625 | 3.39608923e-05 |

所以，历史“首个分歧层L28”指的是**层输出首次不一致**，不是第一个算子差异；第一个已观测算术差异在 **L27的AV**。L28也不是最早直接超过0.02 Logits门槛的概念，0.02是最终有效Logits的平均误差标准，并非中间Hidden逐元素阈值。

完整证据在 `diagnosis_original/operator_comparison.csv`、`first_divergence.json`、`positions_cache_semantics.json`。未发现此次差异始于位置错位、文本KV索引、mRoPE或DeepStack。

## 根因的独立证据

在另一个**不加载模型**的GPU进程，使用实际保存的 Attention probabilities 与文本V，单独重放 L26、L27、L28 的 AV：

- 同一层 Dense/Physical 的真实文本 probabilities 和文本V逐位一致。
- 原始AV的概率矩阵形状为 `[1,32,122,122]`，Physical真实Query压缩后为 `[1,32,26,122]`；补零后的V形状同为 `[1,32,122,128]`。
- 紧凑Query行重放，逐位复现旧Physical的AV。
- 只把Query行数和原始位置恢复为临时零行形状，再取回真实文本行，逐位复现Dense的AV。
- L26作为负对照，两种重放均一致；L27/L28复现原差异并消除它。

因此，直接证据支持**AV批矩阵乘法的Query行几何改变，引发BF16数值计算路径/舍入差异**。本轮没有读取AV内部CUDA算法ID，不能指定某个未测量的Kernel为根因。没有改变 `allow_bf16_reduced_precision_reduction`、dtype或精度设置。

## 修复方式与代价

新实现独立置于 `vlm_research/qwen3vl_svh_l28_repair/engine.py`，继承冻结的 `AlignedExecution`，不覆盖历史源码。

仅对 Physical Prefill 的 L22–L35：AV执行前，将实际文本Query概率行按原始位置放入临时全Query零矩阵；执行AV后立即取回实际文本行。**QK、Softmax、Q/K/V/O投影、Norm、MLP、Residual及Hidden仍只计算真实剩余行；没有恢复视觉Hidden、真实视觉投影或持久视觉KV。** Decode计算路径本身不修改。

对失败题，14层新增AV矩阵计算合计 **1,343,225,856 FLOPs**。每层临时概率矩阵952,576字节，临时完整AV输出999,424字节；这是张量载荷，不是独立测量的峰值显存增量。额外对齐开销随输入规模增长，不能据此宣称加速；本轮没有性能Benchmark。原本的零列对齐开销继续保留，以上是本次额外部分。

## 修复前后独立复测

修复在新进程中复测，诊断PID 2428805、验收PID 2437932。自然生成均为 `wood`，Token IDs `[6660,151645]`。

| 失败题有效Logits | 修复前 mean/max | 修复后 mean/max |
|---|---:|---:|
| t=0 Prefill | 0.03515845 / 0.25 | **0 / 0** |
| t=1 Decode，消费wood并预测EOS | 0.05127071 / 0.3046875 | **0 / 0** |

Native与No-op在六题全部严格一致。失败题还将自定义Native与HF原生 `generate(logits_to_keep=1)` 的全部有效Logits、Token及EOS逐位核对通过。修复后的失败题逐算子比较没有剩余真实文本差异；mRoPE、DeepStack、文本Cache K/V及原始位置检查通过，禁止连接概率为0。

**两步Decode边界：** 该题自然输出wood后即EOS，只有一次自然增量Decode。为满足第二次缓存调用检查，另用同一Dense自然前缀 `[wood,EOS]` 比较 t=0、t=1、t=2；t=2消费EOS，是明确标记的**诊断调用，不属于自然生成结果**。这三个位置的Dense/Physical Logits、已观测中间算子及文本KV均一致，不能把它表述为长答案Decode验收。

| 此前通过样本 | 修复后最坏mean error | 最坏max error | 原门禁 |
|---|---:|---:|---|
| 20705744 | 0 | 0 | PASS |
| 201752756 | 0.0033345383 | 0.125 | PASS，与历史误差相同 |
| 20204868 | 0 | 0 | PASS |
| 202003652 | 0 | 0 | PASS |
| 202004255 | 0 | 0 | PASS |

不能声称六题全部逐位一致：`201752756`保留原先门槛内的非零差异，本轮没有扩展修复它。六题的历史Native/Dense基线Logits与此次记录也经CPU逐位核验一致。

## 测试、审计与资源

- 修复前CPU测试5项通过；冻结修复后CPU测试7项通过。
- CPU独立复核33条已保存Logits比较记录，采用FP64计算差值作独立核对；GPU记录的原FP32误差保留，不改门槛。
- 每次调用检查输入Tensor哈希、参数地址/版本、Hook数量及原函数恢复；正常结束无残留。沿用历史异常清理上下文与CPU测试。
- 仅使用物理GPU3，UUID `GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`。GPU0/1/2的既有任务未终止、未抢占。
- 验收进程已退出，外部 `nvidia-smi` 确认GPU3无计算进程，约5MiB基础占用、利用率0%。详见 `gpu_release_audit.json`。

复核命令：

```bash
env CUDA_VISIBLE_DEVICES= /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_l28_repair_report
```

关键文件：`fixed_protocol.json`、`numerical_audit.csv`、`regression_summary.csv`、`validation.json`、`completion.json`、`isolated_AV/`、`independent_verification/`、`artifact_hashes.json`。

原始实际中间Tensor保存在 `/data/lk/ATE-resources/experiments/qwen3vl_svh_l28_20705853_repair/original_operator_trace.pt`，SHA256 `7773c3ff8e6accae93dbbdae4edfd14133dfd5617dbaeaf1f05f510b450ff0e8`，547,726,848字节；不上传权重、图片或该大数组。轻量逐算子CSV及AV重放证据随本地提交保留。

**最终判断：此次20705853的数值失败已解决，证据闭环是实际AV分歧、独立算子重放、最小形状修复及新进程回归。范围外长输入、长自然Decode和六任务验收仍未建立。到此停止；后续扩大验收需要另行授权，不自动执行。**
