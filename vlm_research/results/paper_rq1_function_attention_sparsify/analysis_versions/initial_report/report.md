# Function-Token Identification and Attention Source Sparsification

**最终判断：C。存在局部低敏感现象，但当前证据不足以证明能够可靠转化为低损害、低开销的实际 Attention 加速。**

## 设计、冻结与样本

- 基线提交：`6964407e6b4c1291cb0ca7c41128947d3025520d`。本分支不修改 main，不覆盖历史实验。
- Stage A：两轮原有400张图、1181个冻结位置、24801次单步干预；只研究 L14/L26/L31。
- Stage B：按图片输入哈希固定300训练/100校准；所有实际 Decode 位置，包含 EOS、歧义、未知、多 Token、不对齐和标点。t=0 是 Prefill，不属于在线裁剪范围。
- 独立留出200张：ID、图像字节 SHA256、解码 RGB 均与开发数据隔离，并排除前两轮 Smoke 及更早 COCO Discovery。没有根据输出选题。
- 使用物理 GPU1/2/3，每张图所有配对条件在同卡执行。GPU0 的外部任务未被终止或使用。
- LLaVA-1.5-7B 固定权重，FP16/eager，Transformers4.49.0，Greedy，正常 KV Cache，最多64 Token。只训练小型识别器，不训练 VLM。
- Stage A、B 是规则选择/校准，不是独立确认。全 Caption 质量和识别器最终指标仅使用冻结留出200图。

## Stage A：联合来源保留

A1 保留原始 Softmax 分母，联合减去移除来源的 FP32 投影贡献；它已经计算完整 Attention，不能当作加速。A2 在 QK/AV 前真正 gather 保留 K/V，重新 Softmax。模板分隔符全部归入原 Prompt 文本，四类 Key 无遗漏/重叠。

这里研究的是当前层的 Key/Value 访问路径，不是原始信息来源：L31 的 History Value 和残差已经包含 L0–L30 融合后的上下文。即使 History-only 损害较低，也不能解释成预测不依赖图像或原 Prompt。

### L31 Function

| algorithm | mode | n | mean_loss | mean_abs_delta_logp | p95_loss | p99_loss | top1_flips | max_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | No-Image | 400 | -0.00786974 | 0.0167936 | 0.0290037 | 0.0531542 | 10 | 0.0894713 |
| A1 | BOS+History | 400 | -0.00840106 | 0.019987 | 0.0343768 | 0.0652742 | 9 | 0.100108 |
| A1 | History-only | 400 | 0.00901565 | 0.0221529 | 0.0592658 | 0.0752697 | 9 | 0.104869 |
| A2 | No-Image | 400 | -0.013733 | 0.0185889 | 0.0176299 | 0.0485787 | 10 | 0.0700848 |
| A2 | BOS+History | 400 | -0.0186166 | 0.0275689 | 0.0302748 | 0.0673275 | 13 | 0.0924338 |
| A2 | History-only | 400 | -0.0142516 | 0.0556562 | 0.121879 | 0.252076 | 10 | 0.411314 |
| zero | Whole-off | 400 | -0.00278597 | 0.0402861 | 0.0916339 | 0.121028 | 11 | 0.241241 |

冻结的动态来源风险门槛通过：**True**。门槛为 Mean Loss<=0.10nat、P95<=0.50nat、Top1 Flip<=10%，三个都通过才继续；这只是开发门槛，不是安全认证。
历史 Full 最大 Logit 差异：`0.0`；400图原生生成 Token 序列全部一致。没有为历史差异放宽容差。

全部角色、三层、A1/A2 的结果以及负面尾部都保存在 `source_retention_summary.csv`、`source_retention_per_token.csv`、`severe_source_cases.csv`。

![Function source retention](figures/figure1_function_source_retention.png)
![A1 vs A2](figures/figure2_subtraction_vs_sparse.png)

## Stage B：提前识别 Function

A 仅使用已消费前缀最近 Token 的哈希一元/二元/三元特征和已知位置；B 使用真实 L31 层输入隐藏状态及其范数/已知位置；C 合并两者。没有使用当前 Full logits、下一 Token ID、Attention 输出、真实 BOS Mass 或未来生成长度。位置/64 的分母是预先固定上限，不是实际最终长度。
词类是冻结词表和严格单Token对齐的操作性标签，不是人工POS真值。Unknown中即使有语法功能词，本轮仍按非Function计入误判，不通过事后扩词表提高Precision。A在冻结阈值下拒绝部署，因此其误判率0不能解释成识别效果优于其他模型。

### 校准集冻结阈值

| predictor | threshold | precision | precision_ci95_low | recall | coverage | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1 | NA | NA | 0 | 0 | False |
| B | 0.99 | 0.985714 | 0.967564 | 0.557951 | 0.13601 | True |
| C | 0.999 | 0.976048 | 0.951807 | 0.439353 | 0.108161 | True |

选择的部署识别器：`B`；阈值：`0.99`。不因留出成绩重新选模型或调阈值。

### 独立留出集

| predictor | precision | precision_ci95_low | precision_ci95_high | recall | false_positive_rate | coverage | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | NA | NA | NA | 0 | 0 | 0 | 0.104352 |
| B | 0.978313 | 0.963673 | 0.990589 | 0.571027 | 0.0036961 | 0.131914 | 0.0357902 |
| C | 0.979943 | 0.964706 | 0.993846 | 0.481013 | 0.00287474 | 0.110935 | 0.0429857 |

### 部署识别器：非Function类别的误判

| category | n | predicted_function | rate |
| --- | --- | --- | --- |
| Ambiguous | 34 | 1 | 0.0294118 |
| EOS-Special | 200 | 0 | 0 |
| Multi-token-Unaligned | 799 | 0 | 0 |
| Punctuation | 271 | 1 | 0.00369004 |
| Relation | 397 | 1 | 0.00251889 |
| Unknown | 205 | 6 | 0.0292683 |
| Visual | 529 | 0 | 0 |

EOS、未知、歧义、多 Token 及所有其他类别的误判率分别保存在 `holdout_classifier_category_errors.csv`。校准曲线和覆盖率原始点也独立保存，不删除未知类别。高 Function Precision 不意味着裁剪安全。此固定 Prefix-only 基线失败，不等于所有可能的前缀识别器均不可行。

![Function identification](figures/figure3_function_identification_holdout.png)

## Stage C：真实自由生成质量

每个策略独立从图像 Prompt 开始，使用该策略自己的实际前缀和隐藏状态继续生成；没有从 Full 后续 Token teacher-force。Oracle Function 仅用于独立 Full 轨迹的单步参考，从未用于可部署自由生成。

| strategy | n | CIDEr | CIDEr_delta | CIDEr_delta_ci95_low | CIDEr_delta_ci95_high | SPICE | SPICE_delta | SPICE_delta_ci95_low | SPICE_delta_ci95_high | changed_captions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 200 | 1.21025 | 0 | 0 | 0 | 0.248112 | 0 | 0 | 0 | 0 |
| S1 | 200 | 1.21993 | 0.00968373 | -0.0179828 | 0.0377106 | 0.245551 | -0.00256076 | -0.00645691 | 0.00113131 | 49 |
| S2 | 200 | 1.18698 | -0.0232704 | -0.0684218 | 0.0187055 | 0.244258 | -0.00385451 | -0.00928169 | 0.00108416 | 65 |
| S3 | 200 | 1.21205 | 0.00179817 | -4.98225e-05 | 0.0046257 | 0.248574 | 0.00046155 | 0 | 0.00125324 | 3 |
| S4 | 200 | 1.2112 | 0.000954633 | -7.47338e-05 | 0.00293863 | 0.248544 | 0.000431308 | 0 | 0.00118281 | 2 |

S0=Full；S1=静态 No-Image；S2=静态 BOS+History；S3=预测 Function 后 BOS+History；S4=预测 Function 后 No-Image。未列出的策略因冻结门槛失败而未运行，不能当作0损害。
自由生成中的 Function/非Function裁剪数量按各策略实际输出 Caption 事后对齐，仅用于评价，不能作为在线标签；独立原生 Full 轨迹上的识别器精度和误裁风险另表给出。
CIDEr 使用原始0–10口径（不是乘100后的数）；SPICE 为0–1。普通图片级配对 Bootstrap10000次，CIDEr 的参考语料 IDF 固定为同一完整200图。冻结非劣效容忍为 CIDEr0.02、SPICE0.01；下界大于负容忍只是样本内 Bootstrap 判据，200图仍不足以排除罕见严重损害。

| strategy | CIDEr_bootstrap_noninferior | SPICE_bootstrap_noninferior | pruned_decode_steps | decode_steps | pruned_function_steps | pruned_nonfunction_steps | retained_key_ratio | mean_length | eos_count | truncated_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | True | True | 0 | 3146 | 0 | 0 | 1 | 16.73 | 200 | 0 |
| S1 | True | True | 3165 | 3165 | 708 | 2457 | 0.0475748 | 16.825 | 200 | 0 |
| S2 | False | True | 3213 | 3213 | 721 | 2492 | 0.0163321 | 17.065 | 200 | 0 |
| S3 | True | True | 411 | 3158 | 402 | 9 | 0.871959 | 16.79 | 200 | 0 |
| S4 | True | True | 413 | 3160 | 404 | 9 | 0.875524 | 16.8 | 200 | 0 |

### 预测/Oracle裁剪的实际单步风险

| group | mode | n | images | mean_loss | p95_loss | p99_loss | top1_flips | severe_gt1 | false_function_positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_function | No-Image | 711 | 200 | -0.0122984 | 0.0201314 | 0.0406631 | 10 | 0 | 0 |
| oracle_function | BOS+History | 711 | 200 | -0.014951 | 0.0351362 | 0.0663562 | 19 | 0 | 0 |
| predicted_function | No-Image | 415 | 187 | -0.0141042 | 0.00524042 | 0.0286034 | 2 | 0 | 9 |
| predicted_function | BOS+History | 415 | 187 | -0.0194388 | 0.0124444 | 0.0349678 | 3 | 0 | 9 |

对象候选词、颜色、数量、关系词差异保存在逐图 CSV。它们只是词表对齐的文本变化，不是人工验证的视觉语义错误或 Grounding 得分；严重退化案例未删除。

![Quality vs sparsity](figures/figure4_quality_sparsity.png)
![Actual Caption cases](figures/figure6_caption_case_studies.png)

## 实际延迟与计算

固定物理 GPU1，冻结留出顺序前10图，每实现3次 Warmup、5次重复，种子固定的随机条件顺序。时间统计先在每张图内平均重复，再做10图配对 Bootstrap；50次重复不当成50张独立图片。主时间不含 profiler/隐藏状态采集、图像预处理、模型加载和官方评分。CUDA Events 与同步用于 Prefill/Decode 分解，另有完整模型生成 wall time。

| strategy | implementation | mean_caption_seconds | measured_speed_ratio | paired_time_delta_ci95_low | paired_time_delta_ci95_high | decode_per_token_delta_ci95_low | decode_per_token_delta_ci95_high | speed_supported | mean_l31_attention_operator_ms | mean_classifier_host_ms | peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2 | A2 | 1.39334 | 0.991655 | -0.0422676 | 0.0739034 | 0.189071 | 0.342855 | False | 0.668833 | 0 | 13916.5 |
| S2 | A1 | 1.37028 | NA | -0.0972928 | 0.0672628 | 0.765885 | 0.920644 | False | 1.00402 | 0 | 13916.5 |
| S1 | A2 | 1.39278 | 0.992053 | -0.0438531 | 0.0741235 | 0.182368 | 0.253088 | False | 0.674214 | 0 | 13916.5 |
| S0 | A2 | 1.38172 | 1 | 0 | 0 | 0 | 0 | False | 0.49353 | 0 | 13916.5 |
| S3 | A2 | 1.39231 | 0.99239 | 0.00913274 | 0.0121063 | 0.649697 | 0.741955 | False | 0.516727 | 0.63166 | 13916.5 |
| S4 | A2 | 1.39309 | 0.991836 | 0.00990729 | 0.0129459 | 0.673744 | 0.821211 | False | 0.51682 | 0.640802 | 13916.5 |

- 只裁剪 L31 的真实 Decode QK/AV。QK/AV 理论操作数为每步 `4*32*128*retained_keys`；不把理论 FLOPs 比例当成实测速比。
- 全尺寸 K/V 投影、缓存存储、DynamicCache 拼接仍保留，拼接本身仍读取完整历史缓存。稀疏 QK/AV 的输入范围缩小，但不声称整条缓存带宽按同比例降低。
- PyTorch index_select/gather、Python 调度与分类器同步成本均真实存在。`operator_profile.json` 中独立 profiler 的 index_select 时间是全 Caption 运算统计，不冒充纯 L31 gather；不拿 profiler 时间作为端到端速度。
- A1 的实际耗时仅作机制模拟成本对照，不给出加速比。真实速度支持同时要求 Caption 配对时间和 Decode 每 Token 时间的95%CI上界<0，避免把提前 EOS/更短 Caption 单独误报成 Attention 加速。
- 此 eager实现的理论收益仅来自32层中最后1层的部分运算。若 gather/路由抵消收益，可考虑专用 fused retained-key Attention kernel；本轮没有实现或虚构 Triton/CUDA 收益。

![Latency vs quality](figures/figure5_latency_quality.png)

## 九个研究问题

1. **Function 是否只需要 History？** 不能这样概括。History-only 的 Mean/P95 Loss 为 -0.014251557118986966/0.12187901139259331，Top1翻转 10/400，必须保留尾部和反例。
2. **BOS+History 是否更稳？** 尾部幅度更小：P95为0.0303，History-only为0.1219nat；但翻转13比10更多，不能声称所有指标均更稳。
3. **能否同时移除 Image 和 Prompt？** L31 Function 的 BOS+History 已实际测量；再去 Prompt 相对 No-Image 的 Mean Loss 差为 -0.004884nat。只针对固定原生轨迹，不推出整条生成安全。
4. **真正裁剪 Key 后仍低损害吗？** A2 与 A1 分开汇报，重新 Softmax 会改变保留来源权重；以真实 A2 的尾部和完整 Caption 质量判断，不能直接继承减法结论。
5. **能提前识别 Function 吗？** 冻结部署识别器的留出 Precision/Recall 为 0.9783132530120482/0.5710267229254571，并保留 EOS/未知等假阳性。
6. **动态优于静态吗？** 需要同时比较200图官方质量、实际裁剪范围、10图固定硬件时间和路由成本，不根据覆盖率宣称更优；本轮整体判断见开头。
7. **完整 Caption 质量受损吗？** 上表提供 CIDEr/SPICE 配对差异、区间、逐图增减和真实变更案例；CI包含0不等于无损，非劣效阈值没有调整。
8. **哪种策略实际加速？** 满足冻结的两个配对时间判据者为：无。理论稀疏度、A1模拟和更短文本均不单独作为加速证据。
9. **足够声称安全吗？** 不足。还需更大且独立的图像/文本任务确认、更严格的严重风险上界、多硬件重复，以及在质量可控前提下的专用 kernel 验证；本轮不扩展研究范围。

## 正确性与审计

- 原生 cached Full 与 model.generate 的 Token 完全一致；Observer/No-op logits 精确一致。
- 最终CPU测试29项通过；真实GPU Smoke在三个物理GPU均通过。
- 在线特征CPU/GPU独立算术审计：3146个留出Decode位置，概率最大差异5.960464477539062e-07，路由决定不一致0；阈值未调整。非零概率舍入差异明确记录，没有冒充数值完全相同。
- 单步条件各自深克隆原生 cache，目标 Token 时序、原始 RoPE/position、Key 分区与缓存哈希检查通过。
- A1 使用前轮不变的解析 FP16/FP32舍入界；A2 使用独立 bmm参考加全维 masked softmax/AV，正式路径只对保留 K/V 做 QK/AV。
- 初次 Smoke 因参考 QK 的宽度/FP16累计路径不同而被拒绝，未进入正式采集。失败与修正历史完整保留；没有放宽原有数值容差。
- 留出运行前还修复了流水线可变状态写入和自由生成包装的 scaling关键字接口错误；相应失败均归档，未计入正式结果。
- 正常/异常路径恢复 Hook 和临时 eager实现。模型参数版本不变，固定权重文件末尾重新 SHA256 校验通过；原始数据未改。
- 正式完成记录无 NaN、Inf、OOM 或跳过失败样本。冻结代码版本、环境、GPU记录、CPU/GPU日志、逐图检查点和 raw SHA 清单均保存。
- 已校验外置原始结果文件 803 个。raw logits/特征不上传Git，以清单SHA及本机绝对路径绑定；小型识别器权重另行提交。

## 产物

`source_retention_per_token.csv`、`all_development_token_labels.csv`、`development_predictions.csv`、`holdout_token_predictions.csv`、`caption_per_image.csv`、`caption_generation_steps.csv`、`oracle_and_predicted_single_step_risk.csv`、`latency_repetitions.csv`、`operator_latency.csv`、全部PNG/PDF及各阶段日志。

本轮不训练VLM，不扩展层范围、不做Router层扫描。实验已停止并释放本轮GPU资源，不自动合并 main。
