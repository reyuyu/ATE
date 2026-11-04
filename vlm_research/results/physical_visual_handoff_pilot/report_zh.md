# Physical Visual Handoff：部分完成，正确性门槛未通过

## 执行边界

独立分支 `codex/dual-gpu-physical-visual-handoff-pilot`，所有本轮代码、检查点与日志位于 `/data/lk/ATE-worktrees/dual-gpu-physical-visual-handoff-pilot`。
依用户后续指示不使用 `/data1`；没有删除历史数据，没有修改main或模型权重。LLaVA使用物理GPU2，Qwen使用物理GPU3，两张3090用UUID绑定，独立冻结环境。

正式完成标记：**PARTIAL / STOPPED_VALIDATION_FAILURE**。没有完成60题初筛、240题验证或正式性能基准；不报告未经测量的加速比。

## 已实现与验证

- 原生视觉前端，原生语言Decoder模块；显式分离每层物理KV与原始逻辑位置。
- LLaVA进入L20 / Qwen进入L22前确实移除视觉Hidden行，后层QKV与MLP接收较短张量，不是Mask伪装。
- P2/P3压缩持久视觉KV；P4早层答案分支不调用QKV/Attention/o_proj，并不保留这些层的持久KV。
- 原生Question位置、Qwen动态视觉位置、vision_end、原生mRoPE及早层DeepStack得到保留。
- 5项CPU测试通过，追加FP32小模型裁剪等价测试也通过。同形状原生/No-op全部返回Logits严格逐位一致。

## LLaVA：43题 / 387条质量记录，不能外推240题

6张独立Smoke图片的24个密集/物理对照全部通过。Smoke最大Logits误差0.03125、最大文本Hidden相对RMSE 0.004241。
初筛前42题全部等价对照生成Token一致；第43题 `LogicalObj__03567478` 的D1回答“No”，P1回答“Yes”，触发冻结的Greedy一致性停止规则。
这题原生回答“Yes”；物理路径看似纠正D1，但不得把工程数值差异算成算法收益。完整387条记录（含触发题）均保留；43题只是提前停止的子集，任务比例也不再平衡。

`quality_divergence_diagnostic.json`保存该题Top-5首Token和独立自由生成诊断；分叉后的Decode使用不同前缀，其Logits差异不能当作共同前缀下的浮点实现误差。`common_prefix_divergence_audit.json`仅对共同首步进行数值比较：密集Yes/No平局，物理Yes高出一个FP16步长。`physical_execution_trace.json`记录实际短张量行数、原始位置与禁止连接概率。
该题观察到KV实际存储：D1 321,388,544 Bytes，P1 208,142,336 Bytes，P2/P3 19,398,656 Bytes，P4 16,973,824 Bytes。这些是单样本真实Tensor存储，不是代表性平均，也不意味着数值/答案等价已经通过。

## Qwen：首张Smoke停止，未运行质量集

同形状Native/No-op逐位一致。D1与P1保留文本Hidden在L0–L21逐位一致，从L22物理裁剪点开始出现误差，最终Hidden相对RMSE 0.061073，超过GPU前固定0.04门槛；最大Logits误差0.5，首Token和完整Token序列暂未变化。
不因答案相同而忽略Hidden门槛。没有开展P2/P3/P4矩阵或六层候选。
额外诊断仅关闭BF16 reduced-precision reduction，Hidden相对RMSE仍为0.043846，高于原0.04门槛，仍不通过；该诊断不是正式新条件，没有替换原生设置或放宽阈值。

## 十二项问题的当前回答

1. LLaVA后层视觉跳算：已真实实现，六个Smoke通过，但初筛存在Greedy分叉，不能宣称普遍等价。
2. Qwen后层跳算：已实现较短张量；在原始BF16设置下数值门槛未过，不能交付为已验证优化。
3. QKV/MLP行数：执行轨迹证实变短；正式Profiler和时延尚未完成，不宣称实际加速。
4. Prefill后视觉KV删除：LLaVA在Smoke及部分初筛有真实紧凑存储；总体等价结论未完成。
5. 逐层即时释放：代码支持，正式即时释放验证尚未执行，不能标记已验证。
6. LLaVA L2–L5：物理分支与无未来消费者层的持久KV删除已实现并通过六个Smoke；尚无240题联合质量结论。
7. Qwen四层/六层：上游P1门槛失败，未继续叠加，不能给出质量结论。
8. Question-Only结合截止层：仅有部分记录，逐题结果可查，不是完整60/240题结果。
9. Prefill/Decode/端到端加速比：正式测量未运行，没有可靠数值。
10. KV存储减少：有单样本实测和逐题KV形状；没有经公平微基准验证的峰值显存收益。
11. 数学图等价、浮点一致与任务准确率：数学依赖图/FP32小模型支持实现方向；FP16/BF16出现差异，不能将三者混为一谈。
12. 下一步：先固定同Kernel下可重复的形状变化误差来源，消除或量化近似并重新预注册门槛；目前没有可直接扩展3000题的合格配置。

## 工程限制与交付

本轮严格停在验证失败处，没有反复放宽门槛，没有将Mask算成加速，没有补造Latency或Profiler数据，也未把部分准确率标成完整实验。
主性能图A/C及正式性能表标为NOT_RUN；仅提供实际KV存储示意图，不绘制伪造的速度曲线。
所有GPU作业结束后核查资源；其他用户任务保持不变。单图单轮结果不能推广多轮Cache永久删除。
