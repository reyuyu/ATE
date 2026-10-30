# 深层 Attention 来源分解：中文研究总结

状态：已完成新的200图，590个冻结目标，27140次真实单步消融。词类计数：{'Function': 200, 'Visual': 197, 'Relation': 193}。
不是旧200图的重新选样，不因结果排除图片；复用上一轮词表、单Token对齐和位置选择规则。

## 1. Visual 在 L26/L31 对什么来源最敏感？

L26 四个互斥来源中最大带符号平均损失来自 History：0.021048 nat（普通95% CI [0.016553, 0.025806]），平均绝对变化0.025925，Top1翻转7/197。这只是实测排序，不是对所有词/层的普遍断言。
同Token History−Image：0.004107，family41校正95% CI [-0.011555, 0.019304]。
同Token History−Prompt：0.008563，family41校正95% CI [0.000964, 0.016619]。
L31 四个互斥来源中最大带符号平均损失来自 History：0.086091 nat（普通95% CI [0.072193, 0.100754]），平均绝对变化0.094189，Top1翻转15/197。这只是实测排序，不是对所有词/层的普遍断言。
同Token History−Image：0.097592，family41校正95% CI [0.071128, 0.125074]。
同Token History−Prompt：0.077936，family41校正95% CI [0.054866, 0.102989]。

## 2. L31 的高 Whole、低 Image 能否由 History 或 Prompt 解释？

Visual L31 Whole：loss=0.102197，mean|ΔLogP|=0.121824，flip=21/197。
Visual L31 Image：loss=-0.011501，mean|ΔLogP|=0.032966，flip=3/197。
Visual L31 Prompt：loss=0.008155，mean|ΔLogP|=0.014898，flip=2/197。
Visual L31 History：loss=0.086091，mean|ΔLogP|=0.094189，flip=15/197。
Visual L31 BOS：loss=0.020622，mean|ΔLogP|=0.027023，flip=5/197。
只有实际正向且有区间支持的 History/Prompt 移除效应才能支持其局部预测作用；即使来源移除损失很大，也不能线性分摊或完整解释 Whole 损失。负损失说明原Token概率上升，而非该来源没有功能。

## 3. Function 是否表现出不同来源敏感性？

L26 Image Visual−Function（raw，197张同图）：0.013058，校正95% CI [-0.002581, 0.029318]。
L26 Image Visual−Function（adjusted，197张同图）：0.013297，校正95% CI [-0.001770, 0.028367]。
L26 Prompt Visual−Function（raw，197张同图）：0.007783，校正95% CI [0.003470, 0.012478]。
L26 Prompt Visual−Function（adjusted，197张同图）：0.006546，校正95% CI [0.002329, 0.010613]。
L26 History Visual−Function（raw，197张同图）：0.018128，校正95% CI [0.008736, 0.028656]。
L26 History Visual−Function（adjusted，197张同图）：0.017876，校正95% CI [0.008696, 0.027030]。
L31 Image Visual−Function（raw，197张同图）：-0.004205，校正95% CI [-0.015254, 0.006643]。
L31 Image Visual−Function（adjusted，197张同图）：-0.005196，校正95% CI [-0.016166, 0.005118]。
L31 Prompt Visual−Function（raw，197张同图）：0.009397，校正95% CI [0.002864, 0.018136]。
L31 Prompt Visual−Function（adjusted，197张同图）：0.007695，校正95% CI [0.001743, 0.015716]。
L31 History Visual−Function（raw，197张同图）：0.101792，校正95% CI [0.077393, 0.128487]。
L31 History Visual−Function（adjusted，197张同图）：0.105737，校正95% CI [0.082144, 0.134073]。
调整后区间仍排除零才支持超出线性置信度/位置控制后的条件类别差异；区间跨零不能证明两类相同。Function低损失不等于可跳过整层。

## 4. Relation/Action 在中层和后层依赖哪些来源？

L14：Image loss=0.027967, |Δ|=0.049512, flip=12/193；Prompt loss=0.005717, |Δ|=0.021273, flip=6/193；History loss=0.014813, |Δ|=0.039198, flip=8/193。
L16：Image loss=0.035498, |Δ|=0.059673, flip=14/193；Prompt loss=-0.007778, |Δ|=0.022085, flip=6/193；History loss=0.001379, |Δ|=0.048907, flip=4/193。
L26：Image loss=0.010861, |Δ|=0.022311, flip=3/193；Prompt loss=0.006980, |Δ|=0.010013, flip=1/193；History loss=0.012146, |Δ|=0.021899, flip=3/193。
L31：Image loss=-0.007632, |Δ|=0.024869, flip=5/193；Prompt loss=-0.000983, |Δ|=0.007540, flip=0/193；History loss=0.033342, |Δ|=0.050975, flip=8/193。
Relation沿用优先级，介词/空间关系与动作词合并；词频表在classification_audit.json。以上Relation逐来源排序是描述性结果，不冒充新增主对比的显著发现。

## 5. L25–L31 是否形成连续深层敏感区？

预定判据：七层校正区间下界全部>0，才称这七个实测层均有正向平均损失证据。本次满足该判据。
L25：0.013692，校正95% CI [0.002784, 0.024575]。
L26：0.072163，校正95% CI [0.049419, 0.095916]。
L27：0.014231，校正95% CI [0.006800, 0.022406]。
L28：0.018615，校正95% CI [0.006881, 0.030411]。
L29：0.022830，校正95% CI [0.004241, 0.044365]。
L30：0.043921，校正95% CI [0.027763, 0.060510]。
L31：0.102197，校正95% CI [0.074889, 0.129776]。
六个相邻层配对差异完整保留在deep_layer_peak_analysis.csv；不能仅靠L26/L31两个点宣布离散峰，也不能把未测L17–L24补成曲线。

## 6. Attention Mass 与实际输出贡献是否一致？

Visual L14：最大平均Mass来源=BOS（0.465003）；最大平均投影范数来源=IMAGE（7.869725）。Image–History平均Cosine=-0.075386，四组向量抵消比例=0.562961。
Visual L16：最大平均Mass来源=BOS（0.555266）；最大平均投影范数来源=GENERATED_HISTORY（6.848282）。Image–History平均Cosine=0.094329，四组向量抵消比例=0.616499。
Visual L26：最大平均Mass来源=BOS（0.691319）；最大平均投影范数来源=IMAGE（11.508270）。Image–History平均Cosine=0.184361，四组向量抵消比例=0.722548。
Visual L31：最大平均Mass来源=BOS（0.604323）；最大平均投影范数来源=GENERATED_HISTORY（13.404800）。Image–History平均Cosine=0.110355，四组向量抵消比例=0.569125。
Mass没有Value方向或Wo的信息；范数与实际移除损失也不是同一个量。图2分开画Mass、范数、相同残差分母的比例和实际绝对LogP变化，不从未标准化范数推断固有重要性。

## 7. 是否支持生成历史承担重要预测作用？

Visual L26 History-off：0.021048，校正95% CI [0.013937, 0.028947]。
Visual L31 History-off：0.086091，校正95% CI [0.063925, 0.110655]。
Function L26 Recent−Older：0.004939，校正95% CI [-0.001211, 0.010097]；Recent/Older平均绝对变化=0.005993/0.013855，Older为空的t=1目标=0。
Visual L26 Recent−Older：-0.010308，校正95% CI [-0.016719, -0.004338]；Recent/Older平均绝对变化=0.008074/0.019416，Older为空的t=1目标=32。
Relation L26 Recent−Older：-0.002693，校正95% CI [-0.007462, 0.001744]；Recent/Older平均绝对变化=0.009215/0.014748，Older为空的t=1目标=0。
Function L31 Recent−Older：-0.012933，校正95% CI [-0.020058, -0.006304]；Recent/Older平均绝对变化=0.021032/0.016842，Older为空的t=1目标=0。
Visual L31 Recent−Older：-0.003487，校正95% CI [-0.022112, 0.016143]；Recent/Older平均绝对变化=0.045372/0.047422，Older为空的t=1目标=32。
Relation L31 Recent−Older：-0.006373，校正95% CI [-0.017872, 0.012701]；Recent/Older平均绝对变化=0.019352/0.035319，Older为空的t=1目标=0。
Recent包括当前消费的上一个生成Token；Older可为空但不能删除这些样本。只有本表真实效应及不确定性支持的范围内，才讨论短距离或更早历史的作用。

## 8. 本轮仍不能证明什么？

不能证明视觉信息已经从图像迁移到历史；历史Value可能混有多层、多来源信息。不能证明直接图像贡献小就不使用图像。
不能将来源损失相加为Whole损失；各来源方向抵消和后续非线性都可能改变单独移除效应。
不能将词类标签当作严格Grounding或词性的无偏因果效应；原生Caption决定可观察词类，单Token词表过滤限制推广范围。
不能从单步LogP/Top1变化推导完整Caption准确率、整层可跳过、动态路由有效或真实推理加速。

## 验证与复现

GPU1/2/3上的四个逻辑分片Smoke、23项采集/调度CPU测试、独立统计测试通过。原生生成/无操作/观察器Logits精确一致；590次目标父Cache哈希检查通过。
5980组原生数值分解验证通过。容差来自冻结FP16/FP32传播误差界，旧Image审计原样保留；未按输出放宽。
最初GPU0曾被其他用户任务占用，预检拒绝加载；随后用户授权改用GPU1/2/3。原200图及四个逻辑分片保持不变，新绑定重新Smoke；未使用GPU0或干扰其他进程。
全部明细、置信区间、环境、代码及原始Logits哈希见CSV/JSON，五张主图均有PNG和PDF。大体积原始NPZ保留本机并提交路径/哈希，不向Git提交模型权重。
