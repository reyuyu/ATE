# Visual Pathways：视觉来源敏感性、中继交互与稀疏补偿

**总体状态：partial。InternVL 原生 Logits Smoke 失败后按规则停止；没有 InternVL 正式曲线或准确率。其余完成范围如下。**

## 覆盖与执行

- Qwen：3000/3,000 题，108,000/108,000 个逐层条件；其中严格复用历史 240 题 × 36 层，新增 2760 题 × 36 层。状态：complete。
- InternVL：0/240 个正式样本，未启动 3,000 题扩展；三卡均在官方 Chat 与独立 Full 首位置 Logits 精确一致性检查处失败。输入 Token ID 和视觉注入 Embedding 已验证一致，但这不能替代输出一致性。失败日志全部保留，没有放宽容差或继续评分。
- LLaVA 交互：同一批 240 题 / 235 个图片 SHA256 簇，A/B/C/D 共 960 次配对完整生成。
- 稀疏补偿：LLaVA 240 题 × 3 条件，其中 FullImageAccess 严格复用交互 C，新增 480 次生成；Qwen 240 题 × 3 条件，新增 720 次生成。
- 未扩展 Qwen/InternVL 的 L2–L5 交互：LLaVA 未通过事先冻结的正向交互门槛；InternVL 另有 Smoke 停止限制。没有增加层组合、训练或加速算法。
- Qwen/InternVL 历史固定 revision、原生 Chat Template、BF16/eager；Qwen DeepStack 原生。LLaVA 固定 revision、FP16/eager、Transformers 4.49.0、原始 GQA Prompt。模型文件校验和、输入展开及环境见各 protocol/validation 文件。
- 本轮仅使用物理 GPU0/2/3；GPU1 排除。分支 `codex/visual-pathways-seven-hour`，独立 Worktree；没有修改或推送 main。
- 计时起点 2026-11-02T04:18:13+08:00；报告生成 2026-11-02T06:04:37.568179+08:00，已用 1.77 小时；总预算截止 2026-11-02T11:18:13+08:00。最终资源与交付审计时间见 delivery_audit.json。
- 实测正式运行：Qwen 2,760 个新增样本两卡并行约 90.08 分钟；LLaVA 交互 5.26 分钟；LLaVA/Qwen 稀疏补偿分别 3.65/3.71 分钟。后几项与大扫描并行，不能相加作为总墙钟。

## 1. 单层 Image 来源必要性

每次只在一个目标层对原始 Prefill 最后 Query P−1 → 实际 Image Key 加 Softmax 前 −inf；不是 o_proj OFF，也不是 Keep-only。目标是每题原生 Full 自己的首 Token Top1；使用全词表 FP32 log_softmax。主指标为先逐题取绝对值再求均值，不是平均 Signed Loss 的绝对值。Top1 Flip 不是完整答案 Accuracy。

低敏感仅针对当前原生残差、其他层与间接路径背景下的直接来源读取，不表示该层没有视觉信息，也不表示整个 Attention 更新可以省略。

Qwen 全部已完成样本的主指标峰为 L26，均值 0.017883 nat；前三层为 [26, 23, 24]，占全部逐层平均绝对变化之和 15.95%。这一比例仅是描述性的集中程度，不是可跳层比例或独立贡献分解。
原 240 题的峰为 L26；单峰并不代表所有任务共享同一峰，六任务完整曲线与样本数如下。

|任务|题数|Mean Absolute 峰层|峰值 nat|Top3 描述性占比|
|---|---|---|---|---|
|ChooseAttr|500|L26|0.009473|12.47%|
|ChooseCat|500|L21|0.005299|13.31%|
|ChooseRel|500|L26|0.013933|14.69%|
|CompareAttr|500|L26|0.051504|20.23%|
|LogicalObj|500|L21|0.007406|11.90%|
|QueryAttr|500|L26|0.025582|22.76%|

同题 240 问的历史 LLaVA Image OFF 主峰为 L14，前三层 [14, 13, 19] 占比 34.95%；Qwen 的同批峰位与它不同。图中保留各自的全部层，不对齐或平移峰。原生 Prompt、Tokenizer/目标 Token、精度和网络深度不同，因此跨模型比较是层级模式的描述，不是严格控制架构以外因素的因果比较。
对“不同 VLM 是否共享少数视觉热点”的回答：可观察到非均匀层级分布及模型/任务依赖峰，但不能仅凭最大峰声称强稀疏；应同时查看 Top3 占比与非峰背景。InternVL 本轮无有效结果，**不能声称已验证 Qwen 与 InternVL 的共同规律**。
图：`figures/qwen_overall_sensitivity.pdf`、`qwen_six_tasks_absolute/signed/flip.pdf`、`cross_model_paired240.pdf`。跨模型图仅使用完全相同的 240 问，LLaVA 明确标为历史数据；不为 InternVL 填零或画假曲线。

## 2. L2–L5 死亡区是否苏醒

A：原生中继/原生 P−1；B：原生中继/P−1 L2–L5 o_proj OFF；C：中继 OFF/原生 P−1；D：中继 OFF/P−1 L2–L5 o_proj OFF。中继 OFF 只屏蔽 P−1 之前全部非视觉 Prefill Query → Image 的合法连接；P−1 直接读图正常，Decode 完全原生。其他 Prefill Query 不做 o_proj 关闭，视觉 Token、MLP、残差和原始位置保留。

|条件|完整 Accuracy|ΔA pp|配对95%CI pp|原对→错|原错→对|固定 Full 首Token Loss nat|首Token翻转|
|---|---|---|---|---|---|---|---|
|A|88.750%|+0.000|[0.000, 0.000]|0|0|0.000000|0|
|B|88.750%|+0.000|[0.000, 0.000]|0|0|-0.003283|0|
|C|75.417%|-13.333|[-18.828, -7.949]|40|8|0.277286|48|
|D|76.667%|-12.083|[-17.500, -6.695]|37|8|0.262790|45|

- `relay_on_skip_logp_harm`：-0.003283，95%CI [-0.006876, +0.000424]。
- `relay_off_skip_logp_harm`：-0.014496，95%CI [-0.022682, -0.006558]。
- `logp_interaction`：-0.011213，95%CI [-0.019249, -0.003380]。
- `relay_on_skip_accuracy_harm_pp`：+0.000000，95%CI [+0.000000, +0.000000]。
- `relay_off_skip_accuracy_harm_pp`：-1.250000，95%CI [-3.813559, +1.271186]。
- `accuracy_interaction_pp`：-1.250000，95%CI [-3.813559, +1.271186]。

交互定义：(D 的固定 Full Token Loss − C 的 Loss) − (B 的 Loss − A 的 Loss)，正数才表示中继受损后 Skip 损害增加。事前扩展门槛为该交互的图片簇配对双侧95%CI整体大于0。
本轮门槛是否通过：**False**。LogP 交互点估计 -0.011213 nat，95%CI [-0.019249, -0.003380]，不是正向“苏醒”。Accuracy 交互 -1.250 pp，区间跨0，不能据此认定 Skip 改善 Accuracy 或两者等价。
非零负向 LogP 交互说明该干预的影响并非完全不受路径背景影响，但方向不支持“原先低敏感的更新变得更必要”，Accuracy 交互也没有明确证据。
**结论：在本批样本与固定中继干预下，没有观察到 L2–L5 死亡区苏醒；固定 Token LogP 交互反而指向更小的 Skip 损害。** 这是对当前具体假说的反向证据，不是对所有信息路径、所有模型的普遍否定。图：`figures/llava_relay_interaction.pdf`。

## 3. 固定稀疏热点能否补偿中继损伤

三个条件均为 Relay OFF，只改变原始 P−1 的直接 Image 读取；Decode 原生。FullImageAccess 不是原生 Full，而是中继受损、P−1 全层可读图的对照。KeepSparse 仅保留 LLaVA L14/L19 或 Qwen L26，不是只保留这些完整 Transformer 层。

### LLaVA

原生 Full（中继正常）Accuracy：88.750%。

|Relay OFF 条件|Accuracy|固定原生Full Token Loss nat|首Token翻转|EOS|截断|
|---|---|---|---|---|---|
|FullImageAccess|75.417%|0.277286|48|240|0|
|AllImageOFF|64.167%|0.409606|81|240|0|
|KeepSparse|65.833%|0.359220|73|240|0|

|探索性配对比较|ΔAccuracy pp|95%CI pp|参照对→错|参照错→对|
|---|---|---|---|---|
|KeepSparse - AllImageOFF|+1.667|[-1.245, 4.583]|4|8|
|KeepSparse - FullImageAccess|-9.583|[-13.636, -5.809]|24|1|
|FullImageAccess - AllImageOFF|+11.250|[6.695, 15.900]|4|31|
|FullImageAccess - NativeFull|-13.333|[-18.828, -7.949]|40|8|
|AllImageOFF - NativeFull|-24.583|[-31.224, -17.915]|70|11|
|KeepSparse - NativeFull|-22.917|[-29.289, -16.667]|64|9|

### Qwen

原生 Full（中继正常）Accuracy：90.417%。

|Relay OFF 条件|Accuracy|固定原生Full Token Loss nat|首Token翻转|EOS|截断|
|---|---|---|---|---|---|
|FullImageAccess|63.750%|1.347999|83|240|0|
|AllImageOFF|56.250%|1.919066|105|240|0|
|KeepSparse|58.750%|1.707612|97|240|0|

|探索性配对比较|ΔAccuracy pp|95%CI pp|参照对→错|参照错→对|
|---|---|---|---|---|
|KeepSparse - AllImageOFF|+2.500|[0.417, 4.979]|1|7|
|KeepSparse - FullImageAccess|-5.000|[-9.959, 0.000]|24|12|
|FullImageAccess - AllImageOFF|+7.500|[2.881, 12.552]|9|27|
|FullImageAccess - NativeFull|-26.667|[-33.333, -20.332]|72|8|
|AllImageOFF - NativeFull|-34.167|[-40.984, -27.426]|91|9|
|KeepSparse - NativeFull|-31.667|[-38.235, -25.105]|84|8|

LLaVA 固定 L14/L19 未能补偿到完整直接视觉访问；相对全关的小幅增益区间跨0，不能宣称有效恢复。Qwen L26 相对全关有小幅描述性恢复，但仍低于完整直接视觉访问的点估计，远未恢复到原生 Full；探索性未校正区间不是确认性证明，也不能把区间接触0认定等价。
**单层 OFF 的敏感峰不能证明该层 Keep-only 足够。** 这些预先固定热点在中继受损背景下并未充分恢复完整答案表现。图：`figures/sparse_visual_rescue.pdf`。

## 研究结论与边界

1. 不同架构和任务的视觉来源敏感峰不同；本轮支持“非均匀、模型/任务依赖”，但 InternVL 未通过验证，不能完成两新模型的普适性论证。
2. 中继关闭明显损害最终答案，表明指定视觉信息路径干预有实际预测影响；Hard Mask 同时导致概率重分配，这不等于某种视觉语义的独立贡献。
3. 本轮没有支持“Dead Attention Can Wake Up”的关键正交互；固定 LogP 的反向交互削弱这一具体论文叙事。不能为了故事改层、改样本或追加组合。
4. 稀疏热点仅有有限/不确定的恢复，本轮未证实足以补偿受损中继；敏感峰与充分性必须分开。Decode 仍可看图，完整答案结果可能含后续补偿，不能当作纯首 Token 因果分解。
5. 首 Token Loss/Flip 不是最终 Accuracy；所有来源 Mask 都重新归一化其余合法 Key。观察性曲线、局部因果干预与最终生成评分分别报告，不宣称 Attention 冗余、整层可跳过或实际加速。

## 统计与审计

- 图片 SHA256 整簇配对 Bootstrap，10,000次、seed=20261102、题目加权；同图多题一起重采样。层、Head、重叠条件不视为独立图片。曲线区间为逐点探索性95%CI，未用来宣称96/36项扫描的确认性显著性。
- 240题是复用的历史探索队列，部分候选层来自先前相关研究；它不是新的独立确认集。没有根据本轮3,000题新峰位调整固定LLaVA L14/L19或Qwen L26候选。
- Task2 的固定 LogP 交互为事前单一主比较；Accuracy 及其他生成比较辅助/探索性。Task3 的多项比较全部探索性，未进行事后非劣效或等价认定。
- 成对正确性完全相同或没有观察到翻转时，经验Bootstrap区间可能退化为[0,0]；它不能覆盖未观察到的稀有损害，也不表示总体损害率为0。本轮没有据此认定无损或等价。
- CPU Hook/Mask/Bootstrap 检查、真实 GPU No-op/目标概率为0/非目标 Query/原生 DeepStack/Cache隔离/Hook清理见验证记录；正式采集没有 NaN、Inf、OOM、空答案或截断。
- InternVL 首次 Smoke 失败严格保留并停止，不计为正式样本。Qwen 两处启动工程问题（权重加载模式、Smoke 输入设备）在数值实验前修复，原始日志与协议修订均保留，没有放宽数值标准；详见 attempts_audit.json。
- JSON 条件数组字段口径：顶层 `protocol.json/task2/conditions` 为 `[relay_on, skip]`；`llava_relay/protocol.json/conditions` 为 `[relay_off, skip]`。执行代码和本报告 A/B/C/D 表一致，另存 condition_definitions.json 供审阅；没有改动冻结协议或逐题结果。
- CSV、逐题原子检查点、源/模型/图片绑定、环境、执行日志、PNG/矢量PDF及哈希均在当前结果目录。资源最终快照和交付审计时间见 `delivery_audit.json`。

## 下一步建议

不基于本轮增加层组合或样本。若继续 InternVL，应先单独解决原生首 Logits 与官方 Chat 的精确一致性（当前怀疑 LM-head 投影形状不同，但未经本轮复核，不能当作已证实原因），重新通过 Smoke 后再授权采集。对论文故事应优先如实呈现负交互与热点补偿受限，而不是继续寻找正面条件。
