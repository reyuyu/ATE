# Research 2 Experiment 2.1：单层来源 Attention 必要性

状态：完整完成240题（六类各40）、96条件，共23,040次干预 Prefill。另执行240次同机 Full 配对前向；没有完整答案生成、Decode或训练。

## 主要结果

| 来源 | 最大平均 Loss 层 | 平均 Loss（nats） | 探索性95% CI | 首Token翻转 | L14–L16平均 Loss |
| --- | --- | --- | --- | --- | --- |
| Image OFF | L14 | 0.017971 | [0.012367, 0.024124] | 2/240 | 0.005619 |
| Prompt OFF | L1 | 0.063009 | [0.039158, 0.091677] | 10/240 | 0.013619 |
| BOS OFF | L29 | 0.011788 | [0.004608, 0.019067] | 4/240 | 0.005743 |

## 对研究问题的回答

1. 按本轮直接来源 Hard Mask 的平均首Token Loss，L14–L16最大的是 **Prompt** 路径：Image=0.005619, Prompt=0.013619, BOS=0.005743 nats。来源不是互斥的因果贡献分解；Softmax重分配与非线性使三个损害不能相加。
2. Image最大损害三层：L14 (0.01797)，L19 (0.00382)，L20 (0.00189)。
2. Prompt最大损害三层：L1 (0.06301)，L15 (0.02092)，L13 (0.01247)。
2. BOS最大损害三层：L29 (0.01179)，L15 (0.01163)，L14 (0.00923)。

### 分任务峰值

| Task | Image OFF peak | Prompt OFF peak | BOS OFF peak |
| --- | --- | --- | --- |
| ChooseAttr | L14 / 0.01879 | L1 / 0.08011 | L28 / 0.02594 |
| ChooseCat | L14 / 0.00421 | L14 / 0.04191 | L17 / 0.01676 |
| ChooseRel | L14 / 0.03265 | L1 / 0.18570 | L15 / 0.02821 |
| CompareAttr | L19 / 0.01766 | L1 / 0.06709 | L17 / 0.03308 |
| LogicalObj | L14 / 0.00100 | L13 / 0.01227 | L29 / 0.01196 |
| QueryAttr | L14 / 0.05263 | L13 / 0.02262 | L11 / 0.02613 |

### BOS主导层与来源必要性

- L2–L5 Image平均原生来源Mass=0.0893，平均Loss=0.000153，翻转总数=0/960个题层观测（不是960张独立图片）。
- L2–L5 Prompt平均原生来源Mass=0.0886，平均Loss=-0.000041，翻转总数=0/960个题层观测（不是960张独立图片）。
- L2–L5 BOS平均原生来源Mass=0.8041，平均Loss=0.002718，翻转总数=4/960个题层观测（不是960张独立图片）。

### Mass与Loss的描述性关系

- Image：32层均值 Pearson=0.0581，Spearman=0.2269。这混有层深结构，不是控制层号后的机制证据。固定单层内、严格对齐题目的相关性见CSV。
- Prompt：32层均值 Pearson=0.7492，Spearman=0.5345。这混有层深结构，不是控制层号后的机制证据。固定单层内、严格对齐题目的相关性见CSV。
- BOS：32层均值 Pearson=0.0531，Spearman=0.0279。这混有层深结构，不是控制层号后的机制证据。固定单层内、严格对齐题目的相关性见CSV。

## 协议与数值审计

- 模型revision：`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；FP16/eager，Transformers4.49.0；仅物理GPU2/3。
- 固定seed20261031，哈希每类抽40；共235个图片SHA256簇；整簇分片120/120，题目权重保留。
- Prompt OFF屏蔽所有真实文本Key，但排除BOS、Image和P−1自身。旧Prompt Mass包含自身，本轮单独保存自身Mass及真正被阻断的Prompt Mass，不能混用。
- 直接复用既有VisualAttentionKnockout/knockout_mask；只替换目标Query/Key集合。所有Mask在Softmax之前生效，-inf使目标概率严格为0，其余合法Key重新归一化。
- 两张GPU各2题Smoke：No-op全部有效Logits精确一致；三来源覆盖L0/L15/L31；其他Query的逐层状态、Attention和最终Logits逐位一致；正常及异常Hook清理通过。
- 历史Full向量精确一致：240/240；Top1一致：240/240；最大Logit差=0；原生Mass最大差=7.4505806e-08。无论历史是否一致，指标始终使用本轮同卡Full，不放宽容差。
- 最大原生Head行和误差=0.00037288666，沿用历史2e-3界限；无NaN、Inf、OOM；输入与参数版本不变；各题checkpoint哈希及输入绑定完整。
- 两卡正式执行用时：2589.2s、2581.8s；代码冻结commit：`522448b313c7987682ccae412778c3fc5d68b194`。
- 20,000次按图片SHA256整簇配对Bootstrap，普通逐点95%CI。240题仅为探索，96条件及分任务不作未校正确认性显著性宣称。未做独立确认。

## 解释边界与下一步

- Loss固定Full原预测Token，允许负值；不是Gold答案损失。Top1翻转不是完整答案Accuracy。所有数值均来自实际前向。
- 阻断Image直接读取后，视觉信息仍可经残差、Prompt表示和其他层间接传递；低敏感不能证明无需视觉。BOS或Prompt同理。
- 高Mass不等于高功能必要性；接近零或无Flip不能证明路径无功能。来源屏蔽的Softmax重分配也可能产生有利或不利效果。
- 下一步建议仅作为研究决策：预注册本轮中层来源差异，在独立样本上确认后，再考虑完整答案指标。本轮不追加实验。

## 文件

- `figure1_source_necessity.pdf/png`：六任务三曲线，同一纵轴尺度。
- `overall_source_necessity_and_flip.pdf/png`：240题探索汇总与翻转。
- `source_sensitivity_heatmap.pdf/png`：32×3真实均值，统一对称色标。
- `top1_flip_tasks.pdf/png`：六任务翻转率。
- `per_question.csv`、`layer_source_summary.csv`、`task_source_summary.csv`、`mass_loss_correlations.csv`。
- `protocol.json`、`sample_manifest.json`、`integrity.json`、两卡Smoke及逐题checkpoint、当前Full向量。

## 中层来源与任务一致性的直接对照

| 范围 | Image OFF平均Loss | Prompt OFF平均Loss | BOS OFF平均Loss | 最大平均Loss来源 |
| --- | --- | --- | --- | --- |
| Overall L14-L16 | 0.005619 | 0.013619 | 0.005743 | Prompt |
| ChooseAttr L14-L16 | 0.003863 | 0.028581 | 0.002094 | Prompt |
| ChooseCat L14-L16 | 0.000745 | 0.024802 | 0.002731 | Prompt |
| ChooseRel L14-L16 | 0.011393 | 0.013899 | 0.011062 | Prompt |
| CompareAttr L14-L16 | -0.000707 | 0.010998 | 0.002012 | Prompt |
| LogicalObj L14-L16 | 0.000254 | 0.000719 | 0.004613 | BOS |
| QueryAttr L14-L16 | 0.018167 | 0.002715 | 0.011948 | Image |

六类任务的中层最大平均损害来源计数：{'Prompt': 4, 'BOS': 1, 'Image': 1}。中层主导来源并不完全一致，不能将总体均值当作所有任务的共同机制。

### L14、L15、L16逐层排序

- L14：Image (0.017971) > Prompt (0.011698) > BOS (0.009230)。
- L15：Prompt (0.020919) > BOS (0.011627) > Image (-0.000966)。
- L16：Prompt (0.008240) > Image (-0.000148) > BOS (-0.003628)。

### BOS主导与BOS路径干预不是同一指标

L2-L5的BOS OFF平均Loss为0.002718，其余28层平均为0.003181 nats；前者更低。这是层均值的描述性对照，不是图片级独立确认或等价检验。
若某层Image OFF损害小，只能说这条直接P-1到Image边在该条件下影响较小；不能排除已有残差流、其他文本表示及前后层传递的视觉信息。
图中的灰、绿、浅红背景分别标记L0-L1、L2-L5、L13-L16；曲线没有平滑，热图使用统一对称色标。
