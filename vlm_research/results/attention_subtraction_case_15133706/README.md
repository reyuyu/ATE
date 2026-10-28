# 15133706：L22 − L2 注意力相减预览（仅离线）

只使用原生Full的既有Phase3.4逐Head缓存，观察Answer-prediction P−1→576视觉Patch；不运行模型、不占GPU，没有干预后答案分数。图像使用既有真实Processor 336×336裁剪和冻结ROI/Sink映射。层编号沿用零起点：L2为第三层，L22为第二十三层。

- [四列对比图](l22_minus_l2.png)：L2、L22、有符号差分、截断负值并归一化的残差；上排真实图像叠加，下排精确24×24热图。
- [逐Sink候选对比图](sink_patch_comparison.png)：对比L22和残差的每个冻结候选Patch份额，便于看出哪些热点降低、哪些反而升高。
- [整体统计](distribution_summary.csv)、[逐Head统计](per_head_statistics.csv)、[全部576Patch统计](per_patch_statistics.csv)、[实际数组](distributions.npz)、[来源与核验](validation.json)。

## 固定计算

每个Head分别对视觉行条件化，固定β=1，无强度搜索：

```text
p2_h  = FullAttention[L2, h, P-1, visual] / 该行视觉质量
p22_h = FullAttention[L22,h, P-1, visual] / 该行视觉质量
d_h   = p22_h - p2_h
r_h   = max(d_h, 0)
q_h   = r_h / sum(r_h)
```

最后平均全部32Head绘图，不挑Head。ROI/Sink标签仅用来标注和统计，不决定哪些Patch被减掉。32Head残差总和均大于0（范围0.20689～0.89702），未用回退策略。原始视觉Mass不同，L2均值0.17666、L22均值0.06667，故不直接相减原始Mass。NPZ另外保存假设乘回L22各Head原视觉Mass的数组，通过质量恢复检查；这仍是离线构造，未实际进入Attention×V或预测。

L2、L22、残差共用相同的对数概率色标和透明度；差分单独使用以0为中心的对称线性色标，红色增加、蓝色减少，不是概率。黑色表示零或低于显示下限，不等于所有黑块都严格为零。两物体边框为青/品红；绿色13个非ROI Sink候选；橙色为ROI/Sink交集#364。

## 实际观察

| 视觉条件分布 | ROI份额 | 非ROI Sink份额 | 交集#364 | 平均图归一化熵 |
| --- | --- | --- | --- | --- |
| Full L2 | 9.18% | 76.41% | 5.86% | 0.48385 |
| Full L22 | 15.47% | 58.56% | 2.47% | 0.65319 |
| 相减、截断、逐Head归一化后 | 19.18% | 39.91% | 0.097% | 0.74837 |

相减确实降低了部分共有热点，#364明显变暗；但**没有清除全部Sink**。13个非ROI候选中12个仍有正的Head均值，仅#525严格为零；5个候选的相对份额反而升高，例如#214从5.86%升至11.87%。所以不能将结果称为干净的“纯物体注意力”。

表中熵是**先平均Head的图再计算的熵**，不是逐Head熵的均值。逐Head熵均值仅从L22的0.58014变为0.59377；平均图更分散并不意味着每个Head都明显分散。

聚合顺序有影响：若先平均Head、再相减/截断/归一化，ROI为27.37%、非ROI Sink为27.93%，见validation.json的敏感性记录；它是另一种运算，不与主图混为一谈，也不按结果选择更好看的口径。跨层同编号Head的功能对应没有得到验证。

这些只是同一Full缓存上的代数变换。没有新Forward，不能判断答案Margin、准确率或模型能力是否提高，也不证明剩余热点是噪声。没有在此任务中启动Phase3.6或其他GPU实验。

## 复现与保存

```bash
cd /home/zsy/ATE/vlm_research
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python visualize_attention_subtraction.py
```

已有输出目录拒绝覆盖。检查包括来源Phase3.4缓存SHA、既有163文件哈希、非负/有限/逐Head归一化、差分行和为0、正残差质量等于总变差距离，以及假设恢复L22质量；另保存[输出文件指纹](artifacts_sha256.json)。本轮只提交独立`codex/l22-minus-l2-viz`分支，不修改或推送main。
