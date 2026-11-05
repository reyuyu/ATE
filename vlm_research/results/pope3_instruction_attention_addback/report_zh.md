# 三道POPE：Instruction视觉读取与逐Token恢复

状态：COMPLETE。仅1351/2459/301；原始图像、Prompt、权重和C3-V2路径不变。GPU0来自用户明确授权，GPU1/2/3未使用。

## 首要发现：先看模型实际看到了什么

所有框均来自COCO官方instances_val2014，不按热图或模型输出绘制。模型执行短边缩放至336再中心裁剪；逐像素重建与真实Processor严格一致。24x24顺序为CLIP去CLS后的行优先Patch，未重排。
原图中存在物体，不等于模型输入仍包含物体。目标完全裁出时，ROI比例/富集记NA，不能把原图框强行投到模型可见Patch；也不能把正确Yes解释为从不可见目标提取了直接视觉证据。

**花瓶301特别说明：** 用户指定的右上方对应官方vase框1154230和1670746，两框均在裁剪外。但左下方还有官方vase框1154146，在模型输入内；以绿色虚线单列，不合并进指定目标ROI。因此不能说该模型输入“没有任何花瓶”，也不能将Yes恢复直接归因于右上方不可见目标。

| 题号 | 目标 | 实际输入ROI | 可见框面积比例 | C3 Yes/No Margin | E1 Margin |
|---|---|---|---:|---:|---:|
|1351|truck|OFFICIAL_VISIBLE|0.8112%|0.187500|-0.046875|
|2459|boat|OFFICIAL_FULLY_CROPPED_OUT|0.0000%|0.109375|-0.078125|
|301|vase|OFFICIAL_FULLY_CROPPED_OUT|0.0000%|0.078125|-0.140625|

## 恢复实验：全部预先固定窗口

每题28条件：E1、完整10-Token Instruction恢复(C3)、10个单Token、9个连续2-Token、7个连续4-Token；共84条完整Greedy答案。所有恢复在L0-L19执行，P−1/Decode禁图、L2-L5答案物理Attention跳算、L20视觉退出均不变。
主Margin为logP(" Yes")−logP(" No")，空格与大小写在GPU前固定，逐候选完整Teacher Forcing且不计EOS、不额外BOS。大小写/无前导空格的8种形式分别保存在answer_form_logp_audit.csv，不挑有利形式。

| 题号 | 单Token恢复Yes | 连续2个恢复Yes | 连续4个恢复Yes | 完整恢复 |
|---|---|---|---|---|
|1351|无|无|Answer _the _question _using; _single _word _or _phrase|Yes|
|2459|Answer|Answer _the|Answer _the _question _using|Yes|
|301|无|无|无|Yes|

卡车两组有效4-Token窗口的Margin提升分别为+0.125000、+0.062500 nat；船的`Answer`单独恢复为+0.093750 nat。它们是本轮固定窗口中的结果，并非所有子集的最小充分集合。花瓶只在所测完整10-Token正对照恢复，但未测试5-9个或非连续子集，不能断言必须10个。
各条件的完整生成均保留；这些Margin基于FP16网络输出与FP32 log_softmax，数值接近决策边界，不能当作稳定的总体效果。

## 注意力观察：层/Head热点不等于恢复成功

卡车可见框占实际输入面积0.8112%。在L0-L19、Head与区域内Query平均后，Question/Instruction/ASSISTANT的ROI条件比例分别为3.2133% / 0.4284% / 6.1918%，面积富集为3.96 / 0.53 / 7.63。Instruction整体并非最集中读取卡车的区域。
但Instruction的`the`在L14/H29的ROI条件比例达到15.1714%，富集18.70；句号在L14/H12为10.4693%，富集12.91。这是从全部层/Head中按绝对ROI Mass选择的描述性热点，不是显著性检验。单独恢复这两个Token均仍回答No，因此高ROI关注不足以证明单Token恢复有效。
Instruction的平均外侧两圈比例：卡车25.51%、船42.19%、花瓶38.09%；均匀分布对应176/576=30.56%。船/花瓶的边缘注意力指向的是裁剪后图像边缘，不是原图中已被裁掉的指定目标。其目标ROI指标为NA。

## Instruction逐Token配对表

| 题号 | 原生Subword | C3关注层/Head | 平均边缘比例 | 平均ROI比例 | 恢复后答案 | ΔYes/No Margin nat |
|---|---|---|---:|---:|---|---:|
|1351|`Answer`|L0/H14|26.442%|0.385%|No|-0.031250|
|1351|`▁the`|L14/H29|25.875%|0.419%|No|+0.015625|
|1351|`▁question`|L0/H23|25.405%|0.398%|No|+0.000000|
|1351|`▁using`|L0/H23|24.660%|0.394%|No|-0.015625|
|1351|`▁a`|L0/H23|25.560%|0.466%|No|+0.000000|
|1351|`▁single`|L0/H23|24.333%|0.382%|No|+0.000000|
|1351|`▁word`|L0/H30|26.013%|0.451%|No|+0.015625|
|1351|`▁or`|L0/H31|24.779%|0.424%|No|+0.015625|
|1351|`▁phrase`|L0/H23|24.296%|0.384%|No|+0.015625|
|1351|`.`|L14/H12|27.328%|0.573%|No|-0.015625|
|2459|`Answer`|L0/H29|43.669%|NA|Yes|+0.093750|
|2459|`▁the`|L0/H3|42.331%|NA|No|+0.062500|
|2459|`▁question`|L0/H29|42.282%|NA|No|-0.031250|
|2459|`▁using`|L0/H3|41.712%|NA|No|+0.000000|
|2459|`▁a`|L0/H3|42.239%|NA|No|+0.046875|
|2459|`▁single`|L0/H3|41.978%|NA|No|+0.031250|
|2459|`▁word`|L0/H3|42.651%|NA|No|+0.046875|
|2459|`▁or`|L0/H3|41.111%|NA|No|+0.031250|
|2459|`▁phrase`|L0/H3|41.507%|NA|No|+0.046875|
|2459|`.`|L0/H3|42.166%|NA|No|-0.015625|
|301|`Answer`|L0/H29|39.878%|NA|No|+0.000000|
|301|`▁the`|L0/H3|38.507%|NA|No|+0.000000|
|301|`▁question`|L0/H3|38.649%|NA|No|-0.015625|
|301|`▁using`|L0/H3|37.630%|NA|No|+0.015625|
|301|`▁a`|L0/H3|37.469%|NA|No|+0.031250|
|301|`▁single`|L0/H3|37.173%|NA|No|+0.031250|
|301|`▁word`|L0/H3|38.336%|NA|No|+0.015625|
|301|`▁or`|L0/H3|36.836%|NA|No|+0.000000|
|301|`▁phrase`|L0/H3|36.952%|NA|No|+0.015625|
|301|`.`|L0/H3|38.980%|NA|No|+0.015625|

## 解释边界与核心回答

- 观察性Attention：edge/ROI条件比例以实际Image mass为分母；E1 Instruction的视觉Mass严格为0，因此空间条件分布未定义，不把零热图解释为无用。
- ROI为官方BBox覆盖，非物体分割；重叠框按几何并集合并，按Patch覆盖面积分数加权；roi_hit_ratio另记录任何重叠Patch的比例。富集=ROI条件比例/可见框面积占比，不等同于准确定位。
- 重点层/Head预定为可见ROI绝对Attention Mass最大处；目标裁出时改列绝对边缘Mass最大处并标注ROI不可用。它是描述性选择，不是确认性显著结果。
- 恢复输出为Yes的Token/窗口是此算法背景下的功能敏感候选；只有Margin变化而不翻转也完整保留。不能凭Attention热点宣称因果有效。
- 对完全裁出的目标，本实验不支持“Instruction直接看到该边缘物体”的解释。恢复若改善Yes，仍可能涉及可见场景线索、文本/位置依赖、概率重新分配和残差路径，不能区分这些机制。
- 三题是事先挑出的历史失分案例，不是无偏样本；不估计总体Accuracy、不作独立外部验证、不追加Token组合。

**核心回答：** 卡车案例支持Instruction在少数中层Head接收到可见目标框附近的注意力，且部分4-Token窗口足以恢复Yes；但高关注单Token并不充分。船的`Answer`单Token能恢复Yes，却不能说明它重新读取了被裁掉的船。花瓶的局部窗口未恢复，完整Instruction可恢复，但指定右上目标不可见且存在另一个可见花瓶。三例不能共同证明“Instruction发现边缘目标”这一机制；它们支持的是背景依赖、可能多Token协同的直接视觉路径功能敏感性。

## 图与原始数据

每题figures/<qid>/case_overview.png/pdf提供原图框与真实裁剪、全部R列表、区域热图、逐Token和连续窗口恢复结果。token_galleries/覆盖每个R Token的C3/E1全部20层Head均值及L0-L19×32 Head完整图册。所有图基于真实576个位置，不插值生成新观测。
卡车另有instruction_roi_focus.png/pdf，放大`the`与句号的ROI热点，并同时显示单Token仍未恢复Yes；两者按各Token最大绝对ROI Head的富集度事后选作展示，不将其解释为预注册验证。
层图在同一Token的C3/E1各层间共用线性色标；全Head图册采用预先固定log2(1+576*p(patch|Image))范围0..6，灰格表示视觉Mass为0。图像叠加仅用于显示，CSV和NPZ保留原始概率。
每个Head/Token的绝对Mass、边缘比例、ROI比例/富集见per_layer_head_token_attention.csv；用户要求的合并表见token_attention_addback_summary.csv。精确Attention数组在/data资源目录，attention/*.json记录路径、SHA和维度，不将大数组重复进Git。

[卡车图册](figures/1351/README.md) | [船图册](figures/2459/README.md) | [花瓶图册](figures/301/README.md)

## 正确性与复现

CPU回归与3题GPU Smoke后执行；三题C3/E1完整Token和首Token全词表Logits SHA与e936ff3历史严格一致；观测Hook前后所有生成步Logits torch.equal；Native/No-op严格一致；对应Dense参考沿用冻结数值标准；禁止概率0，独立Cache、MLP、位置和Hook清理保留审计。
复现：`python -m vlm_research.pope_instruction_probe.run`（只允许协议GPU0且空闲）；`CUDA_VISIBLE_DEVICES= python -m vlm_research.pope_instruction_probe.analyze`重算CSV与图。完整模型/数据/源码哈希见freeze.json；官方COCO标注只按Range提取验证集成员，来源、ZIP CRC和内容SHA均保留。
本轮没有训练、没有改Prompt、没有额外Decode实验、没有性能加速宣称。GPU完成后释放，到此停止。
