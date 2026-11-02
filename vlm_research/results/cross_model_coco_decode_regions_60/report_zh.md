# COCO Decode视觉重访：Qwen与InternVL跨架构探索

严格复用历史60张COCO和每图5条参考，正式执行960次新增完整生成（必要Smoke另计，不参与评分）；整个Prefill和首Token原生，仅真实增量Decode施加Image来源Hard Mask。
两个模型均为BF16/Eager、独立原生KV Cache、Greedy、最多96 Token；只用物理GPU1顺序运行。

## 首页结论

**这批固定60图不支持“LLaVA的中层重访优势稳定跨架构成立”。** 结果依模型、指标与冻结层段而变，不能强行归纳为统一的中层最优。

1. **Qwen：中层相对早层有收益，但不能独立归因层位。** Middle−Early为CIDEr +0.0520、SPICE +0.0466，但保留层数13比12多一层。Middle−Late的CIDEr为+0.0179 [−0.0388,+0.0757]，SPICE为−0.0231 [−0.0479,+0.0024]，没有一致的中层优势。相对等层数Random12均值，两指标差值区间也均跨0；不能据此证明等价。
2. **InternVL：结果反对这里冻结的中层优先。** Middle−Late的SPICE为−0.0839 [−0.1121,−0.0549]，60图中13图改善、47图退化。Middle相对三个等层数Random9的均值，CIDEr为−0.1707 [−0.2820,−0.0600]，SPICE为−0.0530 [−0.0729,−0.0328]。这些仍为探索性普通区间，不作为多重校正的确认结论。
3. **晚层更好保留语义评分的点估计，但未证明无损。** Qwen Late的SPICE为0.1956，Full为0.1979；InternVL Late为0.2410，Full为0.2291。相应差值区间均跨0，不能称等价、安全或已完成非劣效验证。
4. **CIDEr与SPICE存在重要分歧。** InternVL Decode None的CIDEr点估计高于Full，却使SPICE下降0.0706 [−0.0970,−0.0438]。因此不能只用CIDEr上升宣布Caption质量提高，也不能据此说Decode直接读图无用。
5. **所有预定条件与负面结果均保留。** 960次正式生成全部正常EOS，截断0、空答案0；两个模型的480次生成各自完成，未增加窗口、调Prompt或重选样本。首Token在所有条件严格相同；随后原始生成序列相对Full发生变化的题数见CSV，不能将平均分接近写成逐Token无损。

上述结论只涉及原生Prefill之后、当前Decode Query对原始Image Key的直接读取。不能推广到全视觉处理、物理KV删除、训练或真实GPU加速。本轮结束，不自行追加实验。

## 官方完整Caption评分

以下CIDEr、SPICE均为官方实现的**原始标度**，没有乘100；ΔFull及区间使用相同标度。

| 模型 | 条件 | 保留层数 | CIDEr | ΔFull [95%CI] | SPICE | ΔFull [95%CI] |
|---|---|---:|---:|---|---:|---|
| qwen | Full | 36 | 0.1280 | +0.0000 [+0.0000,+0.0000] | 0.1979 | +0.0000 [+0.0000,+0.0000] |
| qwen | Decode None | 0 | 0.0333 | -0.0947 [-0.1454,-0.0487] | 0.1044 | -0.0936 [-0.1189,-0.0683] |
| qwen | Early | 12 | 0.0619 | -0.0661 [-0.1235,-0.0103] | 0.1258 | -0.0721 [-0.0982,-0.0458] |
| qwen | Middle | 13 | 0.1139 | -0.0141 [-0.0630,+0.0403] | 0.1725 | -0.0255 [-0.0467,-0.0034] |
| qwen | Late | 13 | 0.0960 | -0.0320 [-0.0893,+0.0268] | 0.1956 | -0.0023 [-0.0246,+0.0198] |
| qwen | Random11-20261102 | 12 | 0.1194 | -0.0085 [-0.0648,+0.0529] | 0.1639 | -0.0340 [-0.0557,-0.0128] |
| qwen | Random12-20261103 | 13 | 0.0912 | -0.0368 [-0.0841,+0.0080] | 0.1603 | -0.0376 [-0.0604,-0.0153] |
| qwen | Random12-20261104 | 13 | 0.2223 | +0.0943 [+0.0369,+0.1637] | 0.1919 | -0.0060 [-0.0217,+0.0098] |
| internvl | Full | 28 | 0.4823 | +0.0000 [+0.0000,+0.0000] | 0.2291 | +0.0000 [+0.0000,+0.0000] |
| internvl | Decode None | 0 | 0.5853 | +0.1029 [-0.0335,+0.2527] | 0.1585 | -0.0706 [-0.0970,-0.0438] |
| internvl | Early | 10 | 0.6683 | +0.1860 [+0.0306,+0.3611] | 0.1697 | -0.0594 [-0.0864,-0.0326] |
| internvl | Middle | 10 | 0.5535 | +0.0711 [-0.0650,+0.2153] | 0.1571 | -0.0719 [-0.1000,-0.0432] |
| internvl | Late | 10 | 0.6671 | +0.1848 [+0.0368,+0.3560] | 0.2410 | +0.0119 [-0.0091,+0.0341] |
| internvl | Random9-20261102 | 10 | 0.6562 | +0.1738 [+0.0497,+0.3081] | 0.2003 | -0.0288 [-0.0494,-0.0081] |
| internvl | Random9-20261103 | 10 | 0.8072 | +0.3249 [+0.1676,+0.4980] | 0.2120 | -0.0171 [-0.0405,+0.0066] |
| internvl | Random9-20261104 | 10 | 0.7090 | +0.2267 [+0.0969,+0.3733] | 0.2180 | -0.0111 [-0.0331,+0.0111] |

## 预声明重点比较

| 模型 | 比较 | 指标 | 配对差值 [95%CI] | 改善/退化/平局图片 |
|---|---|---|---|---|
| qwen | Middle − Early | CIDEr | +0.0520 [+0.0001,+0.1078] | 39/21/0 |
| qwen | Middle − Early | SPICE | +0.0466 [+0.0199,+0.0735] | 37/20/3 |
| qwen | Middle − Late | CIDEr | +0.0179 [-0.0388,+0.0757] | 37/23/0 |
| qwen | Middle − Late | SPICE | -0.0231 [-0.0479,+0.0024] | 25/35/0 |
| qwen | Middle − Random12-20261103 | CIDEr | +0.0227 [-0.0208,+0.0733] | 38/22/0 |
| qwen | Middle − Random12-20261103 | SPICE | +0.0122 [-0.0101,+0.0351] | 35/25/0 |
| qwen | Middle − Random12-20261104 | CIDEr | -0.1084 [-0.1964,-0.0298] | 23/37/0 |
| qwen | Middle − Random12-20261104 | SPICE | -0.0195 [-0.0435,+0.0051] | 26/32/2 |
| qwen | Middle − Matched Random mean | CIDEr | -0.0429 [-0.0966,+0.0131] | 22/38/0 |
| qwen | Middle − Matched Random mean | SPICE | -0.0036 [-0.0235,+0.0167] | 25/35/0 |
| internvl | Middle − Early | CIDEr | -0.1149 [-0.2543,+0.0169] | 23/33/4 |
| internvl | Middle − Early | SPICE | -0.0125 [-0.0309,+0.0063] | 21/33/6 |
| internvl | Middle − Late | CIDEr | -0.1136 [-0.2626,+0.0353] | 27/33/0 |
| internvl | Middle − Late | SPICE | -0.0839 [-0.1121,-0.0549] | 13/47/0 |
| internvl | Middle − Random9-20261102 | CIDEr | -0.1027 [-0.2206,+0.0165] | 23/35/2 |
| internvl | Middle − Random9-20261102 | SPICE | -0.0432 [-0.0669,-0.0190] | 16/38/6 |
| internvl | Middle − Random9-20261103 | CIDEr | -0.2537 [-0.3950,-0.1117] | 19/40/1 |
| internvl | Middle − Random9-20261103 | SPICE | -0.0548 [-0.0776,-0.0317] | 12/43/5 |
| internvl | Middle − Random9-20261104 | CIDEr | -0.1556 [-0.2881,-0.0234] | 15/45/0 |
| internvl | Middle − Random9-20261104 | SPICE | -0.0609 [-0.0842,-0.0381] | 16/41/3 |
| internvl | Middle − Matched Random mean | CIDEr | -0.1707 [-0.2820,-0.0600] | 17/43/0 |
| internvl | Middle − Matched Random mean | SPICE | -0.0530 [-0.0729,-0.0328] | 16/44/0 |

## 解释边界

**原生Caption风格与指标限制。** 官方PTB分词后，300条参考平均10.57词；LLaVA Full平均12.37词，Qwen Full为24.92词，InternVL Full为19.73词。固定CIDEr实现包含sigma=6的原生高斯长度惩罚，因此Qwen较低的绝对CIDEr不应直接当作跨模型视觉能力排名。长度是解释限制，不意味着所有分数差异都由长度造成；没有修改或重算一个“去惩罚”替代指标。数据见native_caption_length_review.json。

**原生视觉预算也不同。** 本批Qwen真实Image Token数为80–400，Prompt总长度97–417；InternVL固定256个Image Token、总长度311，max_num=1；LLaVA历史为576个视觉Token。均沿用各自冻结Processor，没有为结果调整图像分辨率。跨架构比较不能排除预处理、Prompt Template、视觉预算及生成风格等差异。

Qwen Early保留12层（L0及L1–L11），Middle和Late保留13层。Middle−Early同时包含层数与层位差异，不能单独归因为中层位置。Middle只与两个Random12集合做等层数对照；Random11仅作为冻结参考。
InternVL的Early/Middle/Late和三个Random9均保留10层，包括L0。三个随机种子不是独立图片；随机均值先在同图内求均值，再做图片配对Bootstrap。
每项区间按60个Image SHA256整簇配对抽样20,000次。这里均为普通探索性95%区间，未作为多重校正确认检验；CIDEr沿用历史官方实现，Bootstrap条件于固定60图参考语料的IDF，不逐次重新拟合IDF。
LLaVA历史32层的Early/Middle/Late分别保留11/11/12层，与36层Qwen、28层InternVL并非相同绝对层段。三模型图相对各自Full，只支持各自协议下的探索性结构对照。
原始图片、题号、五条参考及SHA256完全保留，未根据模型输出筛题。六组原图案例按冻结规则作事后描述性正反例展示，未用于改变任何总体分数或实验条件。
来源Mask只阻断当前Decode Query到原始Image Key的直接读取。其他Prefill Query与P−1均已正常处理图像，文本与历史Value可间接携带视觉信息。不能称为模型不需要视觉，也不是物理删除KV、整层跳过或实际加速。
本轮沿用用于形成研究假说的同60图，属于探索性跨架构复现，不是独立确认；分数接近或区间包含0均不能证明无损。

## 验证与资源

CPU测试通过后冻结源文件、随机层、输入布局、模型文件和官方评分器哈希。每模型真实Smoke确认官方原生与Full/No-op生成Token及逐步有效Logits精确一致，所有条件整个Prefill逐层Hidden与首Token精确不变。
每层每步审计原始Image Key、被阻断视觉概率严格为0、保留层视觉概率大于0、BF16行和原容差、原生Cache长度和独立性。Qwen DeepStack特征哈希与Full一致；InternVL官方chat输入嵌入与原生生成精确一致，max_num=1且logits_to_keep=1。
所有逐图Caption、Token ID、首次分叉、EOS/截断/空答案、官方逐图CIDEr/SPICE与检查点均保存。误差门槛沿用历史实现，无放宽。
启动前有两次工程预检终止：PyTorch UUID对象不带GPU-前缀，以及外层inference_mode加载的参数无法读取版本计数器。二者都发生在首次模型前向之前；分别做UUID规范化精确比较、改用既有no_grad加载方式后重新执行CPU检查并冻结源码。旧协议及异常日志均保留，未产生或混用任何正式结果。没有修改Attention数值容差、层集合、样本或评分规则。两模型正式Worker使用相同代码Commit与协议SHA256。
- qwen: 正式GPU运行 21.42 分钟；含加载/Smoke 21.97 分钟；峰值allocated/reserved 17190/17548 MiB。
- internvl: 正式GPU运行 10.44 分钟；含加载/Smoke 10.86 分钟；峰值allocated/reserved 15289/15456 MiB。

图：figure_cross_model_decode_regions.pdf/png；六组含正反例原图：qualitative_six_images.pdf。
协议、逐图统计与全部审计随分支保存；模型权重不提交。GPU结束后释放记录见gpu_release.json。

完整Caption、Token ID、首次分叉、官方逐图评分与JSON检查点均随Git提交。960个首Token完整词表Logits张量为各检查点同名.pt文件，保留在本机本结果目录，遵循仓库大型张量忽略规则，不推送Git；JSON与artifact_sha256.json保存精确哈希。原服务器可直接安全续跑；迁移到其他机器时须一并恢复这些原始张量及已绑定模型/图片，而不能将Git克隆误当成完整原始数据包。

交付CPU复核脚本为vlm_research/cross_model_caption_delivery_checks.py，记录历史评分绑定、Bootstrap基本性质、官方PTB长度对照，并为每项比较明确写出双方保留层数。比较CSV的层数匹配标签经重新计算，包含相对Full时本应不等层数的比较；所有评分、配对差值、区间和预声明比较集合均保持原值。重新生成统计时依次执行cross_model_caption_regions.analyze和cross_model_caption_delivery_checks；后者不执行任何GPU前向。送审图已检查PNG及六页原图PDF，无文本裁切或重叠。
