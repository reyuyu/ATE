# LLaVA层段Attention更新与Text来源消融

## 协议与边界
固定LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager/Transformers4.49.0；L0始终原生。Early=L1–L10，Middle=L11–L20，Late=L21–L31（后段11层，前/中段10层，不能混同干预层数）。
GQA：原3000题中固定输入哈希选每任务100，共600题，600个图片SHA，未按输出筛选；对P−1及所有真实Decode干预，最多32 Token。
COCO：原60图、原Prompt和官方参考Caption；整个Prefill含P−1原生，只对真实Decode干预，最多96 Token。两数据集干预时机不同，差异不能仅归因任务类型。
Attention OFF仅置零目标Query的o_proj更新，QKV/cache、MLP和Residual不变；Text OFF是Softmax前−inf文本Key Mask，保留原BOS、全部Image与当前Self，并动态屏蔽已生成历史。两者都不是Image Key Mask，不是关闭Block或物理剪枝，也不代表真实加速。

## GQA Accuracy
|条件|Accuracy %|ΔFull pp|95%CI pp|对→错|错→对|首Token翻转|EOS/截断|
|---|---:|---:|---|---:|---:|---:|---|
|Full|84.667|+0.000|[+0.000, +0.000]|0|0|0|600/0|
|Early Attention OFF|76.167|-8.500|[-11.500, -5.500]|70|19|85|600/0|
|Early Text OFF|74.500|-10.167|[-13.000, -7.333]|70|9|47|556/44|
|Middle Attention OFF|71.667|-13.000|[-16.500, -9.500]|100|22|125|600/0|
|Middle Text OFF|59.000|-25.667|[-30.167, -21.333]|187|33|239|600/0|
|Late Attention OFF|74.333|-10.333|[-13.167, -7.667]|71|9|39|547/53|
|Late Text OFF|78.167|-6.500|[-8.833, -4.167]|47|8|28|589/11|

六任务结果详见gqa_task_accuracy.csv；完整答案、Token ID、首Token翻转与LogP Loss、输出变化、长度及EOS见gqa_per_question.csv。

## COCO官方指标（原始尺度）
|条件|CIDEr|ΔFull|相对降幅%|SPICE|ΔFull|相对降幅%|EOS/截断|
|---|---:|---:|---:|---:|---:|---:|---|
|Full|1.382441|+0.000000|+0.00|0.258243|+0.000000|+0.00|60/0|
|Early Attention OFF|0.339299|-1.043142|+75.46|0.171553|-0.086691|+33.57|30/30|
|Early Text OFF|0.064921|-1.317521|+95.30|0.085460|-0.172783|+66.91|0/60|
|Middle Attention OFF|0.886467|-0.495974|+35.88|0.162902|-0.095341|+36.92|60/0|
|Middle Text OFF|1.518801|+0.136360|-9.86|0.231031|-0.027212|+10.54|60/0|
|Late Attention OFF|0.923220|-0.459221|+33.22|0.234329|-0.023914|+9.26|57/3|
|Late Text OFF|1.369561|-0.012880|+0.93|0.249580|-0.008663|+3.35|60/0|

COCO各条件相对Full的同图配对95%CI见coco_summary.csv。负的相对降幅表示点估计提升，不代表已证实提升。

## 同一区段：Text OFF − Attention OFF
|层段|指标|差值|95%CI|Text更高/Attention更高/相同|
|---|---|---:|---|---|
|Early|GQA Accuracy pp|-1.666667|[-4.666667, +1.333333]|38/48/514|
|Middle|GQA Accuracy pp|-12.666667|[-16.666667, -8.666667]|44/120/436|
|Late|GQA Accuracy pp|+3.833333|[+1.000000, +6.666667]|51/28/521|
|Early|COCO CIDEr|-0.274378|[-0.452506, -0.115463]|10/50/0|
|Early|COCO SPICE|-0.086093|[-0.109678, -0.063466]|9/47/4|
|Middle|COCO CIDEr|+0.632334|[+0.430517, +0.840823]|47/12/1|
|Middle|COCO SPICE|+0.068129|[+0.042185, +0.093846]|43/13/4|
|Late|COCO CIDEr|+0.446341|[+0.245556, +0.662937]|40/15/5|
|Late|COCO SPICE|+0.015251|[-0.012015, +0.043137]|31/22/7|

## 本轮数据读数（探索性）
GQA Attention OFF：点估计损失最大为Middle Attention OFF（-13.000 pp），最小为Early Attention OFF（-8.500 pp）；不能仅按排序作显著性或无损判断。
COCO Attention OFF：SPICE点估计损失最大为Middle Attention OFF（-0.095341），最小为Late Attention OFF（-0.023914）；CIDEr、SPICE和人工图文核对可能不一致，应分别阅读。
GQA Text OFF：点估计损失最大为Middle Text OFF（-25.667 pp），最小为Late Text OFF（-6.500 pp）；不能仅按排序作显著性或无损判断。
COCO Text OFF：SPICE点估计损失最大为Early Text OFF（-0.172783），最小为Late Text OFF（-0.008663）；CIDEr、SPICE和人工图文核对可能不一致，应分别阅读。
Early GQA Accuracy pp：Text OFF比Attention OFF点估计-1.666667，普通配对95%CI [-4.666667, +1.333333]。
Middle GQA Accuracy pp：Text OFF比Attention OFF点估计-12.666667，普通配对95%CI [-16.666667, -8.666667]。
Late GQA Accuracy pp：Text OFF比Attention OFF点估计+3.833333，普通配对95%CI [+1.000000, +6.666667]。
Early COCO CIDEr：Text OFF比Attention OFF点估计-0.274378，普通配对95%CI [-0.452506, -0.115463]。
Early COCO SPICE：Text OFF比Attention OFF点估计-0.086093，普通配对95%CI [-0.109678, -0.063466]。
Middle COCO CIDEr：Text OFF比Attention OFF点估计+0.632334，普通配对95%CI [+0.430517, +0.840823]。
Middle COCO SPICE：Text OFF比Attention OFF点估计+0.068129，普通配对95%CI [+0.042185, +0.093846]。
Late COCO CIDEr：Text OFF比Attention OFF点估计+0.446341，普通配对95%CI [+0.245556, +0.662937]。
Late COCO SPICE：Text OFF比Attention OFF点估计+0.015251，普通配对95%CI [-0.012015, +0.043137]。

## 统计与解释限制

### 关键结论与终止诊断

GQA 的中层最敏感：Middle Text OFF 为59.00%，比Full低25.67 pp，比Middle Attention OFF低12.67 pp（普通配对95%CI [-16.67,-8.67] pp）。中层Text Key屏蔽的损失大于整个Attention更新置零，不能将二者当成可加的来源贡献。

COCO 的早层Text OFF最严重：CIDEr下降95.30%、SPICE下降66.91%，且60/60无EOS、全部截断。Middle Text OFF的CIDEr上升9.86%，但SPICE下降10.54%；这是指标分歧，不是整体质量提升的证据。Late Text OFF的点估计变化相对较小（CIDEr -0.93%、SPICE -3.35%），普通CI跨0不表示无损。

GQA的终止异常不可忽略：Early Text OFF截断44题，Late Attention OFF截断53题，Late Text OFF截断11题；Early Attention OFF有1条空答案。Late Attention OFF损害71题，其中45题首Token不变，34题截断（32题同时首Token不变且截断）；Early Text OFF损害70题，其中37题首Token不变，36题截断。因此完整答案损失不能全部归因首Token知识判断变化，也不删除这些失败输出后重算Accuracy。COCO Early Attention OFF截断30/60、Late Attention OFF截断3/60，其余非Early Text条件均正常EOS。

GQA从P−1开始干预，COCO则保留完整Prefill及首Token，仅干预后续Decode；同时二者答案长度、指标、样本量不同。跨任务层段效应只能作为各自协议下的观察，不能仅凭这些差异断言任务专门化、语言/视觉中枢或推理加速。六组图文核对与反例见[qualitative_notes_zh.md](qualitative_notes_zh.md)。

20,000次图片SHA整簇配对Bootstrap，保留题目加权估计；本次GQA和COCO各自均为一图一题。所有层段和分任务比较均探索性，普通95%CI未经多重比较校正；区间跨0不证明无功能损害或等价。
沿用历史官方COCO评分器和统计口径：CIDEr的document frequencies固定于原60图参考语料，重采样官方逐图贡献；SPICE重采样官方逐图F值。区间条件于原参考语料，不是新语料重估IDF的总体区间。
Text OFF的保留概率会重新分配；o_proj OFF则移除整个Attention更新。这两种干预损失不可相加分解成独立来源贡献，不足以发现语言中枢或视觉中枢。
repeated_token_trigrams是Token三元组复读的语法性筛查，不是新的质量或幻觉指标；与EOS、截断、空答案分别记录。
每图词序差异见coco_per_image.csv；coco_object_action_terms.csv及change_counts.csv使用现有SPICE依赖Stanford POS/lemma抽取NN*/VB*名词/动词变化（含助动词）。这些是描述词差异，不是视觉事实标注、对象召回率或幻觉评分；不将名词/动词变化自动判为错误。六案例按预声明规则事后选取，人工对象/动作核对另见qualitative_notes_zh.md，不作为新增质量指标。

## 验证与运行
6项CPU测试及三卡真实Smoke通过：Full/no-op有效生成Logits精确一致，官方原生生成Token一致；非目标Prefill的Attention、Hidden及Logits精确不变。COCO整个Prefill及首Token保持精确一致。
目标o_proj输出严格0；Text目标概率严格0；非目标Key Mask未变、Image实际Mass保持正值，合法Key存在；原始P−1在Decode时正确成为文本历史。检查每层真实Cache位置/长度、独立Cache、Hook清理、输入哈希和参数版本不变；GPU前向无NaN/Inf/OOM。
官方SPICE首次CPU评分在长重复Caption的Stanford解析中耗尽Java堆并失败，原始日志保存在analysis_attempt1.log。仅通过_JAVA_OPTIONS提高堆上限至32 GiB、限制可见CPU核数为2后重试；没有修改Caption、官方评分器或指标定义，没有重跑GPU生成。个别长Caption的缓存键过长警告不等于评分失败，所有最终逐图分数均检查有限且完整。
三卡正式运行分钟：23.02, 19.06, 20.19。GPU1未使用。
执行代码Commit：131f6bf82aa3bb3f25967f10ad1dbc3567fc4fa2。全部4620条为本轮真实配对生成，没有复用历史分数。
