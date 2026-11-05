# Qwen3-VL历史Keep-L0–L21：相同1000题复现与All-Writer质量对照

**历史Keep与最新D通过严格复核；All-Writer为677/1000（67.700%），Keep为585/1000（58.500%）。All-Writer−Keep为+9.200pp，图片整簇配对探索性95%CI [+7.256, +11.221]。**

## 历史Keep是否等价于最新D？

两者均保留L1–L6 Attention正常执行；P−1及所有真实Decode在全部36层禁止读取原始Image Key；Prefill R仅L0–L21可读图。直接调用未修改的历史`qwen3vl_keep_l0_21.engine.generate`，20道冻结Smoke的完整Token序列相同，全部46份有效生成Logits均严格`torch.equal`，逐向量Mean/Max Abs差异为0。最新D同时逐步复现原历史Smoke Logits SHA与输出Token。Native/历史No-op也全部逐位一致。

所有1000题的原始输入、图片、mRoPE与角色位置经CPU重建核对；36层Mask规则CPU逐项相等。依照预设复用门槛，Keep准确率与完整答案复用已完成D，不重新推理1000题Keep。严格逐位运行复核的直接范围是20题，其余980题依据输入、协议及控制规则等价审计复用，不宣称已逐位重测全1000题。

## 冻结协议与样本

基于`c2ab2892994a39149f7c3203241fa6f6e9ceb341`，执行源码冻结于`d27497db9a5091fb87830645bab6f7a2b908bfd6`。模型Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16/Eager，Transformers 4.57.6，Greedy、max_new_tokens=32、logits_to_keep=1，每请求独立完整真实KV。

原样复用最新1000题GQA val_balanced，918个图像SHA簇；Manifest SHA256为`bb09f894a04f111f430841f4d074ff0d770585956a04c5401436351e235d4a88`。R为最后一个Image Token之后至P−1之前的位置，包含原生vision_end标记。只做Softmax前Hard Mask；Image Query、非目标连接、MLP、Residual、Norm、mRoPE、DeepStack、QKV与完整图像KV不变。

沿用此1000题历史评分器：原始`tokenizer.decode(..., skip_special_tokens=True)`后执行`rstrip(".").lower()`，与Gold严格比对；不额外strip。历史生成包装器的strip后文本另字段保存，不用它改写本次评分。模型/原始图片保留在服务器资源目录，Git中保存其绑定和审计。

## 全部质量结果

|条件|R读图层|早层P−1/Decode Attention|正确/1000|ACC|相对Native Δpp|
|---|---|---|---:|---:|---:|
|Native|全部|正常；答案直接读图开放|694|69.400%|+0.000|
|F|L0–L21|L1–L6输出置零|588|58.800%|-10.600|
|Keep-L0-L21|L0–L21|正常|585|58.500%|-10.900|
|E|L0–L35|L1–L6输出置零|680|68.000%|-1.400|
|All-Writer|L0–L35|正常|677|67.700%|-1.700|

Native之外四条件的P−1/Decode均在全部36层直接禁图。F/E中的输出置零是继承的Dense功能对照，不属于本轮新增Keep/All-Writer；本轮没有物理删除、跳算或新消融条件。Native、F、E及D共4000份原Checkpoint经SHA、输入、评分、源码绑定审计后复用，新增All-Writer共1000份生成。

|候选−参考|Δpp|普通配对95%CI|参考对→错|参考错→对|答案变化|首Token翻转|
|---|---:|---|---:|---:|---:|---:|
|All-Writer−Keep-L0-L21|+9.200|[+7.256, +11.221]|10|102|207|203|
|All-Writer−F|+8.900|[+6.893, +10.967]|14|103|218|209|
|All-Writer−E|-0.300|[-1.104, +0.502]|10|7|37|28|
|Keep-L0-L21−F|-0.300|[-1.091, +0.489]|9|6|40|31|
|Keep-L0-L21−E|-9.500|[-11.531, -7.553]|105|10|217|210|
|All-Writer−Native|-1.700|[-3.297, -0.099]|41|24|125|113|

## 关闭深层R视觉写入损失多少？

在早层Attention正常、答案Query全层直接禁图的固定背景下，将R读图从L0–L35缩为L0–L21，Accuracy变化为-9.200pp（Keep−All-Writer），对应95%CI [-11.221, -7.256]。恢复深层R写入使102题由错变对，同时使10题由对变错。该比较定位的是此背景下深层视觉写入权限的总干预效果，不说明每个深层均必要，也不能等同于实际加速收益。

F→Keep仅取消早层输出置零；E→All-Writer也仅取消早层输出置零。两个对照有助于区分早层Attention恢复与深层R写入恢复，不应把不同背景下的效应直接相加解释为独立因果份额。

本轮All-Writer比E低0.300pp，Keep比F低0.300pp；这两项普通配对区间均跨0，不能声称早层Attention恢复带来质量收益。相反，关闭深层R视觉写入在本轮同题控制下造成明显损害。All-Writer仍比Native低1.700pp，不能把恢复全部R写入等同于恢复Native。

两种早层Attention状态下的晚层写入收益点估计在本批均为9.2pp，这是本批探索观察，不据此断言效应在其他数据上完全可加或没有交互。

## 为什么不能套用历史3000题分数？

历史Keep/All-Writer的87.5%/87.9%来自另一份六任务均衡3000题（2409张图），本轮是官方val_balanced冻结1000题，官方types.semantic分为5类。两组仅重合17个Question ID，本轮353题的图片出现在旧3000中。此前87.5%不能作为本轮同题Keep分数，也不能据此认定代码复现失败。本轮样本已用于前轮探索，不是独立确认集。

## 统计与完整性

20,000次Image SHA整簇配对Bootstrap，seed=20261106；每次有放回抽取918个图片簇，保留同图所有问题，以题数加权。全部区间为预设比较的普通探索性95%CI，未作确认性多重检验结论，CI跨0不代表无损或非劣效。任务表保留官方五类，不混入历史六任务标签。

20题GPU Smoke验证禁用实际Attention概率严格为0、非目标Mask不变、P−1/Decode全层禁图、原生位置与DeepStack/MLP保持、Cache独立、Hook清理。全部1000题新增生成逐步检查Logits有限，完整保存Token、EOS、截断、空答案和异常；异常记录共0条（包含复用参考，见output_anomalies.csv）。

本轮只做质量，没有运行测速、FLOPs统计、Kernel搜索或物理优化。图表仅展示准确率和配对质量差异。生成、检查和统计结束后释放两卡，提交独立分支并停止。
