# InternVL 原生前向修复与视觉来源扫描补齐

## 修复与正确性

- 已定位旧独立前向与官方 Chat 不一致的原因：官方生成设置 `logits_to_keep=1`，旧独立前向默认对全部位置投影 LM-head。相同 P−1 Hidden State 在不同 BF16 矩阵乘法形状下出现舍入差异；不是输入、视觉 Token 或模型权重不同。
- 诊断样本旧最大 Logits 差异为 0.0625；改为原生最后位置投影后逐元素完全一致，最大差异 **0**。官方输入重放也完全一致。详见 `diagnosis.json` 的输入、Hidden 和投影形状哈希。
- 没有放宽容差，没有修改第三方 Transformers/模型源码或模型权重。修复仅是独立前向显式传入 `logits_to_keep=1`，其余原生前向和 Softmax 前来源 Mask 不变。
- 9项修复与Mask CPU回归检查、1项补偿控制器检查、2项图片簇配对统计检查通过。每个扫描阶段的GPU0/2/3各用两题Smoke，官方Full/No-op精确一致；目标Image概率为0、非目标Prefill Query状态逐层精确不变、无Cache前向、条件隔离和Hook清理检查通过。
- 旧失败记录及上一轮3843个产物原样保留。本目录是独立恢复交付，不把先前的失败改写成已通过。

## 完成范围

- InternVL：3000/3,000题，84,000/84,000个单层条件，六任务各500；先完成原240题的6,720条件，再扩展，未重新抽样。
- Qwen：严格复用上一轮3,000题、108,000条件，没有重跑；三模型同240题对照中的LLaVA来源为明确标记的历史结果。
- InternVL固定revision `853e3a797a661694b1b8ece0cb72dc2b23e3dac9`，BF16/eager、Transformers4.49.0，官方模板，max_num=1，256视觉Token。每题Full目标为其自己的原生首Token Top1，不是Gold候选。
- 逐层扫描每次只对一个目标层的原始P−1→实际Image Key施加Softmax前−inf；不是o_proj OFF，也不是Keep-only。其他27层、其他Prefill Query、MLP、Residual和视觉注入均原生，不将Decode或EOS Query替代首答案位置。
- 分片按全部3,000题的Image SHA256整簇确定，Pilot和扩展阶段沿用同一GPU归属。同题Full与28个干预在同卡配对，GPU1始终排除。

## 主要结果

|模型与队列|Mean Absolute峰层|峰值 nat|前三层|前三层描述性占比|
|---|---|---|---|---|
|Qwen，3,000题|L26|0.017883|[26, 23, 24]|15.95%|
|InternVL，3,000题|L19|0.091859|[19, 9, 16]|23.31%|

|InternVL任务|题数|Mean Absolute峰层|峰值 nat|
|---|---|---|---|
|ChooseAttr|500|L9|0.074719|
|ChooseCat|500|L9|0.046043|
|ChooseRel|500|L19|0.009741|
|CompareAttr|500|L21|0.039798|
|LogicalObj|500|L9|0.104206|
|QueryAttr|500|L19|0.341758|

InternVL总体L19峰中，QueryAttr的题目加权份额约为62.0%；任务组成对总体峰很重要，不能把均值峰解释成所有任务共享的处理阶段。该份额只是本层平均绝对变化的算术分解，不是独立因果贡献。
Qwen与InternVL均呈非均匀分布，但主峰不在相同绝对层，InternVL分任务也出现L9/L19/L21差异。两者Top3均不到逐层平均绝对变化之和的一半，因此支持“有热点、任务相关”的描述，不支持“直接视觉必要性只存在于极少数层”的强结论。
峰值与Top3占比只能描述曲线集中程度，不构成这些层充分或其余层可跳过的证据。跨模型保留各自原生Prompt、Tokenizer、视觉Token数和精度；同题、同图隔离了数据差异，但没有控制全部架构外因素。

## 与原实验的关系

- 上一轮LLaVA 240题的LogP交互仍为 −0.011213 nat，图片簇95%CI [−0.019249, −0.003380]。没有出现预声明的正交互，因此不新增InternVL的L2–L5 2×2实验，也不修改候选层寻找正面结果。
- LLaVA/Qwen中继受损后的固定稀疏层补偿结果不变，详见上一轮报告。单层OFF峰值不是Keep-only充分性证明。

## InternVL 可选稀疏补偿

仅依据先完成的240题探索曲线、在3,000题扩展前冻结的两层为 [19, 9]。没有使用完整生成Accuracy或3,000题新峰选择层。本项严格沿用原定三个条件，Decode仍原生；额外运行原生Full仅用于同机基线与首Logits绑定。

|240题条件|完整答案Accuracy|固定Full首Token Loss nat|首Token翻转|EOS/截断/空答案|
|---|---|---|---|---|
|原生Full（中继ON）|92.500%|0|0|240/0/0|
|Relay OFF / FullImageAccess|72.083%|1.063728|100|240/0/0|
|Relay OFF / AllImageOFF|59.167%|1.313045|123|240/0/0|
|Relay OFF / KeepSparse|64.167%|1.287771|115|240/0/0|

|探索性配对比较|ΔAccuracy pp|图片簇95%CI pp|参照对→错|参照错→对|
|---|---|---|---|---|
|KeepSparse - AllImageOFF|+5.000|[+1.660, +8.403]|3|15|
|KeepSparse - FullImageAccess|-7.917|[-12.917, -3.306]|28|9|
|FullImageAccess - AllImageOFF|+12.917|[+8.333, +17.647]|3|34|
|FullImageAccess - NativeFull|-20.417|[-26.556, -14.346]|57|8|
|AllImageOFF - NativeFull|-33.333|[-40.083, -26.556]|87|7|
|KeepSparse - NativeFull|-28.333|[-34.855, -21.757]|76|8|

InternVL固定L19/L9相对全部直接视觉关闭恢复了5.000pp，但仍比保留全部直接视觉读取低7.917pp、比原生Full低28.333pp。因此是有限恢复，不是受损中继已被两个热点充分补偿。
本项为同240题、数据驱动挑选两层的探索性充分性检查，不是独立确认。不能将普通CI包含0解释成等价，不能将稀疏访问相对全关的改善解释成已恢复原生Full；来源Hard Mask也会重分配保留Key的概率。

综合三个模型：指定中继联合干预会损害预测，固定稀疏热点只产生有限恢复。该结果与“信息路径背景值得研究”相容，但LLaVA的预声明L2–L5跳过交互仍不支持“Dead Attention Can Wake Up”。没有据此追加候选或改变负面结论。
CSV口径：`accuracy_summary.csv`/`task_accuracy.csv`的`delta_full_pp`指相对Relay OFF的FullImageAccess；相对原生NativeFull的差异在`paired_comparisons.csv`中显式列出。原生Full逐题答案另存`native_full_per_question.csv`。

## 统计、图表与时限

- 复用题目加权、Image SHA256整簇配对Bootstrap，10,000次、seed20261102；输出Mean Absolute、Mean Signed和Top1 Flip的总体/六任务逐层估计及逐点95%CI。探索性曲线不以未经多重比较校正的区间宣称确认性显著性。
- 3,000题仍是原Research 1冻结的六类研究题库，不是新抽取的独立确认集；240题热点选择及充分性检查也属于探索。本轮没有Qwen/InternVL的L2–L5交互实测，不能把LLaVA的负结果泛化成所有模型、所有层都没有条件性。
- PNG与矢量PDF：`figures/internvl_overall_sensitivity`、`internvl_six_tasks_absolute/signed/flip`、`qwen_internvl_same3000`、`three_models_same240`。原始逐题CSV、每层/任务汇总和完整检查点均保留。
- 沿用原总墙钟起点 2026-11-02T04:18:13+08:00 和截止 2026-11-02T11:18:13+08:00，未重新计算7小时预算。报告时间 2026-11-02T07:29:34.250056+08:00，累计 3.19 小时。最终资源状态见 `delivery_audit.json`。

本恢复补齐原定InternVL扫描与可选固定两层补偿，没有新增干预条件。首Token分布变化不是完整答案Accuracy，直接来源低敏感也不代表模型没有视觉信息。
