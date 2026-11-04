# 单GPU三小时Receiver规则探索：最终交付

当前最优无训练规则在3000题上答对2598题（86.600%），平均R预算48.779%。相对Question-Only多23题，距离All-Writer仍少13题。另一较省预算规则（阈值18）为2596/3000、47.870% R。均为探索性结果，未达到或证明完整All-Writer无损。

## 可复现固定规则

所有Question Token均可在32层读取原始Image Key；若英文问题首词为what/which，或原始LLaVA Question Token数至少15，再允许原生Prompt中完整ASSISTANT词组的4个Subword读图。仅改变Hard Mask，不插入文字，不删除原S，不修改Prompt。P−1及真实Decode在全部32层仍禁止直接读取Image Key；其余Attention、Image Query、MLP、Residual和完整KV均原生。

LLaVA沿用固定revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9、FP16/Eager、Transformers4.49.0、Greedy32及logits_to_keep=1。实际逐题位置见final_per_question.csv及phase6_fixed_compositions/winner_freeze.json。50%约束是逐题比例的总体均值，不是每题上限；理论可访问连接预算不代表FLOPs、显存或延迟节省。

## 全部无训练完整3000题结果

|条件|正确/3000|ACC|平均R比例|
|---|---:|---:|---:|
|All-Writer|2611|87.033%|100.000%|
|Question-Only|2575|85.833%|42.143%|
|Semantic-k|2514|83.800%|25.795%|
|Semantic-Q80|2537|84.567%|31.997%|
|S+Assistant|2560|85.333%|39.408%|
|Phase1-whatwhich|2570|85.667%|44.852%|
|Phase2-length20|2566|85.533%|40.291%|
|Phase5-spatial|2574|85.800%|43.213%|
|Fixed-Q-Conditional-Assistant-15|2598|86.600%|48.779%|
|Fixed-Q-Conditional-Assistant-18|2596|86.533%|47.870%|
|Phase8-spatial-length20|2597|86.567%|48.820%|
|Phase9-cached-fallback|2596|86.533%|47.870%|

## 配对损害与纠正

最终规则减基线；单位pp。按2409个Image SHA整簇配对Bootstrap，20,000次，seed=20261104。

|基线|ΔACC|普通95%CI|原对→错|原错→对|
|---|---:|---|---:|---:|
|All-Writer|-0.433|[-1.090, +0.234]|58|45|
|Question-Only|+0.767|[+0.267, +1.282]|19|42|
|Semantic-k|+2.800|[+1.800, +3.801]|76|160|
|S+Assistant|+1.267|[+0.368, +2.163]|75|113|

final_paired_comparisons.csv额外提供四项终局比较的配对图片簇符号翻转检验及Holm校正。该校正不覆盖此前反复的候选搜索，不能将其当作独立确认性证据。所有3000题此前均已被研究；DEV、CHECK与其余图片的分组仅用于开发筛选组织，不是新的外部验证集。CI跨0不是非劣效证明，也不据此宣称严格无损。

## 搜索过程与训练禁令

原3小时从2026-11-04 09:20:25 UTC起算，始终未重置时钟；物理GPU1串行加载每次一份模型，其他GPU不占用。阶段1测试32个预算合格的固定S扩展规则；阶段2测试Question与原生模板/长度规则，曾因大面积并列扩展180题复查，并在任何新增最终3000生成前保留了方案修订记录。阶段5冻结文本类别谓词，阶段6冻结OR组合，阶段7冻结整数长度阈值与有限类别扩展。每轮先锁定候选及位置，再做小样本筛选与600题复查，冻结一个规则后覆盖完整3000题；相同writer集合经SHA审计复用，不重复生成。阶段7 top4含等价/高度重叠的规则，不能称为四个独立证据。

用户后来明确禁止训练。此前阶段3已有CPU ridge/类型收缩试探，现完整保留但排除出候选排名；其中OOF并不构成干净独立确认。阶段4当时只在GPU收集前向答案，收到指令后立即终止，尚未执行CPU拟合，无拟合后的模型或最终策略。其冻结原模型前向记录若与后续writer集合完全一致，可在输入、Checkpoint和raw SHA核验后复用；这不使用学习出的权重。阶段5、6、7和Qwen复核全部只采用明写规则，没有训练、拟合分数或Router。

阶段1名为middle的早期填充规则使用原始绝对位置归一化，不能视为严格的问题词组中心位置控制。全部尝试、失败和中止记录均保留，未改写历史结论。

## Qwen固定60题迁移复核

用户明确授权后新增。完全复用原DEV60，六任务各10题，不重新挑样。Qwen3-VL-8B-Instruct revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b；其历史冻结环境为BF16/Eager、Transformers4.57.6、36层、原生Qwen Prompt/Tokenizer/图像布局与DeepStack/mRoPE。全部36层P−1/Decode禁读图，R包含vision_end，以免额外视觉中继未受控制。

迁移规则保持what/which及原LLaVA长度≥15这个输入判据；Receiver映射到Qwen实际Question Token和原生assistant角色词组，不把LLaVA的4个Subword位置生搬硬套。不同模型预算必须实测，不能假设Qwen也低于50%。

|Qwen条件|正确/60|ACC|实际平均R比例|
|---|---:|---:|---:|
|All-Writer|57|95.000%|100.000%|
|Question-Only|56|93.333%|44.882%|
|Semantic-k|54|90.000%|26.530%|
|Fixed-Q-Conditional-Assistant|56|93.333%|46.764%|

Qwen结果与LLaVA3000主统计分开。qwen_fixed60/paired_comparisons.csv含小样本配对CI；same_sample_cross_model.csv含同60题两模型表现。仅60题、提示词/Tokenizer/精度/架构不同，不能解释为纯架构因果效应或普遍跨模型有效。Qwen加载第一次在任何Forward前因inference参数无版本计数而中止；已保存qwen_attempt0_load_audit，修正为在普通上下文加载、推理仍使用inference_mode后，重新CPU冻结并通过三题全有效Logits严格torch.equal、Mask概率0、独立KV、MLP36、DeepStack、位置与Hook检查。未放宽容差。

## 校验与交付范围

终局LLaVA输出审计：EOS=3000/3000，空答案=0，截断=0，重复3gram=0。完整答案重新评分、S包含性、整词、writer位置、输入哈希及复用来源逐题核验通过。首Token翻转、输出变化、所有相对All/Q/S/SA的损害和纠正均已保存；六任务各500题。

final_provenance.csv、各阶段checkpoint_index.csv及Checkpoint记录原始向量的服务器路径和SHA。大向量不在GitHub，不声称GitHub含这些数组。final_artifact_index.csv提供交付文件SHA。图表为离散实测值，不画虚构平滑线。final_gpu_release.json确认仅GPU1相关进程退出；main保持原Hash。

结论：无需训练的Question接收者加条件式原生答案角色接收者，在这批已反复研究的数据上比仅Question更接近All-Writer；总体准确率86.60%，预算48.78%，仍存在58道All原对变错及45道相反纠正。该收益不是严格无损，不代表物理计算已减少。停止新增规则、训练、搜索和GPU实验；已保留全部负面结果。

本汇总生成时距原起点113.80分钟；最终提交/推送及结束时间另见发布回执。

## 出版前复核补充

阶段8对阶段7已冻结、预算合格的整个规则族统一完成600题复查，避免早期top4多为等价规则而缩窄候选。冻结的胜者为what/which或长度≥20或含空间词时开放assistant，最终2597/3000、48.820% R；未超过阶段6的2598。阶段9冻结少量文本类别触发All-Writer的静态回退规则，只复用精确相同writer集合的历史记录，新增GPU生成数严格为0；其开发胜者退回已有阈值18规则，为2596/3000。全部阶段已停止，不再追加策略。

这里“无训练”不等于规则没有使用实验反馈进行开发选择：规则族是有限手写谓词，ACC用于挑选候选，没有学习权重或打分模型；本轮正向收益仍受同数据反复探索的选择偏差影响。

Qwen新规则与Question-Only均为56/60，未复现LLaVA上超越Question-Only的净提升。不能称为跨模型优势确认。Qwen原生assistant角色词的Tokenizer组与LLaVA ASSISTANT组不同，这种迁移检查只测试明确映射后的规则。

预算另记：最终LLaVA的逐题k/R均值为48.779%，所有问题合并后的总k/总R为50.066%；二者含义不同。完整逐题位置、比例和高预算题均保留。

图表已经人工查看PNG；预算图对重叠编号做了排版调整，数据和区间不变。三个矢量PDF经pdfinfo验证，均为单页。最终CSV、原始Checkpoint、图表和代码均进入文件SHA索引；原始Logits数组仍只在服务器资源目录。

若要求合并后的连接预算也低于50%，阈值18规则符合：逐题平均47.870%，合并总k/总R为49.070%，正确2596/3000（86.533%）。阈值15是原先逐题平均预算口径下的已测最优；不将其50.066%的合并连接预算写成低于50%。两种口径的完整分子、分母见final_budget_definitions.csv。
