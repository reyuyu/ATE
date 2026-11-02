# COCO逐词Decode Attention来源观察

## 核心结论

1. **L31来源分配并非逐词静态。** 非EOS增量位置的每图Image Mass范围平均8.73个百分点，History为19.04个百分点；Image范围的跨图P5/中位数/P95为5.11/8.65/12.77个百分点，History为14.70/18.45/24.02个百分点。
2. **词类关联并不完全被位置与置信度解释。** 但Visual−Function的原始Image Mass优势经调整后接近0；不能把所有原始差异都写成独立的词类效应。非BOS Image占比及部分History关联仍保留，均属描述性、非因果结果。
3. **Head平均Mass相近不代表逐Head分配不变。** 严格近Mass口径只覆盖35/2,843对非EOS相邻位置（1.23%，34/200图）；其中Image贡献变化22对、History变化26对。更重要的是，这35对位置中平均只有0.54%的Head同时满足同样的单Head稳定阈值。因此只能说Head平均Mass不能充分概括贡献，不能说完整Attention分布固定而功能改变。
4. **分离与同步均存在。** L31逐图内部Mass与相对贡献范数的Pearson均值为Image 0.488、History 0.821，不能宣称两者普遍无关。L14/L26也有近平均Mass而贡献改变的案例，非L31独有。
5. **本轮没有测因果必要性。** 来源投影范数和方向不等于来源移除损害，也不能支持安全跳层或加速声明。

## A. L31来源是否逐词变化？

严格复用历史200张COCO图片，3443个原生生成Token，其中3243个为真实增量Decode；348个多Token词和445个非首子词均保留。
以下数值为每张图内部变化的图片级均值与普通95% Bootstrap区间；首Token Prefill不混入增量主分析。

| 来源 | 每图平均Mass | 每图范围(max−min) |
|---|---|---|
| BOS | 0.606494 [0.604209, 0.608684] | 0.139679 [0.132707, 0.147496] |
| IMAGE | 0.170071 [0.168347, 0.171794] | 0.087300 [0.083534, 0.091308] |
| PROMPT_TEXT | 0.069763 [0.069022, 0.070492] | 0.105232 [0.102804, 0.107626] |
| GENERATED_HISTORY | 0.153671 [0.151427, 0.156055] | 0.190413 [0.185678, 0.195432] |

范围刻画同一Caption内非EOS位置的实际变化，不等于不同词类的因果差异。完整含EOS/不含EOS、32层和每图最小/最大/均值/标准差均在CSV。

相邻非EOS增量位置的图片平均来源JS为0.003600 nat，95%CI [0.003408, 0.003819]；去BOS后为0.005754 [0.005537, 0.005972]。逐层结果见adjacent_js_summary.csv；跨图变化范围分位数见l31_across_image_range_quantiles.csv。

## B. 词类关联与位置控制

主分析使用真实词的首子词预测位置，Function/Visual/Relation沿用旧词典及优先级。Ambiguous并入Unknown但保留原标签；Visual只是词表候选，不等于已验证视觉Grounding。
全部词类统计按**正在预测的词**标注；Query本身正在消费的前一个Token及其词类另列保存，二者没有混用。
原始均值采用同图词类均值配对；调整模型采用图片固定效应，每张Caption等权，控制绝对/相对生成位置、原生Token LogP/Entropy及词的Token长度。
Generated History合法Key数量严格等于t，因此t同时控制历史长度；不再加入完全共线的Key数量列。另报告单位Key Mass和非BOS条件占比。

| 结果 | 比较 | 原始同图差异 [95%CI] | 调整后系数 [95%CI] |
|---|---|---|---|
| BOS | Visual-Function | 0.009484 [0.006381, 0.012713]; n=200 | 0.007067 [0.003903, 0.010253] |
| BOS | Relation-Function | 0.013562 [0.009545, 0.017756]; n=197 | 0.015212 [0.011819, 0.018615] |
| IMAGE | Visual-Function | 0.004106 [0.002091, 0.006183]; n=200 | -0.000081 [-0.002017, 0.001939] |
| IMAGE | Relation-Function | -0.023669 [-0.026004, -0.021314]; n=197 | -0.025161 [-0.027498, -0.022791] |
| GENERATED_HISTORY | Visual-Function | -0.025125 [-0.029181, -0.021187]; n=200 | -0.013563 [-0.016619, -0.010613] |
| GENERATED_HISTORY | Relation-Function | 0.003959 [-0.000548, 0.008276]; n=197 | 0.008625 [0.005764, 0.011417] |
| nonbos_IMAGE | Visual-Function | 0.020416 [0.015237, 0.025641]; n=200 | 0.007724 [0.003119, 0.012445] |
| nonbos_IMAGE | Relation-Function | -0.045006 [-0.050369, -0.039672]; n=197 | -0.046631 [-0.051545, -0.041754] |
| nonbos_GENERATED_HISTORY | Visual-Function | -0.054489 [-0.062501, -0.046553]; n=200 | -0.028011 [-0.033777, -0.022375] |
| nonbos_GENERATED_HISTORY | Relation-Function | 0.024921 [0.016270, 0.033363]; n=197 | 0.037868 [0.032200, 0.043404] |
| image_history_ratio | Visual-Function | 0.985996 [0.899993, 1.071638]; n=200 | 0.775508 [0.694409, 0.854374] |
| image_history_ratio | Relation-Function | -0.210243 [-0.256426, -0.161647]; n=197 | -0.310326 [-0.369804, -0.248883] |

调整后的描述性区间排除0的项目：IMAGE/Relation_vs_Function；GENERATED_HISTORY/Visual_vs_Function；GENERATED_HISTORY/Relation_vs_Function；nonbos_IMAGE/Visual_vs_Function；nonbos_IMAGE/Relation_vs_Function；nonbos_GENERATED_HISTORY/Visual_vs_Function；nonbos_GENERATED_HISTORY/Relation_vs_Function。
这些普通区间未作确认性多重比较保证；不能把关联解释为词类驱动注意力的因果效应，也不能从区间跨0宣称分配恒定。

## C. Mass相近但输出贡献改变？

预先固定的描述性“Mass相近”：相邻增量位置**Head平均**四来源JS≤1e−4 nat且非BOS三来源JS≤1e−3 nat。这不是完整逐Head注意力行不变。
“贡献变化”：同来源投影向量范数相对变化≥20%，或相邻向量Cosine≤0.90。阈值只用于可复核描述，不是等价性或必要性检验。

| 层 | 来源 | 相近Mass位置对 | 贡献变化位置对 | 涉及变化图片 | 图片平均变化率 [95%CI] |
|---|---|---:|---:|---:|---|
| L14 | IMAGE | 15 | 7 | 7 | 0.464286 [0.214286, 0.714286]; n=14 |
| L14 | GENERATED_HISTORY | 15 | 15 | 14 | 1.000000 [1.000000, 1.000000]; n=14 |
| L26 | IMAGE | 24 | 10 | 10 | 0.434783 [0.217391, 0.652174]; n=23 |
| L26 | GENERATED_HISTORY | 24 | 23 | 22 | 0.956522 [0.869565, 1.000000]; n=23 |
| L31 | IMAGE | 35 | 22 | 22 | 0.647059 [0.470588, 0.794118]; n=34 |
| L31 | GENERATED_HISTORY | 35 | 26 | 26 | 0.750000 [0.602941, 0.882353]; n=34 |
L31在上述预声明口径下存在“Mass相近而投影贡献变化”的真实位置对。范数和方向仍是观察量，不是独立因果必要性。

本表L31分母为2,843对可用非EOS相邻增量位置中的35对，只有34张图片入选；64.7%/75.0%是这34张图内符合近Mass口径的位置对变化率的图片均值，不是全200图全部位置的变化率。单Head复核见near_mass_head_audit.csv；全部真实差异案例均保存于similar_mass_difference_cases.csv，没有仅挑选最大变化案例。

## D. 是否L31特有？

L14：IMAGE变化7/15对，7图；GENERATED_HISTORY变化15/15对，14图。
L26：IMAGE变化10/24对，10图；GENERATED_HISTORY变化23/24对，22图。
L31：IMAGE变化22/35对，22图；GENERATED_HISTORY变化26/35对，26图。
这种分离并非L31独有；在实测L14/L26也可观察。只在L14/L26/L31做了投影分解，不能推广为所有深层均具有同样贡献结构。

## E. Attention Allocation vs Functional Necessity

支持点：在同一组全部增量位置（包括EOS、排除Prefill）中，L31的BOS平均Mass约0.603，但其投影范数/残差范数约0.0643；History平均Mass约0.155，相对贡献范数约0.0862。来源质量排序不能直接当作输出贡献排序。上述两项使用完全相同位置与图片权重，见contribution_summary.csv。
反证：实际Image/History分配会随词变化，且单Head变化可以被Head平均掩盖；“Attention分配基本保持不变”不适合作为本轮总体结论。
同步证据：包含EOS的逐图内部Mass与相对贡献范数Pearson均值，Image为0.488 [0.460, 0.515]，History为0.821 [0.807, 0.833]，Prompt为0.927 [0.921, 0.932]。这些是观察性同步关系，不能只挑近Mass案例宣称分配与贡献脱钩。
本轮没有移除Attention或来源，没有新的因果必要性/质量实验；Wo(PV)范数不等于去掉该来源造成的LogP损害。来源向量可能抵消，历史Value也可能携带视觉信息。
因此不能据此宣称视觉语义检索、某层可安全跳过或已经实现计算加速。应将本轮观察与历史独立记录的消融损害严格区分。

## 正确性、资源与文件

200图原生与只读采集的全部生成Token及有效Logits严格一致；八图GPU Smoke通过。所有32层、32Head和每步均覆盖，行和容差沿用0.002。
实际来源位置动态构建、互斥完备；生成采用两次独立原生Cache，未保留/修改其他条件Cache。原生输入与权重版本未变，Hook和eager包装均清理。
与历史Caption轨迹一致：200/200；历史逐Logits数值差异逐图记录，不隐去跨运行差异。
GPU1采集耗时16.72分钟；峰值显存见collection_completion.json及environment.json。
原环境未安装SciPy，CPU统计单独使用外部cpu_deps目录中的SciPy 1.13.1，未升级或覆盖任何已有模型环境；冻结采集源码不变，依赖与秩统计验证见cpu_analysis_environment.json。原始模型文件在采集后再次计算SHA256，全部与冻结绑定一致。
per_token.csv保留消费/预测Token与词类、词对齐、LogP/Entropy；per_token_layer_mass.csv保留32层质量、非BOS比例、Key数量和相邻JS。
per_head_mass/*.csv.gz保存每图每步每层每Head全部五类Mass（9位有效数字可还原FP32）；完整词表原生Logits及未平均Mass在本机raw_path指向的NPZ并有SHA256绑定。
per_token_contributions.csv保留三层全部来源投影范数、相对残差比例、方向及相邻向量变化；模型/数据/源码绑定、CPU/GPU验证和每图检查点随分支保存。

图1代表图片在GPU前固定为原Manifest第一图，不按效果选取。首Token Prefill和后续增量位置用虚线分隔；多子词的后续位置标记·2/·3。F/V/R/U/P/E为词类。
图2保留全部生成Token，统一Image/History色标，标记L31。图3上排使用同时具备三类词的同图集合，下排是各自可用同图配对区间。
图4上排每点为一图一来源的均值，纵轴为投影范数/该层残差范数；下排使用预声明Head平均近Mass阈值和贡献变化阈值，排除EOS。入选图片仅L14的14张、L26的23张、L31的34张，不能把这些条件变化率推广到全部位置。
Bootstrap只重采样图片SHA256簇，不将Token/Head/Layer当作独立样本；相同词的其他子词及EOS分析在subword_and_eos_summary.csv。
