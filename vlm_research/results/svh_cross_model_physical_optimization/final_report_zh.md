# 跨模型物理优化与效率复测：完成验证

本轮只测真实发生变化的路径、对应冻结实现及同 Backbone Native；没有重跑全量质量 Benchmark、训练、层搜索、其他外部基线或 GitHub 推送。普通 LLaVA SVH/SVH-R 源码与执行路径未改，未为统一表格新增其 GPU 请求。

已记录正式效率请求：1200。正式协议为原 GQA15 + TextVQA Paper-OCR15，32输出Token、1 Prefill +31真实Decode、Batch1、原精度/Eager、两个独立Session。全部请求固定 GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6。

## 主要结论：两轮配对改善

|Backbone|Method|版本|TTFT改善% S1/S2|TPOT改善% S1/S2|E2E改善% S1/S2|
|---|---|---|---|---|---|
|Qwen|SVH|E|4.904 / 4.789|8.004 / 8.402|7.637 / 7.959|
|Qwen|SVH-R|C|11.134 / 10.974|23.672 / 23.643|21.914 / 21.869|
|InternVL|SVH|B|2.754 / -0.572|0.400 / 0.024|0.572 / -0.002|
|LLaVA|FastV+SVH|B|8.684 / 7.372|-0.102 / -0.252|0.606 / 0.368|
|LLaVA|SparseVLM+SVH|B|7.013 / 7.010|0.964 / 0.698|1.511 / 1.275|
|LLaVA|V2Drop+SVH|B|5.717 / 5.120|0.452 / 0.062|0.895 / 0.487|
|LLaVA|ApET+SVH|B|0.860 / -3.598|19.877 / 19.373|18.653 / 17.869|

Qwen SVH-R 与 LLaVA ApET+SVH 显示稳定的主要 E2E 改善；Qwen SVH 最后E版也在独立两轮中重复获得收益。FastV/V²Drop融合的E2E改善不足1%，不据此宣称显著整体加速。SparseVLM融合改善较小。
InternVL SVH B的正式E2E改善约0.572%和−0.002%，没有稳定的整体收益，本轮不建议据此替换默认原版。其C2/E虽然数值通过，但筛选E2E退化约0.533%，拒绝采用。InternVL SVH-R的B/C未显示有用收益，未强制进行正式复测。
ApET+SVH第二轮TTFT退化约3.598%，尽管Decode与E2E改善明显，仍完整报告该局部负结果。
上表每个S1/S2属于该版本自己的同题配对Session；E补充与早期B/C是不同campaign，不把它们的Native绝对均值混合。所有版本和慢样本均保留。

## 正式效率（每行30题算术均值）

|Backbone|Session|Method|阶段|n|KV Live MiB|TTFT ms|TPOT ms/token|E2E@32 ms|E2E中位数|
|---|---|---|---|---|---|---|---|---|---|
|InternVL|session1|Native|A|30|20.969|144.176|61.886|2062.651|2061.759|
|InternVL|session1|SVH|A|30|5.973|130.414|56.238|1873.783|1867.330|
|InternVL|session1|SVH|B|30|5.973|126.656|56.010|1862.969|1862.060|
|InternVL|session2|Native|A|30|20.969|144.363|60.578|2022.274|2021.492|
|InternVL|session2|SVH|A|30|5.973|126.278|55.035|1832.376|1829.038|
|InternVL|session2|SVH|B|30|5.973|126.740|55.022|1832.410|1823.595|
|LLaVA|session1|ApET+SVH|A|30|55.562|165.329|77.062|2554.240|2553.004|
|LLaVA|session1|ApET+SVH|B|30|55.562|163.831|61.730|2077.461|2069.096|
|LLaVA|session1|FastV+SVH|A|30|55.562|169.708|61.433|2074.127|2068.990|
|LLaVA|session1|FastV+SVH|B|30|55.562|155.121|61.491|2061.340|2058.132|
|LLaVA|session1|Native|A|30|351.500|242.513|66.788|2312.948|2313.856|
|LLaVA|session1|SparseVLM+SVH|A|30|55.562|189.659|62.037|2112.800|2102.826|
|LLaVA|session1|SparseVLM+SVH|B|30|55.562|176.333|61.416|2080.225|2071.858|
|LLaVA|session1|V2Drop+SVH|A|30|55.562|175.097|61.753|2089.440|2086.374|
|LLaVA|session1|V2Drop+SVH|B|30|55.562|165.100|61.467|2070.570|2058.255|
|LLaVA|session2|ApET+SVH|A|30|55.562|163.446|76.410|2532.164|2520.422|
|LLaVA|session2|ApET+SVH|B|30|55.562|169.857|61.598|2079.406|2059.230|
|LLaVA|session2|FastV+SVH|A|30|55.562|168.474|61.246|2067.104|2054.504|
|LLaVA|session2|FastV+SVH|B|30|55.562|156.229|61.392|2059.372|2043.112|
|LLaVA|session2|Native|A|30|351.500|240.067|66.800|2310.854|2303.659|
|LLaVA|session2|SparseVLM+SVH|A|30|55.562|189.758|61.584|2098.853|2102.498|
|LLaVA|session2|SparseVLM+SVH|B|30|55.562|176.389|61.145|2071.885|2063.036|
|LLaVA|session2|V2Drop+SVH|A|30|55.562|173.915|61.404|2077.439|2075.566|
|LLaVA|session2|V2Drop+SVH|B|30|55.562|165.034|61.349|2066.861|2049.929|
|Qwen|refine_session1|Native|A|30|86.245|383.911|89.874|3170.019|3135.489|
|Qwen|refine_session1|SVH|A|30|10.480|365.002|88.867|3119.880|3113.363|
|Qwen|refine_session1|SVH|E|30|10.480|347.683|81.752|2882.003|2856.748|
|Qwen|refine_session2|Native|A|30|86.245|383.426|91.984|3234.921|3187.183|
|Qwen|refine_session2|SVH|A|30|10.480|365.583|91.376|3198.239|3189.883|
|Qwen|refine_session2|SVH|E|30|10.480|349.246|83.684|2943.451|2903.010|
|Qwen|session1|Native|A|30|86.245|382.629|91.570|3221.310|3245.661|
|Qwen|session1|SVH|A|30|10.480|358.777|90.309|3158.361|3113.884|
|Qwen|session1|SVH|B|30|10.480|348.615|85.596|3002.076|2960.708|
|Qwen|session1|SVH-R|A|30|45.268|460.802|104.424|3697.934|3658.440|
|Qwen|session1|SVH-R|C|30|45.268|418.657|79.699|2889.321|2851.529|
|Qwen|session2|Native|A|30|86.245|383.458|91.513|3220.353|3156.243|
|Qwen|session2|SVH|A|30|10.480|364.377|90.201|3160.622|3175.601|
|Qwen|session2|SVH|B|30|10.480|347.278|85.960|3012.045|2959.891|
|Qwen|session2|SVH-R|A|30|45.268|460.597|104.569|3702.245|3649.260|
|Qwen|session2|SVH-R|C|30|45.268|419.139|79.843|2894.271|2836.795|

中位数、Peak Allocated/Reserved、真实逐层Live/Storage/Capacity及全部逐题数据见CSV，不以峰值总显存替代KV。

## 同Session、同输入配对改善

|Backbone|Session|Method|优化阶段|配对n|TTFT改善%|TPOT改善%|E2E改善%|KV改善%|
|---|---|---|---|---|---|---|---|---|
|InternVL|session1|SVH|B|30|2.754|0.400|0.572|0.000|
|InternVL|session2|SVH|B|30|-0.572|0.024|-0.002|0.000|
|LLaVA|session1|ApET+SVH|B|30|0.860|19.877|18.653|0.000|
|LLaVA|session1|FastV+SVH|B|30|8.684|-0.102|0.606|0.000|
|LLaVA|session1|SparseVLM+SVH|B|30|7.013|0.964|1.511|0.000|
|LLaVA|session1|V2Drop+SVH|B|30|5.717|0.452|0.895|0.000|
|LLaVA|session2|ApET+SVH|B|30|-3.598|19.373|17.869|0.000|
|LLaVA|session2|FastV+SVH|B|30|7.372|-0.252|0.368|0.000|
|LLaVA|session2|SparseVLM+SVH|B|30|7.010|0.698|1.275|0.000|
|LLaVA|session2|V2Drop+SVH|B|30|5.120|0.062|0.487|0.000|
|Qwen|refine_session1|SVH|E|30|4.904|8.004|7.637|0.000|
|Qwen|refine_session2|SVH|E|30|4.789|8.402|7.959|0.000|
|Qwen|session1|SVH|B|30|2.910|5.207|4.943|0.000|
|Qwen|session1|SVH-R|C|30|11.134|23.672|21.914|0.000|
|Qwen|session2|SVH|B|30|4.813|4.699|4.711|0.000|
|Qwen|session2|SVH-R|C|30|10.974|23.643|21.869|0.000|

改善百分比为逐题(1−优化/冻结原版)的算术平均，不能与均值比值混淆。正值代表改善；负值与慢样本不删除。两Session均保留，不选最快一轮。相对本轮Native的对应数据在paired_comparisons.csv。

## 共同问题与模型差异

跨模型重复的是控制路径与存储管理成本：重复索引/Mask、调试标量回传、位置Gather，以及InternVL的逐Token KV cat。融合路径还有排名/聚类账本与调试同步；同名API的成本和语义不能直接等同。
Qwen特有问题包括mRoPE/DeepStack约束、原数值修复链的固定Lt descriptor与补零、SVH-R晚层辅助视觉KV构造。B/C没有改变必要构造、模型权重、访问层集合或原始位置。Qwen的热路径isin与反复布局断言可以缓存/移到审计；必要视觉构造仍计时。
LLaVA SVH/SVH-R原本已使用Compact Attention、请求级Mask与预分配KV，无证据支持改写。四个融合版本保留官方排名、Token Recycling、FPS/基底替换及冻结剪枝预算；没有把融合配置冒充独立论文工作点。

## 阶段选择与负结果

|Backbone|Method|E实际指向|筛选状态|全部筛选决策|
|---|---|---|---|---|
|Qwen|SVH|B|ELIGIBLE_FOR_FORMAL|[{"stage": "B", "relative_improvement": {"TTFT_ms": 0.03144450733176363, "TPOT_ms": 0.04990563085993582, "E2E_ms": 0.04731876904567611}, "useful": true}, {"stage": "C", "relative_improvement": {"TTFT_ms": 0.029041621708963183, "TPOT_ms": 0.047877895366397416, "E2E_ms": 0.04523846570367929}, "useful": true}]|
|Qwen|SVH-R|C|ELIGIBLE_FOR_FORMAL|[{"stage": "B", "relative_improvement": {"TTFT_ms": 0.07595451269682885, "TPOT_ms": 0.2335039697182305, "E2E_ms": 0.20984327930605362}, "useful": true}, {"stage": "C", "relative_improvement": {"TTFT_ms": 0.07971640932850277, "TPOT_ms": 0.2335036571984701, "E2E_ms": 0.21040797327810135}, "useful": true}]|
|InternVL|SVH|B|ELIGIBLE_FOR_FORMAL|[{"stage": "B", "relative_improvement": {"TTFT_ms": 0.02060783970859814, "TPOT_ms": 0.00047101110661051937, "E2E_ms": 0.001922844851869665}, "useful": true}]|
|InternVL|SVH-R|NOT_ADOPTED|NO_MEASURED_USEFUL_GAIN|[{"stage": "B", "relative_improvement": {"TTFT_ms": -0.0033386328666258347, "TPOT_ms": 0.0041451053790132475, "E2E_ms": 0.0035691985915977664}, "useful": false}, {"stage": "C", "relative_improvement": {"TTFT_ms": -0.0034918275282476152, "TPOT_ms": -0.00477013570155238, "E2E_ms": -0.004671764228733277}, "useful": false}]|
|LLaVA|FastV+SVH|B|ELIGIBLE_FOR_FORMAL|[{"stage": "B", "relative_improvement": {"TTFT_ms": 0.06313180987321554, "TPOT_ms": 0.0009225593560079348, "E2E_ms": 0.006386661231358226}, "useful": true}]|
|LLaVA|SparseVLM+SVH|B|ELIGIBLE_FOR_FORMAL|[{"stage": "B", "relative_improvement": {"TTFT_ms": 0.04703544096036316, "TPOT_ms": 0.00011347284991303219, "E2E_ms": 0.004579597846615324}, "useful": true}]|
|LLaVA|V2Drop+SVH|B|ELIGIBLE_FOR_FORMAL|[{"stage": "B", "relative_improvement": {"TTFT_ms": 0.05221305601237758, "TPOT_ms": -0.000391944171824532, "E2E_ms": 0.0044207115868566405}, "useful": true}]|
|LLaVA|ApET+SVH|B|ELIGIBLE_FOR_FORMAL|[{"stage": "B", "relative_improvement": {"TTFT_ms": -0.0012082141677822733, "TPOT_ms": 0.18906528876548145, "E2E_ms": 0.17536022603528234}, "useful": true}]|
|Qwen / E补充|SVH|E|ELIGIBLE_FOR_FORMAL|[{"stage": "E", "relative_improvement": {"TTFT_ms": -0.0014752528017756283, "TPOT_ms": 0.03765442720029921, "E2E_ms": 0.03201298797985441}, "useful": true}]|
|InternVL / E补充|SVH|NOT_ADOPTED|NO_MEASURED_USEFUL_GAIN|[{"stage": "E", "relative_improvement": {"TTFT_ms": 0.010273884702162062, "TPOT_ms": -0.006541844599764701, "E2E_ms": -0.005330832135299612}, "useful": false}]|

阶段A/B/C的普通筛选耗时见ablation_efficiency.csv，三种长度输入、各3次并交错执行。E为冻结筛选规则下的阶段别名，不是重新调参数；C相对B的小差值不能单独当作稳定收益。
没有收益或回退的步骤保留，不强制采用。未通过数值门槛的D不进入正式计时，不报告其仪器扰动耗时为有效性能。

D的数值失败/候选记录：

|Backbone|阶段|样本|状态|Max|logit差||Logits相对L2|Hidden相对L2|参考上下文Top-1序列一致|
|---|---|---|---|---|---|---|---|
|Qwen|diagnosis|GQA:201759069|FAILED|1.625|0.104|0.048|False|
|Qwen|screen|GQA:201759069|FAILED|1.625|0.104|0.048|False|
|InternVL|diagnosis|GQA:201889314|FAILED|3.500|0.315|0.000|True|
|InternVL|screen|GQA:201889314|FAILED|3.500|0.315|0.000|True|

候选拒绝 Qwen / screen: {"SVH|D": {"reason": "TOKEN_CHANGE_REQUIRES_INDEPENDENT_QUALITY_NOT_HISTORICAL_BINDING", "first_key": "GQA:201759069"}}

候选拒绝 InternVL / screen: {"SVH|C": {"reason": "AssertionError(('InternVL', 'SVH', 0, 62, 93))", "first_key": "GQA:201889314"}, "SVH-R|D": {"reason": "FROZEN_NUMERICAL_GATE_FAILED", "first_key": "GQA:201889314"}}

E补充版本：Qwen SVH缓存同一步dead-column索引/Mask，原补零保留；InternVL SVH C2修复被有效长度检查拒绝的原C实现，使用独立包。原失败源码与日志保留。E在GPU前另冻结门槛，只有相对B确认额外收益才做正式复测。E与原B/C分开列出，不将不同轮次均值直接作配对比较。

## 数值、质量与论文使用限制

B/C只涉及索引、Mask复用或物理缓存管理，按GPU前冻结要求检查原30题全部32步有效Logits及Token严格一致；原始A还对齐历史冻结Token记录。Native/No-op和真实Audit Mode重新启用检查。逐层数值与缓存生命周期见numerical_*、kv_*及各阶段原始记录。
本轮不是全量Accuracy复评。对B/C，已匹配输入上的完整输出没有回归；不能由30题宣称所有任务绝对无质量回退。未修改Attention算术且数值回归通过的版本可作为对应冻结算法的独立工程实现效率记录；仍须保留原质量协议与配置来源。此前融合质量来源未验证的部分不会因效率优化自动获得质量可比性。
回归参照是冻结的Physical原版；本轮不撤销、不掩盖历史Physical Numerical Waiver。D诊断采用参考序列teacher forcing，记录的是参考上下文的逐步Top-1变化；数值失败后不额外声称已完成自由生成质量测试。
D修改Attention数值执行路径，数值失败版本不能继承历史Accuracy。即使某候选小样本答案不变，也仅能证明该回归集合；没有独立质量证据时，不将其效率与历史全量Accuracy拼成同一论文工作点。
这些计时仅适用于本3090、原30输入及固定32输出协议，不替代正常EOS质量实验、不代表不同分辨率/长度的泛化速度。

## 漂移与计时审计

已发现Native探针>5%告警 0 条；全部保留并列于performance_warnings.json。计时不删除慢请求、不重设基线、不修改GPU功耗/频率或系统配置。

Qwen调度顺序差异已记录：全部30输入精确回归先完成，实际Audit Mode在Session1之后、Session2之前运行。Session1只有在Audit Mode PASS后纳入最终验证，未为选择更快成绩重跑。
计时沿用冻结边界，包含方法专用初始化、Vision、注入、Prefill、31步Decode与Context恢复；H2D/预处理/写盘和最终缓存元数据检查/对象释放不在边界内，所有条件一致。Profiler属于额外审计请求，不能将其墙钟耗时纳入正式数据。Kernel数和同步数为固定中等输入的诊断，不能简单相加推断墙钟收益。

## 固定中等输入Profiler实测

|Backbone|Campaign|Method|阶段|Kernel数|CUDA同步API数|isin|cat|index_select|copy|CPU local_scalar|
|---|---|---|---|---|---|---|---|---|---|---|
|Qwen|refinement_E|Native|A|76223|187|0|4727|34|14594|57|
|Qwen|refinement_E|SVH|A|78064|3100|0|2075|1085|16845|1021|
|Qwen|refinement_E|SVH|B|75152|1132|0|2075|1045|15841|57|
|Qwen|refinement_E|SVH|E|73354|233|0|2075|1045|14942|57|
|Qwen|primary|Native|A|76223|187|0|4727|34|14594|57|
|Qwen|primary|SVH-R|A|100049|7821|1152|3237|93|24996|4698|
|Qwen|primary|SVH-R|B|69356|188|0|2085|93|15098|65|
|Qwen|primary|SVH-R|C|69302|174|0|2085|93|15030|65|
|Qwen|primary|SVH|A|78064|3100|0|2075|1085|16845|1021|
|Qwen|primary|SVH|B|75152|1132|0|2075|1045|15841|57|
|Qwen|primary|SVH|C|75120|1132|0|2075|1045|15809|57|
|InternVL|refinement_E|Native|A|44542|67|0|3562|32|7481|64|
|InternVL|refinement_E|SVH|A|41745|68|0|3101|100|7003|65|
|InternVL|refinement_E|SVH|B|41438|70|0|3070|69|6913|65|
|InternVL|refinement_E|SVH|E|41470|70|0|1582|69|8433|65|
|InternVL|primary|Native|A|44542|67|0|3562|32|7481|64|
|InternVL|primary|SVH-R|A|44526|68|0|3574|64|7039|98|
|InternVL|primary|SVH-R|B|44219|70|0|3543|64|6918|67|
|InternVL|primary|SVH-R|C|44249|70|0|2055|64|8436|67|
|InternVL|primary|SVH|A|41745|68|0|3101|100|7003|65|
|InternVL|primary|SVH|B|41438|70|0|3070|69|6913|65|
|LLaVA|primary|ApET+SVH|A|63269|1034|0|2828|98|10995|6|
|LLaVA|primary|ApET+SVH|B|51015|103|0|1880|98|7275|5|
|LLaVA|primary|FastV+SVH|A|50121|66|0|1836|107|7099|3|
|LLaVA|primary|FastV+SVH|B|50105|22|0|1836|104|7057|2|
|LLaVA|primary|Native|A|53639|3|0|4065|33|5507|0|
|LLaVA|primary|SparseVLM+SVH|A|51100|134|0|1857|108|7165|39|
|LLaVA|primary|SparseVLM+SVH|B|51032|74|0|1857|108|7125|22|
|LLaVA|primary|V2Drop+SVH|A|50819|78|4|1839|225|7123|16|
|LLaVA|primary|V2Drop+SVH|B|50479|48|4|1839|115|7097|12|

## 文件与资源状态

服务器持久化交付目录：/home/featurize/work/ATE/vlm_research/results/svh_cross_model_physical_optimization/
本地SSD完整工作区：/home/featurize/ate-experiments/svh-cross-opt-20261113/
源码快照、实际修改diff、模型Revision回执、独立适配包、命令、SQLite检查点、逐题Token与输入SHA、数值诊断、监控和完整Profiler均保留。SHA256SUMS与evidence_manifest.json说明文件完整性。无模型权重或图片复制到结果目录；没有推送GitHub。

最终释放状态以resource_release.json为准。未完成独立验证及持久盘SHA复核前，不标记最终PASS。

最终独立验收：PASS_WITH_REJECTED_CANDIDATES；正式请求 1200 个。GPU 推理进程已退出，资源释放检查通过。失败数值候选未进入正式表。持久盘文件完整性以结果目录外的 svh_cross_model_physical_optimization_delivery.json 回执为准。
