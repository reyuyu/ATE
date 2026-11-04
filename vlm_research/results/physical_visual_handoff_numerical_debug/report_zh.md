# Physical Visual Handoff：数值诊断、修复与真实微基准

**范围：仅D1/P1/P2；未扩大到60/240题。历史PARTIAL原封不动。**

D1不是原生Full：P−1及Decode全层禁止直接读Image，R只在截止点之前读Image（LLaVA进入L20、Qwen进入L22后关闭）。P1与D1使用同一数学依赖图，物理删除后层视觉Hidden行；P2在P1基础上物理压缩视觉KV。

## 1. 已定位事实

LLaVA `LogicalObj__03567478` 三次确定性重复均为D1=No、P1=Yes：密集Yes/No精确平局，物理Yes高0.015625。共享首步最大Logits误差0.015625、最大逐文本行Hidden相对RMSE 0.002343478；原数值门槛通过，但Argmax/Greedy失败，不能整体PASS。
两路径首Token分叉后各自前缀不同，后续6.03125的Logits差仅作为分叉后果，不作为同输入数值误差。
历史43题已逐题重查首步：接近决策边界的保守风险计数为1/43（D1 top1–top2 margin ≤ 2×逐词表最大误差；不是新增Accuracy损害计数）。

在同一份真实层入口Hidden上，两模型的Layer Input和RMSNorm均精确一致，首处不一致均为Q投影；K/V也有舍入差异。Qwen对应L22，LLaVA对应L20。实际物理入口的Hidden、Mask、原始位置及RoPE cos/sin均与密集文本切片严格一致，禁止视觉连接实测概率为0。
这些证据支持低精度GEMM形状/归约差异为首要误差来源，不支持位置重编号、错误Mask或DeepStack漏执行的解释。只能对本轮已审计样本作此判断。
算子表中Masked Score含causal finfo.min哨兵，全局FP32归一化RMSE不可解释，已在根目录表中置空并保留原诊断值；该项看max_abs和Mask/合法连接实测，不把0误解为逐位一致。层Hidden门槛仍使用最大逐文本行RMSE，不使用算子表的全局RMSE。

## 2. 修复、三类验收与原生漂移

所有阈值保持历史值不变。局部FP32控制只改变单个层内算子的累加，不把整个8B模型转为FP32。候选按冻结顺序测试，失败不删除。
`fp32_linears`/`fp32_linears_attention`在截止点后的投影/MLP和LM Head临时FP32计算，回转原激活dtype；后者还改变QK/AV累加。`canonical_tiles`保持原FP16/BF16，在后层和LM Head使用固定32行投影Tile，并把严格断开的文本/视觉Attention子图分开计算，使Dense/Physical文本内核输入形状一致；最多31行零Padding，不重建被裁剪视觉行。
canonical方案的密集参考仍计算视觉行，物理路径不计算这些行；未改变权重、数学合法边、原始位置或写入截止点。KV Cache按逐层原始逻辑位置映射，不以紧凑张量长度充当RoPE位置。

|模型|选择的候选|小回归G2|新题数|原生D1→修复D1的Greedy保持|
|---|---|---|---:|---|
|llava|fp32_linears_attention|PASS|10|False|
|qwen|canonical_tiles|PASS|10|True|

**最终原生等价判定仍未通过。** LLaVA修复D1把原生D1的No改为Yes；Qwen六道Smoke虽然答案保持，但原生D1→canonical D1的最大逐行Hidden RMSE超出原0.04阈值。因此匹配内核内部通过，不等于原生实现等价，全部速度统一归为`UNVALIDATED_PERFORMANCE`。
上表G2针对**同修复内核的D1/P1/P2**，不是与未经修改的原生密集路径逐位一致。原生→修复的漂移单独保存，若首Token或答案变化，不能宣称恢复了所有原生答案，也不能以修复参考替换历史PARTIAL。

|模型|比较阶段|最大Logits绝对误差|最大逐行Hidden RMSE|Greedy一致/比较数|
|---|---|---:|---:|---:|
|llava|原生失败重复|0.015625|0.002343478|0/3|
|llava|修复后独立回归|0.03125|0.003206178|34/34|
|llava|原生D1→修复D1|0.01953125|0.002287252|6/7|
|qwen|原生失败重复|0.5|0.06107337|3/3|
|qwen|修复后独立回归|0.625|0|32/32|
|qwen|原生D1→修复D1|0.6054688|0.07472774|6/6|

Qwen同内核D1/P1的Prefill文本Hidden及各共享步Logits全部精确一致；P2移除早层视觉KV后改变了Decode注意力矩阵形状，最大Logits差0.625，但仍满足冻结的组合atol/rtol门槛、Greedy一致。不能把P1的精确一致扩大到P2或原生参考。
G1：CPU小模型FP32数学等价对照、实际入口/Mask/位置对齐与目标概率审计提供支持。G2：数值和Greedy分别记录，并检查P2。G3：仅10道输入无关新题与原Smoke的小样本风险检查，不估计总体性能或等价性。

## 3. 真实KV与物理执行

逐层记录真实K/V `untyped_storage().nbytes()`、原始Key位置和物理长度，而非只记录逻辑shape。压缩前张量采用Weakref审计；不得存在旧全量KV的存活引用。P2每层均排除原始视觉位置，原始逻辑长度仍用于下一步位置。详细记录见`kv_memory_validation.csv`及各模型`kv_audit.json`。

## 4. 实测GPU性能

每模型一张RTX3090，2次Warmup、5次实测；16/64为总预测位置，分别包含15/63个增量Decode步骤。固定长度微基准可越过EOS，仅用于工程计时，质量回归仍正常EOS。Vision/Prefill/Decode用CUDA Event，E2E为同步墙钟。不同GPU上的两个模型不可直接作速度优劣比较。
下表为均值；完整标准差和每次结果在CSV。`UNVALIDATED_PERFORMANCE`不得解释为通过正确性验证的加速。

|模型|内核|预测位置数|路径|Prefill ms|Decode ms/token|E2E ms|KV MiB|状态|
|---|---|---:|---|---:|---:|---:|---:|---|
|llava|native|16|D1|212.93|121.60|2086.25|313.50|UNVALIDATED_PERFORMANCE|
|llava|native|16|P1|184.37|122.94|2077.94|205.50|UNVALIDATED_PERFORMANCE|
|llava|native|16|P2|201.58|122.66|2091.26|25.50|UNVALIDATED_PERFORMANCE|
|llava|native|64|D1|216.88|122.39|7990.27|337.50|UNVALIDATED_PERFORMANCE|
|llava|native|64|P1|184.14|123.17|8007.70|229.50|UNVALIDATED_PERFORMANCE|
|llava|native|64|P2|200.80|122.14|7957.83|49.50|UNVALIDATED_PERFORMANCE|
|llava|fp32_linears_attention|16|D1|366.21|145.34|2595.59|313.50|UNVALIDATED_PERFORMANCE|
|llava|fp32_linears_attention|16|P1|204.96|145.37|2434.74|205.50|UNVALIDATED_PERFORMANCE|
|llava|fp32_linears_attention|16|P2|223.19|144.87|2445.54|25.50|UNVALIDATED_PERFORMANCE|
|llava|fp32_linears_attention|64|D1|367.67|145.37|9588.51|337.50|UNVALIDATED_PERFORMANCE|
|llava|fp32_linears_attention|64|P1|205.00|145.42|9428.78|229.50|UNVALIDATED_PERFORMANCE|
|llava|fp32_linears_attention|64|P2|223.52|145.32|9441.06|49.50|UNVALIDATED_PERFORMANCE|
|qwen|native|16|D1|194.11|150.50|2508.37|28.27|UNVALIDATED_PERFORMANCE|
|qwen|native|16|P1|179.88|150.53|2494.62|19.52|UNVALIDATED_PERFORMANCE|
|qwen|native|16|P2|200.21|150.16|2509.28|5.77|UNVALIDATED_PERFORMANCE|
|qwen|native|64|D1|195.40|150.84|9768.68|35.02|UNVALIDATED_PERFORMANCE|
|qwen|native|64|P1|179.97|150.61|9738.55|26.27|UNVALIDATED_PERFORMANCE|
|qwen|native|64|P2|199.95|150.24|9735.57|12.52|UNVALIDATED_PERFORMANCE|
|qwen|canonical_tiles|16|D1|340.81|189.96|3247.30|28.27|UNVALIDATED_PERFORMANCE|
|qwen|canonical_tiles|16|P1|225.68|188.58|3111.54|19.52|UNVALIDATED_PERFORMANCE|
|qwen|canonical_tiles|16|P2|246.38|188.46|3130.30|5.77|UNVALIDATED_PERFORMANCE|
|qwen|canonical_tiles|64|D1|341.50|190.30|12400.83|35.02|UNVALIDATED_PERFORMANCE|
|qwen|canonical_tiles|64|P1|225.82|188.86|12194.60|26.27|UNVALIDATED_PERFORMANCE|
|qwen|canonical_tiles|64|P2|246.05|188.60|12198.71|12.52|UNVALIDATED_PERFORMANCE|

需要同时看匹配修复内核的D1/P1/P2和原生D1，避免用较慢修复参考夸大收益。当前实现保留Python Mask、位置及缓存审计开销，不是已优化部署内核；峰值allocated/reserved均实测，不能以KV节省直接代替总显存或时延收益。
实测结论：P1缩短了Language Prefill，但没有可靠Decode提速；稳定内核增加的成本使端到端仍慢于原生D1，未建立原生等价的真实加速。P2显著降低真实KV存储，压缩本身也增加Prefill耗时。

## 5. 结论边界与下一步

Qwen/LLaVA首个数值差异均已定位到投影GEMM而非位置/Mask。LLaVA近乎平局分叉稳定复现，说明小误差门槛通过不保证Argmax一致。修复是否足够进入更大质量验证，必须结合上表G2、原生漂移和真实耗时，不因为KV确实压缩就自动推荐扩大。
**当前不建议作为原生等价加速算法进入60/240题。** 两模型与原生参考的严格总体G2仍未通过；下一步优先解决匹配内核与原生密集参考间的舍入漂移及修复开销。如果后续接受近似实现风险，应先另行预注册小规模质量协议，而不是把本轮匹配内核的PASS换成原生PASS。本轮不自动执行。

本轮不涉及P4/早层Attention跳算/Question-Only，不搜索新截止点、不训练、不清理旧数据。CPU、启动环境及API失败记录均保留；修复后的源码重新提交冻结，未在运行中改写Worker。

## 6. 执行与产物

冻结正式源码Commit：`e955335e166e2309730fa5710e6d86821064a836`；原型基线：`e4272ec80fbf54579d4dc10d1d057a5302aee0b4`。
最终冻结版本Worker墙钟：LLaVA 541.78秒，Qwen 666.46秒；另有前序诊断、失败启动及开发整理时间，不能把Worker时长当作全任务总时间。
分支：`codex/physical-visual-handoff-numerical-debug`。最终结果Commit由Git提交日志给出，避免自引用哈希。
重现入口：`vlm_research/physical_handoff_debug/README.md`。未覆盖历史结果及源码；新结果全部位于`/data`。GPU2/3任务完成释放，GPU0其他任务不终止；资源快照见`resource_audit.json`。
