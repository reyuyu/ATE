# InternVL3-SVH-R L20数值诊断与最小修复

**最终状态：PARTIAL_REPAIR。已确认本例K/V差异与321→65行的计算形状及实际CUDA执行路径变化有关；临时补零能使K/V逐位恢复，但两项局部修复均不能使完整R0通过原数值门槛。不能启动SVH-R正式搜索。**

## 核心结论

1. **形状原因有可复核证据。** 相同原始文本Hidden、Norm输出、Q/K/V输入、权重与Bias，在BF16、同一原生Linear模块上：Dense321行与补零321行的真实文本输出，Q/K/V均torch.equal；65行紧凑计算的Q仍一致，K最大差0.0625、V最大差0.03125。A3的256个非文本行严格为零，未复制原图像Hidden。
2. **补零恢复K/V，不等于恢复整个模型。** C1只修复L20 Prefill K/V，新的首个差异是L20 o_proj；C2在C1基础上只修复此o_proj，新的首个差异是L20 down_proj。没有移动到L21，因为L20仍存在其他形状敏感算子。
3. **Q一致是该输入上的观测，不是普遍保证。** 原生Q输出宽3584、K/V输出宽512，实际CUDA Kernel选择不同。Profiler确认紧凑K/V启动一个GEMM和额外cublasLt::splitKreduce_kernel（GEMM grid Z=5）；Dense/补零K/V使用同一单条sliced1x2 Kernel路径。Q的Dense/紧凑Kernel也不相同，但没有相应独立Split-K归约事件，本例输出相同。证据支持形状影响GEMM累加路径；没有反汇编或强制Kernel对照，不能声称证明了具体哪条舍入指令。详见l20_projection_inputs.json及服务器原始Profiler Trace。
4. **新增计算与延迟真实存在。** C1多1.879048192 GFLOPs的L20 Prefill K/V乘加；C2再多6.576668672 GFLOPs的O投影，累计多8.455716864 GFLOPs。单个补零Hidden为2.194336 MiB；K/V局部实测峰值增量从1.126953 MiB到3.876953 MiB。没有完整请求E2E或加速结论。

## 环境、输入与证据保护

GPU：GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e，RTX3090 24GB；模型OpenGVLab/InternVL3-8B，Revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9。Torch2.6.0+cu124、Transformers4.49.0、BF16/Eager、Batch1/Greedy、448单图块/256视觉Token、logits_to_keep=1、质量诊断max_new_tokens=16、原生EOS。未切换精度、Attention Backend或reduced-precision reduction标志，未训练、修改权重或改变退出层。

仅使用上轮冻结问题201337043，完整实际生成3 Token（含EOS），1次Prefill+2次真实Decode。图片、Prompt、Token IDs与BF16 pixels SHA绑定与上轮一致，见sample_binding.json。没有为了数值结果更换问题或增加输出上限。

父Commit：9602d93869d4ccd11986573369d8fecc84b80f78。完整检查并保留上轮diagnosis、accumulation、逐层差异和独立Logits复算CSV。CPU独立审计确认父文件和冻结源码SHA均未变化。本轮原件使用AB_v002、B_v003、C1_v001、C2_v001、cost_installed_v001等唯一目录，以原子no-clobber写入和SHA收据绑定；旧文件存在即拒绝写入，未覆盖原始证据。

## A：严格投影对照

| 比较 | Q | K | V |
|---|---|---|---|
| A1 Dense321 vs A2 Compact65 | torch.equal | MaxAbs0.0625，15126元素不同 | MaxAbs0.03125，15639元素不同 |
| A1 Dense321 vs A3 ZeroPad321 | torch.equal | torch.equal | torch.equal |
| A2 Compact65 vs A3 ZeroPad321 | torch.equal | MaxAbs0.0625 | MaxAbs0.03125 |

所有结果保存完整原始BF16 Tensor，严格相等直接在BF16上判断；仅误差统计转换Float64，不改变模型计算。Tensor SHA、Shape、Stride、Contiguous、权重/Bias SHA和Kernel记录见l20_projection_inputs.json；9个比较的Max/Mean/RelRMS、元素数和SHA见l20_projection_comparison.csv。A1/A3原生模块、形状、Stride和实际Kernel路径匹配。

## B：独立逐算子回放

实际Qwen2没有Q/K Norm。RoPE使用原始逻辑位置。Attention按原始逻辑文本Key对齐，只比较共同合法且未Mask的连接；非文本Image列仍被禁止。独立回放的Norm、QKV、O、PostNorm、Gate/Up/Down及最终Hidden均与实际捕获的模块输出逐位相同。

| 算子 | torch.equal | MaxAbs | MeanAbs | RelativeRMS |
|---|---|---:|---:|---:|
|hidden_input|True|0|0|0|
|input_norm|True|0|0|0|
|q_proj|True|0|0|0|
|k_proj|False|0.0625|0.0020118336|0.0032361332|
|v_proj|False|0.03125|0.0013912219|0.0032332789|
|rope_q|True|0|0|0|
|rope_k|False|0.0625|0.0021112776|0.0033452116|
|attention_scores|False|0.078125|0.00847829|0.004258608|
|masked_scores|False|0.078125|0.00847829|0.004258608|
|softmax|False|0.01953125|0.0001402814|0.0060134513|
|AV|False|0.0703125|0.0014834524|0.0094096544|
|o_proj_input|False|0.0703125|0.0014834524|0.0094096544|
|o_proj|False|0.25|0.0021082667|0.00991575|
|attention_residual|False|0.25|0.0021058342|0.00010766146|
|post_norm|False|0.0625|0.0014797306|0.0039805714|
|gate_proj|False|0.0390625|0.0021161868|0.0037149088|
|up_proj|False|0.0390625|0.0022204733|0.0046183841|
|mlp_product|False|0.25|0.00081535346|0.0062811649|
|down_proj|False|0.25|0.0030024323|0.0062719725|
|mlp_residual|False|0.5|0.0038067922|0.0001765324|
|layer_hidden|False|0.5|0.0038067922|0.0001765324|

第一差异是K投影，不是Norm或Q。其后Scores最大差0.078125、Softmax0.01953125、AV0.0703125、O0.25，L20最终Hidden0.5；不同算子的尺度不同，不能把MaxAbs序列简单视为单调放大。后续深层与自回归进一步改变完整词表Logits。

审计中修正了一个回放错误：最初Dense前置Hook在功能Mask Hook之前捕获到了原生Mask，因此B_v002回放未匹配实际模型。B_v003只修正独立回放，使用冻结Dense-R实际生效Mask，并验证共享Mask及全部记录输出严格匹配；**没有修改模型推理Mask或协议**。失败记录保留，不混入有效B结论。

## C/D：两项冻结修复与完整R0结果

C1在实验A/B成立后启用，仅L20 Prefill的K/V输入临时补零到原始长度，输出只提取真实文本位置。C2在C1失败后依据新的首次分歧独立冻结，仅追加同层o_proj补零。两个候选使用相同adapter及真实Cache；所有Norm、MLP、Residual、位置、RoPE和Mask均未更改。

| 条件 | Prefill首次分歧算子 | 全步骤最大Logits Abs | 最大Mean Logits Abs | 最大Hidden RelRMS | 原门槛 |
|---|---|---:|---:|---:|---|
|C0_unrepaired|k_proj|1.51562|0.470425|—|未通过|
|C1_v001|o_proj|1.25|0.455521|0.09804393602373751|未通过|
|C2_v001|down_proj|3|1.1693|0.15607615239608674|未通过|

原门槛始终为MaxLogits≤0.125、MeanLogits≤0.01、HiddenRelRMS≤0.015625，逐步Top1及完整Token一致、无NaN/Inf。C1、C2完整自由生成Token仍与Dense相同，但均数值失败；C2最终误差更大，不能说修复越多就越接近Dense。没有根据答案、Gold或Accuracy选择修复，没有沿用历史Waiver。

逐步Logits、28层共享Hidden及L20–L27全部捕获的Norm、Attention投影、Residual与MLP算子记录均保存，含同前缀Teacher replay和独立自由生成。CPU独立重新计算全部A/B比较、C1/C2每步Logits、共享Hidden和逐算子差异，确认结果。只完成首个冻结样本，不运行余下7题；R4/R8兼容诊断、Discovery、Holdout、正式质量和E2E@32均NOT_RUN。两候选之后停止，不尝试down_proj第三修复。

## Physical语义及真实KV

L20入口仍从321行真实删除到65行，原始图像Hidden未进入L20。补零仅用于指定投影，真实文本值与原始逻辑位置不变；投影后先提取真实文本行，再更新Compact Cache，补零行及其Bias结果不写入持久KV。

L2–L5的P−1/Decode Attention仍真实跳算，Norm/MLP/Residual执行。K为空，全部28层持久原始视觉KV为0；禁止连接概率为0，RoPE逻辑位置和独立Cache检查通过。继承上一版临时零Key Scratch宽度兼容路径，没有声称消除了所有屏蔽列计算。

C1/C2在3 Token请求结束后的真实KV Live与底层Storage Capacity均为 **3.140625 MiB**；与原R0相同。它们不是KV@32，不用理论容量替代测量。逐层真实Shape/dtype/Live/Capacity见kv_layout_C1/C2及kv_integrity.json。

## 局部计算成本

下表主计时来自projection_cost_installed_v001.csv：每组预热25次、轮换次序100次CUDA Event重复；固定同一份实际输入，计入补零分配、清零、scatter及gather，计时内无Tensor转CPU或全量审计。模块安装/恢复在局部Event区间之外。KV和O分开测量，**不能相加后冒充完整请求E2E**。

| 投影组 | 路径 | 均值ms | 中位数ms | 实测峰值增量MiB | 相对紧凑新增GFLOPs |
|---|---|---:|---:|---:|---:|
|KV|Dense|0.158935|0.153600|1.626953|0.000000|
|KV|Compact|0.121866|0.120832|1.126953|0.000000|
|KV|C1_v001|0.246395|0.242688|3.876953|1.879048|
|KV|C2_v001|0.241715|0.237568|3.876953|1.879048|
|O|Dense|0.348856|0.347136|2.944336|0.000000|
|O|Compact|0.137984|0.137216|0.444336|0.000000|
|O|C1_v001|0.137677|0.136192|0.444336|0.000000|
|O|C2_v001|0.390451|0.388096|5.444336|6.576669|

峰值增量包含投影输出、临时Tensor及算子Workspace，不等同于单个补零Tensor大小。每个补零Hidden2300928字节；C1的K和V各自临时创建，生命周期短且不持久。O修复局部峰值增量由0.444336到5.444336 MiB。FLOPs按乘加2 FLOPs、真实投影行数计算，不包括scatter、bias等逐元素操作，也不是实测硬件FLOP计数。

projection_cost.csv是A3共享一份补零输入的诊断计时；projection_cost_C1/C2包含每次局部调用重装adapter的额外CPU时间，完整保留但不作主成本表。本表专门排除了该安装边界差异，不能据此与旧完整E2E比较。只测一个冻结输入长度321/65，没有把单题成本外推到不同长度分布。

## 仍有的风险及后续前提

- 本例形状因果证据充分，但BF16、不同矩阵形状和后续GEMM仍可能产生不同数值路径；无法保证其他输入长度或回看集合一致。
- C2已出现下一个独立的形状敏感down_proj，继续逐算子补行可能显著恢复Dense计算量。本轮不越过两个候选限制、不进行Kernel搜索或大范围补行。
- **R0尚不具备进入SVH-R搜索的数值前提。** 本轮没有任何Accuracy改善结论，也没有完整E2E加速结论。
- 最终状态仅为PARTIAL_REPAIR，不是R0_NUMERICAL_PASS，更不是所有SVH-R配置通过。

## 文件、复现和GitHub

源码：/home/featurize/work/ATE-internvl3-svh-r-l20-numerical-diagnosis-3090/vlm_research/internvl3_svh_r_l20_numerical_diagnosis

结果：/home/featurize/work/ATE-internvl3-svh-r-l20-numerical-diagnosis-3090/vlm_research/results/internvl3_svh_r_l20_numerical_diagnosis_3090

SSD原始Tensor与日志：/home/featurize/ate-runs/internvl3-svh-r-l20-numerical-diagnosis-3090

持久化完整原始证据：/home/featurize/work/ATE-internvl3-svh-r-l20-numerical-diagnosis-3090/vlm_research/results/internvl3_svh_r_l20_numerical_diagnosis_3090/server_only/。约2.05GB原始文件仅保留在服务器，GitHub上传轻量SHA/Shape/Dtype收据、配置、比较CSV、源码及报告，不上传模型、图片或大Tensor。原始Tensor清单见raw_tensor_manifest.json及raw_tensor_receipts/；全文件范围与SHA见artifact_manifest.json、SHA256SUMS。

首次启动A_v001因环境记录缺少CUDA_VISIBLE_DEVICES在推理前退出；A_v002显式固定UUID重启；前述B回放修正另存v003。两次工程失败及修正均有独立源码、日志与绑定，没有覆盖或隐藏。

分支：[codex/internvl3-svh-r-l20-numerical-diagnosis-3090](https://github.com/reyuyu/ATE/tree/codex/internvl3-svh-r-l20-numerical-diagnosis-3090)。最终Commit和远端所有Blob核验收据保存在server_only/delivery_receipt.json，避免把Commit SHA递归写进自身文件。

GPU已释放，最终GPU无计算进程、显存约1MiB。保持服务器运行，不自动关闭实例，不启动SVH-R正式搜索或其他实验。
