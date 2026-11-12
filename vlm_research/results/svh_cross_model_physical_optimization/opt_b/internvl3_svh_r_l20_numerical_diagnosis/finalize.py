"""Write the audited PARTIAL_REPAIR report; no GPU and no historical edits."""
from .common import *
def loadcsv(n):return list(csv.DictReader((OUT/n).open()))
def main():
 v=readj(OUT/"validation.json");assert v["status"]=="PARTIAL_REPAIR" and v["evidence_integrity"]=="PASS"
 a=readj(OUT/"l20_projection_inputs.json");summ=loadcsv("repair_summary.csv");ops=loadcsv("l20_operator_comparison.csv");cost=loadcsv("projection_cost_installed_v001.csv")
 nums="\n".join("|"+ "|".join([x["candidate"],x["first_prefill_operator"],f'{float(x["max_logits_abs"]):.6g}',f'{float(x["max_mean_logits_abs"]):.6g}',x.get("max_hidden_relative_RMS") or "—","未通过"])+"|" for x in summ)
 optable="\n".join("|"+ "|".join([x["operator"],x["torch_equal"],f'{float(x["max_abs"]):.8g}',f'{float(x["mean_abs"]):.8g}',f'{float(x["relative_RMS"]):.8g}'])+"|" for x in ops)
 ctable="\n".join("|"+ "|".join([x["group"],x["variant"],f'{float(x["CUDA_event_mean_ms"]):.6f}',f'{float(x["CUDA_event_median_ms"]):.6f}',f'{int(x["measured_peak_incremental_allocated_bytes"])/1048576:.6f}',f'{int(x["extra_FLOPs_vs_compact"])/1e9:.6f}'])+"|" for x in cost)
 report=f"""# InternVL3-SVH-R L20数值诊断与最小修复

**最终状态：PARTIAL_REPAIR。已确认本例K/V差异与321→65行的计算形状及实际CUDA执行路径变化有关；临时补零能使K/V逐位恢复，但两项局部修复均不能使完整R0通过原数值门槛。不能启动SVH-R正式搜索。**

## 核心结论

1. **形状原因有可复核证据。** 相同原始文本Hidden、Norm输出、Q/K/V输入、权重与Bias，在BF16、同一原生Linear模块上：Dense321行与补零321行的真实文本输出，Q/K/V均torch.equal；65行紧凑计算的Q仍一致，K最大差0.0625、V最大差0.03125。A3的256个非文本行严格为零，未复制原图像Hidden。
2. **补零恢复K/V，不等于恢复整个模型。** C1只修复L20 Prefill K/V，新的首个差异是L20 o_proj；C2在C1基础上只修复此o_proj，新的首个差异是L20 down_proj。没有移动到L21，因为L20仍存在其他形状敏感算子。
3. **Q一致是该输入上的观测，不是普遍保证。** 原生Q输出宽3584、K/V输出宽512，实际CUDA Kernel选择不同。Profiler确认紧凑K/V启动一个GEMM和额外cublasLt::splitKreduce_kernel（GEMM grid Z=5）；Dense/补零K/V使用同一单条sliced1x2 Kernel路径。Q的Dense/紧凑Kernel也不相同，但没有相应独立Split-K归约事件，本例输出相同。证据支持形状影响GEMM累加路径；没有反汇编或强制Kernel对照，不能声称证明了具体哪条舍入指令。详见l20_projection_inputs.json及服务器原始Profiler Trace。
4. **新增计算与延迟真实存在。** C1多1.879048192 GFLOPs的L20 Prefill K/V乘加；C2再多6.576668672 GFLOPs的O投影，累计多8.455716864 GFLOPs。单个补零Hidden为2.194336 MiB；K/V局部实测峰值增量从1.126953 MiB到3.876953 MiB。没有完整请求E2E或加速结论。

## 环境、输入与证据保护

GPU：{GPU}，RTX3090 24GB；模型OpenGVLab/InternVL3-8B，Revision {REV}。Torch2.6.0+cu124、Transformers4.49.0、BF16/Eager、Batch1/Greedy、448单图块/256视觉Token、logits_to_keep=1、质量诊断max_new_tokens=16、原生EOS。未切换精度、Attention Backend或reduced-precision reduction标志，未训练、修改权重或改变退出层。

仅使用上轮冻结问题201337043，完整实际生成3 Token（含EOS），1次Prefill+2次真实Decode。图片、Prompt、Token IDs与BF16 pixels SHA绑定与上轮一致，见sample_binding.json。没有为了数值结果更换问题或增加输出上限。

父Commit：{PARENT}。完整检查并保留上轮diagnosis、accumulation、逐层差异和独立Logits复算CSV。CPU独立审计确认父文件和冻结源码SHA均未变化。本轮原件使用AB_v002、B_v003、C1_v001、C2_v001、cost_installed_v001等唯一目录，以原子no-clobber写入和SHA收据绑定；旧文件存在即拒绝写入，未覆盖原始证据。

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
{optable}

第一差异是K投影，不是Norm或Q。其后Scores最大差0.078125、Softmax0.01953125、AV0.0703125、O0.25，L20最终Hidden0.5；不同算子的尺度不同，不能把MaxAbs序列简单视为单调放大。后续深层与自回归进一步改变完整词表Logits。

审计中修正了一个回放错误：最初Dense前置Hook在功能Mask Hook之前捕获到了原生Mask，因此B_v002回放未匹配实际模型。B_v003只修正独立回放，使用冻结Dense-R实际生效Mask，并验证共享Mask及全部记录输出严格匹配；**没有修改模型推理Mask或协议**。失败记录保留，不混入有效B结论。

## C/D：两项冻结修复与完整R0结果

C1在实验A/B成立后启用，仅L20 Prefill的K/V输入临时补零到原始长度，输出只提取真实文本位置。C2在C1失败后依据新的首次分歧独立冻结，仅追加同层o_proj补零。两个候选使用相同adapter及真实Cache；所有Norm、MLP、Residual、位置、RoPE和Mask均未更改。

| 条件 | Prefill首次分歧算子 | 全步骤最大Logits Abs | 最大Mean Logits Abs | 最大Hidden RelRMS | 原门槛 |
|---|---|---:|---:|---:|---|
{nums}

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
{ctable}

峰值增量包含投影输出、临时Tensor及算子Workspace，不等同于单个补零Tensor大小。每个补零Hidden2300928字节；C1的K和V各自临时创建，生命周期短且不持久。O修复局部峰值增量由0.444336到5.444336 MiB。FLOPs按乘加2 FLOPs、真实投影行数计算，不包括scatter、bias等逐元素操作，也不是实测硬件FLOP计数。

projection_cost.csv是A3共享一份补零输入的诊断计时；projection_cost_C1/C2包含每次局部调用重装adapter的额外CPU时间，完整保留但不作主成本表。本表专门排除了该安装边界差异，不能据此与旧完整E2E比较。只测一个冻结输入长度321/65，没有把单题成本外推到不同长度分布。

## 仍有的风险及后续前提

- 本例形状因果证据充分，但BF16、不同矩阵形状和后续GEMM仍可能产生不同数值路径；无法保证其他输入长度或回看集合一致。
- C2已出现下一个独立的形状敏感down_proj，继续逐算子补行可能显著恢复Dense计算量。本轮不越过两个候选限制、不进行Kernel搜索或大范围补行。
- **R0尚不具备进入SVH-R搜索的数值前提。** 本轮没有任何Accuracy改善结论，也没有完整E2E加速结论。
- 最终状态仅为PARTIAL_REPAIR，不是R0_NUMERICAL_PASS，更不是所有SVH-R配置通过。

## 文件、复现和GitHub

源码：{ROOT/PKG}

结果：{ROOT/REL}

SSD原始Tensor与日志：{RUN}

持久化完整原始证据：{ROOT/REL}/server_only/。约2.05GB原始文件仅保留在服务器，GitHub上传轻量SHA/Shape/Dtype收据、配置、比较CSV、源码及报告，不上传模型、图片或大Tensor。原始Tensor清单见raw_tensor_manifest.json及raw_tensor_receipts/；全文件范围与SHA见artifact_manifest.json、SHA256SUMS。

首次启动A_v001因环境记录缺少CUDA_VISIBLE_DEVICES在推理前退出；A_v002显式固定UUID重启；前述B回放修正另存v003。两次工程失败及修正均有独立源码、日志与绑定，没有覆盖或隐藏。

分支：[{BRANCH}](https://github.com/reyuyu/ATE/tree/{BRANCH})。最终Commit和远端所有Blob核验收据保存在server_only/delivery_receipt.json，避免把Commit SHA递归写进自身文件。

GPU已释放，最终GPU无计算进程、显存约1MiB。保持服务器运行，不自动关闭实例，不启动SVH-R正式搜索或其他实验。
"""
 atomic_bytes(OUT/"report_zh.md",report.encode())
 writej(OUT/"source_manifest_final.json",{"parent_commit":PARENT,"original_frozen_manifest_sha256":sha(OUT/"source_manifest.json"),"original_protocol_sha256":sha(OUT/"protocol.json"),"all_added_source_sha256":{str(p.relative_to(RUN/"code")):sha(p) for p in sorted((RUN/"code"/PKG).glob("*.py"))},"phases":{"A2":sha(OUT/"execution_attempt_AB_v002.json"),"B3":sha(OUT/"execution_attempt_B_v003.json"),"C1":sha(OUT/"C1_v001_frozen.json"),"C2":sha(OUT/"C2_v001_frozen.json"),"cost":sha(OUT/"cost_plan_installed_v001.json")},"model_unchanged":True})
 readme=f"""# InternVL3-SVH-R L20 numerical diagnosis

Final status: PARTIAL_REPAIR. No search, no quality scoring, no E2E32 benchmark.
Actual parent code inherited from {PARENT}. Paths in common.py point to this recorded server.
The atomic evidence writer refuses to overwrite old files. Reproduction needs a NEW run directory,
new source binding and same frozen image/prompt/input SHA/model/UUID, never an in-place rerun.

CPU tests: python -m vlm_research.internvl3_svh_r_l20_numerical_diagnosis.cpu_tests
A: diagnose_ab_v002.py (A results valid; initial B mask capture replay failed and is preserved)
B: complete_b_v003.py (correct effective frozen Dense mask; exact native operator reproduction)
C1/C2: repair_trial.py reads immutable candidate-specific frozen JSON, stops after first failed
sample and does not enter remaining seven inputs. adapter.py only pads specified prefill projections.
Cost: projection_cost_installed.py, existing frozen candidates only, local CUDA Events.
Audit: final_audit.py is CPU only; map_location=cpu, no CUDA/model loading.
Final report: finalize.py. Publication: publish.py (explicit scope, no force/main/PR).

GPU processes used CUDA_VISIBLE_DEVICES={GPU}, ATE_SEVEN_ROOT=/home/featurize/work/ATE-internvl3-svh-stop-policy-v2-3090,
ATE_SEVEN_RUN=/home/featurize/ate-runs/internvl3-svh-stop-policy-v2-3090, PYTHONPATH={RUN}/code,
Python /home/featurize/.venvs/ate-frozen/bin/python.
Previous missing visibility variable failed before generation, saved as AB_failure_v001.json.
Original mask capture error affected independent replay only, corrected in separately saved B_v003.
This package preserves actual successful and failed analysis versions; do not mistake earlier replay
failure or partial projection repair for full R0 numerical acceptance. Raw BF16 Tensor files stay server-only.
"""
 atomic_bytes(RUN/"code"/PKG/"README.md",readme.encode())
 print("PARTIAL_REPAIR_REPORT_WRITTEN",flush=True)
if __name__=="__main__":main()
