# LLaVA-1.5-7B-SVH-R：相同128题 TextVQA Image-only
LLaVA Native 为 50.156250%，SVH-R 为 43.515625%，SVH-R相对Native变化 -6.640625 个百分点。图片SHA簇配对Bootstrap普通95%CI：[-11.328125, -2.500000] pp。

| 方法 | 题数 | 官方十答案软分 Accuracy | 分数和 |
|---|---:|---:|---:|
| Native | 128 | 50.156250% | 64.200000/128 |
| llava-SVH-R | 128 | 43.515625% | 55.700000/128 |

## 实验含义
严格复用此前InternVL小测试集的128道题、图片字节/SHA、题目顺序与十答案Gold，但LLaVA有其自身冻结Prompt、Tokenizer与图像预处理。该128题已用于探索，不能视为独立确认集。核心结论是本轮同机LLaVA内部的Native配对差值，不把跨模型分数差异全部归因于算法。

模型llava-hf/llava-1.5-7b-hf，Revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9；FP16/Eager，Torch2.6.0+cu124，Transformers4.49.0，Greedy，Batch1，logits_to_keep=1，use_cache=True，max_new_tokens=32，原生EOS。原HF CLIP shortest-edge resize336、center-crop336（未新增square-pad），原生HF USER/image/ASSISTANT chat，问题后追加“Answer the question using a single word or phrase.”。Image-only，无OCR文字。Gold仅评分读取，不进模型输入。

## 冻结SVH-R
逐字节复用 5ee2a0dbe78df336229b55fed14a4a6d0d880bed 的MiddleWriter及依赖，不修改冻结算法。P−1和真实Decode仅L11–L24可读原始Image Key；L20起R→Image禁止；L25入口真正删除576行视觉Hidden。L2–L5 P−1的Q/Attention/o_proj输出计算被跳过，其他Prefill Query正常计算；固定历史实现的这些层Prefill K/V仍对全Prefill行计算，真实Decode全部QKV/Attention/o_proj不执行，且这些层无持久KV。所有层Norm、MLP与Residual保留。其余层只在L11–L24持久保存原始视觉KV，逻辑RoPE位置不重编号。

## 完整性与正确性
GPU GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27，Native和SVH-R同卡同输入，各128次正式生成，共256次；最终正式Worker加载模型一次，含初始Smoke失败和独立诊断共3次GPU模型加载。正式生成前3题Smoke与主集不重叠，覆盖短中长输入。Native与No-op有效末位Logits严格torch.equal，原生HF generate与共享Native循环完整Token一致。Dense/Physical通过预先冻结的历史FP16门槛（Logits allclose atol0.10 rtol0.003，Hidden relativeRMSE<=0.01、cosine error<=1e-4），自然EOS有效生成步通过上述数值阈值，完整自然EOS生成Tokens一致；另9步强制续跑只用于真实缓存/结构检查。逐层隐藏行、真实投影调用、Mask禁图概率、RoPE和实际K/V布局均核验。smoke_numerical.csv保存实测差异，不宣称所有输入数值逐位等同。

初始统一9步数值门槛曾失败：长Smoke题正常2个Token已EOS，强制EOS后第3步Logit MaxAbs0.104492、MeanAbs0.015672，原allclose条件未通过。该失败、旧协议、旧源码和诊断均保留于initial_attempt和smoke_diagnostic_v1；并未改写为通过。本轮用户请求的是正常EOS质量评测，GPU正式运行前建立范围明确的v2协议：有效生成步保持原数值门槛（3题最大有效差0.020508），EOS后合成步骤保留非有限、位置、缓存、物理结构检查，并记录数值失败。未改变算法、精度、Backend或抬高数值阈值，也不宣称固定长度生成的数值门槛已通过。

CPU32层float64 Mock内部沿用官方FP32 Softmax，初始1e-10断言观察到3.31e-8 Decode Logit舍入差；保留初始诊断，正式GPU前将CPU浮点容差明确冻结为1e-6（Hidden差0），未修改算法或正式GPU既定门槛。

## 配对变化与异常
软分上升2题、下降12题、不变114题；首Token变化19题、完整Tokens变化41题、文本变化41题。完整正确→非完整正确8题，反向1题。全部损害案例见harm_cases.csv，未剔除错误答案、重复或截断。
Native空输出0、截断0、重复标记1；SVH-R空输出1、截断1、重复标记3。重复标记仅为Token三元组启发式，不自动认定异常或停止。
长度P50/P95/最大：Native 3.0/7.6499999999999915/12；SVH-R 3.0/8.299999999999983/32。

Bootstrap：按128个Image SHA整簇配对重采样20000次，seed20261112，普通未调整95%CI。CI跨0时不能解释为等价或独立非劣效证明。小样本仅供探索，不调整层或Prompt。

## 交付与范围
本地SSD完整检查点：/home/featurize/ate-runs/llava-svh-r-textvqa128-3090/predictions.sqlite
持久盘轻量结果与完整原始检查点：/home/featurize/work/ATE-llava-svh-r-textvqa128-3090/vlm_research/results/llava_svh_r_textvqa128_3090/
独立复现代码：/home/featurize/work/ATE-llava-svh-r-textvqa128-3090/vlm_research/llava_svh_r_textvqa128/
独立Git分支：codex/llava-svh-r-textvqa128-3090。artifact_manifest.json区分GitHub上传文件与仅服务器原件；SHA256SUMS覆盖轻量证据与实现来源。模型、图片、prepared Tensor与密钥不上传。
未测速度，缓存为自然EOS实际终点，不能当作KV@32或性能节省结果。内部退出快照仍有32MiB CUDA分配；Worker进程退出后，外部nvidia-smi核验GPU仅1MiB、无计算进程，相关记录均保留。最终进程状态以server_only/git_publish_receipt.json为准。未执行其他Benchmark、训练或新候选。
