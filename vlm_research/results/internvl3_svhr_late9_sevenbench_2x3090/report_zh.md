# InternVL3 SVH-R Late9 七项质量评测

状态：PARTIAL

| Benchmark | 配对覆盖/计划 | Native | 原SVH | Late9 | SVH−Native | Late9−Native | Late9−SVH | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GQA | 12578/12578 | 62.1561 | 53.1086 | 61.4327 | -9.0475 | -0.7235 | 8.3241 | FULL_COMPLETE |
| ScienceQA-IMG | 2017/2017 | 97.5211 | 97.4715 | 97.4219 | -0.0496 | -0.0992 | -0.0496 | FULL_COMPLETE |
| TextVQA | 5000/5000 | 66.3420 | 10.5980 | 56.0040 | -55.7440 | -10.3380 | 45.4060 | FULL_COMPLETE |
| POPE | 9000/9000 | 90.4444 | 90.2222 | 90.4444 | -0.2222 | 0.0000 | 0.2222 | FULL_COMPLETE |
| MME | 2372/2374 | — | — | — | — | — | — | PARTIAL |
| MMBench | 4377/4377 | 83.6735 | 83.4184 | 83.5884 | -0.2551 | -0.0850 | 0.1701 | FULL_COMPLETE |
| OCRBench | 3/1000 | — | — | — | — | — | — | PARTIAL |

MME为原始积分，OCRBench为原始0–1000分；其他为百分比。不同单位不混算。未完整项目的共同子集在另表及CSV中报告，不填充为全量Benchmark分数。

| 方法 | 持久原始视觉KV层数 | 视觉Hidden存活Block数 | 固定32 Token E2E/TTFT/TPOT |
|---|---:|---:|---|
| 原SVH | 0（0 MiB视觉数据） | 20 | 尚未测量 |
| Late9 | 9（4.5 MiB视觉数据） | 28 | 尚未测量 |

视觉数据大小来自全部逐请求实际K/V形状：2×4 KV Heads×256行×128维×2字节，每保留层0.5 MiB。Native为28层/14 MiB，Late9较Native少9.5 MiB（67.86%），但较原SVH多4.5 MiB；这不是总KV、分配器峰值或实测速率。详见visual_kv_payload_audit.json。
效率测试须在七项质量全部完成后执行。不能用自然EOS生成耗时或历史GPU测速替代本轮固定32 Token测试；Late9保留全部28层视觉Hidden，没有L20 Prefill早退收益。

## 配对差异

Image SHA256整簇配对Bootstrap 20,000次，seed=20261110，普通未校正95%CI。保留原生评分单位，TextVQA为软分变化而非二值纠错。

| Benchmark | 比较 | N | 差值 | 95%CI | 得分提高/降低 |
|---|---|---:|---:|---|---|
| GQA (FULL_COMPLETE) | physical_d−native | 12578 | -9.0475 | [-10.0444, -8.0664] | 401/1539 |
| GQA (FULL_COMPLETE) | late9−native | 12578 | -0.7235 | [-1.0888, -0.3498] | 263/354 |
| GQA (FULL_COMPLETE) | late9−physical_d | 12578 | +8.3241 | [+7.3966, +9.2598] | 1389/342 |
| ScienceQA-IMG (FULL_COMPLETE) | physical_d−native | 2017 | -0.0496 | [-0.2538, +0.1506] | 2/3 |
| ScienceQA-IMG (FULL_COMPLETE) | late9−native | 2017 | -0.0992 | [-0.3005, +0.0981] | 1/3 |
| ScienceQA-IMG (FULL_COMPLETE) | late9−physical_d | 2017 | -0.0496 | [-0.2551, +0.1510] | 2/3 |
| TextVQA (FULL_COMPLETE) | physical_d−native | 5000 | -55.7440 | [-57.2574, -54.2066] | 182/3102 |
| TextVQA (FULL_COMPLETE) | late9−native | 5000 | -10.3380 | [-11.4821, -9.2042] | 281/854 |
| TextVQA (FULL_COMPLETE) | late9−physical_d | 5000 | +45.4060 | [+43.9178, +46.9024] | 2560/170 |
| POPE (FULL_COMPLETE) | physical_d−native | 9000 | -0.2222 | [-0.5444, +0.0778] | 35/55 |
| POPE (FULL_COMPLETE) | late9−native | 9000 | +0.0000 | [-0.2444, +0.2444] | 29/29 |
| POPE (FULL_COMPLETE) | late9−physical_d | 9000 | +0.2222 | [-0.0556, +0.5111] | 47/27 |
| MME (PARTIAL) | physical_d−native | 2372 | -6.6157 | [-32.1751, +16.3655] | 6/10 |
| MME (PARTIAL) | late9−native | 2372 | -20.5340 | [-48.6802, +2.9276] | 3/11 |
| MME (PARTIAL) | late9−physical_d | 2372 | -13.9183 | [-35.3600, +2.2887] | 3/7 |
| MMBench (FULL_COMPLETE) | physical_d−native | 4377 | -0.2551 | [-0.6745, +0.0856] | 5/18 |
| MMBench (FULL_COMPLETE) | late9−native | 4377 | -0.0850 | [-0.4939, +0.2584] | 9/16 |
| MMBench (FULL_COMPLETE) | late9−physical_d | 4377 | +0.1701 | [-0.1688, +0.5141] | 10/4 |
| OCRBench (PARTIAL) | physical_d−native | 3 | -66.6667 | [-100.0000, +0.0000] | 0/2 |
| OCRBench (PARTIAL) | late9−native | 3 | +0.0000 | [+0.0000, +0.0000] | 0/0 |
| OCRBench (PARTIAL) | late9−physical_d | 3 | +66.6667 | [+0.0000, +100.0000] | 2/0 |

## 未完整项目共同子集

以下不是正式全量成绩；MME单位为原始积分，OCR子集为百分比而非0–1000总分。

| 项目/共同子集 | 题数 | Native | 原SVH | Late9 |
|---|---:|---:|---:|---:|
| MME Perception | 2112/2114 | 1695.7560 | 1699.1403 | 1692.3649 |
| MME Cognition | 260/260 | 625.7143 | 615.7143 | 608.5714 |
| OCRBench common_score | 3/1000 | 66.6667 | 0.0000 | 66.6667 |

## 十项研究问题

1. GQA Query与Non-Query分别比较；均值大幅恢复不等于逐题或完全恢复Native。
- GQA Query（6805题）：Late9−native=-1.1462pp，95%CI [-1.8044,-0.4981]；Late9−physical_d=+15.3564pp，95%CI [+13.7171,+17.0222]。CI跨0不等于等价；不能把均值恢复视为全部Query逐题恢复。
- GQA Non-Query（5773题）：Late9−native=-0.2252pp，95%CI [-0.5658,+0.1217]；Late9−physical_d=+0.0346pp，95%CI [-0.3340,+0.4164]。CI跨0不等于等价；不能把均值恢复视为全部Query逐题恢复。
2. TextVQA全5000题：Native/原SVH/Late9=66.3420/10.5980/56.0040。Late9−原SVH=+45.4060pp，95%CI [+43.9178,+46.9024]；软分提高/降低=2560/170题，0分变正分=2506题。按总体软分差计算恢复原损失的81.45%，不是二值纠错率。
3. ScienceQA、POPE、MME、MMBench：不能用净分掩盖逐题新增损害。
- ScienceQA-IMG（FULL_COMPLETE）：差值-0.0992，95%CI [-0.3005,+0.0981]；题目正确变错/错误变对=3/1。MMBench翻转为旋转记录层面，正式主分为Circular组；MME差值为原始积分。
- POPE（FULL_COMPLETE）：差值+0.0000，95%CI [-0.2444,+0.2444]；题目正确变错/错误变对=29/29。MMBench翻转为旋转记录层面，正式主分为Circular组；MME差值为原始积分。
- MME（PARTIAL）：差值-20.5340，95%CI [-48.6802,+2.9276]；题目正确变错/错误变对=11/3。MMBench翻转为旋转记录层面，正式主分为Circular组；MME差值为原始积分。
- MMBench（FULL_COMPLETE）：差值-0.0850，95%CI [-0.4939,+0.2584]；题目正确变错/错误变对=16/9。MMBench翻转为旋转记录层面，正式主分为Circular组；MME差值为原始积分。
4. OCRBench仅3/1000题三方严格配对（PARTIAL），不能给出正式0–1000总分，也不能判断全量OCR改善。
5. Late9相对原SVH的完整配对差值与20,000次图像簇Bootstrap区间见上表；缺失项目只对应共同子集，不能外推全量。
6. 9层回看持久视觉KV为4.5 MiB：较Native节省9.5 MiB，较原SVH增加4.5 MiB；总KV还含文本，按真实Token长度另见kv_layout.csv.gz。
7. 相对原SVH，视觉Hidden多存活8个Decoder Block，即每图多2048个视觉Token×Block计算行；实际Prefill额外毫秒尚未独立测量，不能用行数代替时间。
8. 七项质量未全部完成，因此按任务顺序不启动固定32 Token效率阶段。没有本轮可报告的E2E/TTFT/TPOT加速比，不能宣称Late9比Native快。
9. GQA/TextVQA/OCRBench复用逐记录原始SHA验证通过的历史Native与SVH；ScienceQA/POPE/MME/MMBench本轮重新生成两基线。缺原图的题目不进入三方法共同结果，保留BLOCKED清单。
10. 全部Late9质量仍带BF16历史数值豁免标签。Smoke严格一致只对已测输入成立，物理/Dense未获全域数值等价证明；原单tile、Image-only及暂停策略差异均须保留，不能冒充其他论文排行榜协议。


## 协议与复用

原SVH输入协议来自6c294aae，续跑结果来自3297559f；Late9物理引擎来自1407275c。旧代码和历史结果未修改。当前入口仅适配本地路径、GPU UUID、分片、严格复用和检查点。
GQA、TextVQA、OCRBench历史JSON含全部28层KV布局，重建后逐条匹配发布的原始记录SHA256；再核对本地输入与BF16 pixels后才复用。其他四项缺少可核验的原始逐记录SHA，因此Native/SVH重新配对生成。详见historical_reuse_audit.json与逐题generation_origin。逐题导出明确区分generation_gpu_uuid与input_verification_gpu_uuid；复用预测仍来自历史GPU，不冒充本轮生成。
BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0，448×448单图块、256视觉Token。各Benchmark保持原Prompt与生成上限，TextVQA没有OCR辅助输入。
数值标签继续为EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER，不宣称数值等价。两GPU各七任务Native/No-op/官方Chat严格一致；Late9物理布局、权限、RoPE、Norm/MLP与Cache独立性通过。Smoke的Dense/Physical误差完整保留，不能推广成所有输入数值一致。

## 覆盖与限制

本机MME的一张landmark图与冻结原件SHA不同，涉及同图两题；OCRBench本机HF版本与历史官方ZIP版本的绝大多数图片字节不同。Google Drive原件下载连接超时。未换图、重编码或改SHA，缺失请求完整保留并标记BLOCKED。
本轮续跑规则在GPU前冻结：非结构性长输出、空答、解释、重复、自然截断均保留计分；NaN/Inf/OOM、哈希、映射或KV结构错误停止Worker。该规则与旧v1暂停策略不同，不改变输入、评分或算法。
Late9与原SVH的收益不能解释为整模型仅需九层视觉处理，也不代表已加速。Prompt其他位置仍有视觉信息，且固定1tile协议不是作者多tile排行榜复现。

## 逐题与审计

per_question.csv.gz保存完整输出、Token IDs、Gold、评分、异常与绑定；kv_layout.csv.gz保存真实逐层形状和Live/Capacity。所有预测均独立CPU重评分，并核验原始JSON校验和、样本、Processor及实际KV长度。原始SQLite留server_only、不上传大缓存或权重。
依本轮任务要求仅本地提交独立分支，不修改main、不自动推送GitHub、不创建PR。

## 正式执行耗时与资源释放

- Worker0正式进程墙钟约1.920小时（包含加载、输入检查和复用；不是方法速度测量）。
- Worker1正式进程墙钟约1.933小时（包含加载、输入检查和复用；不是方法速度测量）。
本实验两Worker已退出，nvidia-smi计算进程列表中均已消失；进程退出前的残留allocator字节原样保存在release.json，不冒充提前归零。GPU释放审计保留当时其他GPU进程，不终止或修改它们。
