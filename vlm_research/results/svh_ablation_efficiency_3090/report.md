# ATE Section 5.5：Component Ablation 与生成长度敏感性实测报告

生成时间：2026-11-13T04:44:45.035367+08:00。执行与正确性：PASS；效率整体状态：**COMPLETE_WITH_PERFORMANCE_WARNINGS**。

当前GPU：RTX3090，UUID `GPU-4612d0cb-a47a-674d-b647-3f4065f1627f`。模型 `/home/featurize/ate-local-models/llava-1.5-7b-hf`，Revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。Python3.11.8 / torch2.6.0+cu124 / transformers4.49.0；FP16、Eager、Batch1、Greedy；单GPU串行执行。原算法、旧Manifest、此前18条件结果及论文LaTeX未修改；未Push或创建PR。

## 1. Component Ablation：Accuracy

ScienceQA-IMG 2017道正式图像题；POPE沿用正式8910题版本（Random2910、Popular3000、Adversarial3000），未补齐到9000题。正式vicuna_v1 Prompt、pad-to-square CLIP336 Processor、模型EOS2停止，SQA cap1024、POPE cap128，未强制32输出。

| Configuration | SQA-IMG Accuracy (%) | POPE Overall Accuracy (%) |
| --- | --- | --- |
| Native Full | 69.5092 | 86.8462 |
| SVH w/o Selective Transfer | 68.6663 | 87.2166 |
| SVH w/o Attention Bypass | 69.3604 | 86.3973 |
| SVH (Full) | 68.9638 | 87.2054 |


Full SVH复用同Checkpoint、Processor、Prompt、全部输入/图片/Gold、生成参数、冻结执行源码与评分协议的10927条历史Physical-C3预测；CPU逐条重评分及独立解析检查通过。当前GPU上两项Benchmark的物理/数值Smoke与历史SVH输出一致。Native此前无完整匹配同协议对照，本次与两个消融新增32781条自然EOS预测。全表43708条逐题预测齐全；不会将历史15题Smoke或公开论文分数当作完整Native。

| Condition | Benchmark | N | Correct | Delta vs Full SVH (pp) | Parse failed | Empty | Truncated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native Full | ScienceQA-IMG | 2017 | 1402 | +0.5454 | 0 | 0 | 0 |
| Native Full | POPE | 8910 | 7738 | -0.3591 | 0 | 0 | 0 |
| SVH w/o Selective Transfer | ScienceQA-IMG | 2017 | 1385 | -0.2975 | 12 | 0 | 0 |
| SVH w/o Selective Transfer | POPE | 8910 | 7771 | +0.0112 | 0 | 0 | 0 |
| SVH w/o Attention Bypass | ScienceQA-IMG | 2017 | 1399 | +0.3966 | 0 | 0 | 0 |
| SVH w/o Attention Bypass | POPE | 8910 | 7698 | -0.8081 | 0 | 0 | 0 |
| SVH (Full) | ScienceQA-IMG | 2017 | 1391 | +0.0000 | 11 | 0 | 0 |
| SVH (Full) | POPE | 8910 | 7770 | +0.0000 | 0 | 0 | 0 |


### POPE子集

| Condition | Subset | N | Accuracy (%) |
| --- | --- | --- | --- |
| Native Full | random | 2910 | 88.1787 |
| Native Full | popular | 3000 | 87.2667 |
| Native Full | adversarial | 3000 | 85.1333 |
| SVH w/o Selective Transfer | random | 2910 | 88.8660 |
| SVH w/o Selective Transfer | popular | 3000 | 87.7000 |
| SVH w/o Selective Transfer | adversarial | 3000 | 85.1333 |
| SVH w/o Attention Bypass | random | 2910 | 87.4570 |
| SVH w/o Attention Bypass | popular | 3000 | 86.7333 |
| SVH w/o Attention Bypass | adversarial | 3000 | 85.0333 |
| SVH (Full) | random | 2910 | 88.8660 |
| SVH (Full) | popular | 3000 | 87.7000 |
| SVH (Full) | adversarial | 3000 | 85.1000 |


### 消融语义与正确性证据

- **w/o Selective Transfer**：Prefill Relays在全部L0–31读取真实图像K/V，视觉隐藏状态全32层保留，不能在L20提前退出；答案P−1与Decode仍不可直接访问图像，L2–5答案Attention仍物理跳过，持久化KV仍移除视觉Token。额外保留后12层视觉Prefill计算是该消融必然产生的成本。本任务没有为它另做效率测试。
- **w/o Attention Bypass**：L0–19的Transfer、进入L20前真实移除视觉隐藏状态与全部层视觉KV移除不变；L2–5恢复正常Q/K/V/O Attention和真实文本KV；没有保留“空KV元数据”而仅改方法名。
- 两项仅通过独立子类配置组件，decoder forward、Attention算子、模型权重、视觉编码器和Greedy循环复用冻结代码。CPU掩码测试验证目标组件和非目标掩码；GPU检查原始位置/RoPE、逐层QKV调用、隐藏行数、真实KV、图像概率零及同语义短Dense参考数值等价。Dense只用于Smoke参考，正式SVH/消融均为物理实现。Native/No-op输出与全部Smoke logits逐位一致。

### Accuracy解释

ScienceQA-IMG：关闭Selective Transfer相对Full SVH变化-0.2975个百分点；关闭Attention Bypass变化+0.3966个百分点。这是指定两个Benchmark及冻结协议下的消融结果；不预设两个组件都提高Accuracy。
POPE：关闭Selective Transfer相对Full SVH变化+0.0112个百分点；关闭Attention Bypass变化-0.8081个百分点。这是指定两个Benchmark及冻结协议下的消融结果；不预设两个组件都提高Accuracy。

具体而言，Full SVH相对Native在SQA少答对11题，在POPE多答对32题；关闭Selective Transfer相对Full SVH在SQA少6题、POPE仅多1题。关闭Attention Bypass在SQA多8题、POPE少72题。因此，这组消融支持组件对质量存在任务相关的影响，不能写成两个组件均单调提高Accuracy，也不能凭POPE的一题差距宣称优势。SQA的11/12条解析失败均保留并计错；没有进一步逐题因果证据时，不能将全部Accuracy差距归因于输出格式。

## 2. Generation Length Sensitivity

仅Native、SVH、SVH-R；原GQA15+TextVQA15的30题ID、图像、Prompt/OCR、Gold及三个输入Tensor哈希完全一致。原32-Token3090旧Session存在15.09%漂移且GPU UUID不同，因此本次重新实测这三个必要条件的32 Token，与16/64/128共同构成当前GPU的配对长度曲线，未混入4090数据。

每个Session独立进程/模型加载，正式360请求，两轮共720。输入内交错长度与方法，并以相同确定性轮换计划运行；所有请求真实执行N个Token，1次Prefill和N−1次Decode，忽略EOS。原4090运行器只在独立副本中泛化长度与依赖常量，反向转换AST与原版完全一致；32Token原版/新版实际输出与首logits逐位一致。所有输出长度均完成GPU Smoke与真实KV检查。

计时从GPU Tensor就绪且同步后、控制器/位置/掩码初始化前开始，到第N个Token产生、上下文清理和CUDA同步后结束。包含Vision、Projector、Prefill、Greedy、Decode及必要请求开销；排除加载、CPU图片/Processor、H2D、日志与离线KV审计。TTFT为首Token壁钟时间；TPOT=(E2E−TTFT)/(N−1)；CUDA Event Decode TPOT独立记录。KV为第N输出Token后的实际live K/V Tensor字节，此时包含N−1个Decode输入；不把GPU显存当KV。

### 两轮均值表

以下为两个独立Session均值的平均；“±”是两个Session均值的样本标准差，不是60次独立重复的置信区间。Speedup先在各Session相对同轮Native计算，再取均值。

| Tokens | Method | TTFT ms | TPOT ms/token | E2E ms | Within-session E2E speedup |
| --- | --- | --- | --- | --- | --- |
| 16 | Native | 234.56 ± 3.44 | 68.541 ± 3.476 | 1262.68 ± 55.59 | 1.0000× |
| 16 | SVH | 181.42 ± 2.70 | 62.833 ± 3.042 | 1123.92 ± 48.34 | 1.1234× |
| 16 | SVH-R | 205.90 ± 2.21 | 62.827 ± 2.886 | 1148.31 ± 45.49 | 1.0995× |
| 32 | Native | 241.31 ± 2.22 | 68.492 ± 3.185 | 2364.57 ± 100.95 | 1.0000× |
| 32 | SVH | 186.24 ± 2.61 | 62.734 ± 2.694 | 2131.01 ± 86.13 | 1.1096× |
| 32 | SVH-R | 208.86 ± 3.61 | 62.757 ± 2.773 | 2154.33 ± 89.56 | 1.0976× |
| 64 | Native | 243.65 ± 2.40 | 68.119 ± 3.350 | 4535.14 ± 213.47 | 1.0000× |
| 64 | SVH | 188.43 ± 3.23 | 62.627 ± 2.961 | 4133.95 ± 189.77 | 1.0970× |
| 64 | SVH-R | 213.25 ± 3.21 | 62.526 ± 2.916 | 4152.38 ± 186.91 | 1.0921× |
| 128 | Native | 240.71 ± 2.51 | 68.241 ± 3.648 | 8907.32 ± 465.77 | 1.0000× |
| 128 | SVH | 186.35 ± 2.25 | 62.510 ± 2.993 | 8125.15 ± 382.39 | 1.0961× |
| 128 | SVH-R | 212.15 ± 2.49 | 62.541 ± 2.925 | 8154.89 ± 373.96 | 1.0921× |


### Session独立结果

| Session | Tokens | Method | TTFT ms | TPOT ms/token | CUDA Event TPOT | E2E ms | KV MiB | E2E reduction (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | Native | 232.126 | 66.083 | 66.063 | 1223.372 | 343.5000 | +0.000 |
| 1 | 16 | SVH | 179.511 | 60.682 | 60.654 | 1089.737 | 48.5625 | +10.924 |
| 1 | 16 | SVH-R | 204.342 | 60.787 | 60.761 | 1116.144 | 174.5625 | +8.765 |
| 1 | 32 | Native | 239.748 | 66.240 | 66.224 | 2293.189 | 351.5000 | +0.000 |
| 1 | 32 | SVH | 184.395 | 60.829 | 60.810 | 2070.107 | 55.5625 | +9.728 |
| 1 | 32 | SVH-R | 206.309 | 60.796 | 60.778 | 2091.000 | 181.5625 | +8.817 |
| 1 | 64 | Native | 241.953 | 65.750 | 65.738 | 4384.198 | 367.5000 | +0.000 |
| 1 | 64 | SVH | 186.149 | 60.534 | 60.520 | 3999.764 | 69.5625 | +8.769 |
| 1 | 64 | SVH-R | 210.982 | 60.464 | 60.452 | 4020.213 | 195.5625 | +8.302 |
| 1 | 128 | Native | 238.941 | 65.662 | 65.651 | 8577.977 | 399.5000 | +0.000 |
| 1 | 128 | SVH | 184.753 | 60.394 | 60.383 | 7854.763 | 97.5625 | +8.431 |
| 1 | 128 | SVH-R | 210.385 | 60.473 | 60.462 | 7890.464 | 223.5625 | +8.015 |
| 2 | 16 | Native | 236.992 | 70.999 | 70.984 | 1301.982 | 343.5000 | +0.000 |
| 2 | 16 | SVH | 183.332 | 64.984 | 64.965 | 1158.096 | 48.5625 | +11.051 |
| 2 | 16 | SVH-R | 207.463 | 64.868 | 64.846 | 1180.478 | 174.5625 | +9.332 |
| 2 | 32 | Native | 242.881 | 70.744 | 70.734 | 2435.952 | 351.5000 | +0.000 |
| 2 | 32 | SVH | 188.092 | 64.639 | 64.628 | 2191.908 | 55.5625 | +10.018 |
| 2 | 32 | SVH-R | 211.416 | 64.718 | 64.706 | 2217.661 | 181.5625 | +8.961 |
| 2 | 64 | Native | 245.350 | 70.488 | 70.480 | 4686.088 | 367.5000 | +0.000 |
| 2 | 64 | SVH | 190.715 | 64.721 | 64.713 | 4268.145 | 69.5625 | +8.919 |
| 2 | 64 | SVH-R | 215.520 | 64.588 | 64.579 | 4284.540 | 195.5625 | +8.569 |
| 2 | 128 | Native | 242.486 | 70.820 | 70.814 | 9236.672 | 399.5000 | +0.000 |
| 2 | 128 | SVH | 187.939 | 64.627 | 64.620 | 8395.538 | 97.5625 | +9.106 |
| 2 | 128 | SVH-R | 213.911 | 64.610 | 64.602 | 8419.322 | 223.5625 | +8.849 |


### 性能稳定性与重复性

每轮固定40次Native完整32Token预热，再对全部4长度×3方法各2次完整预热；两固定Native探针各10次建立独立中位数基线。每5题（每60正式请求）进行双探针各3次校准，共6窗口/轮；性能漂移只WARN，不重置基线、不重跑、不丢弃异常请求。5Hz NVML GPU遥测、1Hz CPU遥测及其他计算进程检查均保存。

全部24个“Session×窗口×探针”检查中，12个超过2%门限；最大绝对E2E漂移4.473%，最大绝对TPOT漂移4.936%。状态为COMPLETE_WITH_PERFORMANCE_WARNINGS；执行/正确性通过与性能稳定性分别报告。

| Session | Window | Probe | E2E drift (%) | TPOT drift (%) | Flag |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | GQA:202121628 | +3.069 | +3.587 | WARN |
| 1 | 1 | TextVQA:38571 | +1.956 | +2.388 | WARN |
| 1 | 2 | GQA:202121628 | -0.327 | -0.257 | within2% |
| 1 | 2 | TextVQA:38571 | -0.157 | +0.044 | within2% |
| 1 | 3 | GQA:202121628 | -0.108 | +0.019 | within2% |
| 1 | 3 | TextVQA:38571 | -0.745 | -0.569 | within2% |
| 1 | 4 | GQA:202121628 | -0.040 | +0.027 | within2% |
| 1 | 4 | TextVQA:38571 | -0.768 | -0.698 | within2% |
| 1 | 5 | GQA:202121628 | -0.736 | -0.877 | within2% |
| 1 | 5 | TextVQA:38571 | -0.935 | -0.840 | within2% |
| 1 | 6 | GQA:202121628 | +4.473 | +4.936 | WARN |
| 1 | 6 | TextVQA:38571 | +4.211 | +4.325 | WARN |
| 2 | 1 | GQA:202121628 | -1.600 | -1.798 | within2% |
| 2 | 1 | TextVQA:38571 | -2.139 | -2.271 | WARN |
| 2 | 2 | GQA:202121628 | -3.510 | -3.868 | WARN |
| 2 | 2 | TextVQA:38571 | -3.609 | -3.840 | WARN |
| 2 | 3 | GQA:202121628 | -2.210 | -2.494 | WARN |
| 2 | 3 | TextVQA:38571 | -2.672 | -2.785 | WARN |
| 2 | 4 | GQA:202121628 | -1.773 | -1.824 | within2% |
| 2 | 4 | TextVQA:38571 | -2.780 | -3.044 | WARN |
| 2 | 5 | GQA:202121628 | -0.410 | -0.431 | within2% |
| 2 | 5 | TextVQA:38571 | -1.760 | -2.011 | WARN |
| 2 | 6 | GQA:202121628 | -1.663 | -1.738 | within2% |
| 2 | 6 | TextVQA:38571 | -2.022 | -2.191 | WARN |


| Tokens | Method | Session1 E2E | Session2 E2E | Cross-session change (%) |
| --- | --- | --- | --- | --- |
| 16 | Native | 1223.372 | 1301.982 | +6.426 |
| 16 | SVH | 1089.737 | 1158.096 | +6.273 |
| 16 | SVH-R | 1116.144 | 1180.478 | +5.764 |
| 32 | Native | 2293.189 | 2435.952 | +6.226 |
| 32 | SVH | 2070.107 | 2191.908 | +5.884 |
| 32 | SVH-R | 2091.000 | 2217.661 | +6.057 |
| 64 | Native | 4384.198 | 4686.088 | +6.886 |
| 64 | SVH | 3999.764 | 4268.145 | +6.710 |
| 64 | SVH-R | 4020.213 | 4284.540 | +6.575 |
| 128 | Native | 8577.977 | 9236.672 | +7.679 |
| 128 | SVH | 7854.763 | 8395.538 | +6.885 |
| 128 | SVH-R | 7890.464 | 8419.322 | +6.702 |


### 遥测与警告类别

| Session | GPU samples | SM clock min/max MHz | Temperature min/max C | Mean power W |
| --- | --- | --- | --- | --- |
| 1 | 9263 | 1695/1950 | 52/65 | 193.74 |
| 2 | 9830 | 1695/1950 | 53/65 | 188.87 |


| Session | Warning category | Count |
| --- | --- | --- |
| 1 | calibration_over2pct | 4 |
| 1 | within_request_event_decode_change_over5pct | 60 |
| 2 | calibration_over2pct | 8 |
| 2 | within_request_event_decode_change_over5pct | 26 |


以上遥测范围覆盖Smoke、预热、基线及正式运行，不能直接解释为正式请求间的时钟波动；逐请求对应状态在per_request.csv中。请求内早/晚Decode差异超过5%只作诊断标记，可能还包含序列长度增长、首步和运行时开销差异，不能单凭该标记认定GPU降频。CPU调度、P-State与Throttle原始字段保留，未做缺乏证据的根因断言。

### 配对优势与解释

| Session | Tokens | Method | Decode faster /30 | E2E faster /30 | E2E paired Native−method mean ms | Exploratory95% CI ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | SVH | 30 | 30 | 133.636 | [125.635, 141.637] |
| 1 | 16 | SVH-R | 30 | 30 | 107.229 | [99.024, 115.434] |
| 1 | 32 | SVH | 30 | 30 | 223.081 | [211.705, 234.458] |
| 1 | 32 | SVH-R | 30 | 30 | 202.189 | [186.261, 218.117] |
| 1 | 64 | SVH | 30 | 30 | 384.434 | [369.737, 399.130] |
| 1 | 64 | SVH-R | 30 | 30 | 363.985 | [343.457, 384.513] |
| 1 | 128 | SVH | 30 | 30 | 723.214 | [702.047, 744.380] |
| 1 | 128 | SVH-R | 30 | 30 | 687.512 | [657.538, 717.487] |
| 2 | 16 | SVH | 30 | 30 | 143.885 | [136.279, 151.492] |
| 2 | 16 | SVH-R | 30 | 30 | 121.504 | [115.434, 127.575] |
| 2 | 32 | SVH | 30 | 30 | 244.044 | [234.331, 253.757] |
| 2 | 32 | SVH-R | 30 | 30 | 218.291 | [207.971, 228.611] |
| 2 | 64 | SVH | 30 | 30 | 417.943 | [396.914, 438.972] |
| 2 | 64 | SVH-R | 30 | 30 | 401.548 | [380.376, 422.720] |
| 2 | 128 | SVH | 30 | 30 | 841.133 | [804.100, 878.166] |
| 2 | 128 | SVH-R | 30 | 30 | 817.350 | [779.720, 854.980] |


配对统计按同题、同长度、同Session计算；所有方法的逐题延迟和变化完整保留。区间为30个配对输入的探索性t区间，存在系统状态/请求相关性，未做多重检验校正；尤其约1–2%的差异不能直接宣称显著加速。只有两个Session，不能据此概括所有3090实例。
16Token SVH：同轮Native的平均E2E降低+10.987%，Speedup 1.1234×。是否支持长度下的端到端收益以该实测值和两轮配对区间为准，未由32Token外推。
16Token SVH-R：同轮Native的平均E2E降低+9.049%，Speedup 1.0995×。是否支持长度下的端到端收益以该实测值和两轮配对区间为准，未由32Token外推。
32Token SVH：同轮Native的平均E2E降低+9.873%，Speedup 1.1096×。是否支持长度下的端到端收益以该实测值和两轮配对区间为准，未由32Token外推。
32Token SVH-R：同轮Native的平均E2E降低+8.889%，Speedup 1.0976×。是否支持长度下的端到端收益以该实测值和两轮配对区间为准，未由32Token外推。
64Token SVH：同轮Native的平均E2E降低+8.844%，Speedup 1.0970×。是否支持长度下的端到端收益以该实测值和两轮配对区间为准，未由32Token外推。
64Token SVH-R：同轮Native的平均E2E降低+8.436%，Speedup 1.0921×。是否支持长度下的端到端收益以该实测值和两轮配对区间为准，未由32Token外推。
128Token SVH：同轮Native的平均E2E降低+8.769%，Speedup 1.0961×。是否支持长度下的端到端收益以该实测值和两轮配对区间为准，未由32Token外推。
128Token SVH-R：同轮Native的平均E2E降低+8.432%，Speedup 1.0921×。是否支持长度下的端到端收益以该实测值和两轮配对区间为准，未由32Token外推。
SVH的8个“长度×Session”条件中，8/8的E2E点估计低于同轮Native；8/8的探索性配对区间下界大于0。Decode逐题占优数量的范围为30–30/30。上述是当前实例的观察，性能WARN与仅两个Session的限制仍然成立，不能据此声称所有硬件状态均稳定复现。
SVH-R的8个“长度×Session”条件中，8/8的E2E点估计低于同轮Native；8/8的探索性配对区间下界大于0。Decode逐题占优数量的范围为30–30/30。上述是当前实例的观察，性能WARN与仅两个Session的限制仍然成立，不能据此声称所有硬件状态均稳定复现。

### 对Section 5.5的支持范围

本次在16、32、64、128 Token均观察到SVH和SVH-R相对同Session Native的端到端收益，Decode TPOT也较低。因此，这组实测支持“固定输出长度改变后仍有推理收益”。但E2E降低比例随长度增加而减小：短输出中的TTFT收益占比更高，Decode节省并未使相对E2E收益随长度单调增大，不能写成“输出越长，加速比例越大”。SVH和SVH-R的平均TPOT非常接近，二者的小差异不构成可靠速度排名证据。
Native四种长度的Session2相对Session1 E2E增幅为6.23%–7.68%。因此绝对延迟的跨轮漂移不可忽略；同轮配对收益与跨轮绝对稳定性是两个不同结论。全部结果按性能WARN交付，不可标为稳定PASS。
质量方面，Full SVH在SQA略低于Native、在POPE略高于Native；两个消融的变化方向也不同。这组结果反映质量与效率的折中，不支持“所有任务质量均不降”或“两个组件均提高Accuracy”。

## 3. 结果、图与复现

- 质量：`ablation/summary.csv`、`pope_subsets.csv`、`sqa_results.csv`、`pope_results.csv`、`predictions.jsonl`、`checkpoint.sqlite`、`validation.json`。
- 效率：`generation_length/per_request.csv`（720请求，带时间对齐GPU状态）、`latency_summary.csv`（逐Session完整均值/中位数/标准差）、`latency_two_sessions.csv`、`paired_statistics.csv`、`calibration.json`、两个Session各自SQLite、校准/预热/Smoke/5HzGPU与CPU日志。
- 图：`generation_length/latency_vs_tokens.pdf`与同名PNG；可编辑绘图源码`code/s55_plot.py`。误差线显示两个Session均值的范围，非置信区间；完整数值来自latency_two_sessions.csv。图必须在真实数据和CPU验证完成后绘制。
- 可复现源码：`code/`、`protocol.json`、`source_manifest.json`（测量前冻结）、`artifact_manifest.json`（最终文件SHA）、`generation_length/runtime_length_only.diff`及AST审计。

在资源已恢复且同名结果目录不存在的新实例中，将此任务code目录放到同一项目位置，核对GPU UUID和模型回执并在新独立入口中更新实例身份后运行以下命令；已完成结果禁止直接覆盖。源程序对已有协议/Checkpoint拒绝重跑，不会继续旧Session或挑选最快请求。

```bash
source /home/featurize/work/ATE/vlm_research/results/featurize_22484-20261113-r1_resource_restore/activate.sh
cd /home/featurize/work/ATE/vlm_research/results/svh_ablation_efficiency_3090
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -u -B code/s55_prepare.py
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -u -B code/s55_ablation_run.py
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -u -B code/s55_analyze.py --quality-only
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -u -B code/s55_length_run.py --session 1
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -u -B code/s55_length_run.py --session 2
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -u -B code/s55_analyze.py --length-only
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -u -B code/s55_report.py
```

冻结GPU UUID与恢复回执是有意保留的实例防错门禁；跨实例复现必须在新目录留下映射修订证据，不能谎称新UUID与旧GPU相同。模型、数据和Python环境在本地ext4盘，work只存配置、代码和实验输出。

本报告不改论文Table/Figure，不保证结果符合预期。图和结论只使用本次真实测量；历史FullSVH质量复用来源明确，旧4090效率/旧3090漂移结果没有混入新正式统计。
