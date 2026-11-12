# RTX3090 八方法两轮统一效率与复现性

状态：**PARTIAL_OR_BLOCKED**。执行完整性：**INCOMPLETE**；正式请求 **151/480**。测速稳定性：**WARN**，未将完成240次等同于高精度稳定性PASS。所有慢请求、性能警告和历史Token差异保留。

Session1在正式第151条、输入TextVQA:38508的Native输出触发明确保留的完整32Token历史一致性硬门禁。首个EOS均为第4个Token，首次差异在第12个Token，发生于EOS后。图像/Prompt/输入Tensor及Sample SHA一致，终止KV形状/有效长度/dtype/位置哈希一致，Logits有限。未保存完整逐步Logits或KV数值，不能确定数值分叉根因，也不能仅据此认定算法实现错误或答案质量下降。共75次Native中74次完整历史一致；正式19次中1次失败。首个EOS之前及EOS本身一致，但Native的授权要求仍为严格32Token，因此停止；未自动放宽、重跑或启动Session2。完整证据见hard_failure_audit.json。

若本轮未完成，下表仅为已测前缀的部分统计，包含触发硬门禁的原始请求；不删除该行来美化结果。方法样本数可能不同，仅按共同已测题计算相对Native指标。它不能替代完整30题主表，也不能用于判断两轮复现性。Session1共240次总请求包含预热、Smoke、基线、18次校准和151次正式请求，不能将240次总请求写成240次正式完成。

目录：`/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090_repeatability`。GPU `GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27`，同一张RTX3090。模型`llava-hf/llava-1.5-7b-hf`，固定Revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；FP16/Eager、PyTorch2.6.0+cu124、Transformers4.49.0、batch1、greedy、cache=True、logits_to_keep=1，固定32Token=1Prefill+31真实Decode，忽略EOS。冻结Commit `5ee2a0dbe78df336229b55fed14a4a6d0d880bed`。模型和数据复用，没有重新下载。

## Session 1独立均值

| Method | n | KV MiB | TTFT ms | TPOT ms/token | Event TPOT ms/token | E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native Full | 19 | 342.03 | 232.24 | 69.99 | 69.97 | 2402.02 |
| FastV | 18 | 167.94 | 147.08 | 71.26 | 71.23 | 2356.28 |
| SparseVLM | 19 | 157.28 | 174.26 | 73.51 | 73.49 | 2453.13 |
| V²Drop | 19 | 155.43 | 142.34 | 70.93 | 70.90 | 2341.23 |
| ApET | 19 | 150.03 | 160.83 | 71.60 | 71.58 | 2380.46 |
| DUET-VLM (C+all) | 19 | 149.90 | 160.19 | 72.38 | 72.35 | 2403.98 |
| llava-SVH | 19 | 47.27 | 179.84 | 63.80 | 63.77 | 2157.56 |
| llava-SVH-R | 19 | 173.27 | 202.71 | 64.21 | 64.18 | 2193.12 |

| Method | KV减少% | TTFT降低% | TPOT降低% | E2E降低% | E2E Speedup | TPOT Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native Full | 0.00 | 0.00 | 0.00 | 0.00 | 1.0000 | 1.0000 |
| FastV | 50.89 | 36.36 | -2.09 | 1.62 | 1.0165 | 0.9796 |
| SparseVLM | 54.02 | 24.96 | -5.03 | -2.13 | 0.9792 | 0.9521 |
| V²Drop | 54.56 | 38.71 | -1.34 | 2.53 | 1.0260 | 0.9868 |
| ApET | 56.14 | 30.75 | -2.30 | 0.90 | 1.0091 | 0.9775 |
| DUET-VLM (C+all) | 56.17 | 31.02 | -3.41 | -0.08 | 0.9992 | 0.9670 |
| llava-SVH | 86.18 | 22.56 | 8.85 | 10.18 | 1.1133 | 1.0971 |
| llava-SVH-R | 49.34 | 12.71 | 8.27 | 8.70 | 1.0953 | 1.0901 |

## Session 2独立均值

未启动。

无正式数据。

已执行轮次按实际已测题数输出均值、中位数、样本标准差、P90（线性分位数）、最小最大值及GQA/TextVQA分项，见`efficiency_summary.csv`的n列；完整目标为每方法30题。未执行的Session2不生成虚构统计。KV=2^20字节MiB，从实际终止K/V Tensor形状、FP16 dtype及有效长度独立重算；不是峰值显存。各层布局见`kv_layout_per_layer.csv`。

## 两轮变化

变化定义为(S2均值/S1均值−1)×100%。Speedup每轮均使用该轮Native均值/方法均值，表中的Speedup变化是(S2 Speedup/S1 Speedup−1)×100%。正延迟变化表示S2更慢，正Speedup变化表示相对Native优势增加。

| Method | TTFT变化% | TPOT变化% | E2E变化% | E2E Speedup变化% | TPOT Speedup变化% |
| --- | ---: | ---: | ---: | ---: | ---: |

尚无两轮完整Native结果，跨Session变化及加速比稳定性均不可估计。

Session2未启动，跨轮变化范围和相对Native加速比稳定性不可估计。 两个顺序运行的Session不能据此识别共同硬件因果；有两轮结果时须同时报告绝对和相对变化，不能选择有利的一轮。

Session1 llava-SVH Decode TPOT均值改善8.852%，逐题更低19/19，探索性输入Bootstrap95%区间[8.267,9.430]%。
Session1 llava-SVH-R Decode TPOT均值改善8.267%，逐题更低19/19，探索性输入Bootstrap95%区间[7.819,8.789]%。
Session1 FastV TTFT改善36.36%，TPOT改善-2.09%，E2E改善1.62%。
Session1 V²Drop TTFT改善38.71%，TPOT改善-1.34%，E2E改善2.53%。
Session1 ApET TTFT改善30.75%，TPOT改善-2.30%，E2E改善0.90%。
Session1 DUET-VLM (C+all) TTFT改善31.02%，TPOT改善-3.41%，E2E改善-0.08%。
Session1 SparseVLM TTFT改善24.96%，TPOT改善-5.03%，E2E改善-2.13%。
Session2未启动。

没有两轮完整结果，名次反转不可评估；空的排名文件不表示不存在名次反转。 小于约1%–2%的均值改善及名次变化仅作探索性描述，不直接宣称显著加速。若各方法近似共同变化但相对Native更稳定，应同时保留绝对漂移事实；本报告不跨GPU计算算法Speedup。

## 逐题配对、区间与异常

完整两轮时，同题S1/S2对照写入`cross_session_per_input.csv`，包含每项绝对差、百分比、顺序、开始时间及Token是否相同；未完成时该文件保持空表，不编造第二轮数据。每轮相对Native逐题差见`within_session_paired.csv`。探索性95%Bootstrap区间见`bootstrap_CI95.csv`：按冻结输入成对重采样5000次、种子309032；完整两轮时保持GQA15/TextVQA15构成，部分结果保留各方法共同已测题的实际Benchmark构成。跨轮相对Native区间同时重采样该题两轮Native与方法，保留配对相关性。没有GPU重跑。这些区间描述固定题目组成，不替代跨时段重复性、无法排除时序混杂，未做多重检验校正。

统计异常标记2条，原始请求均保留。预先固定筛查：同题E2E比值偏离中位数>max(0.05,6×1.4826×MAD)。每个异常的Event单步、CPU runqueue、时钟、功耗、温度及P-State/Throttle原始值见`anomaly_evidence.json`。各方法两轮耗时差最大的5题及对均值差的算术贡献见`cross_session_top_contributors.csv`，该贡献不是硬件因果证明。所有请求的遥测时间关联见各轮`telemetry_per_request.csv`。CUDA Event也包含相邻Kernel之间的执行间隔，不能凭Event增加就唯一归因于GPU核心性能。未做Profiler或额外GPU诊断，不编造根因。

## Native校准与性能警告

预先冻结的每轮协议为：固定16次Native完整32Token预热，之后七种方法各2次（共14），然后21次正确性Smoke（含旧失败FastV输入）、独立20次Native双探针基线；每5题双探针各3次，计划六窗36次。每轮所有请求计划347次，正式240次独立统计。两轮预先冻结相同顺序，Schedule SHA `50d84b3010bde68fdd5bee93cd810a1d4702909b235516617b54934e63cf3e30`；计划独立进程、独立模型加载与基线。实际执行数量以Checkpoint、validation与上方统计为准，没有混入旧31条PARTIAL数据。

性能1%CV、1%窗口变化、3%单请求、2%校准漂移及更大波动均只记录WARN；没有等待最快状态、没有无限预热、没有删除/重跑/替换请求。正确性、输入、有限值、OOM、GPU冲突/设备和硬件安全仍为硬门禁。85℃及低可用内存等安全监控沿用原采集器，没有改GPU时钟、Power Limit、驱动或Governor。

Session 1：

| 已完成输入 | 合并E2E漂移% | 合并TPOT漂移% | 按输入/指标最大绝对漂移% | 状态 |
| ---: | ---: | ---: | ---: | --- |
| 5 | 4.504 | 4.934 | 6.221 | WARN |
| 10 | 13.487 | 14.516 | 14.680 | WARN |
| 15 | 14.891 | 16.012 | 16.402 | WARN |

Session 2：

未启动。

每轮`performance_warnings.json`、`warmup_summary.json`和原始SQLite记录所有波动。GPU5Hz、CPU1Hz、进程查询2秒分离采集，频率、温度、功耗、P-State、Throttle以及采集缺口/错误汇总见各轮`gpu_telemetry_summary.json`。没有宣称监控零开销。

## 正确性、计时与历史边界

30题Sample、图像、Prompt、OCR、input_ids/attention_mask/pixel_values SHA均对齐原4090。Native严格32Token历史一致；FastV严格首EOS前及EOS，只有EOS后历史差异可以记录后继续；无EOS时32Token严格。其他方法历史Token差异完整记录，不单凭跨GPU历史输出不同判定实现错误；No-op Token/首Logits、finite、实际Hidden/Mask/RoPE/KV、真实Prefill/Decode与SVH/R物理跳算硬检查不变。正式请求没有Smoke观察Hook。

内层直接复用冻结`e2e32_runtime.generate`，SHA `513a122ff5f4da008b74e4d5560b0b6ef50baf8ccf7feb2fa766a0cdcdc51a3b`；没有重写Attention。E2E从GPU输入就绪同步后、控制器准备前开始，到第32Token生成及必要清理同步后结束。Vision/Projector/Prefill/31Decode/argmax/必要控制器开销计入；加载、CPU图像与Processor/Tokenizer、H2D、日志和离线KV审计排除。逐条核对E2E=TTFT+31×TPOT及Event。V²Drop为历史576→288→144→1工作点，非论文官方192；DUET为C+all；SparseVLM含官方Recycling实际平均206.5视觉表示；没有Retain128或算法调参。

## 与4090并列及用途

| Method | 4090 TTFT | 4090 TPOT | 4090 E2E | 3090 S1 E2E | 3090 S2 E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native Full | 119.90 | 59.05 | 1950.50 | 2402.02 | — |
| FastV | 106.66 | 60.22 | 1973.43 | 2356.28 | — |
| SparseVLM | 145.43 | 60.67 | 2026.05 | 2453.13 | — |
| V²Drop | 107.95 | 59.52 | 1952.97 | 2341.23 | — |
| ApET | 115.03 | 59.05 | 1945.45 | 2380.46 | — |
| DUET-VLM (C+all) | 111.85 | 59.97 | 1970.83 | 2403.98 | — |
| llava-SVH | 107.97 | 53.50 | 1766.34 | 2157.56 | — |
| llava-SVH-R | 111.46 | 53.85 | 1780.82 | 2193.12 | — |

硬件不同，不能用跨GPU绝对延迟计算算法Speedup或替换旧4090基线。本轮仅可作中断及性能波动诊断证据；无法进行两轮复现性比较，不适用于完整效率主表。

原4090、旧3090失败15条、旧31条PARTIAL及上次准入失败目录均未改。完整源码SHA、模型Revision、输入、输出Token SHA、Checkpoint、全部尝试与失败日志、预热及遥测保留。没有自然EOS、质量Benchmark、训练、参数搜索、GitHub Push/PR或完整Windows备份。

GPU释放：`GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 30, 350.00 W`；计算进程为空，服务器保持运行。最终验收`comparison/validation.json`，完整SHA清单为根目录及各目录`SHA256.json`。
