# RTX 3090 Native Full E2E@32 基线复测

状态：**INCOMPLETE_OR_UNSTABLE**；正式请求 **15/90**，三轮各30题；预热、Smoke、基线与校准均独立保留且不计入正式统计。

## 一致性与执行路径

GPU UUID `GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27`。模型 `llava-hf/llava-1.5-7b-hf` / Revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。FP16、Eager、batch1、greedy、use_cache=True、logits_to_keep=1；PyTorch2.6.0+cu124、Transformers4.49.0。每次调用原冻结 `e2e32_runtime.generate(a,layout,Native)`，没有重写模型、Attention或计时循环；固定32个输出，1次Prefill+31次真实Decode，忽略EOS。

30个原始ID、顺序、图片SHA、完整Prompt/OCR、input_ids/attention_mask/pixel_values三个Tensor的SHA与5ee2原件完全一致。GQA此处沿用原效率实验的Paper预处理与Prompt，不混用自然EOS任务的另一套输入。只生成Native Full，No-op仅使用冻结identity_attention包装Native并验证首步Logits及全部32Token精确一致；没有运行任何剪枝/退出方法或质量评测。

历史32Token在每次实际请求中逐项核对；完整原件在 frozen_reference/，新图片路径仅保存在 input_path_mapping.json，未改旧Manifest。模型完整SHA复用本实例已通过的下载流LFS验收，重新核对固定Revision、凭据文件SHA与17个文件尺寸；未重复扫描14GB权重。

## 三个独立Session

| Session | GPU UUID | n | E2E Mean ms | TTFT Mean ms | TPOT Mean ms | Event Decode TPOT ms | Drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27 | 15 | 2411.82 | 231.51 | 70.33 | 70.30 | PAUSED_DRIFT |
| 2 | GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27 | 0 | — | — | — | — | NOT_COMPLETED |
| 3 | GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27 | 0 | — | — | — | — | NOT_COMPLETED |

| Session | E2E Median ms | E2E样本标准差ms | 最大探针E2E偏移 | 最大探针TPOT偏移 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2501.74 | 161.44 | 11.44% | 12.50% | PAUSED_DRIFT |
| 2 | — | — | —% | —% | NOT_STARTED |
| 3 | — | — | —% | —% | NOT_STARTED |

三轮E2E均值的中位数：**—ms**；轮均值范围：—–—ms，极差—ms（中位数的—%）。没有把90次请求直接混成一个基线数。

## 计时与稳定性

CUDA synchronize then start before per-request controller/position/mask setup; TTFT after Vision+Projector+Prefill+argmax+finite enqueue+sync; E2E after token 32 and necessary controller teardown+sync. TPOT=(E2E-TTFT)/31.。Vision, projector, full Prefill, 31 real cached Decode, selection/clustering/pruning, necessary algorithm setup/teardown, greedy argmax and common finite enqueue。排除：File reading, tokenizer/processor, H2D, SHA verification, model loading, warmup, final KV audit, CPU token extraction, telemetry and output I/O。

每个Session独立进程、模型仅加载一次，所有输入CPU预处理/H2D在计时外。Native至少预热180秒，完整双探针循环保留；之后6轮双探针建立本轮基线。每5题执行同样的两题探针，第30题后的探针兼作轮后检查。每项E2E/TPOT独立统计连续超2%窗口数；任一指标连续2窗超限即暂停，不重设本轮基线、不挑选新轮替代。

逐请求严格核对 E2E=TTFT+31×TPOT；CUDA Events独立计量31次Decode。标准差使用样本标准差（ddof=1）。GPU每约2秒记录UUID、P-state、SM/Memory Clock、温度、功耗/限制、各Throttle理由、其他计算进程，同时保存CPU负载与压力；完整CPU线程/亲和性、CUDA与依赖/安装源码SHA见各Session环境记录。未修改GPU时钟、功耗或全局驱动；没有启用FlashAttention/SDPA/compile。

其他GPU进程事件：0；监控查询异常：0。每轮详细探针均值、相对偏移和暂停判断见 drift_summary.json；原始探针见 probes_requests.jsonl。

## 与4090历史数据并列

| 硬件/实验 | Native E2E@32 ms | Native TPOT ms |
| --- | --- | --- |
| 4090 八方法统一192档 | 1950.50 | 59.05 |
| 4090 Retain-128实验 Native | 1879.06 | 56.73 |
| 3090 Session1 | 2411.82 | 70.33 |
| 3090 Session2 | — | — |
| 3090 Session3 | — | — |

本轮Native计时函数与5ee2正式源码逐字节一致。Retain128的整体运行器另有文件，这里只读取其已有Native统计，未执行其任何方法。3090与4090硬件不同，上表用于核对量级和波动，不能将绝对延迟比称为算法Speedup，不能替换4090原主表。

## 验收结论

输入一致性：PASS；三轮Native/No-op与历史32Token一致性：True；完整三轮及轮内漂移门禁：False；三轮均值相对其中位数的最大偏离：—%。

**尚不能认定为通过验收的3090 Native参考。** 不补造缺失Session，不剔除异常请求，不用4090历史数值替代。

推荐标准同时考虑完整90题、历史Token/输入一致、轮内门禁和三轮均值相对中位数不超过2%的重复性。即使通过，后续八方法实验仍需同机同时期校准，不应无条件使用跨轮绝对延迟计算Speedup。

## 持久化与异常

目录：`/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_recheck`。完整CSV、JSONL、SQLite不可变台账、每轮日志/环境/Smoke/探针/GPU遥测、参考源码与SHA均保留。没有修改旧4090结果或main；未Push/PR；无Windows结果备份。

Session1保留异常：`RuntimeError('Native calibration >2% in two consecutive windows; preserve session and pause remaining sessions')`。

一致性通过仅指已执行的请求和Smoke；NOT_STARTED的Session没有产生结果，不能解释为已经完成其一致性验收。Session1如被暂停，不用新的有利轮次替换，也不把模型重载后的结果拼成原来的一次加载Session。

## 暂停后的只读排查

未启动额外GPU请求。六轮基线双探针E2E均值为：[2593.15, 2582.94, 2497.81, 2586.89, 2576.57, 2565.27]ms；峰峰变化相对基线均值为3.71%。这表明已执行180秒以上预热并不等于本机已达到2%以内的稳定状态。

同题同Token的校准窗口从2567.11ms基线降至第10题后2288.14ms、第15题后2273.50ms；对应最终E2E偏移−11.44%、TPOT偏移−12.50%。CUDA Event Decode TPOT同步变化，不能用输出长度差异解释。未观察到其他GPU计算进程；GPU功耗限制未改动，Memory Clock记录保持9501MHz。SM Clock变化、主机调度与请求间I/O占空比仍可能影响结果，现有证据不足以唯一归因。

遥测实际间隔中位数2.10秒，最大20.95秒；超过5秒共4次，因此不能把缺失时段描述成连续覆盖。原始同时间探针/遥测、CPU线程配置和压力记录见diagnosis.json。正式请求仅完成前15道GQA；TextVQA正式15题未执行（TextVQA固定探针和Smoke已执行），该部分均值不能代表原30题均值。

逐请求计时恒等式与CUDA Event汇总校验：`{"formal_requests": 15, "E2E_equals_TTFT_plus_31_TPOT": true, "CUDA_Event_decode_sum_equals_31_event_TPOT": true, "mean_wall_minus_event_TPOT_ms": 0.03195869991689146, "max_absolute_wall_minus_event_TPOT_ms": 0.06083404518206237}`。

历史4090两行都是完整30题；当前3090 Session1仅GQA15题，尚无同一30题完整均值，不能把两者的部分/完整均值差解释为同输入集硬件性能差。主机CPU也不同：旧4090为AMD EPYC7453、14个可用CPU，新3090为Intel Xeon Gold6142、6个可用CPU；Torch intra-op线程均1，默认inter-op分别14/6，未调优或修改。详见diagnosis.json及各环境原件。

## GPU正式阶段实测范围

| Session | 采样数 | 温度℃ | SM Clock MHz | Memory Clock MHz | 功耗W | P-state/Throttle masks |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 22 | 57.0–60.0 | 1695.0–1950.0 | 9501.0–9501.0 | 188.5–224.22 | {'P2': 22}/{'0x0000000000000000': 19, '0x0000000000000004': 3} |
| 2 | 0 | None–None | None–None | None–None | None–None | {}/{} |
| 3 | 0 | None–None | None–None | None–None | None–None | {}/{} |

上述采样包括正式题目之间的探针与间隔；详细理由（SW Power Cap、HW/SW Thermal、HW Power Brake等）均有同次查询的独立字段，见gpu_telemetry.json。不能将P2或某一次时钟采样解释为固定时钟运行。

两个4090历史实验的generate函数AST精确一致：True；整体文件的差异为方法导入、方法特定布局/离线审计等。本轮调用的5ee2原始运行器SHA：`513a122ff5f4da008b74e4d5560b0b6ef50baf8ccf7feb2fa766a0cdcdc51a3b`。

所有表台账行数：{'formal': 15, 'smoke': 8, 'warmup': 44, 'probes': 18, 'failures': 1}。断点复用仅允许跳过完整已通过Session；不把模型重载后的部分数据静默拼成同一个独立Session。

GPU释放验收通过：无计算进程；`GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 30, 350.00 W`。服务器保持运行，未修改GPU设置。详见gpu_release.json。
