# RTX3090 八方法统一推理效率复现

状态：**PARTIAL**。稳定性准入：**PASS**；正式请求 **31/240**；一个模型加载、一个测量Session，没有自动重跑。

结果目录`/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090`。GPU `GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27`，与此前3090相同；4090参考GPU `GPU-724fa9c3-d33a-df3f-e349-533cb4634a53`。模型`llava-hf/llava-1.5-7b-hf`，Revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；FP16/Eager、PyTorch2.6.0+cu124、Transformers4.49.0、batch1、greedy、cache与logits_to_keep=1，固定32输出=1Prefill+31Decode，忽略EOS。冻结参考Commit `5ee2a0dbe78df336229b55fed14a4a6d0d880bed`。

## 部分效率数据：不供论文主表使用

下表均值来自本次正式请求；未完成行保持缺失或明确部分样本，不能作为完整主表。失败的第31条FastV原始记录也保留在内，不将它静默删除。KV为终止实际持久K/V有效Tensor字节，MiB=2^20字节；不是峰值显存或名义预算。Event列是31次Decode独立CUDA Event均值。

| Method | n | KV@32 MiB | TTFT ms | TPOT ms/token | Event TPOT ms/token | E2E@32 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native Full | 4 | 337.00 | 233.48 | 63.01 | 63.00 | 2186.69 |
| FastV | 4 | 163.00 | 154.64 | 63.91 | 63.90 | 2135.70 |
| SparseVLM | 3 | 152.08 | 159.59 | 65.98 | 65.97 | 2204.91 |
| V²Drop | 4 | 150.41 | 137.88 | 63.74 | 63.73 | 2113.77 |
| ApET | 4 | 145.00 | 148.40 | 64.62 | 64.62 | 2151.56 |
| DUET-VLM (C+all) | 4 | 144.88 | 148.12 | 64.74 | 64.73 | 2154.91 |
| llava-SVH | 4 | 42.88 | 182.60 | 58.05 | 58.04 | 1982.15 |
| llava-SVH-R | 4 | 168.88 | 201.24 | 58.35 | 58.34 | 2010.22 |

完整逐题、中位数、标准差、最小最大和GQA/TextVQA分项见`efficiency_per_input.csv`、`efficiency_summary.csv`；逐层布局见`kv_layout_per_layer.csv`。没有删除不利E2E或异常请求。

## 相对本轮Native的变化

正值表示减少/改善，负值表示变慢。此表为同一3090 Session**共同已测输入**上的均值比；部分结果按各方法与Native的共同输入计算，不能用4题Native均值直接作3题SparseVLM的分母。逐题配对百分比另存`paired_comparison.csv`，两者统计定义不同。未完成时这些数值仅作部分运行诊断。

| Method | KV减少% | TTFT改善% | TPOT改善% | E2E改善% |
| --- | ---: | ---: | ---: | ---: |
| Native Full | 0.00 | 0.00 | 0.00 | 0.00 |
| FastV | 51.63 | 33.77 | -1.43 | 2.33 |
| SparseVLM | 54.85 | 31.22 | -4.69 | -0.88 |
| V²Drop | 55.37 | 40.95 | -1.16 | 3.33 |
| ApET | 56.97 | 36.44 | -2.56 | 1.61 |
| DUET-VLM (C+all) | 57.01 | 36.56 | -2.74 | 1.45 |
| llava-SVH | 87.28 | 21.79 | 7.87 | 9.35 |
| llava-SVH-R | 49.89 | 13.81 | 7.38 | 8.07 |

逐题优势：llava-SVH: TTFT严格更低4/4, TPOT严格更低4/4, E2E严格更低4/4；llava-SVH-R: TTFT严格更低4/4, TPOT严格更低4/4, E2E严格更低4/4。分母不足30时不能声称全部30题更快。配对计数详见`paired_win_counts.json`。

## Smoke、稳定性与漂移

Smoke状态`PASS`，原始记录在SQLite及`all_requests.jsonl.gz`。完整检查Native/BridgeNative、FastV禁用、V²Drop禁用的32Token与首Logits一致性，以及32层实际Hidden/Mask/RoPE/KV、finite、真实Decode及SVH/R L2–L5答案Attention物理跳算。Smoke观察Hook不进入正式计时；所有方法在相同双探针上各追加2次完整32Token无观察Hook预热。

Native稳定性阶段实际100次，累计生成E2E计时218.967秒；上限120。每20次为窗口，按固定GQA/TextVQA各10次独立统计E2E、TPOT与Event。至少60次且180秒，最近三个互不重叠窗口：robust CV≤1%、单请求偏离窗口中位数≤3%、相邻窗口均值变化≤1%、三窗均值范围≤2%；请求间断/遥测缺口≤1秒，并拒绝Native前5与后26次Decode中位数相差>5%的明显状态突变。全部评估（含未通过窗口）见`native_stability_gate.json`，没有选择最快状态。

通过准入后独立20次双探针建立基线，再每5题各输入重复3次校准，校准不计入240。相对基线同时检查按输入及合并的E2E/TPOT，任一>2%记录breach，连续两窗暂停；明显>5%探针/请求内状态突变立即暂停。没有自动重设基线。

| 已完成输入 | 合并E2E漂移% | 合并TPOT漂移% | 任一输入/指标最大绝对漂移% | 状态 |
| ---: | ---: | ---: | ---: | --- |
| — | — | — | — | 未启动正式校准 |

运行结束状态：`PARTIAL`；失败信息：`AssertionError('Historical token mismatch: GQA:201759069 FastV')`。原始失败堆栈见`completion.json`与`runner.log`，不得将BLOCKED/PARTIAL解释为正式通过。

本次停止来自复用旧4090入口时保留的**FastV逐Token历史相等硬断言**。本轮用户明确要求Native32Token与历史完全一致，对外部方法明确要求物理/No-op正确性；FastV这一额外跨GPU硬门禁比该要求更严格。它触发后进程已经退出，因此不能通过重新加载模型续写为同一Session，也没有自动重跑。输出不一致本身不能证明FastV算法错误。CPU核对见`stop_audit.json`，没有为调查该问题额外执行GPU生成。

具体记录：GQA:201759069、FastV、正式序号31，当前与历史都以`No</s>`开头，前5个Token一致；第6个Token开始不同，已位于首次EOS之后。该输入、Sample SHA、剪枝事件、32层终止KV形状和有效长度均相同。这里不能把后续文本差异认定为答案质量下降或算法实现错误，也未进行质量Benchmark。累计128次已执行Native请求（含Smoke/预热/基线/正式4次）全部32Token匹配；余下Native正式输入尚未执行。

这是入口校验策略造成的提前终止，当前证据未显示性能漂移触发停止。后续若获准新Session，应在新目录中事先冻结输出策略：Native历史32Token严格一致；外部方法历史输出差异单独记录，保留其Source/Input/No-op/布局/Finite硬门禁。最小候选补丁另存`proposed_native_history_gate.diff`，未修改本次实际执行入口，也未运行新GPU请求。

## 计时和实现边界

每次直接调用原`e2e32_runtime.generate`，运行器SHA `513a122ff5f4da008b74e4d5560b0b6ef50baf8ccf7feb2fa766a0cdcdc51a3b`。同步后在控制器准备前开始；Vision、Projector、Prefill、31Decode、argmax、必要控制器清理和同步均计入。加载、CPU图像/Tokenizer、H2D、逐条日志、最终离线KV审计均排除。核对每条E2E=TTFT+31×TPOT及Event计算。没有修改Attention、KV算法、Flash/SDPA/compile替换。

八方法配置沿用4090：FastV K3-192静态KV；SparseVLM官方recycling、实际平均206.5；V²Drop历史576→288→144→1演示点，**不是论文官方192配置**；ApET288→96/L16；DUET **C+all** 307→153→0；SVH L20退出与SVH-R L25退出/仅L11–L24视觉KV。未混入Retain128。

外部算法选择、聚类、回收、额外QKV与必要兼容管理开销均包含在相同计时边界内。统计异常标记0条，详情`anomaly_summary.json`；当前每方法只有3–4题，未达到预设至少5题的稳健异常筛查样本量，零标记不代表无异常；保留全部数据。没有执行独立兼容层剥离或Profiler，不能把TTFT差额直接归因为兼容层，也不能将稀疏KV等同于必然更低TTFT/E2E。

遥测为GPU5Hz、CPU1Hz、进程查询每2秒的三个独立进程，避免旧诊断中CLI阻塞GPU采样；实际GPU样本2364、CPU样本474、最大GPU采样间隔0.2080秒，错误0、其他GPU进程事件0。实际频率/温度/功耗/P-State/Throttle范围和采样器CPU开销见`gpu_telemetry_summary.json`与原始JSONL；不能宣称监控零开销。没有改GPU时钟、Power Limit、驱动或Governor。

## 与4090历史结果并列

以下仅并列原实测绝对量；不跨GPU计算算法Speedup，不替换原4090表的Native基线。模型、输入、配置和内层计时器一致，GPU与稳态门禁、外层监控/本地日志生命周期有明确差别。硬件差异和这些外层差别都限制直接归因。**本次未完成时，4090列是完整30题，而3090列只有前面已测子集，连题目组成也不同，不能由此推断硬件延迟比例或全30题算法结论。**

| Method | 4090 KV MiB | 4090 TTFT ms | 4090 TPOT ms | 4090 E2E ms | 本次3090 E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native Full | 351.50 | 119.90 | 59.05 | 1950.50 | 2186.69 |
| FastV | 177.50 | 106.66 | 60.22 | 1973.43 | 2135.70 |
| SparseVLM | 166.75 | 145.43 | 60.67 | 2026.05 | 2204.91 |
| V²Drop | 164.91 | 107.95 | 59.52 | 1952.97 | 2113.77 |
| ApET | 159.50 | 115.03 | 59.05 | 1945.45 | 2151.56 |
| DUET-VLM (C+all) | 159.38 | 111.85 | 59.97 | 1970.83 | 2154.91 |
| llava-SVH | 55.56 | 107.97 | 53.50 | 1766.34 | 1982.15 |
| llava-SVH-R | 181.56 | 111.46 | 53.85 | 1780.82 | 2010.22 |

## 验收和用途

完整3090同机复现证据资格：**False**。只有完整240、Smoke与稳定性准入通过、六次校准符合冻结停止规则、输入与源码一致、真实KV核对通过时才作为本次效率表候选；单Session不证明跨时段重复性。

46项冻结执行源码与原30题输入哈希逐项验证；Native正式Token核对4/30。183个历史保护文件未变，旧4090结果及失败3090的15条记录保持原状。Checkpoint和全部预热/Smoke/探针原始数据均保留。没有质量Benchmark、自然EOS、训练、参数搜索、Push/PR或Windows备份。

GPU释放：`GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 27, 350.00 W`；计算进程为空，服务器保持运行。完整校验见`validation.json`，全目录SHA清单见`artifact_SHA256.json`。
