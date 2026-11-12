# RTX3090 八方法统一效率实验 Session 2

状态：**BLOCKED**；本轮正式请求 **0/240**；一个模型加载、一个新的正式Session。原31条PARTIAL与旧4090、旧3090记录未变。稳定性准入 **FAILED**，Smoke **PASS**，正式校准 **0/6**。完整复现证据资格：**False**。

持久化结果目录：`/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090_session2`。本地测量Spool独立目录：`/home/featurize/ate-local-experiments/unified_e2e32_eight_methods_3090_session2`。GPU `GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27`，与此前3090相同。未重下模型或数据。

## 部分数据，不供正式主表使用

KV来自32Token终止时实际K/V Tensor形状、dtype和有效长度，MiB=2^20字节。每个请求固定32Token，1次Prefill+31次真实Decode，忽略EOS。全部均值和逐题数据均保留，包括不利结果。

| Method | n | KV@32 MiB | TTFT ms | TPOT ms/token | Event TPOT ms/token | E2E@32 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native Full | 0 | — | — | — | — | — |
| FastV | 0 | — | — | — | — | — |
| SparseVLM | 0 | — | — | — | — | — |
| V²Drop | 0 | — | — | — | — | — |
| ApET | 0 | — | — | — | — | — |
| DUET-VLM (C+all) | 0 | — | — | — | — | — |
| llava-SVH | 0 | — | — | — | — | — |
| llava-SVH-R | 0 | — | — | — | — | — |

各Benchmark分项、均值、中位数、标准差、最小最大值见`efficiency_summary.csv`；逐题见`efficiency_per_input.csv`，32层物理布局见`kv_layout_per_layer.csv`。所有已测正式请求的实际KV有效字节与4090同题同方法记录相符。

## 相对同轮Native

正值为减少/改善，负值为变慢；使用同题配对的均值比。部分数据仅限共同已测输入，不把未测输入混入分母。

| Method | KV减少% | TTFT降低% | TPOT降低% | E2E降低% |
| --- | ---: | ---: | ---: | ---: |
| Native Full | — | — | — | — |
| FastV | — | — | — | — |
| SparseVLM | — | — | — | — |
| V²Drop | — | — | — | — |
| ApET | — | — | — | — |
| DUET-VLM (C+all) | — | — | — | — |
| llava-SVH | — | — | — | — |
| llava-SVH-R | — | — | — | — |

逐题优势：llava-SVH: TTFT严格更低0/0, TPOT严格更低0/0, E2E严格更低0/0；llava-SVH-R: TTFT严格更低0/0, TPOT严格更低0/0, E2E严格更低0/0。只有分母30且对应指标30/30才可说本轮全部30题更低。方法整体均值更低也不代表每题均更低。

## 配对差与不确定性

下表保留同题Native−方法的E2E配对差标准差（ms）以及E2E均值比改善的CPU分层配对Bootstrap 95%区间（5000次，种子309032，GQA/TextVQA构成固定）。它描述这30道固定题的组成不确定性，不能代替多Session测量、估计跨时段重复性或排除轮换顺序的时间混杂。正负方向及小于约1%–2%的改善均如实报告，不直接宣称显著加速。

| Method | E2E均值改善% | Bootstrap 95%区间% | 配对差标准差ms | E2E更低题数 |
| --- | ---: | ---: | ---: | ---: |


全部KV/TTFT/TPOT/Event/E2E的配对绝对差、百分比、离散度、方向计数、符号检验及Bootstrap区间见`paired_comparison.csv`和`paired_uncertainty.csv`。没有因为不利结果删题；未做多重检验校正。

FastV、V²Drop、ApET与DUET的实际E2E变化如上表，不能从KV减少直接推断加速。SparseVLM的TTFT、Decode TPOT和Event按同一边界实测，存在的选择、聚类、回收及兼容管理开销都包含在内；未做兼容层剥离或Profiler，不能把延迟差额全部归因于兼容层。异常筛查见`anomaly_summary.json`：统计标记0条，全部保留；该筛查不构成算法正确性证明。

## 本轮准入失败的CPU核对

Native预热61–80，TextVQA:38571：E2E/TPOT/Event robust CV分别1.079%/1.088%/1.089%；最大单请求偏离分别1.120%/1.161%/1.161%。

Native预热81–100，TextVQA:38571：E2E/TPOT/Event robust CV分别0.311%/0.461%/0.460%；最大单请求偏离分别3.035%/3.287%/3.285%。

超限请求：Native预热90（全局请求127），E2E=2236.984ms、TPOT=64.48379ms/token、Event=64.47441ms/token；主线程runqueue=0.264853ms，期间GPU采样11个，范围{'SM_MHz': {'min': 1935, 'max': 1935}, 'memory_MHz': {'min': 9501, 'max': 9501}, 'temperature_C': {'min': 60, 'max': 60}, 'power_mW': {'min': 216736, 'max': 228249}}，P-State=[2]，原始Throttle mask=[0, 4]。

相邻窗口均值超1%的原始数值见admission_adjacent_mean_failures.json。同一61–80或81–100窗口在滚动最近60请求的后续评估中重复出现，不能当作多次独立异常。

CUDA Event耗时也同步增加；采样时钟未显示该超限请求内明显降频，CPU主线程runqueue也很小。这些证据不足以确立硬件、CPU launch或动态频率中的具体根因；未追加GPU诊断，也没有放宽阈值。

## FastV最小验收修订与Smoke

上轮停止由额外FastV跨GPU全32Token相等断言触发：首个EOS是0-based位置1，首次差异位置5，输入和KV/剪枝一致。上轮Native-only候选补丁未采用；本轮按新授权使用更严格的FastV首EOS规则。

Native继续严格32Token历史一致。FastV比较截至第一个EOS并包括EOS，EOS位置必须一致；任一EOS前差异立即停止；两者均无EOS时必须32Token完全一致。只有EOS后差异被允许且完整记录Token IDs、首次差异位置、全部EOS位置、KV/剪枝及耗时数值。其他Smoke历史比较、No-op/有限Logits/实际Hidden/Mask/RoPE/KV/真实Decode/物理Attention跳算检查均保留。

CPU10例及AST回归检查通过，证明冻结`generate`调用之前及调用后原有计时/有限值/布局计算未改，稳态门禁函数未改；详情`patch_audit.json`和`session2_entry.diff`。所有新增验收在`generate()`返回后执行，未改变其计时区间、算法或生成结果。额外使用上轮失败输入进行一次有观察Hook的FastV32Token Smoke，状态`PASS`；Smoke不计入240，Hook不进入正式请求。

本轮正式FastV仅EOS后存在历史差异的请求数：0。每条完整证据见`FastV_history_acceptance.json`与原始Checkpoint，未隐去差异。Native正式32Token历史一致性已核对0/30。

## 新进程预热、独立基线与六次漂移

八方法统一双探针各2次无观察Hook预热。Native稳定性阶段实际120次完整32Token，累计生成E2E计时259.951秒；最少60次且180秒，上限120。最近三个互不重叠20请求窗口各含固定GQA/TextVQA各10次；逐输入检查E2E/TPOT/Event：robust CV≤1%、单请求偏离中位数≤3%、相邻均值变化≤1%、三窗均值范围≤2%，并检查>5%请求内Native状态跳变、>1秒请求间断/遥测缺口。每次失败准入评估也保留在`native_stability_gate.json`。

通过后独立20次固定双探针建立本轮基线，未使用旧进程基线或指定最快状态。每5题执行双探针各3次，按输入及合并检查E2E与TPOT，连续两窗超过2%暂停，明显>5%探针/请求内Native状态突变立即暂停；不重设基线、不挑最快区间。

| 已完成输入 | 合并E2E漂移% | 合并TPOT漂移% | 任一输入/指标最大绝对漂移% | 状态 |
| ---: | ---: | ---: | ---: | --- |
| — | — | — | — | 尚未启动 |

最大按输入绝对校准偏移：—%。原始探针与基线均保留。GPU5Hz、CPU1Hz、进程查询每2秒独立采集，GPU样本2024，最大采样间隔0.2169秒；遥测错误0，其他GPU计算进程事件0。正式请求期间频率、温度、功耗、P-State/Throttle见`formal_telemetry_ranges.json`，全部原始遥测保留。

## 与4090并列

模型`llava-hf/llava-1.5-7b-hf`固定Revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；FP16/Eager，PyTorch2.6.0+cu124、Transformers4.49.0、batch1、greedy、cache=True、logits_to_keep=1。30题输入SHA与原4090完全一致；冻结Commit `5ee2a0dbe78df336229b55fed14a4a6d0d880bed`。所有方法直接复用原`e2e32_runtime.generate`，SHA `513a122ff5f4da008b74e4d5560b0b6ef50baf8ccf7feb2fa766a0cdcdc51a3b`。V²Drop为历史576→288→144→1演示点，非论文官方192；DUET为C+all；没有混入Retain128。

计时从GPU输入就绪并同步后、控制器初始化前开始，到第32Token生成、必要控制器清理并同步后结束。Vision/Projector/Prefill/31Decode/argmax及必要控制器开销计入；加载、图像读取、CPU Processor/Tokenizer、H2D、日志写盘、离线KV审计排除。逐条校验E2E=TTFT+31×TPOT，并独立记录Event。

| Method | 4090 KV MiB | 4090 TTFT ms | 4090 TPOT ms | 4090 E2E ms | 本轮3090 E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native Full | 351.50 | 119.90 | 59.05 | 1950.50 | — |
| FastV | 177.50 | 106.66 | 60.22 | 1973.43 | — |
| SparseVLM | 166.75 | 145.43 | 60.67 | 2026.05 | — |
| V²Drop | 164.91 | 107.95 | 59.52 | 1952.97 | — |
| ApET | 159.50 | 115.03 | 59.05 | 1945.45 | — |
| DUET-VLM (C+all) | 159.38 | 111.85 | 59.97 | 1970.83 | — |
| llava-SVH | 55.56 | 107.97 | 53.50 | 1766.34 | — |
| llava-SVH-R | 181.56 | 111.46 | 53.85 | 1780.82 | — |

硬件不同，不能由跨GPU绝对延迟计算算法Speedup；不替换旧4090主表基线。算法、模型和内层计时一致；稳态准入、外层监控和本地日志生命周期不同，限制硬件归因。若本轮PARTIAL，题目组成也与完整4090不一致。

## 保存、边界与释放

状态`BLOCKED`，运行退出状态`BLOCKED`，失败信息`RuntimeError('Native stability admission failed within120 warmups; formal240 NOT STARTED')`。完整PASS要求240条正式请求、30题输入和Native历史一致、Smoke/物理检查与稳态准入通过、六次校准符合冻结停止规则。单Session不能证明跨时段可重复性。

46项冻结执行源码SHA未变；历史保护文件237项全部验证未变，包含上轮31条PARTIAL的完整目录。全部尝试、失败堆栈、Checkpoint、源码、准入、独立基线、逐题结果和SHA清单已保存。没有继续上轮统计，没有自动第三轮，没有更改GPU时钟/功耗/驱动/Governor，没有质量Benchmark/自然EOS/训练/调参/GitHub Push/PR/Windows备份。

GPU释放查询：`GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 33, 350.00 W`；计算进程为空，服务器保持运行。全目录清单`artifact_SHA256.json`，验收`validation.json`。
