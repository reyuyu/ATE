"""Write the diagnostic report from actual evidence; no GPU operations."""
from pathlib import Path
import json,statistics,csv,hashlib
from datetime import datetime,timezone
D=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_rootcause_diagnostic');O=D.parent/'native_e2e32_3090_recheck';A=D.parent/'native_e2e32_3090_stability_audit_20261110'
def read(n):return json.loads((D/n).read_text())
def save(n,v):(D/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
e=read('rootcause_evidence.json');pf=read('profiler_summary.json');x=read('additional_rootcause_evidence.json');rs=[json.loads(l) for l in (D/'requests.jsonl').read_text().splitlines()];env=read('environment.json');plan=read('protocol.json')
conditions=e['conditions'];continuous=next(c for c in conditions if c['condition']=='continuous_measured');slow=[r for r in rs if r['phase'] in ['A','B'] and not r['profiled'] and r['condition']!='fixed_warmup' and r['state_band']=='slow_historical_band'];fast=[r for r in rs if r['phase'] in ['A','B'] and not r['profiled'] and r['condition']!='fixed_warmup' and r['state_band']=='fast_historical_band'];prebands={c['preceding_unprofiled_band'] for c in pf['captures']}
def f(x,d=3):return '—' if x is None else f'{x:.{d}f}'
def bandrow(st):
 m=st['metrics'];h=st['hardware'];return '| '+st['band']+' | '+str(st['n'])+' | '+' | '.join(f(m[k]['mean']) for k in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms'])+' | '+f(st['main_thread_runqueue_wait_ms']['mean'])+' |'
def conditionrow(c):
 return '| '+c['phase']+'/'+c['condition']+' | '+str(c['requests'])+' | '+str(c['intentional_idle_s'])+' | '+' | '.join(f(c[k]['mean']) for k in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms'])+' | '+f(c['E2E_ms']['min'],2)+'–'+f(c['E2E_ms']['max'],2)+' |'
slowlines=[]
for r in slow:
 g=r['generation'];slowlines.append(f"- 请求{r['request']}，{r['condition']}，UTC {datetime.fromtimestamp(r['begin_unix_ns']/1e9,timezone.utc).isoformat()}：E2E {g['E2E_ms']:.3f}ms，TTFT {g['TTFT_ms']:.3f}ms，TPOT {g['TPOT_ms']:.3f}ms，Event Decode TPOT {g['CUDA_Event_Decode_TPOT_ms']:.3f}ms；主线程runqueue等待{r['main_thread_runqueue_wait_delta_ms']:.3f}ms。")
kernelrows=[]
for c in pf['captures']:
 d=c.get('decode',{});label='首次Profiler之前' if c['preceding_unprofiled_request']==69 else '前序Profiler之后';kernelrows.append('| '+str(c['request'])+' | '+str(c['preceding_unprofiled_request'])+'/'+label+' | '+f(c['profiled_E2E_ms'])+' | '+f(c['profiled_TPOT_ms'])+' | '+f(d.get('Event_sum_ms'))+' | '+f(d.get('traced_GPU_activity_union_sum_ms'))+' | '+f(d.get('internal_GPU_gaps_sum_ms'))+' |')
hwrows=[]
for st in e['unprofiled_nonwarmup_states']:
 if not st['n']:continue
 h=st['hardware'];hwrows.append('| '+st['band']+' | '+str(h['GPU_samples'])+' | '+f(h['GPU']['SM_MHz']['min'],0)+'–'+f(h['GPU']['SM_MHz']['max'],0)+' / '+f(h['GPU']['SM_MHz']['mean'],1)+' | '+f(h['GPU']['memory_MHz']['min'],0)+'–'+f(h['GPU']['memory_MHz']['max'],0)+' | '+f(h['GPU']['power_W']['mean'],2)+' | '+f(h['GPU']['temperature_C']['min'],0)+'–'+f(h['GPU']['temperature_C']['max'],0)+' | '+str(h['SW_power_cap_sample_count'])+' |')
over='\n'.join('| '+k+' | '+f(v['mean'],4)+' | '+f(v['max'],4)+' |' for k,v in e['overhead_nonprofiled'].items())
recommend=dict(status='RECOMMENDED_NOT_EXECUTED_FORMAL',formal90_resumed=False,original15_remain_stability_failed=True,selected_actions=['A_strict_prewarm_and_admission','B_local_durable_spooling_and_minimize_unnecessary_idle','D_native_bracketed_paired_measurement_and_independent_sessions','E_root_cause_not_uniquely_resolved'],C_fixed_GPU_CPU_conditions='Not changed. Seek separate authorization only if later evidence identifies frequency/CPU conditions as decisive.',minimum_warmup_full32=60,minimum_sum_actual_generation_E2E_s=180,per_input_window_repetitions=10,window_full32_requests=20,consecutive_nonoverlapping_windows=3,window_robust_CV_limit=.01,adjacent_window_median_change_limit=.01,three_window_median_range_limit=.02,independent_baseline_requests=20,calibration_every_formal=5,calibration_requests=6,drift_threshold=.02,consecutive_drift_windows=2,unexpected_gap_s=1,telemetry_gap_s=5,warmup_max_requests=180,warmup_max_wall_s=900,formal_sessions=3,formal_inputs_per_session=30,report_each_session_and_median_of_means=True,pairing_rule='For future eight-method comparison, measure fixed Native before and after each method block on the same hardware and matched input; reject drifting blocks without choosing fastest Native. No such methods were run here.',proof_limit='Current diagnostic uses one fixed GQA input, one model load. It does not validate all30 inputs,3 formal sessions,or eliminate every historical drift mechanism.')
recommend.update(profiler_process_isolation_required=True,telemetry_split_independent_GPU_CPU_and_process_inventory_collectors=True,monitoring_overhead_and_gaps_must_be_reported=True)
recommend.update(stability_statistics=dict(metrics=['E2E_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms'],per_input=True,robust_CV='1.4826*MAD/median',three_window_range='(max(window_medians)-min(window_medians))/median(window_medians)',calibration='Alternating fixed GQA and TextVQA probes, three repeats per input; compare per-input medians to its admitted baseline, never merge the two input scales.'),warmup_execution='Alternate the two original frozen probes; require >=60 full32 calls AND >=180 seconds summed measured E2E, then the latest three complete nonoverlapping windows must pass. Stop without formal requests at the warmup budget limit.')
save('recommended_protocol.json',recommend)
confirmed=['All completed requests used one frozen input, unchanged inner Native@32, and exact historical output Tokens.','Controlled A after fixed12 warmups completed28 continuous requests with recorded distribution; no formal results replaced.','Read-only high-frequency GPU telemetry and per-request main-thread schedstat directly recorded; historical workload was not resumed.']
if slow:confirmed.append('An unprofiled historical slow Decode band recurred in the controlled experiment; all such records are retained.')
e.update(confirmed_observations=confirmed,not_established=['No unique physical root cause established.','Observed idle/clock transitions do not prove every idle causes slow state.','Profiler results are instrumented and cannot replace unprofiled latency or prove a missed historical slow-vs-fast contrast.'],ruled_out_with_scope=['Fewer output Tokens or changed frozen inputs as explanation for recorded latency differences.','Per-request explicit gc.collect or CUDA empty_cache in the original frozen Native entry.','Large OS main-thread runqueue delay as the sole explanation for an observed slow request when its measured wait is far smaller than the latency excess.'],unresolved=['GPU wakeup/power-management vs CPU/driver submission changes','Host physical CPU frequency not exposed by this virtualized environment','Unprofiled slow request kernel execution vs inter-kernel idle decomposition','Original mixed-input/NFS monitor effects were not independently factorially isolated'],root_cause_uniquely_identified=False,stable_environment_fix_validated=False)
save('rootcause_evidence.json',e)
profiler_contrast='C段请求69为首个Profiler之前的快速普通探针；请求71以后紧邻之前的Profiler，TPOT明显升高到约90ms，已超出历史约74ms慢态。不能把这些后继探针当作未经采样影响的普通慢态。未取得可证明对应历史快慢两态的Profiler配对，该对照证据不完整；新增观测提示同一进程经历Profiler后须隔离或重新验证稳态，但尚未用新进程对照证实其具体残留机制。'
report=f'''# ATE RTX 3090 Native E2E@32 受控性能诊断

状态：**诊断已执行；物理根因尚未唯一确定；原正式实验继续保留稳定性失败状态**。

生成时间UTC：{datetime.now(timezone.utc).isoformat()}。结果目录：`{D}`。

本轮实际完成**{len(rs)}次完整Native@32生成**：A预热12次、A/B非预热普通请求56次、C段未插桩探针8次、C段Profiler插桩8次。C段探针可能受前次Profiler影响，全部与A/B普通分布隔离。事先固定84次，硬上限100次；没有新增其他算法、正式Benchmark、质量评测或恢复原75次队列。所有完成的输入/输出一致性和32层KV检查通过。未挑最快结果作为新基线。

## 已缩小的原因范围

固定12次预热后，A阶段28次连续请求E2E均值{continuous['E2E_ms']['mean']:.3f}ms、范围{continuous['E2E_ms']['min']:.3f}–{continuous['E2E_ms']['max']:.3f}ms，TPOT均值{continuous['TPOT_ms']['mean']:.3f}ms。连续A阶段状态转换记录为 `{json.dumps(e['A_continuous_state_transitions'],ensure_ascii=False)}`。

A/B普通非预热请求中，历史快速Decode带（TPOT≤69ms）{len(fast)}次，历史慢速带（TPOT≥71ms）{len(slow)}次，中间带另列。该标签只是预先固定的描述区间，不是剔除标准；首个冷启动预热TTFT显著增加，不与稳态普通请求混合。C段虽有未插桩探针，仍全部单列，不并入本表。

{'本轮确实复现了历史慢态，但其次数有限，不代表存在两个稳定、可重复控制的分布。' if slow else '本轮未在普通请求中复现历史慢态，不能声称问题已修复。'}

{chr(10).join(slowlines) if slowlines else '无普通慢态请求记录。'}

请求49逐步Event显示：前5次Decode平均{x['slow_request49_first5_Event_decode_mean_ms']:.3f}ms，第6次{x['slow_request49_step6_Event_ms']:.3f}ms，最后25次平均{x['slow_request49_last25_Event_decode_mean_ms']:.3f}ms。慢态是在**同一次请求中途**形成并持续到结束，而不是只有首个Token的唤醒惩罚。所有8次60秒条件的32步Event均保存在 `idle60_all_step_events.csv`，包含快速反例。

| 普通非预热状态 | n | E2E均值 ms | TTFT均值 ms | TPOT均值 ms | Event Decode TPOT均值 ms | 主线程runqueue等待均值 ms |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(bandrow(st) for st in e['unprofiled_nonwarmup_states'])}

小样本慢态均值不能视为稳定慢态基线，n=1时没有可估计的组内波动。全部逐请求记录见 `request_timeline.csv`，没有删除反例。

## A/B：连续、15秒、60秒和反序结果

单一输入为冻结 `GQA:202121628`，与原30题Manifest中对应题完全一致。B阶段按预定顺序：连续4→15秒×4→60秒×4→连续4→60秒×4→15秒×4→连续4；每个idle请求**之前**都独立等待指定时间。两次重复顺序为回文安排，仍不能消除所有时间趋势。

| 阶段/条件 | n | 每次主动空闲秒 | E2E均值 ms | TTFT均值 ms | TPOT均值 ms | Event Decode TPOT均值 ms | E2E范围 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(conditionrow(c) for c in conditions if c['phase'] in ['A','B'])}

`continuous_vs_idle.csv`提供每条件中位数、标准差和全范围，未把正序/反序简单混成唯一结论。`request_timeline.csv`另列真实请求间墙钟间隔和实际主动等待；`overhead_timeline.jsonl`列CPU准备、校验及本地写盘调用耗时。15/60秒等待均不包含在E2E中，E2E/TTFT/TPOT仍由原冻结计时器返回。

空闲后的GPU唤醒与延迟有部分对应，但同样60秒条件也有快速反例，所以不能把“60秒等待”认定为充分或必然原因。连续A的短窗口稳定，也不能证明未来全部输入/全部Session稳定。

## GPU状态、功耗与采样限制

GPU UUID `GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27`，RTX3090；驱动610.57.04。开始时空闲P8、SM210MHz、显存405MHz，0计算进程；功耗限制350W。运行时常见P2、SM约1.7–1.965GHz、显存9501MHz。完整样本在 `gpu_telemetry.csv`。

| 普通状态 | GPU样本数 | SM范围/采样均值 MHz | 显存时钟范围 MHz | 采样平均功耗 W | 核心温度范围 ℃ | SW Power Cap激活样本 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(hwrows)}

这些均值按落在请求wrapper内的采样汇总，并非精确按GPU活动加权；空闲恢复早期也可能被计入，不能把该均值当作31次Decode的恒定频率。200ms读取NVML也不保证设备计数器每200ms独立更新，尤其功耗/利用率存在自身采样窗口。

SW Power Cap掩码0x4表示驱动功耗相关时钟限制状态，定义见 [NVIDIA NVML官方说明](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html)。实际记录中它可在P8低功耗或唤醒阶段出现；不能用低平均功耗否定瞬时/驱动事件，也不能仅凭该位判定供电故障或将350W限制改高。本轮未修改任何时钟、功耗、驱动或系统设置。

采样中位间隔{e['GPU_sample_interval_s']['median']:.4f}s、最大{e['GPU_sample_interval_s']['max']:.4f}s。采集器单独进程运行，CPU共{e['telemetry_collector']['CPU_s']:.3f}s / 墙钟{e['telemetry_collector']['wall_s']:.3f}s（约{100*e['telemetry_collector']['CPU_s']/e['telemetry_collector']['wall_s']:.3f}%的一个逻辑核）；nvidia-smi子进程CPU不计入该process_time，不能据此宣称监控零开销。其他GPU进程事件{len(e['telemetry_collector']['other_GPU_process_events'])}次，采集错误{len(e['telemetry_collector']['errors'])}条。

实际仍有{x['telemetry_gaps_over1s'].__len__()}个超过1秒的采样缺口，最长28.018秒；请求57以及C段76–78被缺口跨过，其余较长缺口落在请求外。慢态49自身有13个样本，单次NVML查询最长0.958ms，不在长缺口内。缺口详见 `telemetry_gaps_over1s.csv`。缺口前NVML查询通常不足约1ms，说明长间断发生在同一采集循环的其他操作或调度时段，具体调用没有分项计时，不能唯一归因；后续应把GPU、CPU/procfs及nvidia-smi进程查询分成独立采集器，避免慢操作阻断5Hz流。本轮不能声称全程连续5Hz覆盖。

没有显存结温、VRM温度或板级电气瞬态数据，也未做锁频对照，无法唯一证实/排除功耗管理作为全部根因。

## CPU与调度的直接记录

CPU为Intel Xeon Gold6142，实际可用6个逻辑CPU（0–5）；Torch intra-op=1、interop=6，OMP/MKL/OpenBLAS均1，未设置新亲和性或调度优先级。Governor/cpufreq接口未暴露；/proc/cpuinfo显示约2600MHz但可能是虚拟化固定值，不能当作实际物理核心频率证据。

每请求读取主线程 `/proc/.../schedstat`、上下文切换和运行核心，并保存CPU时间；每秒记录CPU压力、loadavg、cgroup各层cpu.stat/cpu.max、/proc/stat（含steal）及可见频率。全部原始记录见 `cpu_telemetry.csv`。

runqueue等待可以检验“线程已可运行但被OS排队”的时间；它不等于所有CUDA等待、驱动自旋、物理CPU降频或虚拟机调度损失。若慢态请求只等待不足数毫秒，就不能把数百毫秒延迟全部归因于主线程被本机OS抢占。对应逐请求数据已提供；cgroup限流的连续计数增量见 `rootcause_evidence.json/CPU_throttling_counters`，不是用一次初始快照替代全程。

本轮没有安装/启用特权调度跟踪、perf内核采样或更改系统权限，因此缺少每个CPU抢占点的完整sched_switch时间线，不能对更低层宿主机调度作绝对排除。

可读cgroup层级的nr_throttled/throttled_usec计数全程增量均为0；这排除了这些**已观测配额**发生限流的解释，不能排除未暴露的宿主机限制。

## C：Profiler实际证据及缺口

环境未找到nsys；使用已有PyTorch2.6 Profiler（CPU与可用CUDA活动），无安装或环境升级。原始Chrome/Kineto trace在 `profiler/`，索引见 `profiler_capture_index.json`。共成功分析{len(pf['captures'])}份；CUDA EventRecord能按64个记录映射32步的trace有{sum(c['step_pair_mapping_available'] for c in pf['captures'])}份；分析异常{len(pf['analysis_errors'])}条。

{profiler_contrast}

| Prof请求 | 前置普通请求/状态 | 插桩E2E ms | 插桩TPOT ms | Decode Event总量 ms | 匹配GPU活动并集总量 ms | Kernel/活动之间内部空隙总量 ms |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(kernelrows)}

所有Profiler请求均单独报告，不并入A/B或普通快慢状态均值。Profiler会增加开销，采集本身也可能改变提交节奏，参见 [PyTorch2.6 Profiler文档](https://docs.pytorch.org/docs/2.6/profiler.html)；因此不能把其绝对TPOT当作历史普通慢态的复现。

八份插桩轨迹的31步Decode平均：Event总量{x['profiler_decode_aggregates']['Event_sum_ms']['mean']:.3f}ms，已匹配GPU活动并集{x['profiler_decode_aggregates']['traced_GPU_activity_union_sum_ms']['mean']:.3f}ms，活动之间内部空隙{x['profiler_decode_aggregates']['internal_GPU_gaps_sum_ms']['mean']:.3f}ms。每份完整请求均53642个Kernel及对应Launch API。按相邻同stream GPU活动与后继Launch时间对齐，平均有{x['profiler_decode_aggregates']['GPU_gap_before_next_API_start_ms']['mean']:.3f}ms的空隙发生在后继API尚未开始前，{x['profiler_decode_aggregates']['GPU_gap_during_next_API_ms']['mean']:.3f}ms与该API调用重叠，{x['profiler_decode_aggregates']['GPU_gap_after_next_API_end_ms']['mean']:.3f}ms发生在API返回后、GPU尚未执行期间。

这是**插桩状态下**提交不连续的直接时间线证据：大量GPU空隙不能称为纯GPU算术耗时，也不等于OS抢占。它仍不能证明历史未插桩65→74ms的差额全来自CPU提交，更不能把Profiler制造的时间直接扣减为正式结果。

映射方法：先选择CPU user_annotation（同名GPU annotation另存），并将65个cudaEventRecord中带aten External id的1个算子内部记录单列，保留原外层64个Event的**CPU提交顺序**取得32个提交窗口；将窗口内CUDA Runtime/Driver API correlation id链接到GPU活动，再计算并集及间隙。它没有伪造GPU Event绝对时间戳。Event减已追踪活动的剩余时间还含边缘开销和未映射活动，不能全部命名为CPU等待；具体每步映射数量在 `profiler_per_step.csv`。首次CPU解析因未区分同名annotation而失败的摘要另存 `profiler_summary_parse_attempt_01.json`，修正后8份均完成映射，没有重跑GPU或修改原始trace。

`profiler_summary.json`同时给出Launch次数、API耗时分布、提交间隔、GPU kernel总耗时、Top Kernel名称、GEMM/softmax/normalization/copy等aten算子类别。当前为降低采样扰动关闭shapes/stack；没有精确模块标记，不能把所有GEMM无依据地分成Attention与MLP。softmax等可识别算子提供局部线索，但完整Attention/MLP语义时间对照仍缺失。

例如请求70全程（含Prefill）GPU Kernel类别累计：GEMM/matmul 745.337ms，softmax 7.768ms，copy/concat 59.184ms，已按aten归类的normalization 0.266ms，其他/逐元素81.057ms。RMSNorm拆成的逐元素操作落在其他类别，故不能将0.266ms解释为所有归一化成本。完整逐名Kernel统计保留在JSON。

CUDA Event是在GPU stream中两个标记之间的持续时间，既可能包含Kernel运行，也可能包含GPU尚无下一条工作时的空隙。CPU侧CUDA API持续时间、CPU提交间隔和Kernel执行时间是不同量；Launch API到Kernel开始还可能包含已排队工作。解释依据见 [NVIDIA官方时间线说明](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)。本次trace提供了实际分解；若没有同状态的快慢双侧采样，就不能把历史约9ms/token差额全部分配给某一种机制。

## D：原运行器与外围开销

冻结 `e2e32_runtime.generate` 原样导入，SHA `513a122ff5f4da008b74e4d5560b0b6ef50baf8ccf7feb2fa766a0cdcdc51a3b`。模型、Attention、KV扩展及原同步/Event位置均未改。每请求新建position/mask、动态KV及Event，Native对get_image_features的临时Patch沿用原实现，无正式观察Hook或算法controller。

原 `run_native.py` 的clean仅断言函数身份和无Hook；gc.collect/empty_cache只在Session finally执行。该代码没有每请求显式清缓存，因此“每次被empty_cache冷启动”解释不成立。自动GC没有关闭，新诊断用回调记录GC事件，原件未变。

诊断外围在本地ext4目录逐条flush/fsync JSONL，由独立遥测进程采样，结束后同步持久盘；不在每请求间访问NFS或执行UUID子进程查询。**这与原外围执行器不同**，但所有A/B条件均保持这一外围设置，内层计时路径一致。本轮未将“是否NFS/是否UUID查询/是否单输入”做完整因素分离，不能把全部改善归因某一项。

| A/B请求外围操作/范围（含预热） | 平均 ms | 最大 ms |
| --- | --- | --- |
{over}

H2D列为.to调用耗时，包含调用/同步开销，不是PCIe纯传输带宽测试。wrapper时间包括原计时结束后的Token/KV离线验收，因此大于内层E2E。日志和同步写盘始终在原计时外，完整CPU准备/主动等待/真实请求间隔分别保留。

## 根因可信度、可操作建议与正式复测条件

**已证实**：固定输入和Token下存在记录中的延迟变化；连续A有一段较稳定窗口；空闲与GPU电源状态恢复确实发生；不存在足以解释全部差额的输入/输出长度改变或每请求显式GC/empty_cache。

**与证据一致但未证实**：GPU唤醒/功耗管理可能贡献；CPU/驱动提交节奏和GPU活动间隙可能贡献；原外围NFS同步及混合输入可能影响占空比/分配器。不能把一般机制当作这次物理根因。

**本轮不支持的强结论**：所有60秒空闲都变慢、SM频率一个数即可解释全部差额、CPU被抢占数百毫秒、Profiler慢态就是历史慢态、连续28题通过即永久修复。

**仍待验证**：同一未插桩慢态附近的可比快慢GPU活动时间线；宿主机实际CPU频率/驱动行为；原双探针、NFS外围因素各自的独立作用。验收不要求强行找到唯一根因，本轮按真实证据保留不确定性。

以下按下一步调查的证据优先级排列，并非已确认物理根因的概率排名：

| 优先级 | 候选因素 | 直接证据与可信度边界 |
| --- | --- | --- |
| 1 | CPU/驱动提交节奏及运行时状态 | 插桩C中大量空隙发生在下一Launch API开始前，且Profiler之后普通探针持续变慢；对C的证据强，对历史65→74ms的归因仍弱，缺少未插桩快慢配对。 |
| 2 | GPU动态功耗/频率与唤醒交互 | 慢态49有P8→P2和SW Power Cap记录；但相同唤醒、功耗标记也出现在快速反例中，且慢化在第6步Decode才形成，只能视为可能贡献。 |
| 3 | 外围I/O、请求占空比及混合输入状态 | 新外围连续A的28次稳定；原NFS、UUID查询、双输入没有逐因素对照，不能认定哪一项导致历史漂移。 |
| 低 | 本机OS主线程排队或已观测cgroup配额 | 慢态49排队仅0.276ms、可见配额无throttling，作为数百毫秒差额的单一解释不受支持；宿主机隐藏限制仍未排除。 |

最合理的工作假设是运行时/提交节奏与设备动态状态的交互，而非一个已证实的单一升频原因；目前没有足够证据指定唯一最可能物理根因。

建议优先A+B+D，同时保留E：连续完整预热和稳态准入、减少计时外不必要空闲与同步I/O、未来算法块前后用Native包围配对并保留多Session统计。Profiler必须与正式测量分进程，结束后不直接恢复正式计时；本轮C的后继探针变化支持隔离这一保守操作。尚未验证一个能消除所有漂移的修复方法。C（固定GPU/CPU条件）仅在将来证据与权限支持时另行申请，本轮没有更改。

推荐正式协议只写入 `recommended_protocol.json`：每Session至少60次完整预热且累计生成计时≥180秒，三个互不重叠20请求窗口逐输入检验离散度/趋势，通过后独立20次建基线；每5题做6次双探针校准，保留2%双侧、连续两窗暂停规则。发生>1秒未解释间断需停止或重新作为新阶段验证稳态，不能静默重设旧基线；仍用原30题×3独立Session、逐轮均值及三轮均值中位数。新正式实验应新建目录，不能接续旧第16题冒充同一模型加载Session。

窗口对两个原冻结探针分别统计E2E、TPOT及Event TPOT：robust CV=1.4826×MAD/中位数≤1%，相邻窗口中位数变化≤1%，三个窗口中位数范围/其中位数≤2%；不得合并不同输入的绝对延迟。60次和180秒为同时满足的下限，不是保证稳定的固定魔法次数；最大180次或15分钟仍未准入就停止。保留2%门禁是合理的漂移检测要求，当前失效点主要是原基线准入不足，而不是应该放宽门禁。本建议尚未经正式三Session验证。

## 诊断图

![全部84次诊断时间线，A/B与插桩阶段分别展示](diagnostic_timeline.png)

![8次60秒条件逐步Event与插桩区间分解](decode_and_profiled_gaps.png)

## 保存、原件与GPU释放

持久目录 `{D}`；原始本地暂存 `/home/featurize/ate-local-diagnostics/native_e2e32_3090_rootcause_diagnostic`，结束后已同步本轮输出。保留protocol/environment、全部requests及attempts、CPU/GPU原始CSV、逐请求与条件CSV、Profiler原始文件/汇总、GC和外围日志、源码与SHA清单。

原实验 `{O}` 及原审计 `{A}` 均只读，保留原15条历史稳定性失败结果；最终SHA/Checkpoint验收见 `validation.json` 和 `protected_files_after.json`。没有Push/PR、修改main或关闭服务器。GPU释放验收以 `gpu_release.json` 及最终 `validation.json` 的实际进程查询为准，不以模型对象置空推断成功。
'''
(D/'report_zh.md').write_text(report)
print(json.dumps(dict(report_bytes=(D/'report_zh.md').stat().st_size,recommendation_file='recommended_protocol.json',root_cause_uniquely_identified=False,formal_resumed=False)))
