"""CPU-only exports and verification of this single session, including failed attempts."""
from pathlib import Path
import json,csv,gzip,sqlite3,statistics,hashlib,subprocess,datetime,math
D=Path('/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090');U=D.parent/'unified_e2e32_4090'
def read(n):return json.loads((D/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(n,v):(D/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(n,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with (D/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def jl(n):return [json.loads(l) for l in (D/n).read_text().splitlines() if l.strip()] if (D/n).exists() else []
def fmt(v,d=2):return '—' if v is None else f'{v:.{d}f}'
def avg(xs):return statistics.mean(xs) if xs else None
def metric(r,m):return r['generation']['cache']['live_bytes']/2**20 if m=='KV_MiB' else r['generation'][m]
plan=read('protocol.json');completion=read('completion.json');assert (D/'runner.exit').exists(),'Do not analyze a still-running session'
db=sqlite3.connect('file:'+str(D/'checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True);records=[]
for rid,cat,key,method,payload,h in db.execute('SELECT * FROM records ORDER BY id'):
 r=json.loads(payload);assert digest(r)==h and r['request_id']==rid and r['category']==cat and r['key']==key and r['method']==method;records.append(r)
db.close();formal=[r for r in records if r['category']=='formal'];metrics=['KV_MiB','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','E2E_ms'];methods=plan['requested_methods'];labels=plan['method_labels'];indexed={(r['key'],r['method']):r for r in formal};gate=read('native_stability_gate.json');smoke=read('smoke_summary.json');drift=read('drift_summary.json') if (D/'drift_summary.json').exists() else dict(status='NOT_STARTED',checks=[])
assert len(indexed)==len(formal)
with gzip.open(U/'formal_requests.jsonl.gz','rt') as f:historical_records={(r['key'],r['method']):r for r in map(json.loads,f)}
for r in formal:assert r['generation']['cache']['live_bytes']==historical_records[r['key'],r['method']]['generation']['cache']['live_bytes']
if not (D/'drift_summary.json').exists():save('drift_summary.json',dict(drift,reason='Formal run stopped before the first scheduled calibration after five complete inputs.',expected_checks=6,completed_checks=0))
for r in records:
 g=r['generation'];assert r['GPU_UUID']==plan['GPU_UUID'] and g['prefill_calls']==g['vision_forward_calls']==1 and g['real_decode_calls']==31 and g['output_tokens']==32 and g['all_logits_finite']
 assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6 and abs(statistics.mean(g['step_event_ms'][1:])-g['CUDA_Event_Decode_TPOT_ms'])<1e-6
 live=0
 for row in g['cache']['layers']:
  if row['K_shape'] is None:assert row['V_shape'] is None and row['live_bytes']==row['used_tokens']==0;continue
  assert row['K_shape']==row['V_shape'] and row['dtype']=='torch.float16';shape=row['K_shape'];recount=2*shape[0]*shape[1]*row['used_tokens']*shape[3]*2;assert recount==row['live_bytes'];live+=recount
 assert live==g['cache']['live_bytes']==g['cache']['storage_bytes']
 if r['method']=='Native':assert r['historical_tokens_exact'] is True
for r in formal:assert not r['generation']['formal_layer_observer_hooks'] and not r['generation']['smoke']
for path,h in read('source_manifest.json')['frozen_sources'].items():assert sha(path)==h,path
for n,h in plan['entry_source_SHA256'].items():assert sha(D/'code'/n)==h,n
for n,h in plan['reference_files_SHA256'].items():assert sha(U/n)==h,n
for path,h in read('historical_protected_SHA256.json').items():assert sha(path)==h,path
per=[];layers=[]
for r in formal:
 v=dict(key=r['key'],benchmark=r['benchmark'],method=r['method'],method_label=labels[r['method']],session=1,GPU_UUID=r['GPU_UUID'],formal_index=r['formal_index'],input_index=r['input_index'],order_in_input=r['order_in_input'],input_tokens=r['input_tokens'],output_tokens=32,real_decode_calls=31,begin_ns=r['begin_ns'],end_ns=r['end_ns'],gap_s=r['gap_s'],main_thread_runqueue_ms=r['main_thread_runqueue_ms'],sample_sha256=r['sample_sha256'],input_ids_SHA256=r['input_hashes']['input_ids'],attention_mask_SHA256=r['input_hashes']['attention_mask'],pixel_values_SHA256=r['input_hashes']['pixel_values'])
 v.update({m:metric(r,m) for m in metrics});v.update(KV_storage_MiB=r['generation']['cache']['storage_bytes']/2**20,visual_representations_mean_per_layer=r['generation']['cache']['visual_tokens_sum']/32,wall_minus_Event_TPOT_ms=metric(r,'TPOT_ms')-metric(r,'CUDA_Event_Decode_TPOT_ms'));per.append(v)
 for row in r['generation']['cache']['layers']:
  row=dict(row);row.update(key=r['key'],method=r['method'],GPU_UUID=r['GPU_UUID'])
  if r['method']=='SparseVLM':row['position_convention']='compact (merged visual representations); annotation corrected only in CPU export, raw frozen label retained'
  for k in ['K_shape','V_shape']:row[k]=json.dumps(row[k])
  layers.append(row)
csvout('efficiency_per_input.csv',per, list(per[0]) if per else ['key','benchmark','method','KV_MiB','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','E2E_ms']);csvout('kv_layout_per_layer.csv',layers,list(dict.fromkeys(k for r in layers for k in r)) if layers else ['key','method','layer','used_tokens','live_bytes'])
summaries=[]
for m in methods:
 for split in ['All30','GQA','TextVQA']:
  rs=[r for r in formal if r['method']==m and (split=='All30' or r['benchmark']==split)];s=dict(method=m,method_label=labels[m],split=split,n=len(rs),complete_session=completion['status']=='PASS' and len(formal)==240)
  for k in metrics:
   vs=[metric(r,k) for r in rs];s.update({k+'_mean':avg(vs),k+'_median':statistics.median(vs) if vs else None,k+'_stdev':statistics.stdev(vs) if len(vs)>1 else None,k+'_min':min(vs) if vs else None,k+'_max':max(vs) if vs else None})
  summaries.append(s)
for s in summaries:
 common=[key for key in plan['keys'] if (key,s['method']) in indexed and (key,'Native') in indexed and (s['split']=='All30' or indexed[key,s['method']]['benchmark']==s['split'])];s['paired_inputs_vs_Native']=len(common)
 for k in metrics:
  nv=avg([metric(indexed[key,'Native'],k) for key in common]);mv=avg([metric(indexed[key,s['method']],k) for key in common]);s[k+'_reduction_vs_Native_pct']=100*(1-mv/nv) if nv and mv is not None else None
csvout('efficiency_summary.csv',summaries)
pairs=[];wins={}
for m in methods:
 common=[k for k in plan['keys'] if (k,m) in indexed and (k,'Native') in indexed];wins[m]=dict(paired_inputs=len(common))
 for k in metrics:wins[m][k+'_strictly_lower']=sum(metric(indexed[key,m],k)<metric(indexed[key,'Native'],k) for key in common)
 for key in common:
  row=dict(key=key,method=m,method_label=labels[m],benchmark=indexed[key,m]['benchmark'])
  for k in metrics:
   n=metric(indexed[key,'Native'],k);v=metric(indexed[key,m],k);row[k+'_Native']=n;row[k+'_method']=v;row[k+'_reduction_pct']=100*(1-v/n)
  pairs.append(row)
csvout('paired_comparison.csv',pairs,list(pairs[0]) if pairs else ['key','method','KV_MiB_reduction_pct','TTFT_ms_reduction_pct','TPOT_ms_reduction_pct','E2E_ms_reduction_pct']);save('paired_win_counts.json',wins)
timeline=[]
for r in records:
 v={k:r.get(k) for k in ['request_id','category','key','method','begin_ns','end_ns','gap_s','main_thread_runqueue_ms','early5_vs_late26_decode_median_change']};v.update({k:metric(r,k) for k in metrics});timeline.append(v)
csvout('all_request_timeline.csv',timeline)
with gzip.open(D/'all_requests.jsonl.gz','wt') as out:
 for r in records:out.write(json.dumps(r,ensure_ascii=False)+'\n')
gpu=jl('gpu_telemetry.jsonl');cpu=jl('cpu_telemetry.jsonl');processes=jl('processes_telemetry.jsonl');others=[r for r in processes if r.get('other_processes')];errors=[r for r in gpu+cpu+processes if 'error' in r]
ranges={}
for k in ['SM_MHz','memory_MHz','temperature_C','power_mW','power_limit_mW','GPU_util','memory_used']:
 vals=[r[k] for r in gpu if isinstance(r.get(k),(int,float))];ranges[k]=dict(min=min(vals),max=max(vals),mean=avg(vals)) if vals else None
telemetry=dict(GPU_UUID=plan['GPU_UUID'],GPU_samples=len(gpu),CPU_samples=len(cpu),process_samples=len(processes),ranges=ranges,P_states=sorted({r['pstate'] for r in gpu if 'pstate'in r},key=str),throttle_masks=sorted({r['throttle_mask'] for r in gpu if 'throttle_mask'in r},key=str),maximum_GPU_sample_gap_s=max((r.get('gap_s') or 0 for r in gpu),default=None),other_GPU_process_events=others,errors=errors,collectors={m:read(m+'_collector_summary.json') for m in ['gpu','cpu','processes'] if (D/(m+'_collector_summary.json')).exists()})
save('gpu_telemetry_summary.json',telemetry)
group={s['method']:s for s in summaries if s['split']=='All30'};historical=list(csv.DictReader((U/'efficiency_summary.csv').open()));hist={r['method']:r for r in historical if r['split']=='All30'}
complete=completion['status']=='PASS' and len(formal)==240 and gate['status']=='PASS' and smoke['status']=='PASS' and len(drift['checks'])==6 and not others and not errors
for pos,key in enumerate(plan['keys']):
 sub=sorted([r for r in formal if r['key']==key],key=lambda r:r['order_in_input']);want=methods[pos%8:]+methods[:pos%8]
 assert [r['method'] for r in sub]==want[:len(sub)],key
 if complete:assert len(sub)==8
assert len([r for r in records if r['category']=='stability_warmup'])<=120
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip();assert not apps,'GPU not released';release=read('gpu_release.json');assert not release['processes']
release=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,power.limit','--format=csv,noheader'],text=True).strip(),processes=apps,server_left_running=True);save('gpu_release_final.json',release)
validation=dict(status='PASS' if complete else 'NOT_FULLY_VALIDATED' if completion['status']=='PASS' else completion['status'],evidence_integrity='PASS',stability_admission=gate['status'],formal_completed=len(formal),formal_expected=240,formal_sessions=1,model_loads=completion['model_loads'],category_counts=completion['category_counts'],all_record_SHA_verified=True,all_physical_KV_layout_bytes_recounted=True,all_inputs_exact=read('input_verification.json')['status']=='PASS',all_Native_tokens_exact=True,Native_formal_tokens_verified=sum(r['method']=='Native' for r in formal),all_formal_hooks_absent=True,frozen_sources_unchanged=True,reference_4090_unchanged=True,historical_protected_files_unchanged=len(read('historical_protected_SHA256.json')),GPU_released=True,GPU_compute_processes=apps,GitHub_push=False,automatic_repeats=False,quality_benchmark_requests=0,eligible_as_complete_same_GPU_efficiency_evidence=complete,paired_win_counts=wins,telemetry_errors=len(errors))
save('validation.json',validation)
validation.update(extra_FastV_history_gate_triggered=any(r['method']=='FastV' and r.get('historical_tokens_exact') is False for r in formal),matched_4090_KV_byte_counts_verified=len(formal),formal_inputs_started=len({r['key'] for r in formal}),formal_inputs_fully_completed=sum(sum(r['key']==k for r in formal)==8 for k in plan['keys']),calibration_checks_completed=len(drift['checks']))
save('validation.json',validation)
def table(s):return '| '+s['method_label']+' | '+str(s['n'])+' | '+' | '.join(fmt(s[k+'_mean']) for k in ['KV_MiB','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','E2E_ms'])+' |'
reductions=['| '+labels[m]+' | '+' | '.join(fmt(group[m][k+'_reduction_vs_Native_pct']) for k in ['KV_MiB','TTFT_ms','TPOT_ms','E2E_ms'])+' |' for m in methods]
compare=[]
for m in methods:
 h=hist.get(m,{})
 compare.append('| '+labels[m]+' | '+' | '.join(fmt(float(h[k+'_mean'])) if h.get(k+'_mean') else '—' for k in ['KV_MiB','TTFT_ms','TPOT_ms','E2E_ms'])+' | '+fmt(group[m]['E2E_ms_mean'])+' |')
driftrows=[]
for c in drift['checks']:
 driftrows.append('| '+str(c['after_inputs'])+' | '+fmt(c['pooled_relative_change']['E2E_ms']*100,3)+' | '+fmt(c['pooled_relative_change']['TPOT_ms']*100,3)+' | '+fmt(max(abs(v) for x in c['relative_change'].values() for v in x.values())*100,3)+' | '+c['status']+' |')
anomalies=[]
for m in methods:
 vals=[r for r in pairs if r['method']==m]
 if len(vals)<5:continue
 ratios=[r['E2E_ms_method']/r['E2E_ms_Native'] for r in vals];med=statistics.median(ratios);mad=statistics.median(abs(x-med) for x in ratios)
 for r,ratio in zip(vals,ratios):
  if abs(ratio-med)>max(.05,6*1.4826*mad):anomalies.append(dict(method=m,key=r['key'],paired_E2E_ratio=ratio,median=med,MAD=mad,meaning='Statistical flag only; retained, not attributed to compatibility layer'))
save('anomaly_summary.json',dict(status='INSUFFICIENT_SAMPLE_FOR_ROBUST_SCREEN' if all(s['n']<5 for s in group.values()) else 'SCREENED',statistical_flags=anomalies,no_records_removed=True,no_independent_compatibility_overhead_ablation=True))
svh='；'.join(labels[m]+': '+', '.join(k.replace('_ms','')+'严格更低'+str(wins[m][k+'_strictly_lower'])+'/'+str(wins[m]['paired_inputs']) for k in ['TTFT_ms','TPOT_ms','E2E_ms']) for m in ['SVH','SVH-R'])
report=f'''# RTX3090 八方法统一推理效率复现

状态：**{validation['status']}**。稳定性准入：**{gate['status']}**；正式请求 **{len(formal)}/240**；一个模型加载、一个测量Session，没有自动重跑。

结果目录`{D}`。GPU `{plan['GPU_UUID']}`，与此前3090相同；4090参考GPU `{plan['reference_GPU_UUID']}`。模型`{plan['model']}`，Revision `{plan['revision']}`；FP16/Eager、PyTorch2.6.0+cu124、Transformers4.49.0、batch1、greedy、cache与logits_to_keep=1，固定32输出=1Prefill+31Decode，忽略EOS。冻结参考Commit `{plan['reference_commit']}`。

## {'主效率表候选' if complete else '部分效率数据：不供论文主表使用'}

下表均值来自本次正式请求；未完成行保持缺失或明确部分样本，不能作为完整主表。失败的第31条FastV原始记录也保留在内，不将它静默删除。KV为终止实际持久K/V有效Tensor字节，MiB=2^20字节；不是峰值显存或名义预算。Event列是31次Decode独立CUDA Event均值。

| Method | n | KV@32 MiB | TTFT ms | TPOT ms/token | Event TPOT ms/token | E2E@32 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table(group[m]) for m in methods)}

完整逐题、中位数、标准差、最小最大和GQA/TextVQA分项见`efficiency_per_input.csv`、`efficiency_summary.csv`；逐层布局见`kv_layout_per_layer.csv`。没有删除不利E2E或异常请求。

## 相对本轮Native的变化

正值表示减少/改善，负值表示变慢。此表为同一3090 Session**共同已测输入**上的均值比；部分结果按各方法与Native的共同输入计算，不能用4题Native均值直接作3题SparseVLM的分母。逐题配对百分比另存`paired_comparison.csv`，两者统计定义不同。未完成时这些数值仅作部分运行诊断。

| Method | KV减少% | TTFT改善% | TPOT改善% | E2E改善% |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(reductions)}

逐题优势：{svh}。分母不足30时不能声称全部30题更快。配对计数详见`paired_win_counts.json`。

## Smoke、稳定性与漂移

Smoke状态`{smoke['status']}`，原始记录在SQLite及`all_requests.jsonl.gz`。完整检查Native/BridgeNative、FastV禁用、V²Drop禁用的32Token与首Logits一致性，以及32层实际Hidden/Mask/RoPE/KV、finite、真实Decode及SVH/R L2–L5答案Attention物理跳算。Smoke观察Hook不进入正式计时；所有方法在相同双探针上各追加2次完整32Token无观察Hook预热。

Native稳定性阶段实际{completion['category_counts'].get('stability_warmup',0)}次，累计生成E2E计时{gate.get('summed_generation_E2E_s',0):.3f}秒；上限120。每20次为窗口，按固定GQA/TextVQA各10次独立统计E2E、TPOT与Event。至少60次且180秒，最近三个互不重叠窗口：robust CV≤1%、单请求偏离窗口中位数≤3%、相邻窗口均值变化≤1%、三窗均值范围≤2%；请求间断/遥测缺口≤1秒，并拒绝Native前5与后26次Decode中位数相差>5%的明显状态突变。全部评估（含未通过窗口）见`native_stability_gate.json`，没有选择最快状态。

通过准入后独立20次双探针建立基线，再每5题各输入重复3次校准，校准不计入240。相对基线同时检查按输入及合并的E2E/TPOT，任一>2%记录breach，连续两窗暂停；明显>5%探针/请求内状态突变立即暂停。没有自动重设基线。

| 已完成输入 | 合并E2E漂移% | 合并TPOT漂移% | 任一输入/指标最大绝对漂移% | 状态 |
| ---: | ---: | ---: | ---: | --- |
{chr(10).join(driftrows) if driftrows else '| — | — | — | — | 未启动正式校准 |'}

运行结束状态：`{completion['status']}`；失败信息：`{completion.get('failure','无')}`。原始失败堆栈见`completion.json`与`runner.log`，不得将BLOCKED/PARTIAL解释为正式通过。

本次停止来自复用旧4090入口时保留的**FastV逐Token历史相等硬断言**。本轮用户明确要求Native32Token与历史完全一致，对外部方法明确要求物理/No-op正确性；FastV这一额外跨GPU硬门禁比该要求更严格。它触发后进程已经退出，因此不能通过重新加载模型续写为同一Session，也没有自动重跑。输出不一致本身不能证明FastV算法错误。CPU核对见`stop_audit.json`，没有为调查该问题额外执行GPU生成。

具体记录：GQA:201759069、FastV、正式序号31，当前与历史都以`No</s>`开头，前5个Token一致；第6个Token开始不同，已位于首次EOS之后。该输入、Sample SHA、剪枝事件、32层终止KV形状和有效长度均相同。这里不能把后续文本差异认定为答案质量下降或算法实现错误，也未进行质量Benchmark。累计128次已执行Native请求（含Smoke/预热/基线/正式4次）全部32Token匹配；余下Native正式输入尚未执行。

这是入口校验策略造成的提前终止，当前证据未显示性能漂移触发停止。后续若获准新Session，应在新目录中事先冻结输出策略：Native历史32Token严格一致；外部方法历史输出差异单独记录，保留其Source/Input/No-op/布局/Finite硬门禁。最小候选补丁另存`proposed_native_history_gate.diff`，未修改本次实际执行入口，也未运行新GPU请求。

## 计时和实现边界

每次直接调用原`e2e32_runtime.generate`，运行器SHA `{plan['frozen_source_SHA256'].get(str(U/'code/e2e32_runtime.py'),plan['runner_source_SHA256']['e2e32_runtime.py'])}`。同步后在控制器准备前开始；Vision、Projector、Prefill、31Decode、argmax、必要控制器清理和同步均计入。加载、CPU图像/Tokenizer、H2D、逐条日志、最终离线KV审计均排除。核对每条E2E=TTFT+31×TPOT及Event计算。没有修改Attention、KV算法、Flash/SDPA/compile替换。

八方法配置沿用4090：FastV K3-192静态KV；SparseVLM官方recycling、实际平均206.5；V²Drop历史576→288→144→1演示点，**不是论文官方192配置**；ApET288→96/L16；DUET **C+all** 307→153→0；SVH L20退出与SVH-R L25退出/仅L11–L24视觉KV。未混入Retain128。

外部算法选择、聚类、回收、额外QKV与必要兼容管理开销均包含在相同计时边界内。统计异常标记{len(anomalies)}条，详情`anomaly_summary.json`；当前每方法只有3–4题，未达到预设至少5题的稳健异常筛查样本量，零标记不代表无异常；保留全部数据。没有执行独立兼容层剥离或Profiler，不能把TTFT差额直接归因为兼容层，也不能将稀疏KV等同于必然更低TTFT/E2E。

遥测为GPU5Hz、CPU1Hz、进程查询每2秒的三个独立进程，避免旧诊断中CLI阻塞GPU采样；实际GPU样本{len(gpu)}、CPU样本{len(cpu)}、最大GPU采样间隔{fmt(telemetry['maximum_GPU_sample_gap_s'],4)}秒，错误{len(errors)}、其他GPU进程事件{len(others)}。实际频率/温度/功耗/P-State/Throttle范围和采样器CPU开销见`gpu_telemetry_summary.json`与原始JSONL；不能宣称监控零开销。没有改GPU时钟、Power Limit、驱动或Governor。

## 与4090历史结果并列

以下仅并列原实测绝对量；不跨GPU计算算法Speedup，不替换原4090表的Native基线。模型、输入、配置和内层计时器一致，GPU与稳态门禁、外层监控/本地日志生命周期有明确差别。硬件差异和这些外层差别都限制直接归因。**本次未完成时，4090列是完整30题，而3090列只有前面已测子集，连题目组成也不同，不能由此推断硬件延迟比例或全30题算法结论。**

| Method | 4090 KV MiB | 4090 TTFT ms | 4090 TPOT ms | 4090 E2E ms | 本次3090 E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(compare)}

## 验收和用途

完整3090同机复现证据资格：**{complete}**。只有完整240、Smoke与稳定性准入通过、六次校准符合冻结停止规则、输入与源码一致、真实KV核对通过时才作为本次效率表候选；单Session不证明跨时段重复性。

46项冻结执行源码与原30题输入哈希逐项验证；Native正式Token核对{validation['Native_formal_tokens_verified']}/30。{validation['historical_protected_files_unchanged']}个历史保护文件未变，旧4090结果及失败3090的15条记录保持原状。Checkpoint和全部预热/Smoke/探针原始数据均保留。没有质量Benchmark、自然EOS、训练、参数搜索、Push/PR或Windows备份。

GPU释放：`{release['GPU']}`；计算进程为空，服务器保持运行。完整校验见`validation.json`，全目录SHA清单见`artifact_SHA256.json`。
'''
(D/'report_zh.md').write_text(report)
inventory={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='artifact_SHA256.json'};save('artifact_SHA256.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=inventory))
print(json.dumps(validation,ensure_ascii=False,indent=2))
