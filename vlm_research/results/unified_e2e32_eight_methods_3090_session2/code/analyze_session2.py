"""CPU-only exports and verification of this single session, including failed attempts."""
from pathlib import Path
import json,csv,gzip,sqlite3,statistics,hashlib,subprocess,datetime,math
D=Path('/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090_session2');U=D.parent/'unified_e2e32_4090'
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
 v=dict(key=r['key'],benchmark=r['benchmark'],method=r['method'],method_label=labels[r['method']],session=2,GPU_UUID=r['GPU_UUID'],formal_index=r['formal_index'],input_index=r['input_index'],order_in_input=r['order_in_input'],input_tokens=r['input_tokens'],output_tokens=32,real_decode_calls=31,begin_ns=r['begin_ns'],end_ns=r['end_ns'],gap_s=r['gap_s'],main_thread_runqueue_ms=r['main_thread_runqueue_ms'],sample_sha256=r['sample_sha256'],input_ids_SHA256=r['input_hashes']['input_ids'],attention_mask_SHA256=r['input_hashes']['attention_mask'],pixel_values_SHA256=r['input_hashes']['pixel_values'])
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
validation.update(FastV_full32_differences_recorded=sum(r['method']=='FastV' and r.get('historical_tokens_exact') is False for r in formal),matched_4090_KV_byte_counts_verified=len(formal),formal_inputs_started=len({r['key'] for r in formal}),formal_inputs_fully_completed=sum(sum(r['key']==k for r in formal)==8 for k in plan['keys']),calibration_checks_completed=len(drift['checks']))
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

# Independent CPU check of the new output rule on every preserved raw record.
import sys,random
sys.path.insert(0,str(D/'code'))
from history_policy import compare_history
policy_records=[]
for r in records:
 pol=r['history_acceptance'];assert pol['accepted'],(r['request_id'],pol)
 if r['method'] in ('Native','FastV'):
  h=historical_records[r['key'],r['method']]['generation']
  assert compare_history(r['method'],r['generation']['token_ids'],h['token_ids'],2)==pol
  if r['method']=='FastV':
   ev=r['FastV_physical_history_evidence'];assert ev['KV_layout_equal'] and ev['pruning_events_current']==ev['pruning_events_historical']
   policy_records.append(dict(request_id=r['request_id'],category=r['category'],key=r['key'],acceptance=pol,physical_evidence=ev))
audit=read('patch_audit.json');assert audit['status']=='PASS' and audit['request_prefix_AST_unchanged']
assert audit['new_entry_SHA256']==plan['entry_source_SHA256']['run_3090.py']
save('FastV_history_acceptance.json',dict(status='PASS',records=policy_records,full32_differences=sum(not r['acceptance']['full32_exact'] for r in policy_records),post_EOS_only_differences=sum(r['acceptance']['post_EOS_only_difference'] for r in policy_records),all_through_first_EOS_exact=True))
validation.update(session_index=2,FastV_first_EOS_inclusive_policy='PASS',FastV_all_requests_checked=len(policy_records),FastV_formal_post_EOS_only_differences=sum(r['category']=='formal' and r['acceptance']['post_EOS_only_difference'] for r in policy_records),policy_CPU_AST_audit='PASS',policy_GPU_smoke=read('policy_smoke_validation.json')['status'],previous_PARTIAL_results_unchanged=True)
save('validation.json',validation)

uncertainty=[]
def quantile(xs,p):
 xs=sorted(xs);i=(len(xs)-1)*p;lo=int(i);hi=min(lo+1,len(xs)-1);return xs[lo]+(xs[hi]-xs[lo])*(i-lo)
for m in methods:
 sub=[r for r in pairs if r['method']==m];splits=[[i for i,r in enumerate(sub) if r['benchmark']==b] for b in ['GQA','TextVQA']]
 # Resample CPU statistics only, retaining the original benchmark mixture.
 rng=random.Random(309032);draws=[]
 for _ in range(5000):draws.append([rng.choice(g) for g in splits if g for _ in range(len(g))])
 for k in metrics:
  nv=[r[k+'_Native'] for r in sub];mv=[r[k+'_method'] for r in sub]
  if not nv:continue
  absdiff=[a-b for a,b in zip(nv,mv)];pct=[100*(1-b/a) for a,b in zip(nv,mv)];boots=[100*(1-sum(mv[i] for i in d)/sum(nv[i] for i in d)) for d in draws]
  w=sum(x>0 for x in absdiff);loss=sum(x<0 for x in absdiff);nonzero=w+loss
  signp=min(1,2*sum(math.comb(nonzero,i) for i in range(min(w,loss)+1))/2**nonzero) if nonzero else 1
  uncertainty.append(dict(method=m,metric=k,n=len(sub),Native_mean=avg(nv),method_mean=avg(mv),paired_reduction_absolute_mean=avg(absdiff),paired_reduction_absolute_median=statistics.median(absdiff),paired_reduction_absolute_stdev=statistics.stdev(absdiff) if len(sub)>1 else None,paired_reduction_pct_mean=avg(pct),paired_reduction_pct_median=statistics.median(pct),paired_reduction_pct_stdev=statistics.stdev(pct) if len(sub)>1 else None,ratio_of_means_reduction_pct=100*(1-avg(mv)/avg(nv)),bootstrap_ratio_reduction_CI95_low=quantile(boots,.025),bootstrap_ratio_reduction_CI95_high=quantile(boots,.975),strictly_faster=w,strictly_slower=loss,ties=len(sub)-nonzero,paired_sign_test_two_sided_p=signp,bootstrap_replicates=5000,bootstrap_seed=309032,scope='Fixed inputs, stratified GQA/TextVQA paired bootstrap. Single session: excludes cross-session repeatability and temporal confounding; no multiple-testing adjustment.'))
csvout('paired_uncertainty.csv',uncertainty,list(uncertainty[0]) if uncertainty else ['method','metric','n','paired_reduction_absolute_mean','paired_reduction_pct_stdev','bootstrap_ratio_reduction_CI95_low','bootstrap_ratio_reduction_CI95_high'])
for r in pairs:
 for k in metrics:r[k+'_Native_minus_method']=r[k+'_Native']-r[k+'_method']
csvout('paired_comparison.csv',pairs,list(pairs[0]) if pairs else ['key','method'])
e2eunc={r['method']:r for r in uncertainty if r['metric']=='E2E_ms'}
uncrows=['| '+labels[m]+' | '+fmt(e2eunc[m]['ratio_of_means_reduction_pct'])+' | ['+fmt(e2eunc[m]['bootstrap_ratio_reduction_CI95_low'])+', '+fmt(e2eunc[m]['bootstrap_ratio_reduction_CI95_high'])+'] | '+fmt(e2eunc[m]['paired_reduction_absolute_stdev'])+' | '+str(e2eunc[m]['strictly_faster'])+'/'+str(e2eunc[m]['n'])+' |' for m in methods if m in e2eunc]
maxdrift=max((abs(v) for c in drift['checks'] for d in c['relative_change'].values() for v in d.values()),default=None)
validation['maximum_abs_per_input_calibration_drift_pct']=maxdrift*100 if maxdrift is not None else None
validation['drift_checks_all_below_2pct']=len(drift['checks'])==6 and all(not c['breach'] for c in drift['checks'])
save('validation.json',validation)
postdiff=[r for r in policy_records if r['category']=='formal' and r['acceptance']['post_EOS_only_difference']]
formal_times=[(r['begin_ns'],r['end_ns']) for r in formal]
fgpu=[g for g in gpu if any(a<=g['at_ns']<=b for a,b in formal_times)]
franges={}
for k in ['SM_MHz','memory_MHz','temperature_C','power_mW']:
 v=[g[k] for g in fgpu if isinstance(g.get(k),(int,float))];franges[k]=dict(min=min(v),max=max(v),mean=avg(v)) if v else None
save('formal_telemetry_ranges.json',dict(GPU_samples=len(fgpu),ranges=franges,P_states=sorted({g['pstate'] for g in fgpu if 'pstate'in g},key=str),throttle_masks=sorted({g['throttle_mask'] for g in fgpu if 'throttle_mask'in g},key=str)))
gate_diagnosis='未触发稳态准入失败。'
if (D/'gate_failure_audit.json').exists():
 ga=read('gate_failure_audit.json');parts=[]
 for w in ga['flagged_windows']:
  st=w['window_stats'];parts.append(f"Native预热{w['first_warmup']}–{w['last_warmup']}，{w['key']}：E2E/TPOT/Event robust CV分别{100*st['E2E_ms']['robust_CV']:.3f}%/{100*st['TPOT_ms']['robust_CV']:.3f}%/{100*st['CUDA_Event_Decode_TPOT_ms']['robust_CV']:.3f}%；最大单请求偏离分别{100*st['E2E_ms']['max_abs_deviation_from_median']:.3f}%/{100*st['TPOT_ms']['max_abs_deviation_from_median']:.3f}%/{100*st['CUDA_Event_Decode_TPOT_ms']['max_abs_deviation_from_median']:.3f}%。")
  for r in w['requests']:
   if max(abs(v) for v in r['relative_to_window_median'].values())>.03:
    z=r['metrics'];parts.append(f"超限请求：Native预热{r['warmup_index']}（全局请求{r['request_id']}），E2E={z['E2E_ms']:.3f}ms、TPOT={z['TPOT_ms']:.5f}ms/token、Event={z['CUDA_Event_Decode_TPOT_ms']:.5f}ms/token；主线程runqueue={r['main_thread_runqueue_ms']:.6f}ms，期间GPU采样{r['GPU_samples']}个，范围{r['GPU_ranges']}，P-State={r['P_states']}，原始Throttle mask={r['throttle_masks']}。")
 adjacent=[]
 for ev in gate['evaluations']:
  for key in plan['smoke_keys']:
   for m in ['E2E_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']:
    ws=ev['windows']
    for a,b in zip(ws,ws[1:]):
     change=b['per_input'][key][m]['mean']/a['per_input'][key][m]['mean']-1
     if abs(change)>.01:adjacent.append(dict(after_warmup=ev['after_warmup'],key=key,metric=m,from_window=[a['first_warmup'],a['last_warmup']],to_window=[b['first_warmup'],b['last_warmup']],change_pct=100*change))
 save('admission_adjacent_mean_failures.json',dict(records=adjacent))
 parts.append('相邻窗口均值超1%的原始数值见admission_adjacent_mean_failures.json。同一61–80或81–100窗口在滚动最近60请求的后续评估中重复出现，不能当作多次独立异常。')
 parts.append('CUDA Event耗时也同步增加；采样时钟未显示该超限请求内明显降频，CPU主线程runqueue也很小。这些证据不足以确立硬件、CPU launch或动态频率中的具体根因；未追加GPU诊断，也没有放宽阈值。')
 gate_diagnosis='\n\n'.join(parts)
report=f'''# RTX3090 八方法统一效率实验 Session 2

状态：**{validation['status']}**；本轮正式请求 **{len(formal)}/240**；一个模型加载、一个新的正式Session。原31条PARTIAL与旧4090、旧3090记录未变。稳定性准入 **{gate['status']}**，Smoke **{smoke['status']}**，正式校准 **{len(drift['checks'])}/6**。完整复现证据资格：**{complete}**。

持久化结果目录：`{D}`。本地测量Spool独立目录：`{plan['local_spool']}`。GPU `{plan['GPU_UUID']}`，与此前3090相同。未重下模型或数据。

## {'完整效率表候选' if complete else '部分数据，不供正式主表使用'}

KV来自32Token终止时实际K/V Tensor形状、dtype和有效长度，MiB=2^20字节。每个请求固定32Token，1次Prefill+31次真实Decode，忽略EOS。全部均值和逐题数据均保留，包括不利结果。

| Method | n | KV@32 MiB | TTFT ms | TPOT ms/token | Event TPOT ms/token | E2E@32 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table(group[m]) for m in methods)}

各Benchmark分项、均值、中位数、标准差、最小最大值见`efficiency_summary.csv`；逐题见`efficiency_per_input.csv`，32层物理布局见`kv_layout_per_layer.csv`。所有已测正式请求的实际KV有效字节与4090同题同方法记录相符。

## 相对同轮Native

正值为减少/改善，负值为变慢；使用同题配对的均值比。部分数据仅限共同已测输入，不把未测输入混入分母。

| Method | KV减少% | TTFT降低% | TPOT降低% | E2E降低% |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(reductions)}

逐题优势：{svh}。只有分母30且对应指标30/30才可说本轮全部30题更低。方法整体均值更低也不代表每题均更低。

## 配对差与不确定性

下表保留同题Native−方法的E2E配对差标准差（ms）以及E2E均值比改善的CPU分层配对Bootstrap 95%区间（5000次，种子309032，GQA/TextVQA构成固定）。它描述这30道固定题的组成不确定性，不能代替多Session测量、估计跨时段重复性或排除轮换顺序的时间混杂。正负方向及小于约1%–2%的改善均如实报告，不直接宣称显著加速。

| Method | E2E均值改善% | Bootstrap 95%区间% | 配对差标准差ms | E2E更低题数 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(uncrows)}

全部KV/TTFT/TPOT/Event/E2E的配对绝对差、百分比、离散度、方向计数、符号检验及Bootstrap区间见`paired_comparison.csv`和`paired_uncertainty.csv`。没有因为不利结果删题；未做多重检验校正。

FastV、V²Drop、ApET与DUET的实际E2E变化如上表，不能从KV减少直接推断加速。SparseVLM的TTFT、Decode TPOT和Event按同一边界实测，存在的选择、聚类、回收及兼容管理开销都包含在内；未做兼容层剥离或Profiler，不能把延迟差额全部归因于兼容层。异常筛查见`anomaly_summary.json`：统计标记{len(anomalies)}条，全部保留；该筛查不构成算法正确性证明。

## 本轮准入失败的CPU核对

{gate_diagnosis}

## FastV最小验收修订与Smoke

上轮停止由额外FastV跨GPU全32Token相等断言触发：首个EOS是0-based位置1，首次差异位置5，输入和KV/剪枝一致。上轮Native-only候选补丁未采用；本轮按新授权使用更严格的FastV首EOS规则。

Native继续严格32Token历史一致。FastV比较截至第一个EOS并包括EOS，EOS位置必须一致；任一EOS前差异立即停止；两者均无EOS时必须32Token完全一致。只有EOS后差异被允许且完整记录Token IDs、首次差异位置、全部EOS位置、KV/剪枝及耗时数值。其他Smoke历史比较、No-op/有限Logits/实际Hidden/Mask/RoPE/KV/真实Decode/物理Attention跳算检查均保留。

CPU10例及AST回归检查通过，证明冻结`generate`调用之前及调用后原有计时/有限值/布局计算未改，稳态门禁函数未改；详情`patch_audit.json`和`session2_entry.diff`。所有新增验收在`generate()`返回后执行，未改变其计时区间、算法或生成结果。额外使用上轮失败输入进行一次有观察Hook的FastV32Token Smoke，状态`{read('policy_smoke_validation.json')['status']}`；Smoke不计入240，Hook不进入正式请求。

本轮正式FastV仅EOS后存在历史差异的请求数：{len(postdiff)}。每条完整证据见`FastV_history_acceptance.json`与原始Checkpoint，未隐去差异。Native正式32Token历史一致性已核对{validation['Native_formal_tokens_verified']}/30。

## 新进程预热、独立基线与六次漂移

八方法统一双探针各2次无观察Hook预热。Native稳定性阶段实际{completion['category_counts'].get('stability_warmup',0)}次完整32Token，累计生成E2E计时{gate.get('summed_generation_E2E_s',0):.3f}秒；最少60次且180秒，上限120。最近三个互不重叠20请求窗口各含固定GQA/TextVQA各10次；逐输入检查E2E/TPOT/Event：robust CV≤1%、单请求偏离中位数≤3%、相邻均值变化≤1%、三窗均值范围≤2%，并检查>5%请求内Native状态跳变、>1秒请求间断/遥测缺口。每次失败准入评估也保留在`native_stability_gate.json`。

通过后独立20次固定双探针建立本轮基线，未使用旧进程基线或指定最快状态。每5题执行双探针各3次，按输入及合并检查E2E与TPOT，连续两窗超过2%暂停，明显>5%探针/请求内Native状态突变立即暂停；不重设基线、不挑最快区间。

| 已完成输入 | 合并E2E漂移% | 合并TPOT漂移% | 任一输入/指标最大绝对漂移% | 状态 |
| ---: | ---: | ---: | ---: | --- |
{chr(10).join(driftrows) if driftrows else '| — | — | — | — | 尚未启动 |'}

最大按输入绝对校准偏移：{fmt(None if maxdrift is None else 100*maxdrift,3)}%。原始探针与基线均保留。GPU5Hz、CPU1Hz、进程查询每2秒独立采集，GPU样本{len(gpu)}，最大采样间隔{fmt(telemetry['maximum_GPU_sample_gap_s'],4)}秒；遥测错误{len(errors)}，其他GPU计算进程事件{len(others)}。正式请求期间频率、温度、功耗、P-State/Throttle见`formal_telemetry_ranges.json`，全部原始遥测保留。

## 与4090并列

模型`{plan['model']}`固定Revision `{plan['revision']}`；FP16/Eager，PyTorch2.6.0+cu124、Transformers4.49.0、batch1、greedy、cache=True、logits_to_keep=1。30题输入SHA与原4090完全一致；冻结Commit `{plan['reference_commit']}`。所有方法直接复用原`e2e32_runtime.generate`，SHA `513a122ff5f4da008b74e4d5560b0b6ef50baf8ccf7feb2fa766a0cdcdc51a3b`。V²Drop为历史576→288→144→1演示点，非论文官方192；DUET为C+all；没有混入Retain128。

计时从GPU输入就绪并同步后、控制器初始化前开始，到第32Token生成、必要控制器清理并同步后结束。Vision/Projector/Prefill/31Decode/argmax及必要控制器开销计入；加载、图像读取、CPU Processor/Tokenizer、H2D、日志写盘、离线KV审计排除。逐条校验E2E=TTFT+31×TPOT，并独立记录Event。

| Method | 4090 KV MiB | 4090 TTFT ms | 4090 TPOT ms | 4090 E2E ms | 本轮3090 E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(compare)}

硬件不同，不能由跨GPU绝对延迟计算算法Speedup；不替换旧4090主表基线。算法、模型和内层计时一致；稳态准入、外层监控和本地日志生命周期不同，限制硬件归因。若本轮PARTIAL，题目组成也与完整4090不一致。

## 保存、边界与释放

状态`{validation['status']}`，运行退出状态`{completion['status']}`，失败信息`{completion.get('failure','无')}`。完整PASS要求240条正式请求、30题输入和Native历史一致、Smoke/物理检查与稳态准入通过、六次校准符合冻结停止规则。单Session不能证明跨时段可重复性。

46项冻结执行源码SHA未变；历史保护文件{validation['historical_protected_files_unchanged']}项全部验证未变，包含上轮31条PARTIAL的完整目录。全部尝试、失败堆栈、Checkpoint、源码、准入、独立基线、逐题结果和SHA清单已保存。没有继续上轮统计，没有自动第三轮，没有更改GPU时钟/功耗/驱动/Governor，没有质量Benchmark/自然EOS/训练/调参/GitHub Push/PR/Windows备份。

GPU释放查询：`{release['GPU']}`；计算进程为空，服务器保持运行。全目录清单`artifact_SHA256.json`，验收`validation.json`。
'''
(D/'report_zh.md').write_text(report)
inventory={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='artifact_SHA256.json'};save('artifact_SHA256.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=inventory))
print(json.dumps(validation,ensure_ascii=False,indent=2))
