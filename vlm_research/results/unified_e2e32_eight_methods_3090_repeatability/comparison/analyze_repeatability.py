"""CPU-only verification and two-session analysis. No GPU or torch imports."""
from pathlib import Path
import json,gzip,csv,sqlite3,statistics,hashlib,datetime,math,random,subprocess,sys,bisect,itertools,shutil
B=Path('/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090_repeatability');C=B/'comparison';U=B.parent/'unified_e2e32_4090'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jl(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []
def csvout(p,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r)) or ['status']
 with Path(p).open('w',newline='') as out:w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(rows)
def avg(v):return statistics.mean(v) if v else None
def q(v,p):
 if not v:return None
 v=sorted(v);i=(len(v)-1)*p;lo=int(i);hi=min(lo+1,len(v)-1);return v[lo]+(v[hi]-v[lo])*(i-lo)
def stats(v):return dict(n=len(v),mean=avg(v),median=statistics.median(v) if v else None,stdev=statistics.stdev(v) if len(v)>1 else None,P90=q(v,.9),min=min(v) if v else None,max=max(v) if v else None)
def f(v,n=2):return '—' if v is None else f'{v:.{n}f}'
def metric(r,m):return r['generation']['cache']['live_bytes']/2**20 if m=='KV_MiB' else r['generation'][m]
METRICS=['KV_MiB','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','E2E_ms']
assert read(B/'coordinator_status.json')['status']!='RUNNING','Do not analyze while either GPU session is running'
C.mkdir(exist_ok=True);p1=read(B/'session1/protocol.json');p2=read(B/'session2/protocol.json');assert p1['schedule_SHA256']==p2['schedule_SHA256'] and p1['entry_source_SHA256']==p2['entry_source_SHA256']
family=read(B/'protocol.json')
for name,h in family['session_protocol_SHA256'].items():assert sha(B/name/'protocol.json')==h
assert sha(B/'coordinator.py')==sha('/home/featurize/ate-local-experiments/repeatability_preparation/coordinator.py')==family['coordinator_SHA256']
assert sha(B/'authorization_zh.md')==family['authorization_SHA256']
METHODS=p1['requested_methods'];LABELS=p1['method_labels'];KEYS=p1['keys'];S={};summaryrows=[];within=[];cross=[];bootrows=[];flags=[]
with gzip.open(U/'formal_requests.jsonl.gz','rt') as src:H={(r['key'],r['method']):r for r in map(json.loads,src)}
sys.path.insert(0,str(B/'session1/code'))
from history_policy import compare_history
def resamples(keys,seed=309032):
 rng=random.Random(seed);groups=[[i for i,k in enumerate(keys) if H[k,'Native']['benchmark']==b] for b in ['GQA','TextVQA']]
 return [[rng.choice(g) for g in groups if g for _ in range(len(g))] for _ in range(5000)]
def CI(values):return dict(CI95_low=q(values,.025),CI95_high=q(values,.975))
def inventory(root):
 root=Path(root);save(root/'SHA256.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'SHA256.json'}))
for sn in [1,2]:
 D=B/('session'+str(sn));plan=read(D/'protocol.json')
 for name,h in plan['entry_source_SHA256'].items():
  target=D/'code'/name
  if not target.exists():shutil.copyfile(Path(plan['local_spool'])/'code'/name,target)
  assert sha(target)==h
 if not (D/'completion.json').exists():
  save(D/'validation.json',dict(status='NOT_STARTED',execution_complete=False,formal_completed=0));S[sn]=dict(started=False,complete=False,formal=[],idx={},summary={});continue
 assert (D/'runner.exit').exists(),'Worker has not exited'
 completion=read(D/'completion.json');db=sqlite3.connect('file:'+str(D/'checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True);records=[]
 for rid,cat,key,method,payload,h in db.execute('SELECT * FROM records ORDER BY id'):
  r=json.loads(payload);assert r['request_id']==rid and r['category']==cat and r['key']==key and r['method']==method and digest(r)==h;records.append(r)
 db.close();formal=[r for r in records if r['category']=='formal'];idx={(r['key'],r['method']):r for r in formal};assert len(idx)==len(formal)
 layoutrows=[];per=[];policy=[];correctness_failures=[];timeline=[]
 for r in records:
  g=r['generation'];assert r['session']==sn and r['GPU_UUID']==plan['GPU_UUID'];assert g['output_tokens']==32 and g['real_decode_calls']==31 and g['prefill_calls']==g['vision_forward_calls']==1 and g['all_logits_finite']
  assert r['output_token_ids_SHA256']==digest(g['token_ids']);assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6;assert abs(statistics.mean(g['step_event_ms'][1:])-g['CUDA_Event_Decode_TPOT_ms'])<1e-6
  nbytes=0
  for row in g['cache']['layers']:
   if row['K_shape'] is None:assert row['V_shape'] is None and row['used_tokens']==row['live_bytes']==0;continue
   assert row['K_shape']==row['V_shape'] and row['dtype']=='torch.float16';shape=row['K_shape'];b=2*shape[0]*shape[1]*row['used_tokens']*shape[3]*2;assert b==row['live_bytes'];nbytes+=b
  assert len(g['cache']['layers'])==32 and nbytes==g['cache']['live_bytes']==g['cache']['storage_bytes']
  if (r['key'],r['method']) in H:
   h=H[r['key'],r['method']];expected=compare_history(r['method'],g['token_ids'],h['generation']['token_ids'],2);assert expected==r['history_acceptance'];assert r['input_hashes']==h['input_hashes'] and r['sample_sha256']==h['sample_sha256'];assert nbytes==h['generation']['cache']['live_bytes']
   policy.append(dict(request_id=r['request_id'],category=r['category'],key=r['key'],method=r['method'],input_hashes=r['input_hashes'],history_acceptance=r['history_acceptance'],output_token_ids_SHA256=r['output_token_ids_SHA256']))
  if not r['history_acceptance']['accepted']:correctness_failures.append(dict(request_id=r['request_id'],reason='history_output_hard_gate',details=r['history_acceptance']))
  if r['method']=='FastV':
   ev=r['FastV_physical_history_evidence'];assert ev['KV_layout_equal'] and ev['pruning_events_current']==ev['pruning_events_historical']
  t={k:r.get(k) for k in ['request_id','category','key','method','session','begin_ns','end_ns','gap_s','main_thread_runqueue_ms','early5_vs_late26_decode_median_change']};t.update({m:metric(r,m) for m in METRICS});timeline.append(t)
  if r['category']!='formal':continue
  assert not g['smoke'] and not g['formal_layer_observer_hooks']
  v={k:r[k] for k in ['session','request_id','formal_index','input_index','order_in_input','key','benchmark','method','GPU_UUID','begin_ns','end_ns','gap_s','input_tokens','main_thread_runqueue_ms','main_thread_running_ms','sample_sha256','output_token_ids_SHA256']};v.update({m:metric(r,m) for m in METRICS});v.update(input_ids_SHA256=r['input_hashes']['input_ids'],attention_mask_SHA256=r['input_hashes']['attention_mask'],pixel_values_SHA256=r['input_hashes']['pixel_values'],historical_tokens_exact=r['historical_tokens_exact'],performance_warning_count=len(r['performance_warnings']),step_event_ms=json.dumps(g['step_event_ms']),KV_storage_MiB=g['cache']['storage_bytes']/2**20);per.append(v)
  for row in g['cache']['layers']:
   v=dict(row,session=sn,key=r['key'],method=r['method'])
   if r['method']=='SparseVLM':v['position_convention']='compact (merged representations); CPU export annotation only'
   for k in ['K_shape','V_shape']:v[k]=json.dumps(v[k])
   layoutrows.append(v)
 for n,h in plan['entry_source_SHA256'].items():assert sha(D/'code'/n)==h,n
 for path,h in read(D/'source_manifest.json')['frozen_sources'].items():assert sha(path)==h,path
 for n,h in plan['reference_files_SHA256'].items():assert sha(U/n)==h,n
 for path,h in read(D/'historical_protected_SHA256.json').items():assert sha(path)==h,path
 complete=completion['status'].startswith('COMPLETE_') and len(formal)==240 and not correctness_failures
 for pos,key in enumerate(KEYS):
  rows=sorted([r for r in formal if r['key']==key],key=lambda r:r['order_in_input']);want=METHODS[pos%8:]+METHODS[:pos%8];assert [r['method'] for r in rows]==want[:len(rows)]
  if complete:assert len(rows)==8
 if complete:
  assert completion['category_counts']==dict(baseline=20,calibration=36,formal=240,native_warmup=16,other_warmup=14,policy_smoke=1,smoke=20)
  assert read(D/'smoke_summary.json')['status']=='PASS' and read(D/'input_verification.json')['status']=='PASS'
 csvout(D/'efficiency_per_input.csv',per);csvout(D/'kv_layout_per_layer.csv',layoutrows);csvout(D/'all_request_timeline.csv',timeline)
 with gzip.open(D/'all_requests.jsonl.gz','wt') as out:
  for r in records:out.write(json.dumps(r,ensure_ascii=False)+'\n')
 save(D/'historical_output_comparisons.json',dict(records=policy,all_hard_gates_pass=not correctness_failures,correctness_failures=correctness_failures))
 summaries=[];methodsummary={}
 for m in METHODS:
  for split in ['All30','GQA','TextVQA']:
   rs=[r for r in formal if r['method']==m and (split=='All30' or r['benchmark']==split)];v=dict(session=sn,method=m,method_label=LABELS[m],split=split,n=len(rs))
   for k in METRICS:
    for stat,value in stats([metric(r,k) for r in rs]).items():
     if stat!='n':v[k+'_'+stat]=value
   common=[r['key'] for r in rs if (r['key'],'Native') in idx]
   for k in METRICS:
    n=avg([metric(idx[key,'Native'],k) for key in common]);a=avg([metric(idx[key,m],k) for key in common]);v[k+'_reduction_pct']=100*(1-a/n) if n and a is not None else None;v[k+'_speedup']=n/a if n and a else None
   summaries.append(v)
   if split=='All30':methodsummary[m]=v
  common=[k for k in KEYS if (k,m) in idx and (k,'Native') in idx];draws=resamples(common) if common else []
  for key in common:
   r=idx[key,m];n=idx[key,'Native'];v=dict(session=sn,key=key,benchmark=r['benchmark'],method=m)
   for k in METRICS:
    a=metric(r,k);b=metric(n,k);v[k+'_method']=a;v[k+'_Native']=b;v[k+'_Native_minus_method']=b-a;v[k+'_reduction_pct']=100*(1-a/b);v[k+'_speedup']=b/a
   within.append(v)
  for k in METRICS:
   a=[metric(idx[key,m],k) for key in common];n=[metric(idx[key,'Native'],k) for key in common]
   if not a:continue
   pct=[100*(1-x/y) for x,y in zip(a,n)];diff=[y-x for x,y in zip(a,n)];boots=[100*(1-sum(a[i] for i in d)/sum(n[i] for i in d)) for d in draws]
   v=dict(scope='within_session',session=sn,method=m,metric=k,n=len(common),ratio_of_means_reduction_pct=100*(1-avg(a)/avg(n)),paired_absolute_difference_stdev=statistics.stdev(diff) if len(diff)>1 else None,paired_pct_stdev=statistics.stdev(pct) if len(pct)>1 else None,strictly_lower=sum(x<y for x,y in zip(a,n)),strictly_higher=sum(x>y for x,y in zip(a,n)),**CI(boots));bootrows.append(v)
  if len(common)>=5:
   ratios=[metric(idx[key,m],'E2E_ms')/metric(idx[key,'Native'],'E2E_ms') for key in common];med=statistics.median(ratios);mad=statistics.median(abs(x-med) for x in ratios)
   for key,ratio in zip(common,ratios):
    if abs(ratio-med)>max(.05,6*1.4826*mad):flags.append(dict(scope='within_session',session=sn,method=m,key=key,ratio=ratio,median_ratio=med,MAD=mad,flag='paired E2E ratio outlier; retained'))
 csvout(D/'efficiency_summary.csv',summaries);summaryrows.extend(summaries)
 drift=read(D/'drift_summary.json') if (D/'drift_summary.json').exists() else dict(status='NOT_STARTED',checks=[]);save(D/'drift_summary.json',drift)
 maxdrift=max((abs(x) for d in drift['checks'] for changes in d['relative_change'].values() for x in changes.values()),default=None)
 if complete:assert len(drift['checks'])==6
 gpu=jl(D/'gpu_telemetry.jsonl');cpu=jl(D/'cpu_telemetry.jsonl');proc=jl(D/'processes_telemetry.jsonl');errs=[r for r in gpu+cpu+proc if 'error'in r];others=[r for r in proc if r.get('other_processes')]
 ranges={}
 for k in ['SM_MHz','memory_MHz','temperature_C','power_mW','GPU_util']:
  vals=[r[k] for r in gpu if isinstance(r.get(k),(int,float))];ranges[k]=stats(vals)
 telemetry=dict(GPU_UUID=plan['GPU_UUID'],GPU_samples=len(gpu),CPU_samples=len(cpu),maximum_GPU_sample_gap_s=max((r.get('gap_s') or 0 for r in gpu),default=None),ranges=ranges,other_compute_process_events=others,errors=errs,P_states=sorted({r['pstate'] for r in gpu if 'pstate'in r},key=str),throttle_masks=sorted({r['throttle_mask'] for r in gpu if 'throttle_mask'in r},key=str));save(D/'gpu_telemetry_summary.json',telemetry)
 times=[g['at_ns'] for g in gpu];joined=[]
 for r in records:
  samples=gpu[bisect.bisect_left(times,r['begin_ns']):bisect.bisect_right(times,r['end_ns'])];v=dict(session=sn,request_id=r['request_id'],category=r['category'],key=r['key'],method=r['method'],GPU_UUID=r['GPU_UUID'],begin_ns=r['begin_ns'],end_ns=r['end_ns'],GPU_samples=len(samples),P_states=json.dumps(sorted({g['pstate'] for g in samples if 'pstate'in g},key=str)),throttle_masks=json.dumps(sorted({g['throttle_mask'] for g in samples if 'throttle_mask'in g},key=str)))
  for k in ['SM_MHz','memory_MHz','temperature_C','power_mW','GPU_util']:
   values=[g[k] for g in samples if isinstance(g.get(k),(int,float))];z=stats(values)
   for stat in ['mean','median','min','max']:v[k+'_'+stat]=z[stat]
  joined.append(v)
 csvout(D/'telemetry_per_request.csv',joined)
 if complete:assert not others
 warns=read(D/'performance_warnings.json');v=dict(status=completion['status'],execution_integrity='PASS' if complete else 'INCOMPLETE',execution_complete=complete,formal_completed=len(formal),expected=240,input_SHA_status=read(D/'input_verification.json')['status'],all_records_SHA_verified=True,actual_KV_recounted=True,all_Native_tokens_exact=all(r['historical_tokens_exact'] for r in records if r['method']=='Native'),history_correctness_failures=correctness_failures,performance_status='WARN' if warns['warnings'] else 'NO_WARNINGS_RECORDED',warnings=len(warns['warnings']),calibrations=len(drift['checks']),calibration_windows_warn=sum(d['breach'] for d in drift['checks']),maximum_abs_per_input_calibration_drift_pct=None if maxdrift is None else maxdrift*100,frozen_sources_unchanged=True,historical_files_verified=len(read(D/'historical_protected_SHA256.json')),schedule_SHA256=plan['schedule_SHA256'],model_loads=completion['model_loads'],telemetry_errors=len(errs),other_GPU_process_events=len(others));save(D/'validation.json',v)
 S[sn]=dict(started=True,complete=complete,formal=formal,idx=idx,summary=methodsummary,validation=v,drift=drift,warnings=warns,telemetry=telemetry,completion=completion,gpu=gpu)

csvout(C/'session_summaries.csv',summaryrows);csvout(C/'within_session_paired.csv',within)
cross_summary=[]
if all(S[n]['complete'] for n in [1,2]):
 for m in METHODS:
  common=KEYS;draws=resamples(common);s1=S[1]['summary'][m];s2=S[2]['summary'][m];v=dict(method=m,method_label=LABELS[m],n=30)
  for k in METRICS:v[k+'_S1_mean']=s1[k+'_mean'];v[k+'_S2_mean']=s2[k+'_mean'];v[k+'_change_pct']=100*(s2[k+'_mean']/s1[k+'_mean']-1)
  for k in ['TTFT_ms','TPOT_ms','E2E_ms']:
   x=s1[k+'_speedup'];y=s2[k+'_speedup'];v[k+'_speedup_S1']=x;v[k+'_speedup_S2']=y;v[k+'_speedup_delta']=y-x;v[k+'_speedup_change_pct']=100*(y/x-1)
  cross_summary.append(v)
  for key in common:
   a=S[1]['idx'][key,m];b=S[2]['idx'][key,m];assert a['input_hashes']==b['input_hashes'] and a['sample_sha256']==b['sample_sha256'] and a['order_in_input']==b['order_in_input'];z=dict(key=key,benchmark=a['benchmark'],method=m,order_in_input=a['order_in_input'],S1_request_id=a['request_id'],S2_request_id=b['request_id'],S1_begin_ns=a['begin_ns'],S2_begin_ns=b['begin_ns'],output_tokens_identical=a['generation']['token_ids']==b['generation']['token_ids'])
   for k in METRICS:
    x=metric(a,k);y=metric(b,k);z[k+'_S1']=x;z[k+'_S2']=y;z[k+'_S2_minus_S1']=y-x;z[k+'_change_pct']=100*(y/x-1)
   cross.append(z)
  for k in METRICS:
   a=[metric(S[1]['idx'][key,m],k) for key in common];b=[metric(S[2]['idx'][key,m],k) for key in common];n1=[metric(S[1]['idx'][key,'Native'],k) for key in common];n2=[metric(S[2]['idx'][key,'Native'],k) for key in common];pct=[100*(y/x-1) for x,y in zip(a,b)];diff=[y-x for x,y in zip(a,b)];boots=[100*(sum(b[i] for i in d)/sum(a[i] for i in d)-1) for d in draws];speed=[100*((sum(n2[i] for i in d)/sum(b[i] for i in d))/(sum(n1[i] for i in d)/sum(a[i] for i in d))-1) for d in draws]
   bootrows.append(dict(scope='cross_session_absolute_latency',session='2_vs_1',method=m,metric=k,n=30,ratio_of_means_change_pct=100*(avg(b)/avg(a)-1),paired_absolute_difference_mean=avg(diff),paired_absolute_difference_median=statistics.median(diff),paired_absolute_difference_stdev=statistics.stdev(diff),paired_pct_median=statistics.median(pct),paired_pct_stdev=statistics.stdev(pct),**CI(boots)))
   bootrows.append(dict(scope='cross_session_relative_Native_speedup',session='2_vs_1',method=m,metric=k,n=30,ratio_of_means_change_pct=100*((avg(n2)/avg(b))/(avg(n1)/avg(a))-1),**CI(speed)))
  ratios=[metric(S[2]['idx'][key,m],'E2E_ms')/metric(S[1]['idx'][key,m],'E2E_ms') for key in common];med=statistics.median(ratios);mad=statistics.median(abs(x-med) for x in ratios)
  for key,ratio in zip(common,ratios):
   if abs(ratio-med)>max(.05,6*1.4826*mad):flags.append(dict(scope='cross_session',method=m,key=key,ratio=ratio,median_ratio=med,MAD=mad,flag='S2/S1 paired E2E ratio outlier; retained'))
csvout(C/'cross_session_per_input.csv',cross);csvout(C/'cross_session_summary.csv',cross_summary);csvout(C/'bootstrap_CI95.csv',bootrows)
contributors=[]
for m in METHODS:
 rows=sorted([r for r in cross if r['method']==m],key=lambda r:abs(r['E2E_ms_S2_minus_S1']),reverse=True)
 for rank,r in enumerate(rows[:5]):contributors.append(dict(method=m,rank_by_absolute_difference=rank+1,key=r['key'],benchmark=r['benchmark'],S1_E2E_ms=r['E2E_ms_S1'],S2_E2E_ms=r['E2E_ms_S2'],difference_ms=r['E2E_ms_S2_minus_S1'],contribution_to_method_mean_difference_ms=r['E2E_ms_S2_minus_S1']/len(rows),change_pct=r['E2E_ms_change_pct']))
csvout(C/'cross_session_top_contributors.csv',contributors)
reversals=[]
if cross_summary:
 for a,b in itertools.combinations(METHODS,2):
  x=S[1]['summary'][a]['E2E_ms_mean']/S[1]['summary'][b]['E2E_ms_mean']-1;y=S[2]['summary'][a]['E2E_ms_mean']/S[2]['summary'][b]['E2E_ms_mean']-1
  if x*y<0:reversals.append(dict(method_A=a,method_B=b,S1_relative_E2E_gap_pct=100*x,S2_relative_E2E_gap_pct=100*y,at_least_one_gap_below2pct=min(abs(x),abs(y))<.02))
save(C/'ranking_reversals.json',dict(records=reversals,definition='Mean E2E ordering reverses; no significance claim'))
save(C/'anomaly_summary.json',dict(statistical_flags=flags,no_requests_removed=True,rule='Paired E2E ratio distance from median > max(0.05,6*1.4826*MAD); fixed exploratory screen, no causal attribution'))
save(C/'statistics_protocol.json',dict(bootstrap_replicates=5000,seed=309032,stratified_by='Observed paired input counts by benchmark; complete sessions target GQA15/TextVQA15',unit='Frozen input, paired across methods and sessions',interval='Percentile95%',no_multiple_testing_adjustment=True,limitations='Exploratory fixed-input composition intervals. Partial runs cannot estimate cross-session repeatability. Two sequential sessions cannot identify independent hardware/runtime causes, estimate long-term repeatability, or eliminate time-order confounding. No request removed.'))

# Attach contemporaneous measurements to every exploratory anomaly, without GPU reruns.
evidence=[]
for flag in flags:
 for sn in ([flag['session']] if 'session'in flag else [1,2]):
  r=S[sn]['idx'].get((flag['key'],flag['method']))
  if r is None:continue
  gpu=S[sn]['gpu'];samples=[g for g in gpu if r['begin_ns']<=g['at_ns']<=r['end_ns']];rg={}
  for k in ['SM_MHz','memory_MHz','temperature_C','power_mW']:
   vals=[s[k] for s in samples if isinstance(s.get(k),(int,float))];rg[k]=stats(vals)
  evidence.append(dict(flag=flag,session=sn,request_id=r['request_id'],formal_index=r['formal_index'],key=r['key'],method=r['method'],metrics={m:metric(r,m) for m in METRICS},main_thread_runqueue_ms=r['main_thread_runqueue_ms'],early5_vs_late26_decode_change=r['early5_vs_late26_decode_median_change'],step_event_ms=r['generation']['step_event_ms'],GPU_samples=len(samples),GPU_ranges=rg,P_states=sorted({s['pstate'] for s in samples if 'pstate'in s},key=str),throttle_masks=sorted({s['throttle_mask'] for s in samples if 'throttle_mask'in s},key=str),interpretation='Contemporaneous evidence only; no causal attribution'))
save(C/'anomaly_evidence.json',dict(records=evidence))

def session_table(sn):
 if not S[sn]['started']:return '未启动。'
 rows=['| Method | n | KV MiB | TTFT ms | TPOT ms/token | Event TPOT ms/token | E2E ms |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for m in METHODS:
  s=S[sn]['summary'][m];rows.append('| '+LABELS[m]+' | '+str(s['n'])+' | '+' | '.join(f(s[k+'_mean']) for k in METRICS)+' |')
 return '\n'.join(rows)
def reduction_table(sn):
 if not S[sn]['started']:return '无正式数据。'
 rows=['| Method | KV减少% | TTFT降低% | TPOT降低% | E2E降低% | E2E Speedup | TPOT Speedup |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for m in METHODS:
  s=S[sn]['summary'][m];rows.append('| '+LABELS[m]+' | '+' | '.join(f(s[k+'_reduction_pct']) for k in ['KV_MiB','TTFT_ms','TPOT_ms','E2E_ms'])+' | '+f(s['E2E_ms_speedup'],4)+' | '+f(s['TPOT_ms_speedup'],4)+' |')
 return '\n'.join(rows)
def drift_table(sn):
 if not S[sn]['started']:return '未启动。'
 rows=['| 已完成输入 | 合并E2E漂移% | 合并TPOT漂移% | 按输入/指标最大绝对漂移% | 状态 |','| ---: | ---: | ---: | ---: | --- |']
 for r in S[sn]['drift']['checks']:rows.append('| '+str(r['after_inputs'])+' | '+f(r['pooled_relative_change']['E2E_ms']*100,3)+' | '+f(r['pooled_relative_change']['TPOT_ms']*100,3)+' | '+f(100*max(abs(v) for d in r['relative_change'].values() for v in d.values()),3)+' | '+r['status']+' |')
 return '\n'.join(rows)
ct=['| Method | TTFT变化% | TPOT变化% | E2E变化% | E2E Speedup变化% | TPOT Speedup变化% |','| --- | ---: | ---: | ---: | ---: | ---: |']
for r in cross_summary:ct.append('| '+LABELS[r['method']]+' | '+' | '.join(f(r[k+'_change_pct']) for k in ['TTFT_ms','TPOT_ms','E2E_ms'])+' | '+f(r['E2E_ms_speedup_change_pct'])+' | '+f(r['TPOT_ms_speedup_change_pct'])+' |')
both=all(S[n]['complete'] for n in [1,2]);absolute_changes=[r['E2E_ms_change_pct'] for r in cross_summary];relative_changes=[r['E2E_ms_speedup_change_pct'] for r in cross_summary if r['method']!='Native'];stable=sorted([r for r in cross_summary if r['method']!='Native'],key=lambda r:abs(r['E2E_ms_speedup_change_pct']))
performance_warn=any(S[n].get('validation',{}).get('performance_status')=='WARN' for n in [1,2]) or any(abs(x)>2 for x in absolute_changes)
status='COMPLETE_WITH_PERFORMANCE_WARNINGS' if both and performance_warn else 'COMPLETE_NO_PERFORMANCE_WARNINGS' if both else 'PARTIAL_OR_BLOCKED'
decision='本轮仅可作中断及性能波动诊断证据；无法进行两轮复现性比较，不适用于完整效率主表。' if not both else '可保留物理KV与同机相对效率证据；有漂移的绝对延迟及1%–2%的小改善需进一步重复验证。' if performance_warn else '可作为这30题、此GPU与本协议下的两轮效率候选数据；两轮不足以证明长期稳定性，小差异仍需审慎。'
validation=dict(status=status,execution_integrity='PASS' if both else 'INCOMPLETE',formal_completed=sum(len(S[n]['formal']) for n in [1,2]),formal_expected=480,each_session_complete={str(n):S[n]['complete'] for n in [1,2]},independent_model_processes=all(S[n]['started'] for n in [1,2]),actual_model_loads=sum(S[n].get('completion',{}).get('model_loads',0) for n in [1,2]),same_frozen_schedule=True,schedule_SHA256=p1['schedule_SHA256'],all_raw_data_retained=True,performance_status='WARN' if performance_warn else 'NO_WARNINGS_RECORDED',high_precision_stability_proven=False,paper_use=decision,statistical_anomaly_flags=len(flags),GitHub_push=False,automatic_retries=0,server_left_running=True)
release=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,power.limit','--format=csv,noheader'],text=True).strip(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip(),server_left_running=True);assert not release['processes'];save(B/'gpu_release_final.json',release);validation['GPU_released']=True;save(C/'validation.json',validation)
native=next((r for r in cross_summary if r['method']=='Native'),None)
native_text='尚无两轮完整Native结果，跨Session变化及加速比稳定性均不可估计。' if native is None else f"Native E2E：{native['E2E_ms_S1_mean']:.3f}→{native['E2E_ms_S2_mean']:.3f}ms，变化{native['E2E_ms_change_pct']:+.3f}%；TPOT：{native['TPOT_ms_S1_mean']:.5f}→{native['TPOT_ms_S2_mean']:.5f}ms/token，变化{native['TPOT_ms_change_pct']:+.3f}%。"
hard_failure_text='无已记录的硬失败。'
if (C/'hard_failure_audit.json').exists():
 fail=read(C/'hard_failure_audit.json');pol=fail['policy'];hard_failure_text=f"Session1在正式第{fail['formal_index']}条、输入{fail['key']}的Native输出触发明确保留的完整32Token历史一致性硬门禁。首个EOS均为第{pol['current_EOS_positions_0based'][0]+1}个Token，首次差异在第{pol['first_difference_0based']+1}个Token，发生于EOS后。图像/Prompt/输入Tensor及Sample SHA一致，终止KV形状/有效长度/dtype/位置哈希一致，Logits有限。未保存完整逐步Logits或KV数值，不能确定数值分叉根因，也不能仅据此认定算法实现错误或答案质量下降。共{fail['Native_requests']}次Native中{fail['Native_exact']}次完整历史一致；正式{fail['Native_formal']}次中1次失败。首个EOS之前及EOS本身一致，但Native的授权要求仍为严格32Token，因此停止；未自动放宽、重跑或启动Session2。完整证据见hard_failure_audit.json。"
cross_interpretation='Session2未启动，跨轮变化范围和相对Native加速比稳定性不可估计。'
ranking_interpretation='没有两轮完整结果，名次反转不可评估；空的排名文件不表示不存在名次反转。'
if cross_summary:
 cross_interpretation=f"E2E绝对变化跨方法范围：{f(min(absolute_changes),3)}%至{f(max(absolute_changes),3)}%；非Native相对加速比变化范围：{f(min(relative_changes),3)}%至{f(max(relative_changes),3)}%。相对E2E加速比变化最小的方法依次为："+', '.join(LABELS[r['method']]+' '+f(r['E2E_ms_speedup_change_pct'],3)+'%' for r in stable)+'。'
 ranking_interpretation=f"两轮E2E名次见comparison/E2E_rankings.json，名次反转共{len(reversals)}对，逐对相对差值见ranking_reversals.json。"
interpret=[]
for sn in [1,2]:
 if not S[sn]['started']:interpret.append(f'Session{sn}未启动。');continue
 for m in ['SVH','SVH-R']:
  row=next((r for r in bootrows if r['scope']=='within_session' and r['session']==sn and r['method']==m and r['metric']=='TPOT_ms'),None)
  if row:interpret.append(f"Session{sn} {LABELS[m]} Decode TPOT均值改善{row['ratio_of_means_reduction_pct']:.3f}%，逐题更低{row['strictly_lower']}/{row['n']}，探索性输入Bootstrap95%区间[{row['CI95_low']:.3f},{row['CI95_high']:.3f}]%。")
 for m in ['FastV','V2Drop','ApET','DUET','SparseVLM']:
  row=S[sn]['summary'][m];interpret.append(f"Session{sn} {LABELS[m]} TTFT改善{f(row['TTFT_ms_reduction_pct'])}%，TPOT改善{f(row['TPOT_ms_reduction_pct'])}%，E2E改善{f(row['E2E_ms_reduction_pct'])}%。")
ranks={str(n):sorted(METHODS,key=lambda m:S[n]['summary'][m]['E2E_ms_mean']) for n in [1,2] if S[n]['complete']};save(C/'E2E_rankings.json',ranks)
hist=list(csv.DictReader((U/'efficiency_summary.csv').open()));hist={r['method']:r for r in hist if r['split']=='All30'};hc=['| Method | 4090 TTFT | 4090 TPOT | 4090 E2E | 3090 S1 E2E | 3090 S2 E2E |','| --- | ---: | ---: | ---: | ---: | ---: |']
for m in METHODS:hc.append('| '+LABELS[m]+' | '+' | '.join(f(float(hist[m][k+'_mean'])) for k in ['TTFT_ms','TPOT_ms','E2E_ms'])+' | '+' | '.join(f(S[n]['summary'][m]['E2E_ms_mean']) if S[n]['started'] else '—' for n in [1,2])+' |')
report=f'''# RTX3090 八方法两轮统一效率与复现性

状态：**{status}**。执行完整性：**{validation['execution_integrity']}**；正式请求 **{validation['formal_completed']}/480**。测速稳定性：**{validation['performance_status']}**，未将完成240次等同于高精度稳定性PASS。所有慢请求、性能警告和历史Token差异保留。

{hard_failure_text}

若本轮未完成，下表仅为已测前缀的部分统计，包含触发硬门禁的原始请求；不删除该行来美化结果。方法样本数可能不同，仅按共同已测题计算相对Native指标。它不能替代完整30题主表，也不能用于判断两轮复现性。Session1共240次总请求包含预热、Smoke、基线、18次校准和151次正式请求，不能将240次总请求写成240次正式完成。

目录：`{B}`。GPU `{p1['GPU_UUID']}`，同一张RTX3090。模型`{p1['model']}`，固定Revision `{p1['revision']}`；FP16/Eager、PyTorch2.6.0+cu124、Transformers4.49.0、batch1、greedy、cache=True、logits_to_keep=1，固定32Token=1Prefill+31真实Decode，忽略EOS。冻结Commit `{p1['reference_commit']}`。模型和数据复用，没有重新下载。

## Session 1独立均值

{session_table(1)}

{reduction_table(1)}

## Session 2独立均值

{session_table(2)}

{reduction_table(2)}

已执行轮次按实际已测题数输出均值、中位数、样本标准差、P90（线性分位数）、最小最大值及GQA/TextVQA分项，见`efficiency_summary.csv`的n列；完整目标为每方法30题。未执行的Session2不生成虚构统计。KV=2^20字节MiB，从实际终止K/V Tensor形状、FP16 dtype及有效长度独立重算；不是峰值显存。各层布局见`kv_layout_per_layer.csv`。

## 两轮变化

变化定义为(S2均值/S1均值−1)×100%。Speedup每轮均使用该轮Native均值/方法均值，表中的Speedup变化是(S2 Speedup/S1 Speedup−1)×100%。正延迟变化表示S2更慢，正Speedup变化表示相对Native优势增加。

{chr(10).join(ct)}

{native_text}

{cross_interpretation} 两个顺序运行的Session不能据此识别共同硬件因果；有两轮结果时须同时报告绝对和相对变化，不能选择有利的一轮。

{chr(10).join(interpret)}

{ranking_interpretation} 小于约1%–2%的均值改善及名次变化仅作探索性描述，不直接宣称显著加速。若各方法近似共同变化但相对Native更稳定，应同时保留绝对漂移事实；本报告不跨GPU计算算法Speedup。

## 逐题配对、区间与异常

完整两轮时，同题S1/S2对照写入`cross_session_per_input.csv`，包含每项绝对差、百分比、顺序、开始时间及Token是否相同；未完成时该文件保持空表，不编造第二轮数据。每轮相对Native逐题差见`within_session_paired.csv`。探索性95%Bootstrap区间见`bootstrap_CI95.csv`：按冻结输入成对重采样5000次、种子309032；完整两轮时保持GQA15/TextVQA15构成，部分结果保留各方法共同已测题的实际Benchmark构成。跨轮相对Native区间同时重采样该题两轮Native与方法，保留配对相关性。没有GPU重跑。这些区间描述固定题目组成，不替代跨时段重复性、无法排除时序混杂，未做多重检验校正。

统计异常标记{len(flags)}条，原始请求均保留。预先固定筛查：同题E2E比值偏离中位数>max(0.05,6×1.4826×MAD)。每个异常的Event单步、CPU runqueue、时钟、功耗、温度及P-State/Throttle原始值见`anomaly_evidence.json`。各方法两轮耗时差最大的5题及对均值差的算术贡献见`cross_session_top_contributors.csv`，该贡献不是硬件因果证明。所有请求的遥测时间关联见各轮`telemetry_per_request.csv`。CUDA Event也包含相邻Kernel之间的执行间隔，不能凭Event增加就唯一归因于GPU核心性能。未做Profiler或额外GPU诊断，不编造根因。

## Native校准与性能警告

预先冻结的每轮协议为：固定16次Native完整32Token预热，之后七种方法各2次（共14），然后21次正确性Smoke（含旧失败FastV输入）、独立20次Native双探针基线；每5题双探针各3次，计划六窗36次。每轮所有请求计划347次，正式240次独立统计。两轮预先冻结相同顺序，Schedule SHA `{p1['schedule_SHA256']}`；计划独立进程、独立模型加载与基线。实际执行数量以Checkpoint、validation与上方统计为准，没有混入旧31条PARTIAL数据。

性能1%CV、1%窗口变化、3%单请求、2%校准漂移及更大波动均只记录WARN；没有等待最快状态、没有无限预热、没有删除/重跑/替换请求。正确性、输入、有限值、OOM、GPU冲突/设备和硬件安全仍为硬门禁。85℃及低可用内存等安全监控沿用原采集器，没有改GPU时钟、Power Limit、驱动或Governor。

Session 1：

{drift_table(1)}

Session 2：

{drift_table(2)}

每轮`performance_warnings.json`、`warmup_summary.json`和原始SQLite记录所有波动。GPU5Hz、CPU1Hz、进程查询2秒分离采集，频率、温度、功耗、P-State、Throttle以及采集缺口/错误汇总见各轮`gpu_telemetry_summary.json`。没有宣称监控零开销。

## 正确性、计时与历史边界

30题Sample、图像、Prompt、OCR、input_ids/attention_mask/pixel_values SHA均对齐原4090。Native严格32Token历史一致；FastV严格首EOS前及EOS，只有EOS后历史差异可以记录后继续；无EOS时32Token严格。其他方法历史Token差异完整记录，不单凭跨GPU历史输出不同判定实现错误；No-op Token/首Logits、finite、实际Hidden/Mask/RoPE/KV、真实Prefill/Decode与SVH/R物理跳算硬检查不变。正式请求没有Smoke观察Hook。

内层直接复用冻结`e2e32_runtime.generate`，SHA `513a122ff5f4da008b74e4d5560b0b6ef50baf8ccf7feb2fa766a0cdcdc51a3b`；没有重写Attention。E2E从GPU输入就绪同步后、控制器准备前开始，到第32Token生成及必要清理同步后结束。Vision/Projector/Prefill/31Decode/argmax/必要控制器开销计入；加载、CPU图像与Processor/Tokenizer、H2D、日志和离线KV审计排除。逐条核对E2E=TTFT+31×TPOT及Event。V²Drop为历史576→288→144→1工作点，非论文官方192；DUET为C+all；SparseVLM含官方Recycling实际平均206.5视觉表示；没有Retain128或算法调参。

## 与4090并列及用途

{chr(10).join(hc)}

硬件不同，不能用跨GPU绝对延迟计算算法Speedup或替换旧4090基线。{decision}

原4090、旧3090失败15条、旧31条PARTIAL及上次准入失败目录均未改。完整源码SHA、模型Revision、输入、输出Token SHA、Checkpoint、全部尝试与失败日志、预热及遥测保留。没有自然EOS、质量Benchmark、训练、参数搜索、GitHub Push/PR或完整Windows备份。

GPU释放：`{release['GPU']}`；计算进程为空，服务器保持运行。最终验收`comparison/validation.json`，完整SHA清单为根目录及各目录`SHA256.json`。
'''
(C/'report_zh.md').write_text(report)
for sn in [1,2]:
 D=B/('session'+str(sn));short=f"# 独立Session {sn}\n\n状态：{S[sn].get('completion',{}).get('status','NOT_STARTED')}；正式{len(S[sn]['formal'])}/240。完整对照报告见../comparison/report_zh.md。\n\n"+session_table(sn)+'\n\n'+reduction_table(sn)+'\n\n'+drift_table(sn)+'\n';(D/'report_zh.md').write_text(short);inventory(D)
inventory(C);inventory(B)
print(json.dumps(validation,ensure_ascii=False,indent=2));print('S1');print(session_table(1));print('S2');print(session_table(2));print('CROSS');print('\n'.join(ct));print(native_text)
