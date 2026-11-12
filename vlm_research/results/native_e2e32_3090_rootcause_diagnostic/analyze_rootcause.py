"""CPU analysis of the new controlled diagnostic and Kineto timelines."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import os,json,csv,statistics,hashlib,math
os.environ['CUDA_VISIBLE_DEVICES']=''
D=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_rootcause_diagnostic')
def read(n):return json.loads((D/n).read_text())
def save(n,v):(D/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(n,rs):
 if not rs:return
 with (D/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def stats(v):return dict(n=len(v),mean=statistics.mean(v) if v else None,median=statistics.median(v) if v else None,stdev=statistics.stdev(v) if len(v)>1 else 0,min=min(v) if v else None,max=max(v) if v else None)
def union(intervals):
 if not intervals:return 0
 a=sorted(intervals);lo,hi=a[0];total=0
 for l,h in a[1:]:
  if l<=hi:hi=max(hi,h)
  else:total+=hi-lo;lo,hi=l,h
 return total+hi-lo
def num(v):
 try:return float(v)
 except (ValueError,TypeError):return None
def hardware(rows,gpu,cpu):
 gs=[g for g in gpu if any(r['begin_unix_ns']<=int(g['unix_time_ns'])<=r['end_unix_ns'] for r in rows)]
 cs=[c for c in cpu if any(r['begin_unix_ns']<=int(c['unix_time_ns'])<=r['end_unix_ns'] for r in rows)]
 return dict(GPU_samples=len(gs),CPU_samples=len(cs),GPU={k:stats([num(g[k]) for g in gs if num(g[k]) is not None]) for k in ['SM_MHz','memory_MHz','power_W','temperature_C','GPU_util_pct']},pstate_counts=dict(Counter(g['pstate'] for g in gs)),throttle_mask_counts=dict(Counter(g['throttle_mask'] for g in gs)),SW_power_cap_sample_count=sum(bool(int(g['throttle_mask'])&4) for g in gs if num(g['throttle_mask']) is not None),HW_slowdown_sample_count=sum(bool(int(g['throttle_mask'])&8) for g in gs if num(g['throttle_mask']) is not None),thermal_slowdown_sample_count=sum(bool(int(g['throttle_mask'])&(32|64)) for g in gs if num(g['throttle_mask']) is not None))
def profile_trace(path,r):
 trace=json.loads(path.read_text());ev=trace.get('traceEvents',[]);marker=[v for v in ev if v.get('cat')=='user_annotation' and v.get('name')=='FROZEN_NATIVE_FULL32' and v.get('ph')=='X']
 assert len(marker)==1
 start=marker[0]['ts'];end=start+marker[0]['dur']
 kernel=[v for v in ev if v.get('cat')=='kernel' and v.get('ph')=='X']
 runtime=sorted([v for v in ev if v.get('cat') in ['cuda_runtime','cuda_driver'] and v.get('ph')=='X' and start<=v.get('ts',-1)<=end],key=lambda v:v['ts'])
 raw_records=sorted([v for v in runtime if v.get('name','').startswith('cudaEventRecord')],key=lambda v:v['ts'])
 # Outer torch.cuda.Event.record calls have no aten External id. Operator-internal
 # records (e.g. async-copy bookkeeping inside aten::to) must not shift step pairs.
 records=[v for v in raw_records if 'External id' not in v.get('args',{})]
 launches=[v for v in runtime if v.get('name','').startswith(('cudaLaunchKernel','cuLaunchKernel','cudaLaunchCooperativeKernel'))]
 gpu=[v for v in ev if v.get('ph')=='X' and v.get('cat') in ['kernel','gpu_memcpy','gpu_memset']]
 kbycorr=defaultdict(list)
 for v in gpu:
  corr=v.get('args',{}).get('correlation')
  if corr is not None:kbycorr[corr].append(v)
 # cudaEventRecord host pairs identify operations submitted by each frozen step;
 # these are submission boundaries, not fabricated GPU Event timestamps.
 api_by_corr={v.get('args',{}).get('correlation'):v for v in runtime if v.get('args',{}).get('correlation') is not None};steps=[]
 if len(records)==64:
  for step in range(32):
   lo=records[step*2]['ts'];hi=records[step*2+1]['ts'];apis=[v for v in launches if lo<=v['ts']<=hi];ops=[]
   for api in runtime:
    if lo<=api['ts']<=hi:ops+=kbycorr.get(api.get('args',{}).get('correlation'),[])
   busy=union([(v['ts'],v['ts']+v['dur']) for v in ops]);kernels=[v for v in ops if v.get('cat')=='kernel'];ints=sorted((v['ts'],v['ts']+v['dur']) for v in ops)
   span=ints[-1][1]-ints[0][0] if ints else 0;event_ms=r['generation']['step_event_ms'][step]
   stream_groups=defaultdict(list)
   for op in ops:stream_groups[(op.get('args',{}).get('device'),op.get('args',{}).get('stream'))].append(op)
   before_api=inside_api=after_api=unmapped_gap=0
   for stream_ops in stream_groups.values():
    ss=sorted(stream_ops,key=lambda op:op['ts'])
    for first,second in zip(ss,ss[1:]):
     edge=first['ts']+first['dur'];gap=second['ts']-edge
     if gap<=0:continue
     api=api_by_corr.get(second.get('args',{}).get('correlation'))
     if api is None:unmapped_gap+=gap;continue
     b=max(0,min(second['ts'],api['ts'])-edge);a=max(0,second['ts']-max(edge,api['ts']+api['dur']));before_api+=b;after_api+=a;inside_api+=max(0,gap-b-a)
   steps.append(dict(request=r['request'],step=step,kind='Prefill' if step==0 else 'Decode',CUDA_Event_ms=event_ms,CPU_submit_span_ms=(hi-lo)/1000,launches=len(apis),matched_GPU_ops=len(ops),GPU_activity_union_ms=busy/1000,GPU_activity_span_ms=span/1000,internal_GPU_activity_gaps_ms=(span-busy)/1000,Event_minus_traced_GPU_activity_ms=event_ms-busy/1000,Kernel_sum_ms=sum(v['dur'] for v in kernels)/1000,CUDA_launch_API_sum_ms=sum(v['dur'] for v in apis)/1000,GPU_gap_before_next_API_start_ms=before_api/1000,GPU_gap_during_next_API_ms=inside_api/1000,GPU_gap_after_next_API_end_ms=after_api/1000,GPU_gap_unmapped_ms=unmapped_gap/1000,CPU_launch_interarrival_median_us=statistics.median([b['ts']-a['ts'] for a,b in zip(apis,apis[1:])]) if len(apis)>1 else None))
 cpu_ops=[v for v in ev if v.get('cat')=='cpu_op' and v.get('ph')=='X' and start<=v.get('ts',-1)<=end]
 byid={v.get('args',{}).get('External id'):v for v in cpu_ops if v.get('args',{}).get('External id') is not None}
 categories=defaultdict(lambda:dict(count=0,Kernel_ms=0,CPU_dispatch_ops=set()))
 for k in kernel:
  op=byid.get(k.get('args',{}).get('External id'),{});name=op.get('name','UNMAPPED')
  # Operator classes, not assumed semantic attention-vs-MLP attribution.
  cat='GEMM/matmul' if name in ['aten::mm','aten::addmm','aten::bmm','aten::matmul','aten::linear'] else 'softmax' if 'softmax' in name else 'normalization' if any(t in name for t in ['norm','rms']) else 'allocation/copy/concat' if any(t in name for t in ['copy','cat','empty','clone']) else 'elementwise/other'
  categories[cat]['count']+=1;categories[cat]['Kernel_ms']+=k['dur']/1000;categories[cat]['CPU_dispatch_ops'].add(name)
 catout={k:dict(count=v['count'],Kernel_ms=v['Kernel_ms'],CPU_dispatch_ops=sorted(v['CPU_dispatch_ops'])) for k,v in categories.items()}
 byname=defaultdict(lambda:[0,0])
 for k in kernel:byname[k['name']][0]+=1;byname[k['name']][1]+=k['dur']/1000
 summary=dict(request=r['request'],trace_file=str(path.relative_to(D)),profiled=True,profiled_E2E_ms=r['generation']['E2E_ms'],profiled_TPOT_ms=r['generation']['TPOT_ms'],profiled_Event_Decode_TPOT_ms=r['generation']['CUDA_Event_Decode_TPOT_ms'],categories=dict(Counter(v.get('cat','') for v in ev)),CUDA_EventRecord_runtime_count=len(records),step_pair_mapping_available=len(records)==64,CUDA_launch_count=len(launches),kernel_count=len(kernel),kernel_sum_ms=sum(v['dur'] for v in kernel)/1000,all_GPU_activity_union_ms=union([(v['ts'],v['ts']+v['dur']) for v in gpu])/1000,CUDA_launch_API_sum_ms=sum(v['dur'] for v in launches)/1000,CUDA_launch_API_p50_us=statistics.median(v['dur'] for v in launches) if launches else None,CUDA_launch_API_max_us=max([v['dur'] for v in launches],default=0),CUDA_launch_interarrival_p50_us=statistics.median([b['ts']-a['ts'] for a,b in zip(launches,launches[1:])]) if len(launches)>1 else None,operator_class_kernel_totals=catout,top_kernel_names=[dict(name=k,count=v[0],sum_ms=v[1]) for k,v in sorted(byname.items(),key=lambda kv:kv[1][1],reverse=True)[:20]],trace_time_metadata={k:trace.get(k) for k in ['baseTimeNanoseconds','displayTimeUnit','schemaVersion']},note='Profiler overhead is excluded from unprofiled distributions. Kernel categories use linked aten operator classes, not module-level attention/MLP attribution. GPU gaps include runtime submission/dispatch and other effects; no scheduler trace proves why a gap exists.')
 if steps:
  ds=steps[1:];summary['decode']=dict(Event_sum_ms=sum(v['CUDA_Event_ms'] for v in ds),traced_GPU_activity_union_sum_ms=sum(v['GPU_activity_union_ms'] for v in ds),internal_GPU_gaps_sum_ms=sum(v['internal_GPU_activity_gaps_ms'] for v in ds),Event_minus_traced_GPU_activity_sum_ms=sum(v['Event_minus_traced_GPU_activity_ms'] for v in ds),Kernel_sum_ms=sum(v['Kernel_sum_ms'] for v in ds),CPU_submit_span_sum_ms=sum(v['CPU_submit_span_ms'] for v in ds),CUDA_launch_API_sum_ms=sum(v['CUDA_launch_API_sum_ms'] for v in ds),GPU_gap_before_next_API_start_ms=sum(v['GPU_gap_before_next_API_start_ms'] for v in ds),GPU_gap_during_next_API_ms=sum(v['GPU_gap_during_next_API_ms'] for v in ds),GPU_gap_after_next_API_end_ms=sum(v['GPU_gap_after_next_API_end_ms'] for v in ds),GPU_gap_unmapped_ms=sum(v['GPU_gap_unmapped_ms'] for v in ds))
 summary['CUDA_EventRecord_raw_runtime_count']=len(raw_records);summary['operator_internal_EventRecords_excluded']=len(raw_records)-len(records)
 return summary,steps
def main():
 completion=read('request_completion.json');assert completion['completed']<=100
 rs=[json.loads(l) for l in (D/'requests.jsonl').read_text().splitlines()];assert len(rs)==completion['completed'];gpu=list(csv.DictReader((D/'gpu_telemetry.csv').open()));cpu=list(csv.DictReader((D/'cpu_telemetry.csv').open()));ov=[json.loads(l) for l in (D/'overhead_timeline.jsonl').read_text().splitlines()]
 timeline=[]
 for r in rs:
  hw=hardware([r],gpu,cpu);g=r['generation'];row={k:r[k] for k in ['request','phase','condition','repeat','key','profiled','intentional_idle_s','actual_intentional_wait_s','actual_gap_between_generate_wrappers_s','main_thread_running_delta_ms','main_thread_runqueue_wait_delta_ms','main_thread_timeslices_delta','main_thread_voluntary_context_switches_delta','main_thread_nonvoluntary_context_switches_delta','state_band','CUDA_memory_reserved','persistent_allocated_excluding_input','gc_events_seen']}
  row.update(begin_UTC=datetime.fromtimestamp(r['begin_unix_ns']/1e9,timezone.utc).isoformat(),begin_unix_ns=r['begin_unix_ns'],end_unix_ns=r['end_unix_ns'],CPU_begin_last_cpu=r['CPU_before']['last_cpu'],CPU_end_last_cpu=r['CPU_after']['last_cpu'],**{k:g[k] for k in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']},Event_decode_total_ms=sum(g['step_event_ms'][1:]),GPU_samples=hw['GPU_samples'],mean_SM_MHz=hw['GPU']['SM_MHz']['mean'],mean_memory_MHz=hw['GPU']['memory_MHz']['mean'],mean_power_W=hw['GPU']['power_W']['mean'],mean_temperature_C=hw['GPU']['temperature_C']['mean'],mean_GPU_util_pct=hw['GPU']['GPU_util_pct']['mean'],pstate_counts=json.dumps(hw['pstate_counts']),throttle_mask_counts=json.dumps(hw['throttle_mask_counts']));timeline.append(row)
 csvout('request_timeline.csv',timeline)
 conditions=[]
 for phase,condition in dict.fromkeys((r['phase'],r['condition']) for r in rs):
  group=[r for r in rs if r['phase']==phase and r['condition']==condition];c=dict(phase=phase,condition=condition,requests=len(group),profiled=group[0]['profiled'],intentional_idle_s=group[0]['intentional_idle_s'],state_counts=dict(Counter(r['state_band'] for r in group)),hardware=hardware(group,gpu,cpu))
  for k in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']:c[k]=stats([r['generation'][k] for r in group])
  for k in ['actual_gap_between_generate_wrappers_s','main_thread_running_delta_ms','main_thread_runqueue_wait_delta_ms']:c[k]=stats([r[k] for r in group if r[k] is not None])
  conditions.append(c)
 csvout('continuous_vs_idle.csv',[dict(phase=c['phase'],condition=c['condition'],n=c['requests'],profiled=c['profiled'],intentional_idle_s=c['intentional_idle_s'],**{k+'_'+stat:c[k][stat] for k in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms'] for stat in ['mean','median','stdev','min','max']},runqueue_wait_mean_ms=c['main_thread_runqueue_wait_delta_ms']['mean'],actual_gap_mean_s=c['actual_gap_between_generate_wrappers_s']['mean'],state_counts=json.dumps(c['state_counts'])) for c in conditions])
 summaries=[];steps=[];failures=[]
 for cap in read('profiler_capture_index.json'):
  r=next(r for r in rs if r['request']==cap['request'])
  try:
   p,st=profile_trace(D/cap['trace_file'],r);p.update(preceding_unprofiled_request=cap['preceding_unprofiled_request'],preceding_unprofiled_band=cap['preceding_unprofiled_band']);summaries.append(p);steps+=st
  except Exception as ex:failures.append(dict(request=r['request'],error=repr(ex)))
 csvout('profiler_per_step.csv',steps);save('profiler_summary.json',dict(tool='Existing PyTorch Profiler / Kineto',nsys_available=False,captures=summaries,analysis_errors=failures,profiled_requests_not_in_latency_distributions=True))
 states=[]
 for band in ['fast_historical_band','intermediate','slow_historical_band']:
  group=[r for r in rs if r['phase'] in ['A','B'] and not r['profiled'] and r['condition']!='fixed_warmup' and r['state_band']==band]
  states.append(dict(band=band,n=len(group),metrics={k:stats([r['generation'][k] for r in group]) for k in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']},hardware=hardware(group,gpu,cpu),main_thread_runqueue_wait_ms=stats([r['main_thread_runqueue_wait_delta_ms'] for r in group]),main_thread_running_ms=stats([r['main_thread_running_delta_ms'] for r in group])))
 transitions=[]
 a=[r for r in rs if r['phase']=='A' and r['condition']=='continuous_measured']
 for first,second in zip(a,a[1:]):
  if first['state_band']!=second['state_band']:transitions.append(dict(previous=first['request'],request=second['request'],at_unix_ns=second['begin_unix_ns'],previous_band=first['state_band'],new_band=second['state_band']))
 overhead={k:stats([v[k] for v in ov if v['phase'] in ['A','B'] and not v['profiled'] and k in v]) for k in ['H2D_enqueue_ms','seed_ms','clean_before_ms','clean_after_ms','JSON_serialization_ms','local_JSONL_flush_fsync_ms','wrapper_generation_ms','outer_without_intentional_idle_before_trace_export_ms']}
 # Retain all cgroup counter snapshots and verify throttling deltas, not just final values.
 cg=defaultdict(list)
 for c in cpu:
  for path,text in json.loads(c['cgroup']).items():
   if path.endswith('cpu.stat'):
    vals={k:int(v) for k,v in (l.split() for l in text.splitlines())}
    for k in ['nr_throttled','throttled_usec','nr_periods']:
     if k in vals:cg[path+'|'+k].append(vals[k])
 gaps=[(int(b['unix_time_ns'])-int(a['unix_time_ns']))/1e9 for a,b in zip(gpu,gpu[1:])]
 evidence=dict(status='CONTROLLED_DIAGNOSTIC_ANALYZED',completion=completion,conditions=conditions,unprofiled_nonwarmup_states=states,clean_unprofiled_state_scope='A/B only, exclude warmup and ALL C requests including unprofiled post-profiler scouts',C_profiler_adjacent_scouts=[dict(request=r['request'],E2E_ms=r['generation']['E2E_ms'],TPOT_ms=r['generation']['TPOT_ms'],prior_profiler_has_run=r['request']>70) for r in rs if r['condition']=='unprofiled_scout'],A_continuous_state_transitions=transitions,all_output_tokens_exact_historical=all(r['output32_exact_historical'] for r in rs),all_inputs_equal=len({json.dumps(r['input_hashes'],sort_keys=True) for r in rs})==1,all_native_full32=all(r['generation']['output_tokens']==32 and r['generation']['real_decode_calls']==31 and r['generation']['method']=='Native' for r in rs),GPU_sample_interval_s=stats(gaps),CPU_throttling_counters={k:dict(min=min(v),max=max(v),delta=max(v)-min(v)) for k,v in cg.items()},overhead_nonprofiled=overhead,telemetry_collector=read('telemetry_collector_summary.json'),CPU_frequency_governor_note='Governor/cpufreq not exposed; visible /proc/cpuinfo MHz may be virtualized. Per-request schedstat measures runqueue wait and running, not exact launch attribution.',profiler_captures=len(summaries),profiler_step_mapping_captures=sum(s['step_pair_mapping_available'] for s in summaries),root_cause_uniquely_identified=False,all_records_kept=True,formal_checkpoint_resumed=False)
 save('rootcause_evidence.json',evidence)
 print(json.dumps(dict(completion=completion,conditions=[{k:(v['mean'] if isinstance(v,dict) and 'mean' in v else v) for k,v in c.items() if k in ['phase','condition','requests','E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','state_counts','main_thread_runqueue_wait_delta_ms']} for c in conditions],states=[dict(band=s['band'],n=s['n']) for s in states],transitions=transitions,profiler_captures=len(summaries),profiler_step_mapping=sum(s['step_pair_mapping_available'] for s in summaries),profiler_errors=failures),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
