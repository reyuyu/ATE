"""CPU/std-library only; immutable SQLite and source records. No torch import."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter,defaultdict
import json,csv,sqlite3,hashlib,statistics,math,ast
SRC=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_recheck')
OUT=SRC.parent/'native_e2e32_3090_stability_audit_20261110'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mean(a):return statistics.mean(a) if a else None
def med(a):return statistics.median(a) if a else None
def corr(a,b):
 if len(a)<3 or statistics.pstdev(a)==0 or statistics.pstdev(b)==0:return None
 return sum((x-mean(a))*(y-mean(b)) for x,y in zip(a,b))/math.sqrt(sum((x-mean(a))**2 for x in a)*sum((y-mean(b))**2 for y in b))
def csvfile(n,rs):
 if not rs:return
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def ts(v):return datetime.fromisoformat(v).timestamp()
def snapshot():return {str(p.relative_to(SRC)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(SRC.rglob('*')) if p.is_file()}
def main():
 OUT.mkdir(exist_ok=True);before=snapshot();save('original_files_before.json',before)
 assert (SRC/'checkpoint.sqlite-wal').stat().st_size==0,'Immutable SQLite requires checkpointed WAL'
 db=sqlite3.connect('file:'+str(SRC/'checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True);data={};allrows=[]
 for table in ['formal','smoke','warmup','probes','failures']:
  rows=[]
  for key,payload,h in db.execute('select key,payload,sha256 from '+table+' order by rowid'):
   r=json.loads(payload);assert digest(r)==h;r.update(table=table,ledger_key=key);rows.append(r)
  data[table]=rows
  if table!='failures':allrows+=rows
 assert db.execute('pragma integrity_check').fetchone()[0]=='ok';db.close()
 allrows.sort(key=lambda r:r['begin_unix_time']);assert len(allrows)==85 and len(data['formal'])==15
 p=read(SRC/'protocol.json');bindings={s['key']:s for s in read(SRC/'input_verification.json')['records']};refs=read(SRC/'historical_native_tokens.json')
 checks={};source_mismatches={}
 for path,h in p['sources'].items():
  if sha(path)!=h:source_mismatches[path]=dict(expected=h,actual=sha(path))
 for name,h in p['entry_sources_SHA256'].items():assert sha(SRC/'code'/name)==h
 assert not source_mismatches
 artifact=read(SRC/'artifact_SHA256.json')['files'];artifact_mismatches={n:dict(expected=e['SHA256'],actual=sha(SRC/n)) for n,e in artifact.items() if sha(SRC/n)!=e['SHA256']}
 for table in ['formal','smoke','warmup','probes']:
  exported=[json.loads(l) for l in (SRC/(table+'_requests.jsonl')).read_text().splitlines()]
  assert all({k:v for k,v in a.items() if k!='table'}==b for a,b in zip(data[table],exported)) and len(exported)==len(data[table])
 for r in allrows:
  g=r['generation'];s=bindings[r['key']]
  assert r['protocol_SHA256']==digest(p) and r['input_hashes']==s['input_hashes'] and r['sample_SHA256']==s['sample_sha256']
  assert g['token_ids']==refs[r['key']] and g['output_tokens']==32 and g['real_decode_calls']==31 and g['prefill_calls']==g['vision_forward_calls']==1
  assert r['GPU_UUID']==p['GPU_UUID'] and g['method']=='Native' and g['all_logits_finite']
  assert r['runtime_SHA256']==p['sources'][str(SRC.parent/'unified_e2e32_4090/code/e2e32_runtime.py')]
  assert r['entry_SHA256']==p['entry_sources_SHA256'] and r['model_receipt_SHA256']==p['model_receipt_sha256']
  assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6 and len(g['step_event_ms'])==32
  assert abs(sum(g['step_event_ms'][1:])/31-g['CUDA_Event_Decode_TPOT_ms'])<1e-8
  checks.setdefault(r['key'],dict(input_hashes=r['input_hashes'],output32_SHA256=digest(g['token_ids']),sample_SHA256=r['sample_SHA256'],KV_live_bytes=g['cache']['live_bytes'],observed_requests=0))['observed_requests']+=1
  assert checks[r['key']]['KV_live_bytes']==g['cache']['live_bytes']
 start=allrows[0]['begin_unix_time'];tele=[]
 for t in map(json.loads,(SRC/'sessions/session_01/telemetry.jsonl').read_text().splitlines()):
  if 'error' in t:raise RuntimeError('Telemetry error '+str(t))
  fields=t['query_fields'].split(',');values=[s.strip() for s in t['GPU'].split(',')];assert len(fields)==len(values)
  d=dict(zip(fields,values));d.update(at=t['at'],epoch=ts(t['at']),loadavg=t['loadavg'],cpu_pressure=t['cpu_pressure'],other_processes=t['other_processes'])
  # Driver timestamp and collector timestamp agree to sub-second for normal samples; retain both.
  gpu_epoch=datetime.strptime(d['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp();d['gpu_timestamp_epoch_assuming_server_UTC']=gpu_epoch;d['collector_lag_s']=d['epoch']-gpu_epoch
  d['collector_epoch']=d['epoch'];d['epoch']=gpu_epoch
  tele.append(d)
 csvfile('telemetry_normalized.csv',[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in d.items()} for d in tele])
 def aggregate_tele(samples):
  v=dict(samples=len(samples),pstates=dict(Counter(s['pstate'] for s in samples)),throttle_masks=dict(Counter(s['clocks_throttle_reasons.active'] for s in samples)))
  for key in ['clocks.current.sm','clocks.current.memory','temperature.gpu','power.draw','power.limit','utilization.gpu','memory.used']:
   a=[float(s[key]) for s in samples];v[key]=dict(mean=mean(a),min=min(a) if a else None,max=max(a) if a else None)
  for key in ['clocks_throttle_reasons.sw_power_cap','clocks_throttle_reasons.hw_slowdown','clocks_throttle_reasons.hw_thermal_slowdown','clocks_throttle_reasons.hw_power_brake_slowdown','clocks_throttle_reasons.sw_thermal_slowdown']:
   v[key]=dict(Counter(s[key] for s in samples))
  loads=[float(s['loadavg'].split()[0]) for s in samples];v['CPU_load1']=dict(mean=mean(loads),min=min(loads) if loads else None,max=max(loads) if loads else None)
  return v
 timeline=[];profiles=[];prev=None
 for index,r in enumerate(allrows,1):
  g=r['generation'];samples=[s for s in tele if r['begin_unix_time']<=s['epoch']<=r['end_unix_time']];nearest=min(tele,key=lambda s:abs(s['epoch']-(r['begin_unix_time']+r['end_unix_time'])/2));a=aggregate_tele(samples)
  row=dict(sequence=index,table=r['table'],role=r['role'],ledger_key=r['ledger_key'],key=r['key'],begin_UTC=datetime.fromtimestamp(r['begin_unix_time'],timezone.utc).isoformat(),end_UTC=datetime.fromtimestamp(r['end_unix_time'],timezone.utc).isoformat(),begin_Beijing=datetime.fromtimestamp(r['begin_unix_time'],timezone(timedelta(hours=8))).isoformat(),relative_start_s=r['begin_unix_time']-start,wrapper_wall_ms=(r['end_unix_time']-r['begin_unix_time'])*1000,preceding_gap_s=None if prev is None else r['begin_unix_time']-prev['end_unix_time'],E2E_ms=g['E2E_ms'],TTFT_ms=g['TTFT_ms'],TPOT_ms=g['TPOT_ms'],CUDA_Event_prefill_ms=g['step_event_ms'][0],CUDA_Event_Decode_TPOT_ms=g['CUDA_Event_Decode_TPOT_ms'],CUDA_Event_Decode_total_ms=sum(g['step_event_ms'][1:]),CUDA_Event_step_sum_ms=sum(g['step_event_ms']),Decode_wall_ms=31*g['TPOT_ms'],wall_minus_event_TPOT_ms=g['TPOT_ms']-g['CUDA_Event_Decode_TPOT_ms'],output_tokens=32,real_decode_calls=31,observer_hooks=g['smoke'],residual_allocated_bytes=r['persistent_allocated_excluding_input'],KV_live_bytes=g['cache']['live_bytes'],output32_SHA256=digest(g['token_ids']),input_binding_SHA256=digest(r['input_hashes']),telemetry_overlap_samples=len(samples),nearest_telemetry_age_to_request_mid_s=abs(nearest['epoch']-(r['begin_unix_time']+r['end_unix_time'])/2),SM_clock_mean_MHz=a['clocks.current.sm']['mean'],memory_clock_mean_MHz=a['clocks.current.memory']['mean'],temperature_mean_C=a['temperature.gpu']['mean'],power_mean_W=a['power.draw']['mean'],utilization_mean_pct=a['utilization.gpu']['mean'],CPU_load1_mean=a['CPU_load1']['mean'],throttle_masks=json.dumps(a['throttle_masks']))
  timeline.append(row);prev=r
  for step,ms in enumerate(g['step_event_ms']):profiles.append(dict(sequence=index,key=r['key'],role=r['role'],step=step,kind='Prefill' if step==0 else 'Decode',CUDA_Event_ms=ms))
 csvfile('request_timeline.csv',timeline);csvfile('per_step_event_timeline.csv',profiles)
 with (OUT/'request_timeline.jsonl').open('w') as f:
  for r in timeline:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 phase_rows=[];phase_stats={}
 for role in sorted({r['role'] for r in allrows},key=lambda role:next(i for i,r in enumerate(allrows) if r['role']==role)):
  group=[r for r in allrows if r['role']==role];points=[r for r in timeline if r['role']==role];a=dict(requests=len(group),first_begin=group[0]['begin_unix_time'],last_end=group[-1]['end_unix_time'],span_s=group[-1]['end_unix_time']-group[0]['begin_unix_time'],sum_measured_E2E_s=sum(r['generation']['E2E_ms'] for r in group)/1000,sum_wrapper_s=sum(r['end_unix_time']-r['begin_unix_time'] for r in group),observer_hooks=any(r['generation']['smoke'] for r in group),residual_allocations=sorted({r['persistent_allocated_excluding_input'] for r in group}))
  for metric in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']:
   values=[r['generation'][metric] for r in group];a[metric]=dict(mean=mean(values),median=med(values),min=min(values),max=max(values))
  a['telemetry']=aggregate_tele([s for s in tele if any(r['begin_unix_time']<=s['epoch']<=r['end_unix_time'] for r in group)]);phase_stats[role]=a;phase_rows.append(dict(role=role,**{k:v for k,v in a.items() if not isinstance(v,(dict,list))}))
 csvfile('phase_time_budget.csv',phase_rows)
 cycles=[]
 for table,role,block in [('warmup','warmup',2),('probes','baseline',2),('probes','calibration',2)]:
  rows=[r for r in data[table] if r['role']==role]
  for i in range(0,len(rows),block):
   group=rows[i:i+block];assert len(group)==2
   cyc=dict(role=role,cycle=i//2,first_begin=group[0]['begin_unix_time'],last_end=group[-1]['end_unix_time'],begin_UTC=datetime.fromtimestamp(group[0]['begin_unix_time'],timezone.utc).isoformat(),relative_start_s=group[0]['begin_unix_time']-start)
   for metric in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']:cyc[metric]=mean([r['generation'][metric] for r in group])
   cyc['telemetry']=aggregate_tele([s for s in tele if any(r['begin_unix_time']<=s['epoch']<=r['end_unix_time'] for r in group)]);cycles.append(cyc)
 csvfile('fixed_probe_pair_cycles.csv',[{k:v for k,v in r.items() if k!='telemetry'} for r in cycles]);save('fixed_probe_pair_cycles_with_telemetry.json',cycles)
 baseline=[r for r in data['probes'] if r['role']=='baseline'];cal=[r for r in data['probes'] if r['role']=='calibration'];comparisons=[]
 for i in range(0,len(cal),2):
  group=cal[i:i+2];c=dict(after_formal_inputs=(i//2+1)*5)
  for metric in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']:
   b=mean([r['generation'][metric] for r in baseline]);v=mean([r['generation'][metric] for r in group]);c[metric]=dict(baseline=b,measured=v,delta=v-b,relative_change=v/b-1)
  c['Decode_share_of_E2E_drop']=(c['TPOT_ms']['delta']*31)/c['E2E_ms']['delta'];comparisons.append(c)
 # Descriptive change point only: no data are excluded, no baseline redefined.
 warm_cycles=[c for c in cycles if c['role']=='warmup'];base_cycles=[c for c in cycles if c['role']=='baseline']
 segments=[]
 for role,selection in [('baseline',baseline),('after5',cal[:2]),('after10',cal[2:4]),('after15',cal[4:6])]:
  segments.append(dict(phase=role,telemetry=aggregate_tele([s for s in tele if any(r['begin_unix_time']<=s['epoch']<=r['end_unix_time'] for r in selection)])))
 associations={}
 for key in p['probe_keys']:
  pts=[r for r in timeline if r['key']==key and not r['observer_hooks']]
  associations[key]={metric:dict(n=sum(r[metric] is not None for r in pts),Pearson_with_TPOT=corr([r[metric] for r in pts if r[metric] is not None],[r['TPOT_ms'] for r in pts if r[metric] is not None])) for metric in ['SM_clock_mean_MHz','temperature_mean_C','power_mean_W','utilization_mean_pct','CPU_load1_mean']}
 gaps=[b['epoch']-a['epoch'] for a,b in zip(tele,tele[1:])]
 # Audit source text via AST; never import any model/runtime module.
 src=SRC/'code/run_native.py';rt=SRC.parent/'unified_e2e32_4090/code/e2e32_runtime.py';runtime=rt.read_text();entry=src.read_text()
 source_calls=[]
 for path in [src,rt]:
  tree=ast.parse(path.read_text());parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
  for node in ast.walk(tree):
   if not isinstance(node,ast.Call):continue
   name=ast.unparse(node.func)
   if name in ['gc.collect','torch.cuda.empty_cache','torch.cuda.synchronize','torch.cuda.Event','torch.arange','torch.ones','generate','torch.manual_seed','torch.cuda.memory_allocated']:
    contexts=[];par=node
    while par in parents:
     par=parents[par]
     if isinstance(par,(ast.FunctionDef,ast.AsyncFunctionDef)):contexts.append(par.name)
    source_calls.append(dict(file=str(path),line=node.lineno,call=name,functions=contexts,source=path.read_text().splitlines()[node.lineno-1].strip()))
 save('source_call_audit.json',source_calls)
 env=read(SRC/'sessions/session_01/environment.json')
 summary=dict(status='CPU_READONLY_AUDIT',original_experiment_status='HISTORICAL_STABILITY_FAILED',formal_results_preserved=15,requests=len(allrows),table_counts={k:len(v) for k,v in data.items()},role_counts=dict(Counter(r['role'] for r in allrows)),all_input_output_path_bindings_exact=True,source_SHA256_verified=len(p['sources']),artifact_reference_mismatches=artifact_mismatches,phase_stats=phase_stats,fixed_probe_comparisons=comparisons,warmup_pair_E2E_ms=[r['E2E_ms'] for r in warm_cycles],baseline_pair_E2E_ms=[r['E2E_ms'] for r in base_cycles],baseline_pair_TPOT_ms=[r['TPOT_ms'] for r in base_cycles],warmup_span_s=phase_stats['warmup']['span_s'],warmup_E2E_sum_s=phase_stats['warmup']['sum_measured_E2E_s'],old_field_warmup_seconds=read(SRC/'sessions/session_01/drift_baseline.json')['warmup_seconds'],telemetry_summary=aggregate_tele(tele),telemetry_gap_s=dict(median=med(gaps),maximum=max(gaps),over5=sum(v>5 for v in gaps)),telemetry_collector_lag_s=dict(median=med([s['collector_lag_s'] for s in tele]),max=max(s['collector_lag_s'] for s in tele)),other_GPU_process_events=sum(bool(s['other_processes']) for s in tele),same_probe_correlations_descriptive_not_causal=associations,baseline_and_calibration_hardware=segments,CPU_settings={k:env.get(k) for k in ['CPU_models','CPU_affinity','torch_num_threads','torch_num_interop_threads','cgroup_settings','env','cpu_pressure','loadavg']},inputs=checks,no_torch_import=True,GPU_calls_or_generation_performed=False)
 save('audit_statistics.json',summary)
 after=snapshot();assert before==after,'Original experiment changed during audit'
 save('original_files_after.json',after);save('preservation_validation.json',dict(status='PASS',original_scope=str(SRC),files_verified_unchanged=len(before),formal15_preserved=True,SQLite_immutable_read=True,checkpoint_SHA256=sha(SRC/'checkpoint.sqlite'),native_CSV_SHA256=sha(SRC/'native_per_input.csv'),original_SHA_inventory_identical=True,GPU_generation_started=False))
 save('historical_experiment_status.json',dict(path=str(SRC),classification='HISTORICAL_STABILITY_FAILED',formal_requests=15,reference_eligible=False,original_files_edited=False,do_not_delete_or_select_fastest=True))
 print(json.dumps({k:summary[k] for k in ['table_counts','role_counts','warmup_span_s','warmup_E2E_sum_s','old_field_warmup_seconds','warmup_pair_E2E_ms','baseline_pair_E2E_ms','fixed_probe_comparisons','baseline_and_calibration_hardware','same_probe_correlations_descriptive_not_causal','artifact_reference_mismatches']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
