"""CPU-only audit of the admission failure; never performs model requests."""
from pathlib import Path
import json,sqlite3,statistics
D=Path('/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090_session2')
gate=json.loads((D/'native_stability_gate.json').read_text());db=sqlite3.connect('file:'+str(D/'checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True)
records=[json.loads(p) for (p,) in db.execute('SELECT payload FROM records ORDER BY id')];db.close();warm=[r for r in records if r['category']=='stability_warmup']
gpu=[json.loads(l) for l in (D/'gpu_telemetry.jsonl').read_text().splitlines() if l.strip()]
metrics=['E2E_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms'];evaluations=[];details={}
for ev in gate['evaluations']:
 evaluations.append(dict(after_warmup=ev['after_warmup'],status=ev['status'],summed_E2E_s=ev['summed_generation_E2E_s'],reasons=ev['reasons']))
 for window in ev['windows']:
  for key,stats in window['per_input'].items():
   if not any(x['robust_CV']>.01 or x['max_abs_deviation_from_median']>.03 for x in stats.values()):continue
   pair=(window['first_warmup'],window['last_warmup'],key)
   if pair in details:continue
   rows=[r for r in warm if pair[0]<=r['warmup_index']<=pair[1] and r['key']==key];out=[]
   for r in rows:
    rel={m:r['generation'][m]/stats[m]['median']-1 for m in metrics}
    samples=[g for g in gpu if r['begin_ns']<=g['at_ns']<=r['end_ns']]
    ranges={}
    for k in ['SM_MHz','memory_MHz','temperature_C','power_mW']:
     v=[g[k] for g in samples if isinstance(g.get(k),(int,float))];ranges[k]=dict(min=min(v),max=max(v)) if v else None
    out.append(dict(request_id=r['request_id'],warmup_index=r['warmup_index'],key=key,metrics={m:r['generation'][m] for m in metrics},relative_to_window_median=rel,main_thread_runqueue_ms=r['main_thread_runqueue_ms'],main_thread_running_ms=r['main_thread_running_ms'],within_request_Native_decode_change=r['early5_vs_late26_decode_median_change'],GPU_samples=len(samples),GPU_ranges=ranges,P_states=sorted({g['pstate'] for g in samples if 'pstate' in g},key=str),throttle_masks=sorted({g['throttle_mask'] for g in samples if 'throttle_mask' in g},key=str)))
   details[pair]=dict(first_warmup=pair[0],last_warmup=pair[1],key=key,window_stats=stats,requests=out)
report=dict(status='CPU_AUDIT_COMPLETE',new_GPU_requests=0,admission=gate['status'],evaluations=evaluations,flagged_windows=list(details.values()),interpretation='Admission dispersion gate failed. Contemporaneous clocks/runqueue and CUDA Events are evidence only; no counterfactual GPU experiment was performed, so no causal root is claimed. The same fixed61-80 window may persist in successive latest60 evaluations; this does not imply three independent anomaly episodes.',all_raw_records_retained=True)
(D/'gate_failure_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
brief=dict(admission=gate['status'],total_warmup=len(warm),summed_E2E_s=gate['summed_generation_E2E_s'],evaluations=evaluations,flagged_windows=[])
for w in details.values():
 brief['flagged_windows'].append(dict(first=w['first_warmup'],last=w['last_warmup'],key=w['key'],stats=w['window_stats'],requests=w['requests']))
print(json.dumps(brief,ensure_ascii=False,indent=2))
