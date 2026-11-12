"""Additional CPU-only evidence: telemetry holes, per-step changes, profiler gaps."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,statistics
D=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_rootcause_diagnostic')
rs=[json.loads(l) for l in (D/'requests.jsonl').read_text().splitlines()];gpu=list(csv.DictReader((D/'gpu_telemetry.csv').open()));p=json.loads((D/'profiler_summary.json').read_text());e=json.loads((D/'rootcause_evidence.json').read_text())
gaps=[]
for first,second in zip(gpu,gpu[1:]):
 dt=(int(second['unix_time_ns'])-int(first['unix_time_ns']))/1e9
 if dt>1:
  intersects=[r['request'] for r in rs if r['begin_unix_ns']<=int(second['unix_time_ns']) and r['end_unix_ns']>=int(first['unix_time_ns'])]
  gaps.append(dict(start_UTC=datetime.fromtimestamp(int(first['unix_time_ns'])/1e9,timezone.utc).isoformat(),end_UTC=datetime.fromtimestamp(int(second['unix_time_ns'])/1e9,timezone.utc).isoformat(),gap_s=dt,preceding_NVML_query_ms=float(first['query_duration_ms']),intersected_request_ids=json.dumps(intersects)))
with (D/'telemetry_gaps_over1s.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(gaps[0]) if gaps else ['start_UTC','end_UTC','gap_s','preceding_NVML_query_ms','intersected_request_ids']);w.writeheader();w.writerows(gaps)
qstats=[]
for r in rs:
 ts=[g for g in gpu if r['begin_unix_ns']<=int(g['unix_time_ns'])<=r['end_unix_ns']]
 qstats.append(dict(request=r['request'],phase=r['phase'],condition=r['condition'],samples=len(ts),max_NVML_query_ms=max([float(g['query_duration_ms']) for g in ts],default=None)))
with (D/'NVML_query_by_request.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(qstats[0]));w.writeheader();w.writerows(qstats)
metrics=['Event_sum_ms','traced_GPU_activity_union_sum_ms','internal_GPU_gaps_sum_ms','GPU_gap_before_next_API_start_ms','GPU_gap_during_next_API_ms','GPU_gap_after_next_API_end_ms','CUDA_launch_API_sum_ms']
agg={k:dict(mean=statistics.mean(c['decode'][k] for c in p['captures']),min=min(c['decode'][k] for c in p['captures']),max=max(c['decode'][k] for c in p['captures'])) for k in metrics if p['captures'] and all('decode' in c for c in p['captures'])}
first=next(r for r in rs if r['request']==49);decs=first['generation']['step_event_ms'][1:];summary=dict(slow_request49_first5_Event_decode_mean_ms=statistics.mean(decs[:5]),slow_request49_step6_Event_ms=decs[5],slow_request49_last25_Event_decode_mean_ms=statistics.mean(decs[6:]),slow_request49_main_runqueue_wait_ms=first['main_thread_runqueue_wait_delta_ms'],profiler_decode_aggregates=agg,telemetry_gaps_over1s=gaps,NVML_query_by_request_examples=[v for v in qstats if v['request'] in [35,49,50,70,84]],NVML_max_query_ms=max(float(g['query_duration_ms']) for g in gpu),C_scouts_potentially_contaminated=e['C_profiler_adjacent_scouts'])
(D/'additional_rootcause_evidence.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
