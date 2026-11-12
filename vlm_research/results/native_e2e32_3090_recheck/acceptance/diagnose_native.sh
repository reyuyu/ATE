/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python - <<'PY'
import json,sqlite3,statistics,datetime
from pathlib import Path
O=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_recheck')
db=sqlite3.connect('file:'+str(O/'checkpoint.sqlite')+'?mode=ro',uri=True)
rows=[json.loads(r[0]) for r in db.execute('select payload from probes order by rowid')];db.close()
tele=[json.loads(l) for l in (O/'sessions/session_01/telemetry.jsonl').read_text().splitlines()]
groups={}
for r in rows:
 group=r['role'];g=r['generation'];item=groups.setdefault(group,[])
 item.append(dict(key=r['key'],at=r['at'],E2E_ms=g['E2E_ms'],TTFT_ms=g['TTFT_ms'],TPOT_ms=g['TPOT_ms'],event_TPOT_ms=g['CUDA_Event_Decode_TPOT_ms']))
b=groups['baseline'];rounds=[statistics.mean(r['E2E_ms'] for r in b[i:i+2]) for i in range(0,len(b),2)]
times=[datetime.datetime.fromisoformat(t['at']).timestamp() for t in tele];gaps=[b-a for a,b in zip(times,times[1:])]
def state_near(r):
 ts=datetime.datetime.fromisoformat(r['at']).timestamp();t=min(tele,key=lambda t:abs(datetime.datetime.fromisoformat(t['at']).timestamp()-ts))
 return dict(probe=r,nearest_telemetry=t)
old=json.loads((O.parent/'unified_e2e32_4090/environment.json').read_text());new=json.loads((O/'sessions/session_01/environment.json').read_text())
v=dict(baseline_round_E2E_means_ms=rounds,baseline_round_max_minus_min_fraction=(max(rounds)-min(rounds))/statistics.mean(rounds),probes=groups,telemetry_samples=len(tele),telemetry_gap_median_s=statistics.median(gaps),telemetry_gap_max_s=max(gaps),telemetry_gaps_over5s=sum(v>5 for v in gaps),other_compute_process_events=sum(bool(t.get('other_processes')) for t in tele),probe_telemetry_context=[state_near(r) for r in groups['calibration']],CPU_thread_settings={k:dict(old4090=old.get(k),new3090=new.get(k)) for k in ['torch_num_threads','torch_num_interop_threads','CPU_models','CPU_affinity','deterministic','TF32_matmul','FP16_reduced_precision_reduction']},interpretation='Observed drift is real in recorded timed Native requests. Tokens/inputs unchanged. No competing GPU process observed. Sparse telemetry cannot uniquely identify device-clock/host scheduling/I-O duty-cycle causes; no GPU settings changed.')
(O/'diagnosis.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in v.items() if k not in ['probe_telemetry_context','CPU_thread_settings']},ensure_ascii=False,indent=2))
PY
