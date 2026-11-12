"""Analyze historical records with stdlib only. Never import torch or execute the runner."""
from pathlib import Path
import os,json,csv,statistics,hashlib,datetime
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/native_e2e32_3090_recheck';A=O.parent/'native_e2e32_3090_stability_audit_20261110'
def load(p):return json.loads(p.read_text())
def save(n,v):(A/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
rows=[json.loads(l) for l in (A/'request_timeline.jsonl').read_text().splitlines()]
tele=list(csv.DictReader((A/'telemetry_normalized.csv').open()))
cycles=load(A/'fixed_probe_pair_cycles_with_telemetry.json')
stats=load(A/'audit_statistics.json')
def timing(rs):
 return {m:statistics.mean(float(r[m]) for r in rs) for m in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']}
keys=['GQA:202121628','TextVQA:38571'];bykey={}
for key in keys:
 bykey[key]={role:timing([r for r in rows if r['role']==role and r['key']==key]) for role in ['warmup','baseline','calibration']}
 bykey[key]['all_untraced_E2E_range']=[min(r['E2E_ms'] for r in rows if not r['observer_hooks'] and r['key']==key),max(r['E2E_ms'] for r in rows if not r['observer_hooks'] and r['key']==key)]
 bykey[key]['same_sampled_SM_1950MHz_examples']=[{k:r[k] for k in ['sequence','role','begin_Beijing','E2E_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','SM_clock_mean_MHz','memory_clock_mean_MHz','temperature_mean_C','power_mean_W','CPU_load1_mean','telemetry_overlap_samples']} for r in rows if r['key']==key and r['sequence'] in [9,15,31,37,53,63,77,84,10,16,32,38,56,64,78,85]]
profile=list(csv.DictReader((A/'per_step_event_timeline.csv').open()))
stepcompare=[]
for step in range(32):
 b=statistics.mean(float(r['CUDA_Event_ms']) for r in profile if r['role']=='baseline' and int(r['step'])==step)
 f=statistics.mean(float(r['CUDA_Event_ms']) for r in profile if r['role']=='calibration' and int(r['step'])==step and int(r['sequence'])>=84)
 stepcompare.append(dict(step=step,baseline_Event_ms=b,after15_Event_ms=f,delta_ms=f-b,change_pct=100*(f/b-1)))
with (A/'per_step_baseline_vs_after15.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(stepcompare[0]));w.writeheader();w.writerows(stepcompare)
pressure=[]
for t in tele:
 p={k:dict(x.split('=') for x in v.split()) for k,v in (line.split(' ',1) for line in t['cpu_pressure'].splitlines())}
 pressure.append(p)
pressure_summary={kind:{k:{'min':min(float(p[kind][k]) for p in pressure),'max':max(float(p[kind][k]) for p in pressure)} for k in ['avg10','avg60','avg300','total']} for kind in ['some','full']}
gaplist=[]
for first,second in zip(tele,tele[1:]):
 gap=float(second['epoch'])-float(first['epoch'])
 if gap>5:gaplist.append(dict(from_driver_UTC=first['timestamp'],to_driver_UTC=second['timestamp'],gap_s=gap))
entry=(O/'code/run_native.py').read_text().splitlines();runtime=(O.parent/'unified_e2e32_4090/code/e2e32_runtime.py').read_text().splitlines()
cache=Path('/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/lib/python3.11/site-packages/transformers/cache_utils.py')
excerpts=[]
for name,ls,ranges in [('run_native.py',entry,[(25,45),(51,80),(92,116),(120,127)]),('e2e32_runtime.py',runtime,[(95,144)]),('transformers4.49.0/cache_utils.py',cache.read_text().splitlines(),[(398,443)])]:
 for lo,hi in ranges:excerpts.append('\n'+name+':'+str(lo)+'\n'+'\n'.join(str(i)+': '+ls[i-1] for i in range(lo,min(hi,len(ls))+1)))
(A/'source_excerpts.txt').write_text('\n'.join(excerpts)+'\n')
mounts=Path('/proc/mounts').read_text().splitlines();matches=[]
for l in mounts:
 fields=l.split()
 if len(fields)>=3 and str(O).startswith(fields[1].rstrip('/')+'/'):matches.append(dict(device=fields[0],mount=fields[1],filesystem=fields[2]))
info=dict(fixed_probe_by_key=bykey,per_step_decode_change_pct_range=[min(r['change_pct'] for r in stepcompare[1:]),max(r['change_pct'] for r in stepcompare[1:])],CPU_pressure=pressure_summary,telemetry_gaps_over5s=gaplist,telemetry_driver_timestamp_used=True,driver_timestamp_timezone='UTC inferred from <0.42s agreement with UTC collector timestamp',timestamp_note='Request UTC is wrapper begin/end; exact perf_counter boundaries and per-step absolute event timestamps were not logged. Hardware matching uses samples inside wrapper intervals; incomplete sampling cannot characterize clocks throughout every request.',gaps_over1s=[{k:r[k] for k in ['sequence','role','key','begin_Beijing','preceding_gap_s','E2E_ms']} for r in rows if r['preceding_gap_s'] is not None and r['preceding_gap_s']>1],phase_hardware=stats['baseline_and_calibration_hardware'],current_read_only_mount_observation=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),matching_mounts=matches,note='Current mount observation only; historical mount transitions were not logged.'),cache_utils_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),decode_event_all_steps_change=all(r['delta_ms']<0 for r in stepcompare[1:]))
save('additional_evidence.json',info)
print(json.dumps({k:info[k] for k in ['per_step_decode_change_pct_range','CPU_pressure','telemetry_gaps_over5s','gaps_over1s','current_read_only_mount_observation','phase_hardware']},ensure_ascii=False,indent=2))
print('FIXED_GQA',json.dumps(bykey[keys[0]],ensure_ascii=False,indent=2))
print('SOURCE lines58-79\n'+'\n'.join(str(i)+': '+entry[i-1] for i in range(58,80)))
