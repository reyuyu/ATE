"""CPU-only final metadata. Run after all GPU subprocesses have exited."""
from pathlib import Path
import json,hashlib,statistics,subprocess,time,collections
O=Path(__file__).resolve().parent.parent
UUID='GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6'
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={};checked={}
for name in ['source_manifest.json','cross_source_manifest.json']:
 m=read(O/name)
 for p,h in m['files'].items():
  assert sha(p)==h,('FINAL_SOURCE_CHANGED',p)
  sources[p]=h
 checked[name]=len(m['files'])
# Models were streaming-verified at restoration; avoid another full weight scan.
models={}
for d in ['llava-1.5-7b-hf','Qwen3-VL-8B-Instruct','InternVL3-8B']:
 p=Path('/home/featurize/ate-local-models')/d/'.ate_revision.json'
 models[d]=dict(path=str(p.parent),receipt_path=str(p),receipt_SHA256=sha(p),receipt=read(p))
save(O/'model_manifest.json',models)
env={};smoke={};telemetry={};counts=collections.Counter()
for group in ['LLaVA','Qwen','InternVL']:
 sm=O/'smoke_validation.json' if group=='LLaVA' else O/group/'smoke_validation.json'
 smoke[group]=read(sm) if sm.exists() else {'status':'NOT_RUN'}
 for phase in ['smoke','session1','session2']:
  p=O/phase if group=='LLaVA' else O/group/phase
  if (p/'environment.json').exists():env[group+'_'+phase]=read(p/'environment.json')
  if phase!='smoke' and (p/'requests.jsonl').exists():
   with (p/'requests.jsonl').open() as f:
    for line in f:
     r=json.loads(line)
     if r['category']=='formal':counts[(group,phase,r['method'])]+=1
  q=p/'gpu_telemetry.jsonl'
  if q.exists():
   with q.open() as f:a=[json.loads(l) for l in f]
   # Running/loaded samples only, retain full unfiltered raw logs separately.
   active=[r for r in a if r.get('GPU_util',0)>0]
   stats={}
   for k in ['SM_MHz','memory_MHz','temperature_C','power_mW','GPU_util']:
    vals=[r[k] for r in active if isinstance(r.get(k),(int,float))]
    if vals:stats[k]=dict(min=min(vals),median=statistics.median(vals),max=max(vals))
   process=p/'processes_telemetry.jsonl';others=[]
   if process.exists():
    with process.open() as f:
     for line in f:
      r=json.loads(line)
      if r.get('other_processes'):others.append(r)
   telemetry[group+'_'+phase]=dict(samples=len(a),active_samples=len(active),active_stats=stats,other_compute_events=others)
save(O/'environment.json',dict(GPU_UUID=UUID,GPU_settings_modified=False,sessions=env,telemetry_summary=telemetry))
save(O/'all_backbone_smoke_validation.json',smoke)
reg=read(O/'method_registry.json')
if not (O/'source_snapshots/method_registry_preformal.json').exists():save(O/'source_snapshots/method_registry_preformal.json',reg)
for r in reg:
 group='LLaVA' if r['ID'].startswith('L') else 'Qwen' if r['ID'].startswith('Q') else 'InternVL'
 method={'Native Full':'Native','V²Drop':'V2Drop','DUET-VLM (C+all)':'DUET','V²Drop+SVH':'V2Drop+SVH'}.get(r['method'],r['method'])
 r['runner']='code/run_llava_v2.py' if group=='LLaVA' else 'code/run_cross.py --kind '+group
 r['formal_counts']={phase:counts[(group,phase,method)] for phase in ['session1','session2']}
 r['status']='COMPLETE' if list(r['formal_counts'].values())==[30,30] else 'PARTIAL' if sum(r['formal_counts'].values()) else 'BLOCKED'
 r['smoke_status']=smoke[group]['status']
 if group=='Qwen':r['config']={'Native':'BF16/Eager TF4.57.6; all36 native layers; native dynamic visual tokens; full visual KV; no exit/bypass','SVH':'BF16/Eager TF4.57.6; writer0..21; exit22; skip1..6; zero persistent image KV; frozen temporary numeric alignment cost charged','SVH-R':'BF16/Eager TF4.57.6; writer0..21; main hidden exit22; skip1..6; ImageKV19..35; causal-prefix visual KV construction fully charged'}[method]
 if group=='InternVL':r['config']={'Native':'BF16/Eager TF4.49; single448 block/256 visual; all28 native layers; full visual KV; no exit/bypass','SVH':'BF16/Eager TF4.49; writer0..19; physical exit20; skip2..5; zero persistent image KV; frozen numerical waiver','SVH-R':'BF16/Eager TF4.49; writer0..19; keepImageKV19..27; terminal hidden exit28; skip2..5; temporary reduction padding fully charged'}[method]
 if r['status']!='COMPLETE':r['failure_evidence']=group+'/smoke_validation.json'
save(O/'method_registry.json',reg)
compile_count=0
for p in (O/'code').glob('*.py'):
 compile(p.read_text(),str(p),'exec');compile_count+=1;sources[str(p)]=sha(p)
save(O/'cpu_audit.json',dict(status='PASS',source_SHA_checked=checked,new_code_compile_count=compile_count,input_verification_files=[str(p.relative_to(O)) for p in O.rglob('input_verification.json')],model_receipts_reused=True,full_weight_hash_repeated=False,CPU_mock_test_claim='No new synthetic test count claimed; compile, frozen input/source checks and real GPU structural audits are documented separately.'))
save(O/'delivery_source_manifest.json',dict(status='VERIFIED_AFTER_RUN',files=sources,pinned_source_manifest_SHA=sha(O/'source_manifest.json'),cross_source_manifest_SHA=sha(O/'cross_source_manifest.json'),new_adapter_SHA256=sha(O/'code/apet_fusion.py'),runtime_snapshots='source_snapshots/pinned_runtime_tree'))
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
actual=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert actual==UUID
save(O/'GPU_release.json',dict(at_ns=time.time_ns(),GPU_UUID=actual,all_compute_apps=apps,released=not apps,server_closed=False,nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader'],text=True).strip()))
print(json.dumps(dict(method_counts={r['ID']:r['formal_counts'] for r in reg},source_checked=len(sources),GPU_released=not apps)))
