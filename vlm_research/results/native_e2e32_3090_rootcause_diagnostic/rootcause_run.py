"""Controlled Native@32 diagnostics; imports the unchanged frozen generate function."""
import os,sys
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_MODULE_LOADING='LAZY')
sys.dont_write_bytecode=True
from pathlib import Path
import json,time,hashlib,subprocess,gc,traceback,shutil,statistics,contextlib
L=Path('/home/featurize/ate-local-diagnostics/native_e2e32_3090_rootcause_diagnostic');D=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_rootcause_diagnostic');O=D.parent/'native_e2e32_3090_recheck'
sys.path.insert(0,str(O/'code'))
import prepare_native as ref
import torch,transformers
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from vlm_research.physical_handoff_fastpath.engine import Layout
from e2e32_runtime import generate
from e2e32_environment import static_environment
def now():return time.time_ns()
def save(n,v):
 p=L/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def thread_snapshot():
 tid=os.getpid();root=Path('/proc/self/task')/str(tid);s=(root/'stat').read_text();v=s[s.rfind(')')+2:].split();status=(root/'status').read_text()
 d=dict(schedstat=[int(v) for v in (root/'schedstat').read_text().split()],utime_ticks=int(v[11]),stime_ticks=int(v[12]),last_cpu=int(v[36]),thread_CPU_time_ns=time.thread_time_ns())
 for line in status.splitlines():
  if line.startswith(('voluntary_ctxt_switches:','nonvoluntary_ctxt_switches:')):
   k,val=line.split(':',1);d[k]=int(val)
 return d
def check_stop():
 if (L/'STOP.flag').exists():raise RuntimeError('Safety stop: '+(L/'STOP.flag').read_text())
def tag(g):return 'fast_historical_band' if g['TPOT_ms']<=69 else 'slow_historical_band' if g['TPOT_ms']>=71 else 'intermediate'
def main():
 plan=ref.read(L/'protocol.json');old=ref.read(O/'protocol.json');count=0;records=[];a=None;monitor=None;decoder=None;state=dict(status='RUNNING',phase='PRECHECK',attempted=0,completed=0,pid=os.getpid());previous_end=None;gc_events=[];profiler_records=[];overhead=[]
 def progress():state['updated_unix_ns']=now();save('progress.json',state)
 def gcc(phase,info):gc_events.append(dict(unix_ns=now(),phase=phase,info=info))
 gc.callbacks.append(gcc)
 try:
  progress();assert torch.__version__==old['torch'] and transformers.__version__==old['transformers']
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip(),'GPU occupied'
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==ref.UUID
  for path,h in old['sources'].items():assert ref.sha(path)==h,path
  for n,h in old['entry_sources_SHA256'].items():assert ref.sha(O/'code'/n)==h,n
  assert ref.sha(ref.U/'input_manifest.jsonl.gz')==old['input_manifest_SHA256'];assert ref.sha(ref.M/'.ate_revision.json')==old['model_receipt_sha256']
  receipt=ref.read(ref.M/'.ate_revision.json');assert receipt['status']=='VERIFIED' and receipt['revision']==ref.REV and receipt['all_LFS_SHA256_verified']
  for n,item in receipt['files'].items():assert (ref.M/n).stat().st_size==item['bytes']
  sources={n:ref.sha(L/n) for n in ['rootcause_run.py','rootcause_monitor.py']};save('diagnostic_source_SHA256.json',sources)
  monitor=subprocess.Popen([sys.executable,'-B',str(L/'rootcause_monitor.py'),str(L),str(os.getpid()),ref.UUID],stdout=(L/'monitor.log').open('x'),stderr=subprocess.STDOUT)
  state['phase']='LOAD_MODEL';progress();loader.MODELS=dict(loader.MODELS,llava=ref.M);a=frozen.load('B')
  assert {str(v.device) for v in a.model.parameters()}=={'cuda:0'} and {v.dtype for v in a.model.parameters()}=={torch.float16}
  assert a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager' and not a.model.training
  env=static_environment(a);env.update(actual_GPU_UUID=ref.UUID,diagnostic_sources_SHA256=sources,profiler_activities=[str(v) for v in torch.profiler.supported_activities()],nsys=None,single_model_load=True,precheck_file='precheck_readonly.json',sample_before_generation=thread_snapshot());save('environment.json',env)
  key=plan['input_key'];s=next(v for v in ref.lines(ref.U/'input_manifest.jsonl.gz') if v['key']==key);imagepath=ref.read(O/'input_path_mapping.json')[key];refs=ref.read(O/'historical_native_tokens.json')[key]
  assert ref.sha(imagepath)==s['image_sha256'];assert hashlib.sha256(s['prompt'].encode()).hexdigest()==s['prompt_sha256']
  raw=dict(input_ids=torch.tensor([s['input_ids']],dtype=torch.long),attention_mask=torch.ones((1,len(s['input_ids'])),dtype=torch.long),pixel_values=frozen.paper_pixels(imagepath,a.image_processor).half());hashes={k:tensor_hash(v) for k,v in raw.items()};assert hashes==s['input_hashes']
  save('input_verification.json',dict(status='PASS',key=key,image_path=imagepath,image_SHA256=s['image_sha256'],prompt_SHA256=s['prompt_sha256'],input_hashes=hashes,sample_SHA256=s['sample_sha256'],historical32_token_ids=refs,original_manifest_unchanged=True))
  forward=[l.forward.__func__ for l in a.layers];att=[l.self_attn.forward.__func__ for l in a.layers];decoder=a.model.language_model.model;dec=decoder.forward.__func__;top=a.model.forward.__func__
  def clean():
   assert decoder.forward.__func__ is dec and a.model.forward.__func__ is top
   assert all(l.forward.__func__ is f for l,f in zip(a.layers,forward)) and all(l.self_attn.forward.__func__ is f for l,f in zip(a.layers,att))
   assert all(not l._forward_hooks and not l._forward_pre_hooks for l in a.layers)
   assert all(not getattr(l.self_attn,k+'_proj')._forward_pre_hooks for l in a.layers for k in ['q','k','v'])
  out=(L/'requests.jsonl').open('x');of=(L/'overhead_timeline.jsonl').open('x');attempts=(L/'attempts.jsonl').open('x');alloc_ceiling=None
  def request(phase,condition,repeat,idle_s=0,profiled=False):
   nonlocal count,previous_end,alloc_ceiling
   check_stop();assert monitor.poll() is None,'Telemetry collector exited';assert count<plan['hard_request_limit'] and count<plan['planned_requests']
   idle0=now();idle_mono=time.monotonic()
   while time.monotonic()-idle_mono<idle_s:
    check_stop();time.sleep(min(.25,idle_s-(time.monotonic()-idle_mono)))
   idle_end=now();outer0=time.perf_counter_ns();prep0=now();cpu0=thread_snapshot();h0=time.perf_counter_ns();a.inputs={k:v.to('cuda:0') for k,v in raw.items()};h2d_ms=(time.perf_counter_ns()-h0)/1e6
   ids=s['input_ids'];image=tuple(i for i,v in enumerate(ids) if v==32000);text=tuple(i for i,v in enumerate(ids) if v!=32000);relay=tuple(range(image[-1]+1,len(ids)-1));q=question_positions
   layout=Layout(len(ids),image,text,relay,q,20);t0=time.perf_counter_ns();torch.manual_seed(0);seed_ms=(time.perf_counter_ns()-t0)/1e6;t0=time.perf_counter_ns();clean();clean_before_ms=(time.perf_counter_ns()-t0)/1e6
   count+=1;state.update(phase=phase,condition=condition,attempted=count,explicit_idle_s=idle_s);progress();attempts.write(json.dumps(dict(request=count,phase=phase,condition=condition,profiled=profiled,at_ns=now()))+'\n');attempts.flush();os.fsync(attempts.fileno())
   prof=None;prof_start_overhead_ms=0;begin=now();sched0=thread_snapshot()
   if profiled:
    t0=time.perf_counter_ns();acts=[torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():acts.append(torch.profiler.ProfilerActivity.CUDA)
    prof=torch.profiler.profile(activities=acts,record_shapes=False,profile_memory=False,with_stack=False);prof.__enter__();prof_start_overhead_ms=(time.perf_counter_ns()-t0)/1e6
   begin=now()
   with torch.profiler.record_function('FROZEN_NATIVE_FULL32') if profiled else contextlib.nullcontext():g=generate(a,layout,'Native',smoke=False)
   end=now();sched1=thread_snapshot();profile_stop_ms=0
   if profiled:
    t0=time.perf_counter_ns();prof.__exit__(None,None,None);profile_stop_ms=(time.perf_counter_ns()-t0)/1e6
   t0=time.perf_counter_ns();clean();clean_after_ms=(time.perf_counter_ns()-t0)/1e6
   assert g['token_ids']==refs and g['output_tokens']==32 and g['prefill_calls']==g['vision_forward_calls']==1 and g['real_decode_calls']==31 and g['all_logits_finite']
   assert g['cache']['cached_decode_inputs']==31 and g['cache']['live_bytes']==g['cache']['storage_bytes'] and len(g['cache']['layers'])==32
   assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6
   residual=torch.cuda.memory_allocated()-sum(v.numel()*v.element_size() for v in a.inputs.values())
   if not profiled:
    if alloc_ceiling is None:alloc_ceiling=residual
    assert residual<=alloc_ceiling+8*2**20,'Unexpected persistent CUDA allocation increase'
   r=dict(request=count,phase=phase,condition=condition,repeat=repeat,key=key,profiled=profiled,intentional_idle_s=idle_s,idle_start_unix_ns=idle0,idle_end_unix_ns=idle_end,actual_intentional_wait_s=(idle_end-idle0)/1e9,prep_start_unix_ns=prep0,begin_unix_ns=begin,end_unix_ns=end,previous_wrapper_end_unix_ns=previous_end,actual_gap_between_generate_wrappers_s=None if previous_end is None else (begin-previous_end)/1e9,CPU_before=sched0,CPU_after=sched1,main_thread_running_delta_ms=(sched1['schedstat'][0]-sched0['schedstat'][0])/1e6,main_thread_runqueue_wait_delta_ms=(sched1['schedstat'][1]-sched0['schedstat'][1])/1e6,main_thread_timeslices_delta=sched1['schedstat'][2]-sched0['schedstat'][2],main_thread_voluntary_context_switches_delta=sched1['voluntary_ctxt_switches']-sched0['voluntary_ctxt_switches'],main_thread_nonvoluntary_context_switches_delta=sched1['nonvoluntary_ctxt_switches']-sched0['nonvoluntary_ctxt_switches'],GPU_UUID=ref.UUID,input_hashes=hashes,output32_exact_historical=True,runtime_SHA256=old['sources'][str(ref.U/'code/e2e32_runtime.py')],model_receipt_SHA256=old['model_receipt_sha256'],state_band=tag(g),persistent_allocated_excluding_input=residual,CUDA_memory_reserved=torch.cuda.memory_reserved(),CUDA_memory_allocated=torch.cuda.memory_allocated(),gc_events_seen=len(gc_events),generation=g)
   previous_end=end;records.append(r);t0=time.perf_counter_ns();encoded=json.dumps(r,ensure_ascii=False,separators=(',',':'));serialization_ms=(time.perf_counter_ns()-t0)/1e6;t0=time.perf_counter_ns();out.write(encoded+'\n');out.flush();os.fsync(out.fileno());write_ms=(time.perf_counter_ns()-t0)/1e6
   ov=dict(request=count,phase=phase,condition=condition,profiled=profiled,H2D_enqueue_ms=h2d_ms,seed_ms=seed_ms,clean_before_ms=clean_before_ms,clean_after_ms=clean_after_ms,JSON_serialization_ms=serialization_ms,local_JSONL_flush_fsync_ms=write_ms,profile_start_ms=prof_start_overhead_ms,profile_stop_ms=profile_stop_ms,wrapper_generation_ms=(end-begin)/1e6,outer_without_intentional_idle_before_trace_export_ms=(time.perf_counter_ns()-outer0)/1e6)
   if profiled:
    trace=L/'profiler'/('request_%03d.trace.json'%count);trace.parent.mkdir(exist_ok=True);t0=time.perf_counter_ns();prof.export_chrome_trace(str(trace));ov['trace_export_ms']=(time.perf_counter_ns()-t0)/1e6
    profiler_records.append(dict(request=count,condition=condition,trace_file=str(trace.relative_to(L)),trace_bytes=trace.stat().st_size,profiled_E2E_ms=g['E2E_ms'],preceding_unprofiled_request=count-1,preceding_unprofiled_band=records[-2]['state_band'] if len(records)>1 else None,activities=[str(v) for v in acts]));del prof
    assert trace.stat().st_size<512*2**20,'Trace exceeds 512MiB diagnostic resource budget'
   overhead.append(ov);of.write(json.dumps(ov)+'\n');of.flush();state['completed']=len(records);state['last_E2E_ms']=g['E2E_ms'];state['last_TPOT_ms']=g['TPOT_ms'];progress()
   print('REQUEST',count,phase,condition,'E2E',round(g['E2E_ms'],2),'TPOT',round(g['TPOT_ms'],3),r['state_band'],'RUNQUEUE_MS',round(r['main_thread_runqueue_wait_delta_ms'],3),flush=True)
   return r
  question_positions=tuple(ref.read(ref.U/'question_positions.json')[key])
  for i in range(plan['A']['warmup']):request('A','fixed_warmup',i+1)
  for i in range(plan['A']['continuous_measured']):request('A','continuous_measured',i+1)
  for block in plan['B']['order']:
   state.update(phase='B',condition=block['condition'],explicit_idle_s=block['idle_before_each_request_s']);progress()
   for i in range(block['repeats']):request('B',block['condition'],i+1,block['idle_before_each_request_s'])
  for i in range(plan['C']['scout_profile_pairs']):
   request('C','unprofiled_scout',i+1)
   request('C','profiled_native32',i+1,profiled=True)
  state['status']='DIAGNOSTIC_REQUESTS_COMPLETED';assert count==84
 except BaseException as ex:
  state.update(status='BLOCKED_OR_SAFETY_STOP',failure=repr(ex),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
 finally:
  gc.callbacks.remove(gcc);save('automatic_GC_events.json',gc_events);save('profiler_capture_index.json',profiler_records);save('request_completion.json',dict(attempted=count,completed=len(records),planned=84,hard_limit=100,formal_requests=0,profiler_requests=sum(r['profiled'] for r in records),status=state['status']))
  state['completed']=len(records);state['finished_unix_ns']=now();progress();(L/'monitor_stop.flag').write_text('done')
  if monitor:
   try:monitor.wait(timeout=15)
   except subprocess.TimeoutExpired:monitor.terminate();monitor.wait(timeout=10)
  a=None;decoder=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  for f in ['out','of','attempts']:
   if f in locals():locals()[f].close()
  for path,h in old['sources'].items():assert ref.sha(path)==h,path
  # Persist only this new diagnostic; no writes into historical experiment/audit.
  for item in L.iterdir():
   if item.is_dir():shutil.copytree(item,D/item.name,dirs_exist_ok=True)
   elif item.name!='runner.log':shutil.copyfile(item,D/item.name)
  print('FINAL',json.dumps(state),flush=True)
 return 0 if state['status']=='DIAGNOSTIC_REQUESTS_COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
