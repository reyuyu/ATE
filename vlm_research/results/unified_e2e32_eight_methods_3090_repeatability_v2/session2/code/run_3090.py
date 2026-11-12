"""One model, one session; unchanged frozen generate(), fixed warmup and non-blocking performance warnings."""
import os,sys
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_MODULE_LOADING='LAZY')
sys.dont_write_bytecode=True
from prepare_3090 import *
from history_policy import compare_history
sys.path[:0]=[str(R),str(U/'code'),str(R/'vlm_research/external_baselines/fastv/compatibility'),str(R/'vlm_research/external_baselines/v2drop/compatibility'),str(R/'vlm_research/external_baselines/e2e32_compatibility')]
import torch,transformers,gc,sqlite3,statistics,time,traceback,fcntl
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from vlm_research.physical_handoff_fastpath.engine import Layout
from e2e32_runtime import generate
from e2e32_environment import static_environment
def mean(xs):return statistics.mean(xs)
def robust(xs):
 med=statistics.median(xs);return 1.4826*statistics.median(abs(v-med) for v in xs)/med
def thread():
 root=Path('/proc/self/task')/str(os.getpid());return dict(schedstat=[int(x) for x in (root/'schedstat').read_text().split()],thread_CPU_ns=time.thread_time_ns())
def main():
 plan=read(L/'protocol.json');lock=(L/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (L/'checkpoint.sqlite').exists(),'No automatic restart or repeated session'
 db=sqlite3.connect(L/'checkpoint.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY, category TEXT, key TEXT, method TEXT, payload TEXT, sha256 TEXT, UNIQUE(category,key,method,id))');db.execute('CREATE UNIQUE INDEX formal_unique ON records(key,method) WHERE category="formal"');db.commit()
 state=dict(status='RUNNING',phase='PRECHECK',pid=os.getpid(),session=SESSION,formal_completed=0,formal_expected=240,requests_completed=0,requests_attempted=0,model_loads=0)
 monitors=[];a=None;records=[];attempts=(L/'attempts.jsonl').open('x');start=time.time_ns();previous_end=None;warnings=[];drifts=[];smokes={};baseline=[]
 def status():save(L,'progress.json',dict(state,updated_ns=time.time_ns()))
 def verify_sources():
  for path,h in read(L/'source_manifest.json')['frozen_sources'].items():assert sha(path)==h,path
  for n,h in plan['entry_source_SHA256'].items():assert sha(L/'code'/n)==h,n
 def guard():
  if (L/'STOP.flag').exists():raise RuntimeError('Safety stop: '+(L/'STOP.flag').read_text())
  assert all(p.poll() is None for p in monitors),'Collector exited unexpectedly'
 def store(r):
  payload=json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)
  with db:db.execute('INSERT INTO records VALUES (?,?,?,?,?,?)',(r['request_id'],r['category'],r['key'],r['method'],payload,digest(r)))
  records.append(r);state['requests_completed']=len(records);state['formal_completed']=sum(r['category']=='formal' for r in records);status()
 try:
  verify_sources();assert sha(M/'.ate_revision.json')==plan['model_receipt_SHA256'];assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU conflict'
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==UUID
  for mode in ['gpu','cpu','processes']:
   with (L/(mode+'_monitor.log')).open('x') as log:monitors.append(subprocess.Popen([sys.executable,'-B',str(L/'code/monitor_3090.py'),str(L),str(os.getpid()),UUID,mode],stdout=log,stderr=subprocess.STDOUT))
  state['phase']='MODEL_LOAD';state['model_loads']=1;status();loader.MODELS=dict(loader.MODELS,llava=M);a=frozen.load('B')
  assert {str(v.device) for v in a.model.parameters()}=={'cuda:0'} and {v.dtype for v in a.model.parameters()}=={torch.float16}
  assert not a.model.training and a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager'
  assert a.tokenizer.eos_token_id==2 and a.model.generation_config.eos_token_id==2
  env=static_environment(a);env.update(actual_GPU_UUID=UUID,single_model_load=True,new_entry_SHA256=plan['entry_source_SHA256'],model_revision=plan['revision'],reference_commit=plan['reference_commit'],Profiler_used=False);save(L,'environment.json',env)
  rows=lines(L/'input_manifest.jsonl.gz');bykey={r['key']:r for r in rows};questions=read(L/'question_positions.json');mapping=read(L/'input_path_mapping.json');refs=read(L/'historical_token_references.json')
  history_records={r['key']+'|'+r['method']:r for r in lines(U/'formal_requests.jsonl.gz')};fullrefs={k:r['generation']['token_ids'] for k,r in history_records.items()};assert all(len(fullrefs[r['key']+'|Native'])==32 for r in rows);prepared={};checks=[]
  for s in rows:
   path=mapping[s['key']];assert sha(path)==s['image_sha256'] and hashlib.sha256(s['prompt'].encode()).hexdigest()==s['prompt_sha256']
   raw=dict(input_ids=torch.tensor([s['input_ids']],dtype=torch.long),attention_mask=torch.ones((1,len(s['input_ids'])),dtype=torch.long),pixel_values=frozen.paper_pixels(path,a.image_processor).half());hashes={k:tensor_hash(v) for k,v in raw.items()};assert hashes==s['input_hashes']
   prepared[s['key']]=raw;checks.append(dict(key=s['key'],image_path=path,sample_SHA256=s['sample_sha256'],image_SHA256=s['image_sha256'],prompt_SHA256=s['prompt_sha256'],input_hashes=hashes))
  save(L,'input_verification.json',dict(status='PASS',inputs=30,GQA=15,TextVQA=15,all_original_ID_order_and_input_hashes_exact=True,records=checks))
  decoder=a.model.language_model.model;top=a.model.forward.__func__;dec=decoder.forward.__func__;forwards=[l.forward.__func__ for l in a.layers];attention=[l.self_attn.forward.__func__ for l in a.layers];clip=[(l.forward.__func__,l.self_attn.forward.__func__) for l in a.model.vision_tower.vision_model.encoder.layers]
  def clean():
   assert a.model.forward.__func__ is top and decoder.forward.__func__ is dec
   assert all(l.forward.__func__ is f for l,f in zip(a.layers,forwards)) and all(l.self_attn.forward.__func__ is f for l,f in zip(a.layers,attention))
   assert all(not l._forward_hooks and not l._forward_pre_hooks for l in a.layers)
   assert all(not getattr(l.self_attn,op+'_proj')._forward_pre_hooks for l in a.layers for op in ['q','k','v'])
   assert all((l.forward.__func__,l.self_attn.forward.__func__)==f for l,f in zip(a.model.vision_tower.vision_model.encoder.layers,clip))
  def request(key,method,category,smoke=False,**extra):
   nonlocal previous_end
   guard();s=bykey[key];a.inputs={k:v.to('cuda:0') for k,v in prepared[key].items()};ids=s['input_ids'];image=tuple(i for i,v in enumerate(ids) if v==32000);text=tuple(i for i,v in enumerate(ids) if v!=32000);relay=tuple(range(image[-1]+1,len(ids)-1));q=tuple(questions[key]) if s['benchmark']=='GQA' else relay;layout=Layout(len(ids),image,text,relay,q,20)
   torch.manual_seed(0);clean();state['requests_attempted']+=1;rid=state['requests_attempted'];attempts.write(json.dumps(dict(request_id=rid,category=category,key=key,method=method,at_ns=time.time_ns(),**extra))+'\n');attempts.flush();os.fsync(attempts.fileno())
   cpu0=thread();begin=time.time_ns();g=generate(a,layout,method,smoke=smoke);end=time.time_ns();cpu1=thread();clean()
   assert g['all_logits_finite'] and g['output_tokens']==32 and g['real_decode_calls']==31 and g['vision_forward_calls']==g['prefill_calls']==1
   assert g['cache']['live_bytes']==g['cache']['storage_bytes'] and len(g['cache']['layers'])==32 and g['cache']['cached_decode_inputs']==31
   assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6 and abs(mean(g['step_event_ms'][1:])-g['CUDA_Event_Decode_TPOT_ms'])<1e-6
   ref=fullrefs.get(key+'|'+method)
   exact=None if ref is None else g['token_ids']==ref
   early=statistics.median(g['step_event_ms'][1:6]);late=statistics.median(g['step_event_ms'][6:]);jump=late/early-1
   r=dict(request_id=rid,category=category,key=key,benchmark=s['benchmark'],method=method,session=SESSION,GPU_UUID=UUID,seed=0,input_tokens=len(ids),sample_sha256=s['sample_sha256'],input_hashes=s['input_hashes'],begin_ns=begin,end_ns=end,gap_s=None if previous_end is None else (begin-previous_end)/1e9,main_thread_runqueue_ms=(cpu1['schedstat'][1]-cpu0['schedstat'][1])/1e6,main_thread_running_ms=(cpu1['schedstat'][0]-cpu0['schedstat'][0])/1e6,early5_vs_late26_decode_median_change=jump,historical_tokens_exact=exact,CUDA_allocated=torch.cuda.memory_allocated(),CUDA_reserved=torch.cuda.memory_reserved(),generation=g,**extra)
   previous_end=end
   policy=compare_history(method,g['token_ids'],ref,2)
   r['history_acceptance']=policy
   r['output_token_ids_SHA256']=digest(g['token_ids'])
   r['historical_output_token_ids_SHA256']=None if ref is None else digest(ref)
   r['performance_warnings']=[]
   if abs(jump)>.05:
    warning=dict(request_id=rid,key=key,method=method,category=category,type='early5_vs_late26_decode_change_over5pct',relative_change=jump);warnings.append(warning);r['performance_warnings'].append(warning)
   if method=='FastV':
    h=history_records[key+'|FastV']['generation'];fields=['layer','used_tokens','capacity_tokens','visual_tokens','live_bytes','storage_bytes','K_shape','V_shape','dtype']
    r['FastV_physical_history_evidence']=dict(KV_layout_equal=all({k:x.get(k) for k in fields}=={k:y.get(k) for k in fields} for x,y in zip(g['cache']['layers'],h['cache']['layers'])),KV_live_bytes_current=g['cache']['live_bytes'],KV_live_bytes_historical=h['cache']['live_bytes'],pruning_events_current=g.get('pruning_events'),pruning_events_historical=h.get('pruning_events'),current_E2E_ms=g['E2E_ms'],historical_E2E_ms=h['E2E_ms'])
   store(r)
   assert policy['accepted'],'Historical output policy failure: '+key+' '+method+' '+json.dumps(policy)
   if method=='FastV':
    assert r['FastV_physical_history_evidence']['KV_layout_equal'] and g.get('pruning_events')==h.get('pruning_events'),'FastV physical layout/pruning differs from frozen history'
   return r
  state['phase']='FIXED_NATIVE_WARMUP';status();warm=[]
  for i in range(40):warm.append(request(plan['smoke_keys'][i%2],'Native','native_warmup',False,warmup_index=i+1))
  state['phase']='FIXED_OTHER_METHOD_WARMUP';status()
  others=[m for m in plan['requested_methods'] if m!='Native']
  for i,key in enumerate(plan['smoke_keys']):
   for method in others[i:]+others[:i]:request(key,method,'other_warmup',False)
  warmstats={}
  for key in plan['smoke_keys']:
   sub=[r for r in warm if r['key']==key];warmstats[key]={}
   for metric in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']:
    v=[r['generation'][metric] for r in sub];med=statistics.median(v);s=dict(n=len(v),mean=mean(v),median=med,stdev=statistics.stdev(v),robust_CV=robust(v),max_abs_deviation=max(abs(x/med-1) for x in v));warmstats[key][metric]=s
    if s['robust_CV']>.01 or s['max_abs_deviation']>.03:warnings.append(dict(type='fixed_warmup_dispersion',key=key,metric=metric,statistics=s))
  save(L,'warmup_summary.json',dict(status='WARN' if warnings else 'RECORDED',Native_requests=40,other_method_requests_each=2,total_warmup=54,hard_performance_admission=False,statistics=warmstats,warnings=list(warnings)))
  print('FIXED_WARMUP_DONE Native40 other14 performance_warnings',len(warnings),flush=True)
  state['phase']='CORRECTNESS_SMOKE';status()
  for method in plan['requested_methods']:
   for key in plan['smoke_keys']:
    r=request(key,method,'smoke',True);smokes[key+'|'+method]=r;print('SMOKE_PASS',method,key,flush=True)
  for key in plan['smoke_keys']:
   r=request(key,'BridgeNative','smoke',True);base=smokes[key+'|Native']['generation'];assert r['generation']['token_ids']==base['token_ids'] and r['generation']['first_logits']==base['first_logits'];print('BRIDGE_EXACT',key,flush=True)
  for method in ['FastVNoop','V2DropNoop']:
   key=plan['smoke_keys'][0];r=request(key,method,'smoke',True);base=smokes[key+'|Native']['generation'];assert r['generation']['token_ids']==base['token_ids'] and r['generation']['first_logits']==base['first_logits'];print('NOOP_EXACT',method,flush=True)
  extra_smoke=request(plan['extra_policy_smoke_key'],'FastV','policy_smoke',True)
  save(L,'policy_smoke_validation.json',dict(status='PASS',key=extra_smoke['key'],history_acceptance=extra_smoke['history_acceptance'],physical_evidence=extra_smoke['FastV_physical_history_evidence'],frozen_runtime_unchanged=True,timed_call_unchanged=True,request_id=extra_smoke['request_id']))
  native_policy_smoke=request(plan['extra_Native_policy_smoke_key'],'Native','policy_smoke',True)
  save(L,'Native_policy_smoke_validation.json',dict(status='PASS',key=native_policy_smoke['key'],history_acceptance=native_policy_smoke['history_acceptance'],same_GPU_Noops_still_strict=True,request_id=native_policy_smoke['request_id'],frozen_runtime_unchanged=True,timed_call_unchanged=True))
  save(L,'smoke_summary.json',dict(status='PASS',methods=plan['requested_methods'],requests=22,BridgeNative_exact=True,FastV_disabled_exact=True,V2Drop_disabled_exact=True,all_shapes_RoPE_KV_real_decode_finite=True,SVH_attention_skip_layers=[2,3,4,5],observer_steps=[0,1,2,3,4,5,6,7,8,31]))
  state['phase']='INDEPENDENT_NATIVE_BASELINE';status()
  for i in range(20):baseline.append(request(plan['smoke_keys'][i%2],'Native','baseline',False,baseline_index=i+1))
  metrics=plan['drift']['metrics'];base={k:{m:mean(r['generation'][m] for r in baseline if r['key']==k) for m in metrics} for k in plan['smoke_keys']};pooled={m:mean(r['generation'][m] for r in baseline) for m in metrics}
  save(L,'drift_baseline.json',dict(requests=20,per_input=base,pooled=pooled,independent_of_gate=True));consecutive=0
  state['phase']='FORMAL_INTERLEAVED_240';status()
  for pos,s in enumerate(rows):
   methods=plan['requested_methods'];order=methods[pos%8:]+methods[:pos%8]
   for oi,method in enumerate(order):request(s['key'],method,'formal',False,formal_index=state['formal_completed']+1,input_index=pos+1,order_in_input=oi)
   print('INPUT_DONE',pos+1,s['key'],state['formal_completed'],flush=True)
   if (pos+1)%5==0:
    probes=[]
    for repeat in range(3):
     for key in plan['smoke_keys']:probes.append(request(key,'Native','calibration',False,after_inputs=pos+1,repeat=repeat+1))
    measured={k:{m:mean(r['generation'][m] for r in probes if r['key']==k) for m in metrics} for k in plan['smoke_keys']};changes={k:{m:measured[k][m]/base[k][m]-1 for m in metrics} for k in plan['smoke_keys']};pooled_change={m:mean(r['generation'][m] for r in probes)/pooled[m]-1 for m in metrics};breach=any(abs(v)>.02 for v in pooled_change.values()) or any(abs(v)>.02 for d in changes.values() for v in d.values());consecutive=consecutive+1 if breach else 0
    check=dict(after_inputs=pos+1,requests=6,baseline_per_input=base,measured_per_input=measured,relative_change=changes,pooled_relative_change=pooled_change,threshold=.02,breach=breach,consecutive_breaches=consecutive,status='WARN' if breach else 'WITHIN_2PCT');drifts.append(check);save(L,'drift_summary.json',dict(status='RUNNING',checks=drifts));print('DRIFT',json.dumps(check),flush=True)
    if breach:warnings.append(dict(type='calibration_drift_over2pct',after_inputs=pos+1,relative_change=changes,pooled_relative_change=pooled_change))
  assert state['formal_completed']==240 and len(drifts)==6;verify_sources();clean();state.update(status='COMPLETE_WITH_PERFORMANCE_WARNINGS' if warnings else 'COMPLETE_NO_PERFORMANCE_WARNINGS',correctness='PASS',phase='COMPLETE',finished_ns=time.time_ns());save(L,'drift_summary.json',dict(status='WARN' if any(c['breach'] for c in drifts) else 'WITHIN_2PCT',checks=drifts))
 except BaseException as ex:
  state.update(status='PARTIAL' if state['formal_completed'] else 'BLOCKED',failure=repr(ex),traceback=traceback.format_exc(),failed_phase=state['phase']);print(traceback.format_exc(),flush=True)
  if not (L/'smoke_summary.json').exists():save(L,'smoke_summary.json',dict(status='FAILED_OR_NOT_COMPLETED',failure=repr(ex)))
  if not (L/'warmup_summary.json').exists():save(L,'warmup_summary.json',dict(status='NOT_COMPLETED',failure=repr(ex)))
 finally:
  save(L,'performance_warnings.json',dict(status='WARN' if warnings else 'NO_WARNINGS_RECORDED',warnings=warnings,hard_stop=False))
  status();attempts.close();save(L,'completion.json',dict(state,category_counts={c:sum(r['category']==c for r in records) for c in sorted({r['category'] for r in records})},elapsed_s=(time.time_ns()-start)/1e9));(L/'monitor.stop').write_text('done')
  for p in monitors:
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
  a=None;decoder=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
  for item in L.iterdir():
   if item.name in ['runner.log','run.lock']:continue
   if item.is_dir():shutil.copytree(item,D/item.name,dirs_exist_ok=True)
   else:shutil.copyfile(item,D/item.name)
  print('FINAL',json.dumps(state),flush=True)
 return 0 if state['status'].startswith('COMPLETE_') else 1
if __name__=='__main__':raise SystemExit(main())
