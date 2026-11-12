"""One model, one session; unchanged frozen generate(), bounded Native admission."""
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
 state=dict(status='RUNNING',phase='PRECHECK',pid=os.getpid(),session=2,formal_completed=0,formal_expected=240,requests_completed=0,requests_attempted=0,model_loads=0)
 monitors=[];a=None;records=[];attempts=(L/'attempts.jsonl').open('x');start=time.time_ns();previous_end=None;gate_results=[];drifts=[];smokes={};baseline=[]
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
   ref=fullrefs.get(key+'|'+method) if method in ['Native','FastV'] else refs.get(key+'|'+method) if smoke else None
   exact=None if ref is None else g['token_ids']==ref
   early=statistics.median(g['step_event_ms'][1:6]);late=statistics.median(g['step_event_ms'][6:]);jump=late/early-1
   r=dict(request_id=rid,category=category,key=key,benchmark=s['benchmark'],method=method,session=2,GPU_UUID=UUID,seed=0,input_tokens=len(ids),sample_sha256=s['sample_sha256'],input_hashes=s['input_hashes'],begin_ns=begin,end_ns=end,gap_s=None if previous_end is None else (begin-previous_end)/1e9,main_thread_runqueue_ms=(cpu1['schedstat'][1]-cpu0['schedstat'][1])/1e6,main_thread_running_ms=(cpu1['schedstat'][0]-cpu0['schedstat'][0])/1e6,early5_vs_late26_decode_median_change=jump,historical_tokens_exact=exact,CUDA_allocated=torch.cuda.memory_allocated(),CUDA_reserved=torch.cuda.memory_reserved(),generation=g,**extra)
   previous_end=end
   policy=compare_history(method,g['token_ids'],ref,2)
   r['history_acceptance']=policy
   if method=='FastV':
    h=history_records[key+'|FastV']['generation'];fields=['layer','used_tokens','capacity_tokens','visual_tokens','live_bytes','storage_bytes','K_shape','V_shape','dtype']
    r['FastV_physical_history_evidence']=dict(KV_layout_equal=all({k:x.get(k) for k in fields}=={k:y.get(k) for k in fields} for x,y in zip(g['cache']['layers'],h['cache']['layers'])),KV_live_bytes_current=g['cache']['live_bytes'],KV_live_bytes_historical=h['cache']['live_bytes'],pruning_events_current=g.get('pruning_events'),pruning_events_historical=h.get('pruning_events'),current_E2E_ms=g['E2E_ms'],historical_E2E_ms=h['E2E_ms'])
   store(r)
   assert policy['accepted'],'Historical output policy failure: '+key+' '+method+' '+json.dumps(policy)
   if method=='FastV':
    assert r['FastV_physical_history_evidence']['KV_layout_equal'] and g.get('pruning_events')==h.get('pruning_events'),'FastV physical layout/pruning differs from frozen history'
   if category in ['formal','calibration','baseline'] and method=='Native' and abs(jump)>.05:raise RuntimeError('Abrupt within-request Native decode transition: '+str(rid))
   return r
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
  save(L,'smoke_summary.json',dict(status='PASS',methods=plan['requested_methods'],requests=21,BridgeNative_exact=True,FastV_disabled_exact=True,V2Drop_disabled_exact=True,all_shapes_RoPE_KV_real_decode_finite=True,SVH_attention_skip_layers=[2,3,4,5],observer_steps=[0,1,2,3,4,5,6,7,8,31]))
  state['phase']='EQUAL_METHOD_WARMUP';status()
  for k,key in enumerate(plan['smoke_keys']):
   order=plan['requested_methods'][k:]+plan['requested_methods'][:k]
   for method in order:request(key,method,'equal_warmup',False)
  state['phase']='NATIVE_STABILITY_ADMISSION';status();warm=[];admitted=False
  def evaluate():
   n=len(warm);selected=warm[-60:];blocks=[selected[i:i+20] for i in [0,20,40]];windows=[];reasons=[];metrics=plan['native_stability']['metrics']
   for wi,block in enumerate(blocks):
    w=dict(first_warmup=block[0]['warmup_index'],last_warmup=block[-1]['warmup_index'],per_input={})
    for key in plan['smoke_keys']:
     sub=[r for r in block if r['key']==key];assert len(sub)==10;stats={}
     for metric in metrics:
      v=[r['generation'][metric] for r in sub];med=statistics.median(v);stats[metric]=dict(mean=mean(v),median=med,stdev=statistics.stdev(v),CV=statistics.stdev(v)/mean(v),robust_CV=robust(v),max_abs_deviation_from_median=max(abs(x/med-1) for x in v))
      if stats[metric]['robust_CV']>.01 or stats[metric]['max_abs_deviation_from_median']>.03:reasons.append(f'window{wi+1} {key} {metric} dispersion/single-request anomaly')
     w['per_input'][key]=stats
    if any(abs(r['early5_vs_late26_decode_median_change'])>.05 for r in block):reasons.append(f'window{wi+1} within-request Native state transition')
    if any(r['gap_s'] is not None and r['gap_s']>1 for r in block):reasons.append(f'window{wi+1} request gap>1s')
    samples=[json.loads(l) for l in (L/'gpu_telemetry.jsonl').read_text().splitlines(keepends=True) if l.endswith('\n') and l.strip()];samples=[r for r in samples if block[0]['begin_ns']<=r['at_ns']<=block[-1]['end_ns']]
    w['GPU_samples']=len(samples);w['GPU_max_sample_gap_s']=max((r.get('gap_s') or 0 for r in samples),default=999)
    if not samples or w['GPU_max_sample_gap_s']>1:reasons.append(f'window{wi+1} telemetry gap>1s')
    windows.append(w)
   for key in plan['smoke_keys']:
    for metric in metrics:
     means=[w['per_input'][key][metric]['mean'] for w in windows]
     if any(abs(b/a-1)>.01 for a,b in zip(means,means[1:])):reasons.append(key+' '+metric+' adjacent mean shift>1%')
     if (max(means)-min(means))/mean(means)>.02:reasons.append(key+' '+metric+' three-window mean range>2%')
   total=sum(r['generation']['E2E_ms'] for r in warm)/1000
   if total<180:reasons.append('Summed actual generation E2E<180s')
   result=dict(after_warmup=n,summed_generation_E2E_s=total,windows=windows,reasons=reasons,status='PASS' if not reasons else 'NOT_ADMITTED');gate_results.append(result)
   save(L,'native_stability_gate.json',dict(status='PASS' if not reasons else 'RUNNING' if n<120 else 'FAILED',warmup_requests=n,summed_generation_E2E_s=total,maximum_requests=120,evaluations=gate_results));print('ADMISSION',n,result['status'],'seconds',round(total,2),'reasons',json.dumps(reasons),flush=True)
   return not reasons
  for i in range(120):
   r=request(plan['smoke_keys'][i%2],'Native','stability_warmup',False,warmup_index=i+1);warm.append(r)
   if (i+1)%20==0:print('WARMUP',i+1,'summed_E2E_s',round(sum(r['generation']['E2E_ms'] for r in warm)/1000,2),flush=True)
   if i+1 in [60,80,100,120] and evaluate():admitted=True;break
  if not admitted:raise RuntimeError('Native stability admission failed within120 warmups; formal240 NOT STARTED')
  state['phase']='INDEPENDENT_NATIVE_BASELINE';status()
  for i in range(20):baseline.append(request(plan['smoke_keys'][i%2],'Native','baseline',False,baseline_index=i+1))
  metrics=plan['drift']['metrics'];base={k:{m:mean(r['generation'][m] for r in baseline if r['key']==k) for m in metrics} for k in plan['smoke_keys']};pooled={m:mean(r['generation'][m] for r in baseline) for m in metrics}
  save(L,'drift_baseline.json',dict(requests=20,per_input=base,pooled=pooled,independent_of_gate=True));consecutive=0
  # The independent baseline must still agree with the admitted final window.
  last=gate_results[-1]['windows'][-1]
  for key in plan['smoke_keys']:
   for metric in metrics:assert abs(base[key][metric]/last['per_input'][key][metric]['mean']-1)<=.02,'Independent baseline left admitted state'
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
    check=dict(after_inputs=pos+1,requests=6,baseline_per_input=base,measured_per_input=measured,relative_change=changes,pooled_relative_change=pooled_change,threshold=.02,breach=breach,consecutive_breaches=consecutive,status='BREACH_OBSERVED' if breach else 'PASS');drifts.append(check);save(L,'drift_summary.json',dict(status='RUNNING',checks=drifts));print('DRIFT',json.dumps(check),flush=True)
    if consecutive>=2 or any(abs(v)>.05 for d in changes.values() for v in d.values()):raise RuntimeError('Native calibration drift stop: '+str(pos+1))
  assert state['formal_completed']==240 and len(drifts)==6;verify_sources();clean();state.update(status='PASS',phase='COMPLETE',finished_ns=time.time_ns());save(L,'drift_summary.json',dict(status='PASS_WITH_FLAGGED_WINDOW' if any(c['breach'] for c in drifts) else 'PASS',checks=drifts))
 except BaseException as ex:
  state.update(status='PARTIAL' if state['formal_completed'] else 'BLOCKED',failure=repr(ex),traceback=traceback.format_exc(),failed_phase=state['phase']);print(traceback.format_exc(),flush=True)
  if not (L/'smoke_summary.json').exists():save(L,'smoke_summary.json',dict(status='FAILED_OR_NOT_COMPLETED',failure=repr(ex)))
  if not (L/'native_stability_gate.json').exists():save(L,'native_stability_gate.json',dict(status='NOT_REACHED',failure=repr(ex)))
 finally:
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
 return 0 if state['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
