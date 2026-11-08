"""One foreground GPU process; resumable quality and fixed-length KV/TPOT."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from seven_common import *
from datetime import datetime,timezone
from collections import Counter,OrderedDict
import fcntl,gc,sqlite3,subprocess,traceback,time
import torch,transformers
from fusion_engine import PrefixSVH,greedy
from run_fusion import trace_execution,structural
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.physical_handoff_fastpath.engine import Layout
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
def now():return datetime.now(timezone.utc).isoformat()
def cmd(*args):return subprocess.check_output(args,cwd=R,text=True).strip()

@torch.inference_mode()
def timed(a,layout,length):
 ctl=PrefixSVH(a,layout,length,True,False);n=layout.length;device=a.inputs['input_ids'].device
 positions=torch.arange(n+length-1,device=device);mask=torch.ones((1,n+length-1),dtype=torch.long,device=device)
 tokens=[];finite=[];events=[];cache=None
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 with ctl.context():
  for step in range(length):
   kw=dict(a.inputs) if step==0 else dict(input_ids=tokens[-1].reshape(1,1),attention_mask=mask[:,:n+step])
   kw.update(use_cache=True,return_dict=True,past_key_values=cache,cache_position=positions[:n] if step==0 else positions[n+step-1:n+step],logits_to_keep=1)
   begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
   out=a.model(**kw);last=out.logits[:,-1];tokens.append(last.argmax(-1));finite.append(torch.isfinite(last).all());end.record();events.append((begin,end));cache=out.past_key_values
   if step==0:torch.cuda.synchronize() # Same prefill boundary as archived runtime.
   del out,last
 torch.cuda.synchronize();assert torch.stack(finite).all().item(),'Nonfinite profiling logits'
 step_ms=[x.elapsed_time(y) for x,y in events];kv=ctl.cache.audit()
 assert kv['retained_image_layers']==[]
 for r in kv['layers']:
  expected=[] if r['layer'] in (2,3,4,5) else list(layout.text)+list(range(n,n+length-1))
  assert r['original_positions']==expected
 result=dict(output_tokens=length,decode_steps=length-1,fixed=True,EOS_ignored_for_profiling_only=True,all_logits_finite=True,
  token_ids=torch.cat(tokens).tolist(),step_gpu_ms=step_ms,decode_TPOT_ms=sum(step_ms[1:])/(length-1),
  KV_live_MiB=kv['live_bytes']/2**20,KV_storage_MiB=kv['storage_bytes']/2**20,visual_KV_residency_percent=0.0,
  cache=kv,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
  pruning_events=ctl.selection_events,counter_enabled=False,timing_method='CUDA events per model forward+greedy+finite check, decode steps only; frozen runtime.py convention')
 return result

def main():
 guard=(O/'run.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'failure.json').exists(),'Review a critical failure before any restart'
 audit=read(O/'input_audit.json');assert audit['status']=='PASS'
 for n,h in audit['frozen_sources'].items():assert sha(R/n)==h,n
 for n,h in audit['fusion_sources'].items():assert sha(F/n)==h,n
 assert cmd('git','rev-parse','HEAD')==audit['code_commit'] and not cmd('git','status','--porcelain','--untracked-files=no')
 scorer.text_evaluator();scorer.matcher()
 samples=gz(O/'input_manifest.jsonl.gz');by={s['key']:s for s in samples};assert len(samples)==len(by)==36256
 refs={r['key']:r for r in gz(O/'reference_bindings.jsonl.gz')}
 protocol=dict(condition='V2Drop-prefix+SVH',model=audit['model'],revision=audit['revision'],code_commit=audit['code_commit'],dtype='float16',attention='eager',batch_size=1,greedy=True,EOS=2,
  expected=audit['expected'],max_new_tokens=audit['max_new_tokens'],manifest_sha256=sha(O/'input_manifest.jsonl.gz'),input_audit_sha256=sha(O/'input_audit.json'),
  reused_predictions_sha256=sha(O/'reused_predictions.jsonl.gz'),reference_bindings_sha256=sha(O/'reference_bindings.jsonl.gz'),
  sources={n:sha(O/n) for n in ['prepare_seven.py','seven_common.py','run_seven.py']},fusion_sources=audit['fusion_sources'],frozen_sources=audit['frozen_sources'],
  physical_gate=audit['inherited_strict_physical_gate'],semantics=read(F/'protocol.json')['semantics'],
  efficiency=dict(keys=audit['efficiency_keys'],fixed_lengths=[32,128],primary_length=128,warmup_requests_per_length=2,condition_only='V2Drop-prefix+SVH',timing='frozen CUDA event per-step method',KV='actual used rows; excludes reserved capacity'),
  model_and_data_downloaded=False,no_other_condition_generated=True,no_GitHub_push=True)
 if (O/'protocol.json').exists():assert read(O/'protocol.json')==protocol
 else:save('protocol.json',protocol)
 binding=digest(protocol);db=sqlite3.connect(O/'predictions.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 db.execute('CREATE TABLE IF NOT EXISTS predictions(key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
 db.execute('CREATE TABLE IF NOT EXISTS timings(key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)');db.commit()
 existing={}
 for k,p,h in db.execute('SELECT key,payload,sha256 FROM predictions'):
  r=json.loads(p);assert digest(r)==h and r['binding']==binding and r['sample_sha256']==digest(by[k]);assert score(by[k],r['generation']['text'])==r['scoring'];existing[k]=r
 for source in gz(O/'reused_predictions.jsonl.gz'):
  k=source['key'];s=by[k]
  if k in existing:continue
  old=source['record'];r=dict(key=k,benchmark=s['benchmark'],question_id=s['question_id'],condition=protocol['condition'],binding=binding,sample_sha256=digest(s),image_sha256=s['image_sha256'],gold=s['answer'],generation=old['generation'],scoring=old['scoring'],pruning_events=old['pruning_events'],provenance='REUSED_VERIFIED_PREDICTION',source_directory=source['source_directory'],source_record_sha256=source['source_record_sha256'],source_protocol_sha256=source['source_protocol_sha256'],at=now())
  with db:db.execute('INSERT INTO predictions VALUES(?,?,?)',(k,json.dumps(r,ensure_ascii=False,separators=(',',':')),digest(r)))
  existing[k]=r
 timings={k:json.loads(p) for k,p,h in db.execute('SELECT key,payload,sha256 FROM timings') if digest(json.loads(p))==h}
 state=dict(status='RUNNING',phase='GPU_LOAD',completed=len(existing),expected=36256,reused=2574,new_generated=len(existing)-2574,counts=dict(Counter(r['benchmark'] for r in existing.values())),expected_counts=audit['expected'],timings_completed=len(timings),timings_expected=60,pid=os.getpid(),started=now(),at=now(),protocol_binding=binding)
 a=None;ctl=None;cases=read(O/'integration_smoke.json')['cases'] if (O/'integration_smoke.json').exists() else []
 try:
  assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'),'GPU occupied'
  assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
  assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090' and torch.cuda.mem_get_info()[0]>20*2**30
  state['GPU']=cmd('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.free','--format=csv,noheader');save('progress.json',state)
  loader.MODELS=dict(loader.MODELS,llava=M);print('SEVEN_MODEL_LOAD_ONCE',flush=True);a=frozen.load('B')
  assert {str(p.device) for p in a.model.parameters()}=={'cuda:0'} and {p.dtype for p in a.model.parameters()}=={torch.float16}
  assert a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager' and a.model.generation_config.eos_token_id==2
  pixels=OrderedDict()
  def prepare(s):
   key=s['image_sha256']
   if key not in pixels:
    p=frozen.paper_pixels(s['image_path'],a.image_processor);pixels[key]=(p,frozen.tensor_sha(p));
    if len(pixels)>32:pixels.popitem(last=False)
   p,p32=pixels[key];pixels.move_to_end(key)
   raw=frozen.image_ids(s['prompt'],a.tokenizer,-200);assert raw.count(-200)==1
   ids=[t for x in raw for t in ([32000]*576 if x==-200 else [x])]
   x=dict(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),pixel_values=p.half())
   hashes={k:tensor_hash(t) for k,t in x.items()}
   if s['benchmark']=='GQA':assert ids==s['input_ids'] and hashes==s['input_hashes'],'Official GQA tensor binding mismatch: '+s['key']
   else:
    ref=refs[s['key']];assert old_digest(s['prompt'])==ref['prompt_sha256'] and old_digest(ids)==ref['input_ids_sha256'] and p32==ref['pixel_values_sha256'] and len(ids)==ref['expanded_length'],'Paper tensor binding mismatch: '+s['key']
   a.inputs={k:t.to('cuda:0') for k,t in x.items()};image=tuple(i for i,t in enumerate(ids) if t==32000);text=tuple(i for i,t in enumerate(ids) if t!=32000);relay=tuple(range(image[-1]+1,len(ids)-1))
   assert len(image)==576
   return Layout(len(ids),image,text,relay,relay,20),dict(input_hashes=hashes,prompt_sha256=digest(s['prompt']),pixel_values_FP32_sha256=p32)
  # Newly encountered Paper tasks are audited before all missing quality rows.
  order=audit['integration_smoke_keys']+[s['key'] for s in samples if s['key'] not in audit['integration_smoke_keys']]
  state['phase']='QUALITY';save('progress.json',state)
  for key in order:
   if key in existing:continue
   assert not (O/'STOP_REQUESTED').exists(),'Explicit stop request'
   s=by[key];layout,inputs=prepare(s);is_smoke=key in audit['integration_smoke_keys'];ctl=PrefixSVH(a,layout,s['max_new_tokens'],True,is_smoke)
   if is_smoke:
    with trace_execution(a,layout) as trace:g,_=greedy(a,layout,ctl,s['max_new_tokens'])
    kv=structural(ctl,layout,trace,len(g['token_ids']));cases.append(dict(key=key,status='PASS',counted_once_in_quality=True,pruning_events=ctl.selection_events,layer_events=ctl.layer_events,visual_access=ctl.visual_events,KV=kv,execution_trace=trace))
    save('integration_smoke.json',dict(status='PASS' if len(cases)==len(audit['integration_smoke_keys']) else 'RUNNING',cases=cases,inherited_strict_numerical_gate='PASS unchanged'))
    print('INTEGRATION_SMOKE_PASS',key,flush=True)
   else:g,_=greedy(a,layout,ctl,s['max_new_tokens'])
   r=dict(key=key,benchmark=s['benchmark'],question_id=s['question_id'],condition=protocol['condition'],binding=binding,sample_sha256=digest(s),image_sha256=s['image_sha256'],gold=s['answer'],generation=g,scoring=score(s,g['text']),pruning_events=ctl.selection_events,input_binding=inputs,provenance='NEW_GPU_GENERATION',at=now())
   with db:db.execute('INSERT INTO predictions VALUES(?,?,?)',(key,json.dumps(r,ensure_ascii=False,separators=(',',':')),digest(r)))
   existing[key]=r;ctl=None;a.inputs={};state.update(completed=len(existing),new_generated=len(existing)-2574,counts=dict(Counter(x['benchmark'] for x in existing.values())),last_key=key,at=now());save('progress.json',state)
   if len(existing)%100==0:print('SEVEN_PROGRESS',len(existing),'/36256',state['counts'],flush=True)
  assert len(existing)==36256 and read(O/'integration_smoke.json')['status']=='PASS'
  state['phase']='FIXED_LENGTH_KV_TPOT';save('progress.json',state)
  for length in [32,128]:
   for key in audit['efficiency_keys'][:2]:layout,_=prepare(by[key]);timed(a,layout,length)
   for key in audit['efficiency_keys']:
    tk=key+'|'+str(length)
    if tk in timings:continue
    assert not (O/'STOP_REQUESTED').exists(),'Explicit stop request'
    layout,inputs=prepare(by[key]);r=dict(key=key,length=length,binding=binding,input_binding=inputs,result=timed(a,layout,length),at=now())
    with db:db.execute('INSERT INTO timings VALUES(?,?,?)',(tk,json.dumps(r,separators=(',',':')),digest(r)))
    timings[tk]=r;a.inputs={};state.update(timings_completed=len(timings),at=now());save('progress.json',state)
    if len(timings)%10==0:print('FUSION_KV_TPOT',len(timings),'/60',flush=True)
  assert len(timings)==60;state.update(status='COMPLETE',phase='GPU_WORK_DONE',at=now());save('progress.json',state)
 except BaseException as e:
  state.update(status='PARTIAL',error=repr(e),at=now());save('failure.json',dict(phase=state['phase'],error=repr(e),traceback=traceback.format_exc()[-12000:],at=now()));print('SEVEN_STOPPED',traceback.format_exc(),flush=True)
 finally:
  for name,rows in [('predictions.jsonl.gz',[existing[s['key']] for s in samples if s['key'] in existing]),('raw_timings.jsonl.gz',list(timings.values()))]:
   with gzip.GzipFile(filename=str(O/name),mode='wb',mtime=0) as out:
    for row in rows:out.write((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode())
  db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();a=None;ctl=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  state.update(process_exit_releases_CUDA_context=True,at=now());save('progress.json',state);print('SEVEN_FINAL',state['status'],len(existing),'/36256',len(timings),'/60',flush=True)
if __name__=='__main__':main()
