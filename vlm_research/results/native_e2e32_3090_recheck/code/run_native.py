"""Minimal Native-only entry; the complete timed loop is imported unchanged."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import sys,time,gc,sqlite3,fcntl,threading,statistics,traceback,subprocess,argparse
from datetime import datetime,timezone
from prepare_native import *
import torch,transformers
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.mme_seed_quality.engine import identity_attention
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from vlm_research.physical_handoff_fastpath.engine import Layout
from e2e32_runtime import generate
from e2e32_environment import static_environment
from contextlib import nullcontext
def now():return datetime.now(timezone.utc).isoformat()
def cmd(*args):return subprocess.check_output(args,text=True,timeout=20).strip()
def session(index):
 lock=(O/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=read(O/'protocol.json');assert p['GPU_UUID']==UUID and p['reference_commit']==COMMIT
 for path,h in p['sources'].items():assert sha(path)==h,path
 for name,h in p['entry_sources_SHA256'].items():assert sha(O/'code'/name)==h,name
 assert sha(U/'input_manifest.jsonl.gz')==p['input_manifest_SHA256'] and sha(M/'.ate_revision.json')==p['model_receipt_sha256']
 mapping=read(O/'input_path_mapping.json');refs=read(O/'historical_native_tokens.json');assert digest(mapping)==p['input_path_mapping_SHA256'] and digest(refs)==p['historical_native_token_SHA256']
 rows=lines(U/'input_manifest.jsonl.gz');assert [s['key'] for s in rows]==p['keys'];bykey={s['key']:s for s in rows};questions=read(U/'question_positions.json')
 seg=O/'sessions'/('session_%02d'%index);seg.mkdir(parents=True,exist_ok=True)
 if (seg/'progress.json').exists():raise RuntimeError('Refuse silently repeating a started session; inspect its preserved state')
 db=sqlite3.connect(O/'checkpoint.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 for table in ['formal','smoke','warmup','probes','failures']:db.execute('CREATE TABLE IF NOT EXISTS '+table+' (key TEXT PRIMARY KEY,payload TEXT,sha256 TEXT)')
 db.commit()
 def put(table,key,r):
  with db:db.execute('INSERT INTO '+table+' VALUES(?,?,?)',(str(index)+'|'+key,json.dumps(r,ensure_ascii=False,separators=(',',':')),digest(r)))
 state=dict(status='RUNNING',session=index,pid=os.getpid(),phase='PRECHECK',created_at=now(),formal_completed=0,formal_expected=30,GPU_UUID=UUID)
 def status():state['updated_at']=now();save(str(seg.relative_to(O)/'progress.json'),state);save('progress.json',state)
 a=None;stop=threading.Event();thread=None
 def monitor():
  fields='timestamp,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_thermal_slowdown'
  with (seg/'telemetry.jsonl').open('a') as f:
   while not stop.is_set():
    try:
     gpu=cmd('nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits');processes=cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader');others=[l for l in processes.splitlines() if l and l.split(',')[0].strip()!=str(os.getpid())]
     r=dict(at=now(),query_fields=fields,GPU=gpu,compute_processes=processes,other_processes=others,loadavg=Path('/proc/loadavg').read_text(),cpu_pressure=Path('/proc/pressure/cpu').read_text())
     if others:save(str(seg.relative_to(O)/'gpu_conflict.json'),r)
    except Exception as e:r=dict(at=now(),error=repr(e))
    f.write(json.dumps(r)+'\n');f.flush();stop.wait(2)
 try:
  status();assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'),'GPU occupied'
  assert cmd('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==UUID
  assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
  save(str(seg.relative_to(O)/'GPU_before.json'),dict(at=now(),query=cmd('nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version,power.limit','--format=csv,noheader')))
  thread=threading.Thread(target=monitor,daemon=True);thread.start();state['phase']='GPU_LOAD';status()
  loader.MODELS=dict(loader.MODELS,llava=M);a=frozen.load('B')
  assert {str(v.device) for v in a.model.parameters()}=={'cuda:0'} and {v.dtype for v in a.model.parameters()}=={torch.float16}
  assert a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager' and not a.model.training
  env=static_environment(a);env.update(actual_GPU_UUID=UUID,session=index,new_entry_sources=p['entry_sources_SHA256'],protocol_SHA256=digest(p),model_receipt_SHA256=p['model_receipt_sha256'],single_model_load_this_session=True)
  save(str(seg.relative_to(O)/'environment.json'),env)
  forward=[l.forward.__func__ for l in a.layers];att=[l.self_attn.forward.__func__ for l in a.layers];decoder=a.model.language_model.model;dec=decoder.forward.__func__;top=a.model.forward.__func__
  def clean():
   assert decoder.forward.__func__ is dec and a.model.forward.__func__ is top
   assert all(l.forward.__func__ is f for l,f in zip(a.layers,forward)) and all(l.self_attn.forward.__func__ is f for l,f in zip(a.layers,att))
   assert all(not l._forward_hooks and not l._forward_pre_hooks for l in a.layers)
   assert all(not getattr(l.self_attn,k+'_proj')._forward_pre_hooks for l in a.layers for k in ['q','k','v'])
  prepared={}
  for s in rows:
   raw=dict(input_ids=torch.tensor([s['input_ids']],dtype=torch.long),attention_mask=torch.ones((1,len(s['input_ids'])),dtype=torch.long),pixel_values=frozen.paper_pixels(mapping[s['key']],a.image_processor).half())
   assert {k:tensor_hash(v) for k,v in raw.items()}==s['input_hashes'];prepared[s['key']]=raw
  ceiling=None
  def request(s,role,smoke=False,noop=False):
   assert not (seg/'gpu_conflict.json').exists(),'Other GPU process detected'
   assert cmd('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==UUID
   a.inputs={k:v.to('cuda:0') for k,v in prepared[s['key']].items()};ids=s['input_ids'];image=tuple(i for i,v in enumerate(ids) if v==32000);text=tuple(i for i,v in enumerate(ids) if v!=32000);relay=tuple(range(image[-1]+1,len(ids)-1));q=tuple(questions[s['key']]) if s['benchmark']=='GQA' else relay;layout=Layout(len(ids),image,text,relay,q,20)
   torch.manual_seed(0);clean();begin=time.time()
   with identity_attention(a) if noop else nullcontext():g=generate(a,layout,'Native',smoke=smoke)
   end=time.time();clean();assert g['token_ids']==refs[s['key']],'Historical Native 32-token mismatch '+s['key']
   assert g['output_tokens']==32 and g['real_decode_calls']==31 and g['prefill_calls']==g['vision_forward_calls']==1 and g['all_logits_finite']
   assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6 and g['cache']['cached_decode_inputs']==31
   residual=torch.cuda.memory_allocated()-sum(v.numel()*v.element_size() for v in a.inputs.values())
   if ceiling is not None:assert residual<=ceiling+2**20,('Persistent CUDA allocation growth',residual,ceiling)
   return dict(session=index,key=s['key'],benchmark=s['benchmark'],method='Native Full',role=role,GPU_UUID=UUID,at=now(),begin_unix_time=begin,end_unix_time=end,protocol_SHA256=digest(p),sample_SHA256=s['sample_sha256'],input_hashes=s['input_hashes'],model_receipt_SHA256=p['model_receipt_sha256'],runtime_SHA256=p['sources'][str(U/'code/e2e32_runtime.py')],entry_SHA256=p['entry_sources_SHA256'],historical_32_tokens_exact=True,persistent_allocated_excluding_input=residual,generation=g)
  state['phase']='NATIVE_AND_IDENTITY_NOOP_SMOKE';status();smokes=[]
  for key in p['probe_keys']:
   original=request(bykey[key],'native_smoke',True);put('smoke','native|'+key,original)
   noop=request(bykey[key],'identity_noop_smoke',True,True);put('smoke','identity_noop|'+key,noop)
   assert original['generation']['first_logits']==noop['generation']['first_logits'] and original['generation']['token_ids']==noop['generation']['token_ids'];smokes.append(dict(key=key,status='PASS',Native_identity_logits_and_32tokens_exact=True))
  allocation=[]
  for cycle in range(2):
   values=[]
   for key in p['probe_keys']:
    r=request(bykey[key],'allocation_stability');put('smoke','allocation|'+str(cycle)+'|'+key,r);values.append(r['persistent_allocated_excluding_input'])
   allocation.append(values)
  assert all(b<=a+2**20 for a,b in zip(*allocation));ceiling=max(allocation[-1])
  save(str(seg.relative_to(O)/'smoke_validation.json'),dict(status='PASS',representatives=smokes,full_32_tokens=True,prefill=1,real_decode=31,physical_hidden_mask_RoPE_KV_32_layers_checked=True,no_nonfinite=True,allocation_cycles=allocation,allocation_ceiling=ceiling,other_algorithms_run=False))
  state['phase']='WARMUP';status();begin=time.monotonic();cycle=0
  while cycle==0 or time.monotonic()-begin<p['warmup']['minimum_seconds']:
   for key in p['probe_keys']:
    r=request(bykey[key],'warmup');put('warmup',str(cycle)+'|'+key,r)
   cycle+=1
   if cycle%5==0:state['warmup_cycles']=cycle;status();print('WARMUP',index,cycle,round(time.monotonic()-begin),flush=True)
  state['phase']='BASELINE';status();baseline=[]
  for repeat in range(p['drift']['baseline_rounds']):
   for key in p['probe_keys']:
    r=request(bykey[key],'baseline');put('probes','baseline|'+str(repeat)+'|'+key,r);baseline.append(r)
  metrics=p['drift']['metrics'];means={m:statistics.mean(r['generation'][m] for r in baseline) for m in metrics}
  save(str(seg.relative_to(O)/'drift_baseline.json'),dict(status='PASS',at=now(),means=means,requests=len(baseline),keys=p['probe_keys'],warmup_cycles=cycle,warmup_seconds=time.monotonic()-begin))
  state['phase']='FORMAL';status();streak={m:0 for m in metrics};checks=[]
  for position,s in enumerate(rows,1):
   r=request(s,'formal');r['formal_index']=position;put('formal',s['key'],r);state['formal_completed']=position;status();print('FORMAL',index,position,s['key'],round(r['generation']['E2E_ms'],2),flush=True)
   if position%5==0:
    probes=[]
    for key in p['probe_keys']:
     r=request(bykey[key],'calibration');put('probes','after_'+str(position)+'|'+key,r);probes.append(r)
    measured={m:statistics.mean(r['generation'][m] for r in probes) for m in metrics};change={m:measured[m]/means[m]-1 for m in metrics}
    for m in metrics:streak[m]=streak[m]+1 if abs(change[m])>p['drift']['threshold'] else 0
    paused=any(v>=p['drift']['consecutive_windows'] for v in streak.values());check=dict(after_inputs=position,baseline=means,measured=measured,relative_change=change,consecutive_over_2pct=dict(streak),status='PAUSED_DRIFT' if paused else 'PASS');checks.append(check)
    save(str(seg.relative_to(O)/'drift_summary.json'),dict(status=check['status'],checks=checks));print('DRIFT',index,json.dumps(check),flush=True)
    if paused:state['status']='PAUSED_DRIFT';raise RuntimeError('Native calibration >2% in two consecutive windows; preserve session and pause remaining sessions')
  for path,h in p['sources'].items():assert sha(path)==h,path
  state.update(status='PASS',phase='COMPLETE',finished_at=now())
 except BaseException as e:
  if state['status']!='PAUSED_DRIFT':state['status']='BLOCKED'
  state.update(failure=repr(e),traceback=traceback.format_exc());put('failures','fatal',dict(at=now(),failure=repr(e),traceback=traceback.format_exc()));print(state['traceback'],flush=True)
 finally:
  stop.set()
  if thread:thread.join(timeout=25)
  a=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  state['finished_at']=now();status();db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();print('FINAL',json.dumps(state),flush=True)
 return 0 if state['status']=='PASS' else 1
def all_sessions():
 for index in range(1,4):
  seg=O/'sessions'/('session_%02d'%index);seg.mkdir(parents=True,exist_ok=True)
  if (seg/'progress.json').exists():
   assert read(seg/'progress.json')['status']=='PASS','Cannot automatically rerun or rebaseline an incomplete session'
   continue
  with (seg/'runner.log').open('x') as f:r=subprocess.run([sys.executable,__file__,'--session',str(index)],stdout=f,stderr=subprocess.STDOUT)
  (seg/'runner.exit').write_text(str(r.returncode)+'\n')
  assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'),'Compute process remains after session exit'
  if r.returncode:return r.returncode
 return 0
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--session',type=int,choices=[1,2,3]);args=parser.parse_args();raise SystemExit(session(args.session) if args.session else all_sessions())
