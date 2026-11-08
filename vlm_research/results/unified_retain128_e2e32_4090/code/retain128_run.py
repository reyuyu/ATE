"""Unified single-GPU 32-token efficiency, smoke gates, immutable checkpoints."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import sys,json,gzip,sqlite3,subprocess,time,traceback,gc,fcntl,statistics,argparse
from datetime import datetime,timezone
from retain128_prepare import *
sys.path[:0]=[str(R),str(COMP),str(VCOMP),str(NEWCOMP),str(OLDNEWCOMP)]
import torch,transformers
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from vlm_research.physical_handoff_fastpath.engine import Layout
from retain128_runtime import generate
from e2e32_environment import static_environment
def now():return datetime.now(timezone.utc).isoformat()
def cmd(*args):return subprocess.check_output(args,text=True,timeout=20).strip()
def write_db(db,table,key,record):
 payload=json.dumps(record,ensure_ascii=False,separators=(',',':'))
 with db:db.execute('INSERT INTO '+table+' VALUES (?,?,?)',(key,payload,digest(record)))
def load_db(db,table):
 result={}
 for key,payload,h in db.execute('SELECT key,payload,sha256 FROM '+table):
  r=json.loads(payload);assert digest(r)==h;result[key]=r
 return result
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows,p,questions,refs=prepare(freeze=not args.prepare_only);bykey={r['key']:r for r in rows}
 db=sqlite3.connect(OUT/('preparation.sqlite' if args.prepare_only else 'checkpoint.sqlite'),timeout=30);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 for table in ('requests','smoke','probes'):db.execute('CREATE TABLE IF NOT EXISTS '+table+' (key TEXT PRIMARY KEY,payload TEXT,sha256 TEXT)')
 db.commit();records=load_db(db,'requests')
 sessions=OUT/('preparation_attempts' if args.prepare_only else 'sessions');sessions.mkdir(exist_ok=True);index=len(list(sessions.glob('session_*')))+1;segment=sessions/('session_%03d'%index);segment.mkdir()
 state=dict(status='RUNNING',phase='GPU_LOAD',at=now(),pid=os.getpid(),session=index,prepare_only=args.prepare_only,formal_completed=len(records),formal_expected=150,model_loads_this_process=1)
 def status():save(str(segment.relative_to(OUT)/'progress.json'),state);save('preparation_progress.json' if args.prepare_only else 'progress.json',state)
 a=monitor=None;failed_method=None;method_status=dict(BLOCKED)
 try:
  assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'),'GPU occupied'
  assert cmd('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==UUID
  assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
  state['GPU_before']=cmd('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,power.limit','--format=csv,noheader');status()
  with (segment/'monitor.log').open('w') as f:monitor=subprocess.Popen([sys.executable,str(OUT/'code/retain128_monitor.py'),str(segment),str(os.getpid())],stdout=f,stderr=subprocess.STDOUT)
  loader.MODELS=dict(loader.MODELS,llava=M);a=frozen.load('B')
  assert {str(v.device) for v in a.model.parameters()}=={'cuda:0'} and {v.dtype for v in a.model.parameters()}=={torch.float16}
  assert a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager'
  e=static_environment(a);e.update(GPU_UUID=UUID,phase='preparation' if args.prepare_only else 'formal',source_manifest=p['source_SHA256'],retain128_protocol_SHA256=sha(OUT/'protocol.json'),imports={m:dict(path=__import__(m).__file__,sha256=sha(__import__(m).__file__)) for m in ['retain128_runtime','retain128_adapter','e2e32_official_compat','hf_adapter']})
  save(str(segment.relative_to(OUT)/'environment.json'),e)
  if not args.prepare_only and not (OUT/'environment.json').exists():save('environment.json',e)
  save(str(segment.relative_to(OUT)/'protocol.json'),p)
  forwards=[l.forward.__func__ for l in a.layers];attention_forwards=[l.self_attn.forward.__func__ for l in a.layers];decoder=a.model.language_model.model;decoder_forward=decoder.forward.__func__;top_forward=a.model.forward.__func__
  clip_forwards=[(l.forward.__func__,l.self_attn.forward.__func__) for l in a.model.vision_tower.vision_model.encoder.layers]
  prepared={};bindings=[]
  for s in rows:
   raw={'input_ids':torch.tensor([s['input_ids']],dtype=torch.long),'attention_mask':torch.ones((1,len(s['input_ids'])),dtype=torch.long),'pixel_values':frozen.paper_pixels(s['image_path'],a.image_processor).half()}
   hashes={k:tensor_hash(v) for k,v in raw.items()};assert hashes==s['input_hashes'],'Input SHA mismatch: '+s['key']
   prepared[s['key']]=raw;bindings.append(dict(key=s['key'],sample_sha256=s['sample_sha256'],image_sha256=s['image_sha256'],prompt_sha256=hashlib.sha256(s['prompt'].encode()).hexdigest(),input_hashes=hashes))
  save(str(segment.relative_to(OUT)/'input_verification.json'),dict(status='PASS',inputs=30,records=bindings))
  if not args.prepare_only:save('input_verification.json',dict(status='PASS',inputs=30,records=bindings))
  def inputs(s):
   a.inputs={k:v.to('cuda:0') for k,v in prepared[s['key']].items()};image=tuple(i for i,v in enumerate(s['input_ids']) if v==32000);text=tuple(i for i,v in enumerate(s['input_ids']) if v!=32000);relay=tuple(range(image[-1]+1,len(s['input_ids'])-1));q=tuple(questions[s['key']]) if s['benchmark']=='GQA' else relay
   return Layout(len(s['input_ids']),image,text,relay,q,20)
  def clean():
   assert decoder.forward.__func__ is decoder_forward and a.model.forward.__func__ is top_forward
   assert all(l.forward.__func__ is f for l,f in zip(a.layers,forwards))
   assert all(l.self_attn.forward.__func__ is f for l,f in zip(a.layers,attention_forwards))
   assert all(not l._forward_hooks and not l._forward_pre_hooks for l in a.layers)
   assert all(not getattr(l.self_attn,op+'_proj')._forward_pre_hooks for l in a.layers for op in ('q','k','v'))
   assert all((l.forward.__func__,l.self_attn.forward.__func__)==f for l,f in zip(a.model.vision_tower.vision_model.encoder.layers,clip_forwards))
  def request(s,method,smoke):
   if (segment/'gpu_conflict.json').exists():raise RuntimeError('Other GPU compute process')
   assert cmd('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==UUID
   layout=inputs(s);torch.manual_seed(0);clean();begin=time.time();g=generate(a,layout,method,smoke=smoke);end=time.time();clean()
   assert g['all_logits_finite'] and g['output_tokens']==32 and g['real_decode_calls']==31
   return dict(key=s['key'],benchmark=s['benchmark'],method=method,sample_sha256=s['sample_sha256'],input_hashes=s['input_hashes'],input_tokens=layout.length,session=index,GPU_UUID=UUID,seed=0,begin_unix_time=begin,end_unix_time=end,generation=g)
  state['phase']='SMOKE_AND_EQUAL_WARMUP';status();smoke_results={};smoke_count=0
  # Existing five methods first. New candidates are admitted only after both inputs pass.
  for method in METHODS:
   failed_method=method
   try:
    for key in p['smoke_keys']:
     r=request(bykey[key],method,True);ids=r['generation']['token_ids'];ref=refs.get(key+'|'+method)
     if ref is not None:
      r['historical_first32_tokens_equal']=ids==ref;assert ids==ref,'Frozen method token mismatch'
     item='%03d|%s|%s'%(index,key,method);write_db(db,'smoke',item,r);smoke_results[key+'|'+method]=r;smoke_count+=1
     print('SMOKE_PASS',method,key,'KV_MiB',r['generation']['cache']['live_bytes']/2**20,flush=True)
    method_status[method]=dict(status='PASS',smoke_requests=2,actual_shapes_RoPE_KV_and_Decode=True)
   except Exception as exc:
    method_status[method]=dict(status='BLOCKED',failure=repr(exc),traceback=traceback.format_exc());print('SMOKE_BLOCKED',method,method_status[method]['traceback'],flush=True);clean();gc.collect();torch.cuda.empty_cache()
    if method=='Native':raise
   save(str(segment.relative_to(OUT)/'method_status.json'),method_status);state['smoke_completed']=smoke_count;status()
  # Common compact-position bridge must be an exact no-op on stock Native.
  if any(method_status.get(m,{}).get('status')=='PASS' for m in ('ApET','DUET')):
   for key in p['smoke_keys']:
    r=request(bykey[key],'BridgeNative',True);ref=smoke_results[key+'|Native']['generation']
    assert r['generation']['token_ids']==ref['token_ids'] and r['generation']['first_logits']==ref['first_logits'],'Native bridge not exact'
    r['Native_logits_and_32tokens_exact']=True;write_db(db,'smoke','%03d|%s|BridgeNative'%(index,key),r)
   print('BRIDGE_NATIVE_EXACT',flush=True)
  for noop in ('V2DropNoop',):
   r=request(bykey[p['smoke_keys'][0]],noop,True);ref=smoke_results[p['smoke_keys'][0]+'|Native']['generation']
   assert r['generation']['token_ids']==ref['token_ids'] and r['generation']['first_logits']==ref['first_logits'],'Noop not exact: '+noop
   write_db(db,'smoke','%03d|%s|%s'%(index,p['smoke_keys'][0],noop),r)
  available=[m for m in METHODS if method_status.get(m,{}).get('status')=='PASS'];assert available[0]=='Native'
  summary=dict(status='PASS' if len(available)==len(REQUESTED) else 'PARTIAL',available=available,blocked={m:v for m,v in method_status.items() if v['status']!='PASS'},smoke_requests=smoke_count,tokens_each=32,no_op_Native_exact=True,physical_cache_32_layers_checked=True,real_decode_steps_checked=[1,2,3,4,5,6,7,8,31],GPU_UUID=UUID)
  save(str(segment.relative_to(OUT)/'smoke_summary.json'),summary)
  state['available_methods']=available;state['formal_expected_audited']=len(available)*30
  if args.prepare_only:
   state.update(status='PREPARATION_COMPLETE',phase='COMPLETE',formal_completed=0);return
  save('method_status.json',method_status);save('smoke_summary.json',summary)
  paths=sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if '/' in l and any(k in l for k in ('libcuda','libcublas','libcudnn','libtorch','libc10','libnvrtc'))})
  save(str(segment.relative_to(OUT)/'loaded_CUDA_libraries.json'),dict(paths=paths))
  state['phase']='DRIFT_BASELINE';status();baseline=[]
  for repeat in range(p['drift']['baseline_rounds']):
   for key in p['smoke_keys']:
    r=request(bykey[key],'Native',False);write_db(db,'probes','%03d|baseline|%d|%s'%(index,repeat,key),r);baseline.append(r)
  metrics=p['drift']['metrics'];means={m:statistics.mean(r['generation'][m] for r in baseline) for m in metrics};save(str(segment.relative_to(OUT)/'drift_baseline.json'),dict(means=means,requests=len(baseline),keys=p['smoke_keys']))
  state['phase']='FORMAL_EFFICIENCY_ONLY';status()
  for position,s in enumerate(rows):
   order=available[position%len(available):]+available[:position%len(available)]
   for j,method in enumerate(order):
    item=s['key']+'|'+method
    if item in records:continue
    failed_method=method;r=request(s,method,False);r.update(formal_index=len(records)+1,order_in_input=j)
    if method=='Native':
     r['historical_first32_tokens_equal']=r['generation']['token_ids']==refs[item];assert r['historical_first32_tokens_equal']
    write_db(db,'requests',item,r);records[item]=r;state.update(formal_completed=len(records),updated_at=now());status()
   print('INPUT_DONE',position+1,s['key'],len(records),len(available)*30,flush=True)
   if (position+1)%p['drift']['after_each_inputs']==0:
    probes=[]
    for key in p['smoke_keys']:
     r=request(bykey[key],'Native',False);write_db(db,'probes','%03d|phase%d|%s'%(index,position+1,key),r);probes.append(r)
    measured={m:statistics.mean(r['generation'][m] for r in probes) for m in metrics};change={m:measured[m]/means[m]-1 for m in metrics}
    check=dict(phase_after_inputs=position+1,baseline=means,measured=measured,relative_change=change,threshold=.05,status='PASS' if all(abs(x)<=.05 for x in change.values()) else 'PAUSED_DRIFT',GPU_UUID=UUID)
    save(str(segment.relative_to(OUT)/('drift_after_%02d.json'%(position+1))),check);print('DRIFT',json.dumps(check),flush=True)
    if check['status']!='PASS':raise RuntimeError('Native >5% performance drift: '+json.dumps(check))
  assert len(records)==len(available)*30;verify_sources();clean();db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
  state.update(status='PASS' if len(available)==len(REQUESTED) else 'PARTIAL',phase='COMPLETE',finished_at=now(),quality_requests=0)
 except BaseException as exc:
  state.update(status='BLOCKED',failed_method=failed_method,failure=repr(exc),traceback=traceback.format_exc(),at=now());print(state['traceback'],flush=True)
 finally:
  (segment/'monitor.stop').write_text(now())
  if monitor:
   try:monitor.wait(timeout=15)
   except subprocess.TimeoutExpired:monitor.terminate()
  a=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  status();db.close();print('FINAL',json.dumps(state),flush=True)
 if state['status']=='BLOCKED':raise SystemExit(1)
if __name__=='__main__':main()
