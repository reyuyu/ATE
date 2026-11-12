import os,sys,json,gzip,hashlib,subprocess,sqlite3,time,statistics,traceback,gc,shutil,argparse
from pathlib import Path
W=Path(__file__).resolve().parents[4];O=Path(__file__).resolve().parent.parent;R=Path('/home/featurize/work/ATE');U=R/'vlm_research/results/unified_e2e32_4090';D=R/'vlm_research/results/rtx3090_unified_efficiency_18conditions'
UUID='GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6'
os.environ.update(CUDA_VISIBLE_DEVICES=UUID,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_MODULE_LOADING='LAZY')
sys.path[:0]=[str(O/'code'),str(W),str(W/'vlm_research'),str(U/'code'),str(R/'vlm_research/external_baselines/fastv/compatibility'),str(R/'vlm_research/external_baselines/v2drop/compatibility'),str(R/'vlm_research/external_baselines/e2e32_compatibility')]
import torch,transformers
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from vlm_research.physical_handoff_fastpath.engine import Layout
from unified_llava import generate,METHODS,NAMES
from e2e32_environment import static_environment
from contextlib import nullcontext

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def sync():
 D.mkdir(exist_ok=True)
 for p in O.rglob('*'):
  if p.is_file() and not p.name.endswith(('.sqlite','.sqlite-wal','.sqlite-shm','.lock')) and p.name!='runner.log':
   d=D/p.relative_to(O);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
def check_sources():
 for p,h in read(O/'source_manifest.json')['files'].items():assert sha(p)==h,('SOURCE_CHANGED',p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','smoke_sparse','session1','session2'],required=True);args=ap.parse_args();phase=args.phase
 L=O/phase;L.mkdir();state=dict(status='RUNNING',phase='PRECHECK',model='LLaVA',session=phase,pid=os.getpid(),formal_completed=0);warnings=[];records=[];monitors=[];a=None;db=None
 def progress():save(L/'progress.json',dict(state,updated_ns=time.time_ns()))
 progress()
 try:
  check_sources();assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==UUID
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU conflict'
  torch.set_num_threads(1);torch.set_num_interop_threads(6);os.sched_setaffinity(0,set(range(6)))
  for mode in ['gpu','cpu','processes']:
   log=(L/(mode+'_monitor.log')).open('x');monitors.append(subprocess.Popen([sys.executable,'-B',str(O/'code/monitor.py'),str(L),str(os.getpid()),UUID,mode],stdout=log,stderr=subprocess.STDOUT))
  state['phase']='MODEL_LOAD';progress();loader.MODELS=dict(loader.MODELS,llava=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf'));a=frozen.load('B')
  assert len(a.layers)==32 and not a.model.training and {p.dtype for p in a.model.parameters()}=={torch.float16}
  save(L/'environment.json',dict(static_environment(a),GPU_UUID=UUID,torch_import=torch.__file__,transformers_import=transformers.__file__))
  with gzip.open(O/'input_manifest.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
  bykey={s['key']:s for s in rows};mapping=read(O/'input_path_mapping.json');questions=read(O/'question_positions.json');prepared={}
  for s in rows:
   assert sha(mapping[s['key']])==s['image_sha256'];raw=dict(input_ids=torch.tensor([s['input_ids']],dtype=torch.long),attention_mask=torch.ones((1,len(s['input_ids'])),dtype=torch.long),pixel_values=frozen.paper_pixels(mapping[s['key']],a.image_processor).half());assert {k:tensor_hash(v) for k,v in raw.items()}==s['input_hashes'];prepared[s['key']]=raw
  save(L/'input_verification.json',dict(status='PASS',count=30,keys=[s['key'] for s in rows],all_input_hashes_exact=True))
  baseline_hooks=[(l.forward.__func__,l.self_attn.forward.__func__) for l in a.layers]
  def clean():
   for l,(lf,af) in zip(a.layers,baseline_hooks):
    assert l.forward.__func__ is lf and l.self_attn.forward.__func__ is af
    assert not l._forward_hooks and not l._forward_pre_hooks
    assert all(not getattr(l.self_attn,x+'_proj')._forward_pre_hooks for x in ('q','k','v','o'))
  db=sqlite3.connect(L/'checkpoint.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE requests (request_id INTEGER PRIMARY KEY,category TEXT,key TEXT,method TEXT,payload TEXT,sha256 TEXT)');db.execute('CREATE UNIQUE INDEX formal_once ON requests(key,method) WHERE category="formal"');db.commit()
  out=(L/'requests.jsonl').open('x')
  def request(key,method,category,smoke=False,**extra):
   if (L/'STOP.flag').exists():raise RuntimeError((L/'STOP.flag').read_text())
   assert all(x.poll() is None for x in monitors),'Telemetry unavailable'
   clean();s=bykey[key];a.inputs={k:v.to('cuda:0') for k,v in prepared[key].items()};ids=s['input_ids'];im=tuple(i for i,v in enumerate(ids) if v==32000);text=tuple(i for i,v in enumerate(ids) if v!=32000);relay=tuple(range(im[-1]+1,len(ids)-1));layout=Layout(len(ids),im,text,relay,tuple(questions.get(key,relay)),20)
   torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();begin=time.time_ns();g=generate(a,layout,method,smoke);end=time.time_ns();clean()
   assert g['output_tokens']==32 and g['real_decode_calls']==31 and g['vision_forward_calls']==g['prefill_calls']==1 and g['all_logits_finite']
   assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6
   r=dict(request_id=len(records)+1,category=category,key=key,benchmark=s['benchmark'],method=method,session=phase,GPU_UUID=UUID,input_hashes=s['input_hashes'],sample_sha256=s['sample_sha256'],begin_ns=begin,end_ns=end,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),first_EOS_position=next((i+1 for i,t in enumerate(g['token_ids']) if t==2),None),generation=g,**extra)
   payload=json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)
   with db:db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)',(r['request_id'],category,key,method,payload,hashlib.sha256(payload.encode()).hexdigest()))
   out.write(payload+'\n');out.flush();os.fsync(out.fileno());records.append(r);state['formal_completed']+=int(category=='formal');progress();return r
  keys=[rows[0]['key'],rows[15]['key']];smoke_keys=list(dict.fromkeys(keys+[max(rows,key=lambda s:len(s['input_ids']))['key']]))
  if phase in ('smoke','smoke_sparse'):
   state['phase']='CORRECTNESS_SMOKE';progress();checks={};native={};blocked={}
   for key in (smoke_keys if phase=='smoke' else []):
    native[key]=request(key,'Native','smoke',True)
    for noop in ['BridgeNative','FastVNoop','V2DropNoop']:
     b=request(key,noop,'smoke',True);ref=native[key]['generation'];assert b['generation']['token_ids']==ref['token_ids'] and b['generation']['first_logits']==ref['first_logits'];checks[key+'|'+noop]='STRICT_EXACT'
   if phase=='smoke_sparse':
    old=read(O/'smoke_validation.json');checks=dict(old['checks']);blocked=dict(old['blocked']);blocked.pop('SparseVLM+SVH',None)
   for method in (METHODS[1:] if phase=='smoke' else ['SparseVLM+SVH']):
    try:
     for key in smoke_keys:
      r=request(key,method,'smoke',True);checks[key+'|'+method]='PASS';print('SMOKE_PASS',method,key,flush=True)
      if method=='ApET+SVH':
       d=request(key,'ApETDenseOracle','diagnostic',True)['generation'];x=torch.tensor(r['generation']['decode_logits']);y=torch.tensor(d['decode_logits']);diff=(x-y).abs();top=(x.argmax(-1)==y.argmax(-1)).all().item();gate=dict(max_abs=diff.max().item(),mean_abs=diff.mean().item(),top1_exact=top);checks[key+'|ApET_physical_vs_dense']=gate
       assert gate['max_abs']<=.10 and gate['mean_abs']<=.005 and top,'Frozen ApET numerical gate failed'
    except Exception as e:
     blocked[method]=dict(reason=repr(e),traceback=traceback.format_exc());print('SMOKE_BLOCKED',method,repr(e),flush=True);gc.collect();torch.cuda.empty_cache()
     if isinstance(e,torch.OutOfMemoryError):raise
   usable=[m for m in METHODS if m not in blocked];save(O/'smoke_validation.json',dict(status='PASS' if not blocked else 'PARTIAL',backbone='LLaVA',usable=usable,blocked=blocked,checks=checks,all_gates_frozen_before_GPU=True))
   state.update(status='COMPLETE' if not blocked else 'PARTIAL',phase='COMPLETE',usable=usable,blocked=blocked)
  else:
   smoke=read(O/'smoke_validation.json');methods=smoke['usable'];state['phase']='FIXED_WARMUP';progress()
   for i in range(12):request(keys[i%2],'Native','warmup',warmup_index=i+1)
   for key in keys:
    for method in methods:request(key,method,'warmup')
   state['phase']='CALIBRATION_BASELINE';progress();base=[]
   for repeat in range(3):
    for key in keys:base.append(request(key,'Native','baseline',repeat=repeat))
   metrics=['E2E_ms','TPOT_ms'];bm={k:{m:statistics.mean(r['generation'][m] for r in base if r['key']==k) for m in metrics} for k in keys};save(L/'baseline.json',bm)
   state['phase']='FORMAL_INTERLEAVED';progress();drifts=[]
   for i,s in enumerate(rows):
    shift=(i+(0 if phase=='session1' else len(methods)//2))%len(methods);order=methods[shift:]+methods[:shift]
    for oi,method in enumerate(order):request(s['key'],method,'formal',input_index=i+1,order_in_input=oi)
    print('INPUT_DONE',phase,i+1,state['formal_completed'],flush=True)
    if (i+1)%5==0:
     pp=[]
     for rep in range(2):
      for key in keys:pp.append(request(key,'Native','calibration',after_inputs=i+1,repeat=rep))
     change={key:{m:statistics.mean(r['generation'][m] for r in pp if r['key']==key)/bm[key][m]-1 for m in metrics} for key in keys};warn=any(abs(v)>.05 for dd in change.values() for v in dd.values());dd=dict(after_inputs=i+1,relative_change=change,threshold=.05,status='WARN' if warn else 'WITHIN_5PCT');drifts.append(dd)
     if warn:warnings.append(dd)
     save(L/'drift_checks.json',drifts);sync()
   state.update(status='COMPLETE_WITH_PERFORMANCE_WARNINGS' if warnings else 'COMPLETE',phase='COMPLETE',formal_expected=len(methods)*30)
  check_sources();out.close()
 except BaseException as ex:
  state.update(status='PARTIAL' if state['formal_completed'] else 'BLOCKED',error=repr(ex),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
 finally:
  save(L/'performance_warnings.json',warnings);progress();save(L/'completion.json',state);(L/'monitor.stop').write_text('done')
  for m in monitors:
   try:m.wait(timeout=8)
   except subprocess.TimeoutExpired:m.terminate();m.wait(timeout=5)
  if db:db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
  a=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  sync();print('FINAL',json.dumps(state,ensure_ascii=False),flush=True)
 return 0 if state['status'].startswith('COMPLETE') or (phase=='smoke' and state.get('usable')) else 1
if __name__=='__main__':raise SystemExit(main())
