import os,sys,argparse,time,json,hashlib,gzip,subprocess,sqlite3,traceback,statistics,gc,shutil
from pathlib import Path
O=Path(__file__).resolve().parent.parent;W=O.parents[2];D=Path('/home/featurize/work/ATE/vlm_research/results/rtx3090_unified_efficiency_18conditions');UUID='GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6'
os.environ.update(CUDA_VISIBLE_DEVICES=UUID,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_MODULE_LOADING='LAZY')
sys.path[:0]=[str(O/'code'),str(W)]
import torch,transformers
from cross_runtime import Backend,th

def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync():
 for p in O.rglob('*'):
  if p.is_file() and p.suffix not in ['.sqlite','.partial','.so'] and not any(t in p.parts for t in ['deps','cccl']) and not p.name.endswith(('.sqlite-wal','.sqlite-shm')):
   d=D/p.relative_to(O);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['Qwen','InternVL'],required=True);ap.add_argument('--phase',choices=['smoke','session1','session2'],required=True);args=ap.parse_args();kind,phase=args.kind,args.phase;L=O/kind/phase;L.mkdir(parents=True)
 state=dict(status='RUNNING',phase='PRECHECK',backbone=kind,session=phase,pid=os.getpid(),formal_completed=0);records=[];warnings=[];model=None;mon=[];db=None
 def progress():save(L/'progress.json',dict(state,updated_ns=time.time_ns()))
 progress()
 try:
  assert torch.__version__=='2.6.0+cu124' and transformers.__version__==('4.57.6' if kind=='Qwen' else '4.49.0')
  source=read(O/'cross_source_manifest.json')
  for p,h in source['files'].items():assert sha(p)==h,('SOURCE_CHANGED',p)
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==UUID
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU conflict'
  torch.set_num_threads(1);torch.set_num_interop_threads(6);os.sched_setaffinity(0,set(range(6)))
  for mode in ['gpu','cpu','processes']:
   log=(L/(mode+'_monitor.log')).open('x');mon.append(subprocess.Popen([sys.executable,'-B',str(O/'code/monitor.py'),str(L),str(os.getpid()),UUID,mode],stdout=log,stderr=subprocess.STDOUT))
  state['phase']='MODEL_LOAD';progress();model=Backend(kind,O)
  save(L/'environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torch_import=torch.__file__,transformers=transformers.__version__,transformers_import=transformers.__file__,CUDA=torch.version.cuda,GPU_UUID=UUID,nvidia_smi=subprocess.check_output(['nvidia-smi','-q'],text=True),dtype='bfloat16',backend='eager',layers=len(model.layers)))
  with gzip.open(O/'input_manifest.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
  mapping=read(O/'input_path_mapping.json');bykey={r['key']:r for r in rows};prepared={};bindings={}
  for row in rows:
   key=row['key'];assert sha(mapping[key])==row['image_sha256'];x,l,b=model.prepare(row,mapping[key]);prepared[key]=(x,l);bindings[key]=b
  save(L/'input_verification.json',dict(status='PASS',same_frozen_images_questions=True,Paper_OCR_preserved=True,Gold_separate=True,bindings=bindings))
  prior=O/kind/'smoke/input_verification.json'
  if phase!='smoke':assert bindings==read(prior)['bindings'],'BACKBONE_INPUT_HASH_CHANGED'
  db=sqlite3.connect(L/'checkpoint.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE requests(id INTEGER PRIMARY KEY,category TEXT,key TEXT,method TEXT,payload TEXT,sha256 TEXT)');db.execute('CREATE UNIQUE INDEX formal_once ON requests(key,method) WHERE category="formal"');db.commit();out=(L/'requests.jsonl').open('x')
  hooks=lambda:sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.model.modules())
  clean_hooks=hooks()
  def request(key,method,category,smoke=False,**extra):
   if (L/'STOP.flag').exists():raise RuntimeError((L/'STOP.flag').read_text())
   assert all(p.poll() is None for p in mon),'Telemetry unavailable'
   x,l=prepared[key];assert {k:th(v) for k,v in x.items()}==bindings[key]['input_hashes'];inputs={k:v.to('cuda:0') for k,v in x.items()};assert hooks()==clean_hooks
   torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();g,logits=model.generate(inputs,l,method,smoke);assert hooks()==clean_hooks
   assert len(g['token_ids'])==32 and g['real_decode_calls']==31 and abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6
   eos=model.model.generation_config.eos_token_id if kind=='Qwen' else model.tok.convert_tokens_to_ids(model.model.conv_template.sep.strip());eos=[eos] if isinstance(eos,int) else eos
   r=dict(request_id=len(records)+1,category=category,key=key,benchmark=bykey[key]['benchmark'],backbone=kind,method=method,session=phase,GPU_UUID=UUID,input_hashes=bindings[key]['input_hashes'],prompt_sha256=bindings[key]['prompt_sha256'],image_sha256=bykey[key]['image_sha256'],sample_sha256=bykey[key]['sample_sha256'],first_EOS_position=next((i+1 for i,t in enumerate(g['token_ids']) if t in eos),None),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),generation=g,**extra)
   payload=json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)
   with db:db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)',(r['request_id'],category,key,method,payload,hashlib.sha256(payload.encode()).hexdigest()))
   out.write(payload+'\n');out.flush();os.fsync(out.fileno());records.append(r);state['formal_completed']+=int(category=='formal');progress();return r,logits
  keys=[rows[0]['key'],rows[15]['key']];methods=['Native','SVH','SVH-R']
  if phase=='smoke':
   state['phase']='CORRECTNESS_SMOKE';progress();ordered=sorted(rows,key=lambda r:bindings[r['key']]['input_shapes']['input_ids' if kind=='Qwen' else 'ids'][-1]);probe=list(dict.fromkeys([ordered[0]['key'],ordered[len(ordered)//2]['key'],ordered[-1]['key']]))
   checks={};blocked={}
   for method in methods:
    try:
     for key in probe:
      g,lv=request(key,method,'smoke',True)
      if method=='Native':
       z,zv=request(key,'Noop','smoke',True);assert g['generation']['token_ids']==z['generation']['token_ids'] and all(torch.equal(a,b) for a,b in zip(lv,zv)),'NATIVE_NOOP_FAILURE'
      checks[key+'|'+method]='PASS';print('SMOKE_PASS',kind,method,key,flush=True)
    except BaseException as e:
     blocked[method]=dict(reason=repr(e),traceback=traceback.format_exc());print('SMOKE_BLOCKED',kind,method,repr(e),flush=True)
     if method=='Native' or isinstance(e,torch.OutOfMemoryError):raise
     gc.collect();torch.cuda.empty_cache()
   usable=[m for m in methods if m not in blocked];save(O/kind/'smoke_validation.json',dict(status='PASS' if not blocked else 'PARTIAL',usable=usable,blocked=blocked,checks=checks,waivers='InternVL frozen Physical-D numerical waiver retained; Qwen validated aligned BF16 path and SVHR causal prefix construction fully charged'))
   state.update(status='COMPLETE' if not blocked else 'PARTIAL',phase='COMPLETE',usable=usable)
  else:
   methods=read(O/kind/'smoke_validation.json')['usable'];state['phase']='FIXED_WARMUP';progress()
   for i in range(12):request(keys[i%2],'Native','warmup')
   for key in keys:
    for method in methods:request(key,method,'warmup')
   base=[]
   for repeat in range(3):
    for key in keys:base.append(request(key,'Native','baseline',repeat=repeat)[0])
   metrics=['E2E_ms','TPOT_ms'];bm={k:{m:statistics.mean(r['generation'][m] for r in base if r['key']==k) for m in metrics} for k in keys};save(L/'baseline.json',bm);drifts=[];state['phase']='FORMAL_INTERLEAVED';progress()
   for i,row in enumerate(rows):
    shift=(i+(0 if phase=='session1' else len(methods)//2))%len(methods);order=methods[shift:]+methods[:shift]
    for oi,method in enumerate(order):request(row['key'],method,'formal',input_index=i+1,order_in_input=oi)
    print('INPUT_DONE',kind,phase,i+1,state['formal_completed'],flush=True)
    if (i+1)%5==0:
     pp=[]
     for rep in range(2):
      for key in keys:pp.append(request(key,'Native','calibration',after_inputs=i+1,repeat=rep)[0])
     change={key:{m:statistics.mean(r['generation'][m] for r in pp if r['key']==key)/bm[key][m]-1 for m in metrics} for key in keys};warn=any(abs(v)>.05 for a in change.values() for v in a.values());dd=dict(after_inputs=i+1,relative_change=change,threshold=.05,status='WARN' if warn else 'WITHIN_5PCT');drifts.append(dd)
     if warn:warnings.append(dd)
     save(L/'drift_checks.json',drifts);sync()
   state.update(status='COMPLETE_WITH_PERFORMANCE_WARNINGS' if warnings else 'COMPLETE',phase='COMPLETE',formal_expected=len(methods)*30)
  out.close()
 except BaseException as e:
  state.update(status='PARTIAL' if state['formal_completed'] else 'BLOCKED',error=repr(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
  if not (O/kind/'smoke_validation.json').exists():save(O/kind/'smoke_validation.json',dict(status='BLOCKED',usable=[],reason=repr(e),traceback=traceback.format_exc()))
 finally:
  save(L/'performance_warnings.json',warnings);progress();save(L/'completion.json',state);(L/'monitor.stop').write_text('done')
  for pp in mon:
   try:pp.wait(timeout=8)
   except subprocess.TimeoutExpired:pp.terminate();pp.wait(timeout=5)
  if db:db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
  model=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  sync();print('FINAL',json.dumps(state,ensure_ascii=False),flush=True)
 return 0 if state['status'].startswith('COMPLETE') or phase=='smoke' and state.get('usable') else 1
if __name__=='__main__':raise SystemExit(main())
