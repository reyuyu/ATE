import os,sys,json,time,gzip,hashlib,subprocess,gc,traceback,importlib,argparse,statistics,collections,contextlib,sqlite3,shutil
from pathlib import Path
from unittest.mock import patch
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113');W=Path('/home/featurize/ate-experiments/rtx3090-18conditions-20261113')
R=Path('/home/featurize/work/ATE');O=W/'vlm_research/results/rtx3090_unified_efficiency_18conditions';D=R/'vlm_research/results/rtx3090_unified_efficiency_18conditions';U=R/'vlm_research/results/unified_e2e32_4090'
UUID='GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6'
os.environ.update(CUDA_VISIBLE_DEVICES=UUID,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_MODULE_LOADING='LAZY')
sys.path[:0]=[str(A),str(O/'code'),str(W),str(W/'vlm_research'),str(R),str(U/'code'),str(R/'vlm_research/external_baselines/fastv/compatibility'),str(R/'vlm_research/external_baselines/v2drop/compatibility'),str(R/'vlm_research/external_baselines/e2e32_compatibility')]
import torch,transformers
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def th(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['Qwen','InternVL','LLaVA'],required=True);ap.add_argument('--phase',choices=['screen','session1','session2','profile','diagnosis','audit'],required=True);args=ap.parse_args();K=args.kind;phase=args.phase
G=A/K/phase;G.mkdir(parents=True,exist_ok=False);records=[];mon=[];backend=None;a=None;numeric=[];audit_snapshots=[];db=None
state=dict(status='RUNNING',kind=K,phase=phase,pid=os.getpid(),completed=0,formal_completed=0)
def progress():save(G/'progress.json',dict(state,updated_ns=time.time_ns()))
progress()
try:
 assert read(A/'cpu_tests.json')['status']=='PASS_CPU_BEFORE_GPU'
 assert torch.__version__=='2.6.0+cu124' and transformers.__version__==('4.57.6' if K=='Qwen' else '4.49.0')
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==UUID
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU_CONFLICT'
 for x in read(A/'optimized_source_manifest.json'):assert sha(x['path'])==x['sha256'],('OPT_SOURCE_MISMATCH',x['path'])
 # Verify actual original runtime and all selected optimization dependencies.
 for x in read(A/'runtime_source_guard.json'):assert sha(x['path'])==x['sha256'],('ORIGINAL_SOURCE_MISMATCH',x['path'])
 torch.set_num_threads(1);torch.set_num_interop_threads(6);os.sched_setaffinity(0,set(range(6)))
 for mode in ['gpu','cpu','processes']:
  log=(G/(mode+'_monitor.log')).open('x');mon.append(subprocess.Popen([sys.executable,'-B',str(O/'code/monitor.py'),str(G),str(os.getpid()),UUID,mode],stdout=log,stderr=subprocess.STDOUT))
 save(G/'environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torch_import=torch.__file__,transformers=transformers.__version__,transformers_import=transformers.__file__,CUDA=torch.version.cuda,GPU_UUID=UUID,nvidia_smi=subprocess.check_output(['nvidia-smi','-q'],text=True),threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),affinity=sorted(os.sched_getaffinity(0)),deterministic=torch.are_deterministic_algorithms_enabled(),TF32=torch.backends.cuda.matmul.allow_tf32))
 with gzip.open(D/'input_manifest.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
 mapping=read(D/'input_path_mapping.json');questions=read(D/'question_positions.json')
 bykey={r['key']:r for r in rows};prepared={};bindings={};layouts={}
 mods={}
 if K!='LLaVA':
  from cross_runtime import Backend
  backend=Backend(K,O);a=backend
  layers=backend.layers;model=backend.model
  for ns in ['opt_b','opt_c','opt_d']:
   if K=='Qwen':
    c=importlib.import_module(ns+'.qwen3vl_c3v2_l22_kernel_diagnosis.candidates');c.PRIVATE=O/'resources'
    mods[ns]={'S':importlib.import_module(ns+'.qwen3vl_svh_sqa7908_repair.engine').EarlyAlignedExecution,'R':importlib.import_module(ns+'.qwen3vl_svhr_search.physical').Physical}
   else:
    mods[ns]={'base':importlib.import_module(ns+'.internvl3_svh_physical_d_pilot.engine'),'late':importlib.import_module(ns+'.internvl3_svh_r_terminal_windows_textvqa.engine')}
  oldS=backend.S if K=='Qwen' else None;oldR=backend.R;oldbase=backend.base
  for row in rows:
   key=row['key'];assert sha(mapping[key])==row['image_sha256']
   x,l,b=backend.prepare(row,mapping[key]);prepared[key]=x;layouts[key]=l;bindings[key]=b
  assert bindings==read(D/K/'smoke/input_verification.json')['bindings'],'INPUT_BINDING_CHANGED'
  # Frozen kv auditor corrected only to expose actual buffer capacity of new C caches.
  original_kv=backend.kv
  def kv(cache,layout,method):
   result=original_kv(cache,layout,method)
   if isinstance(getattr(cache,'buffers',None),list):
    for r in result['layers']:
     pair=cache.buffers[r['layer']]
     if pair is not None:r['capacity_tokens']=pair[0].shape[-2]
   return result
  backend.kv=kv
  def raw_generate(key,method,stage,audit):
   ns={'B':'opt_b','C':'opt_c','D':'opt_d'}.get(stage)
   if K=='Qwen':
    backend.S=mods[ns]['S'] if ns else oldS;backend.R=mods[ns]['R'] if ns else oldR
   else:
    backend.base=mods[ns]['base'] if ns else oldbase;backend.R=mods[ns]['late'] if ns else oldR
   try:return backend.generate(current_inputs,layouts[key],method,audit)[0]
   finally:
    if K=='Qwen':backend.S=oldS
    backend.R=oldR;backend.base=oldbase
  methods=['SVH','SVH-R']
  model_for_capture=model if K=='Qwen' else model.language_model
 else:
  from vlm_research.information_horizon_alignment import engine as frozen
  from vlm_research.mme_seed_quality import engine as loader
  from vlm_research.physical_handoff_fastpath.engine import Layout
  import unified_llava as ll
  import e2e32_runtime
  loader.MODELS=dict(loader.MODELS,llava=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf'));a=frozen.load('B')
  model=a.model;layers=a.layers;model_for_capture=model.language_model
  for row in rows:
   key=row['key'];assert sha(mapping[key])==row['image_sha256']
   x=dict(input_ids=torch.tensor([row['input_ids']],dtype=torch.long),attention_mask=torch.ones((1,len(row['input_ids'])),dtype=torch.long),pixel_values=frozen.paper_pixels(mapping[key],a.image_processor).half())
   assert {k:th(v) for k,v in x.items()}==row['input_hashes']
   prepared[key]=x;n=len(row['input_ids']);im=tuple(i for i,t in enumerate(row['input_ids']) if t==32000);text=tuple(i for i in range(n) if i not in set(im))
   layouts[key]=Layout(n,im,text,tuple(range(im[-1]+1,n-1)),tuple(questions.get(key,[])),20)
   bindings[key]=dict(input_hashes=row['input_hashes'],prompt_sha256=row['prompt_sha256'] if 'prompt_sha256' in row else hashlib.sha256(row['prompt'].encode()).hexdigest(),image_sha256=row['image_sha256'])
  fast=importlib.import_module('opt_b.fastv_svh.engine');fast.OFFICIAL=ll.fast.OFFICIAL
  sparse=importlib.import_module('opt_b.sparsevlm_svh.engine')
  importlib.import_module('opt_b.sparsevlm_svh.official').OFFICIAL=R/'vlm_research/external_baselines/sparsevlm_icml2025/official_source'
  prefix=importlib.import_module('opt_b.prefix_fusion');apet=importlib.import_module('opt_b.apet_fusion')
  @contextlib.contextmanager
  def stage_context(stage):
   if stage=='A':yield;return
   with patch.object(ll,'fast',fast),patch.object(ll,'SparseFusion',sparse.Fusion),patch.object(ll,'prefix',prefix),patch.object(ll,'ApETFusion',apet.ApETFusion):yield
  def raw_generate(key,method,stage,audit):
   a.inputs=current_inputs
   with stage_context(stage):return ll.generate(a,layouts[key],method,audit)
  methods=['FastV+SVH','SparseVLM+SVH','V2Drop+SVH','ApET+SVH']
 save(G/'runtime_flags.json',dict(recorded_after_model_load=True,torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,CPU_affinity=sorted(os.sched_getaffinity(0))))
 save(G/'input_validation.json',dict(status='PASS',bindings=bindings,count=30,Gold_separate=True))
 # Frozen shape-based smoke selection, before any candidate timing.
 ordered=sorted([r['key'] for r in rows],key=lambda k:((layouts[k].length if K=='LLaVA' else layouts[k].get('input_length',layouts[k].get('n'))),k))
 smoke=list(dict.fromkeys([ordered[0],ordered[len(ordered)//2],ordered[-1]]));save(G/'smoke_selection.json',dict(keys=smoke,rule='min/median/max original input length'))
 histdir=D/('session2' if K=='LLaVA' else K+'/session2')
 historical={(r['key'],r['method']):r['generation'] for r in map(json.loads,(histdir/'requests.jsonl').read_text().splitlines()) if r['category']=='formal'}
 clean_hooks=lambda:sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())
 initial_hooks=clean_hooks()
 db=sqlite3.connect(G/'checkpoint.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 db.execute('CREATE TABLE requests(id INTEGER PRIMARY KEY,category TEXT,key TEXT,method TEXT,stage TEXT,payload TEXT,sha256 TEXT)')
 db.execute('CREATE UNIQUE INDEX formal_once ON requests(key,method,stage) WHERE category="formal"');db.commit()
 def cache_snapshot(cache,step,method,stage,key):
  pairs=cache.entries if hasattr(cache,'entries') else list(zip(cache.key_cache,cache.value_cache)) if hasattr(cache,'key_cache') else [(x.keys,x.values) for x in cache.layers]
  data=[]
  for i,pair in enumerate(pairs):
   if pair is None or any(x is None for x in pair):data.append(dict(layer=i,present=False));continue
   used=cache.used[i] if hasattr(cache,'used') else pair[0].shape[-2]
   data.append(dict(layer=i,present=True,K_shape=list(pair[0].shape),V_shape=list(pair[1].shape),used=used,stride=list(pair[0].stride()),storage_bytes=[x.untyped_storage().nbytes() for x in pair],storage_pointers=[x.untyped_storage().data_ptr() for x in pair],live_bytes=2*pair[0].shape[0]*pair[0].shape[1]*used*pair[0].shape[-1]*pair[0].element_size()))
  audit_snapshots.append(dict(key=key,method=method,stage=stage,step=step,layers=data,late_KV_remaining=len(cache.late) if hasattr(cache,'late') else None))
 @contextlib.contextmanager
 def capture(key,method,stage,hidden=False,forced=None,profile_mode=False):
  vectors=[];hidden_states=[];step=[0];handles=[]
  original_forward=model_for_capture.forward
  def fwd(*ar,**kw):
   s=step[0]
   if forced is not None and s>0:kw=dict(kw,input_ids=torch.tensor([[forced[s-1]]],device='cuda:0'))
   if profile_mode:
    with torch.profiler.record_function('MODEL_PREFILL' if s==0 else 'MODEL_DECODE_'+str(s)):out=original_forward(*ar,**kw)
   else:out=original_forward(*ar,**kw)
   if not profile_mode:vectors.append(out.logits[:,-1].detach().float().cpu())
   if hidden and s in [0,1,8,31]:cache_snapshot(out.past_key_values,s,method,stage,key)
   step[0]+=1;return out
  def hhook(i):
   def h(mod,ar,out):
    if step[0] not in [0,1,8,31]:return
    x=out[0] if isinstance(out,tuple) else out
    if torch.is_tensor(x):hidden_states.append((step[0],i,x.detach().float().cpu()))
   return h
  try:
   if hidden:
    for i,l in enumerate(layers):handles.append(l.register_forward_hook(hhook(i)))
   with patch.object(model_for_capture,'forward',fwd):yield vectors,hidden_states
  finally:
   for h in handles:h.remove()
 def request(key,method,stage,category,capt=False,hidden=False,audit=False,forced=None,profile_mode=False,**extra):
  global current_inputs
  assert not (G/'STOP.flag').exists(),'STOP_REQUESTED'
  assert all(p.poll() is None for p in mon),'MONITOR_EXIT'
  assert clean_hooks()==initial_hooks,'HOOK_LEAK_BEFORE'
  x=prepared[key];assert {k:th(v) for k,v in x.items()}==bindings[key]['input_hashes']
  current_inputs={k:v.to('cuda:0') for k,v in x.items()}
  if K=='LLaVA':a.inputs=current_inputs
  torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();start_ns=time.time_ns()
  if capt or profile_mode:
   with capture(key,method,stage,hidden,forced,profile_mode) as (vs,hs):g=raw_generate(key,method,stage,audit)
  else:g=raw_generate(key,method,stage,audit);vs=[];hs=[]
  assert clean_hooks()==initial_hooks,'HOOK_LEAK_AFTER'
  assert g['output_tokens']==32 and g['real_decode_calls']==31 and g['all_logits_finite']
  assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6
  if stage=='A' and method in methods+['Native'] and forced is None:
   assert g['token_ids']==historical[key,method]['token_ids'],('ORIGINAL_TOKEN_MISMATCH',key,method)
  item=dict(id=len(records)+1,category=category,kind=K,key=key,method=method,stage=stage,session=phase,
   begin_ns=start_ns,end_ns=time.time_ns(),GPU_UUID=UUID,input_hashes=bindings[key]['input_hashes'],
   prompt_sha256=bindings[key]['prompt_sha256'],image_sha256=bykey[key]['image_sha256'],
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   instrumentation_perturbed=capt or audit or profile_mode,teacher_forced_reference=forced is not None,generation=g,**extra)
  payload=json.dumps(item,ensure_ascii=False,separators=(',',':'),allow_nan=False)
  with db:db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?,?)',(item['id'],category,key,method,stage,payload,hashlib.sha256(payload.encode()).hexdigest()))
  with (G/'requests.jsonl').open('a') as f:f.write(payload+'\n');f.flush();os.fsync(f.fileno())
  records.append(item);state['completed']=len(records);state['formal_completed']+=int(category=='formal');progress()
  print('DONE',K,phase,len(records),category,key,method,stage,round(g['TTFT_ms'],2),round(g['TPOT_ms'],2),flush=True)
  del current_inputs;return item,vs,hs
 def compare(ref,r,rv,vs,rh,hs,method,stage,key):
  import math
  equal=len(rv)==len(vs)==32 and all(torch.equal(x,y) for x,y in zip(rv,vs))
  max_abs=max(float((x-y).abs().max()) for x,y in zip(rv,vs))
  rel=max(float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp_min(1e-12)) for x,y in zip(rv,vs))
  hr=0.;hidden_compatible=len(rh)==len(hs)
  if hidden_compatible:
   for (s,i,x),(t,j,y) in zip(rh,hs):
    if (s,i,x.shape)!=(t,j,y.shape):hidden_compatible=False;break
    hr=max(hr,float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp_min(1e-12)))
  tokens=r['generation']['token_ids']==ref['generation']['token_ids']
  finite=all(torch.isfinite(x).all() for x in vs)
  if stage in ['B','C']:passed=equal and tokens and hidden_compatible and hr==0
  else:passed=finite and hidden_compatible and hr<=.0078125 and rel<=.00390625 and max_abs<=.125
  info=dict(key=key,method=method,stage=stage,logits_torch_equal=equal,max_abs=max_abs,logits_relative_L2_max=rel,
   hidden_relative_L2_max=hr,hidden_shapes_equal=hidden_compatible,tokens_exact=tokens,
   top1_changes=sum(a!=b for a,b in zip(ref['generation']['token_ids'],r['generation']['token_ids'])),finite=bool(finite),status='PASS' if passed else 'FAILED',
   gate='exact B/C; frozen BF16 bounded D',quality_historical_binding='only unchanged outputs and arithmetic eligible; D requires independent quality record')
  details=[]
  if len(rh)==len(hs):
   for (s,i,x),(t,j,y) in zip(rh,hs):
    if x.shape!=y.shape:continue
    details.append(dict(step=s,layer=i,shape=list(x.shape),torch_equal=torch.equal(x,y),max_abs=float((x-y).abs().max()),mean_abs=float((x-y).abs().mean()),relative_L2=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp_min(1e-12))))
  info['layerwise_differences']=details
  info['logits_per_step']=[dict(step=i,max_abs=float((x-y).abs().max()),mean_abs=float((x-y).abs().mean()),relative_L2=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp_min(1e-12)),top1_equal=int(x.argmax())==int(y.argmax())) for i,(x,y) in enumerate(zip(rv,vs))]
  numeric.append(info);save(G/'numerical_validation.json',numeric);return passed
 if phase=='screen':
  # Native no-op strict32-logit smoke.
  k=smoke[0];r,v,h=request(k,'Native','A','native_noop',True)
  noop='FastVNoop' if K=='LLaVA' else 'Noop';z,zv,zh=request(k,noop,'A','native_noop',True)
  assert r['generation']['token_ids']==z['generation']['token_ids'] and all(torch.equal(x,y) for x,y in zip(v,zv))
  save(G/'native_noop.json',dict(status='PASS',all32_logits_equal=True,tokens_exact=True));del v,zv
  stages=['B'] if K=='LLaVA' else ['B','C','D']
  eligible={(m,s):True for m in methods for s in stages};failures={}
  for key in smoke+[k for k in [r['key'] for r in rows] if k not in smoke]:
   is_smoke=key in smoke
   for method in methods:
    ref,rv,rh=request(key,method,'A','numerical_original',True,hidden=is_smoke)
    for stage in stages:
     if not eligible[method,stage]:continue
     if stage=='C' and not eligible.get((method,'B'),False):eligible[method,stage]=False;continue
     if stage=='D' and not eligible.get((method,'C'),False):eligible[method,stage]=False;continue
     if stage=='D' and not (K=='Qwen' and method=='SVH' or K=='InternVL' and method=='SVH-R'):
      eligible[method,stage]=False;continue
     try:
      r,vs,hs=request(key,method,stage,'numerical_candidate',True,hidden=is_smoke,forced=ref['generation']['token_ids'] if stage=='D' else None)
      if not compare(ref,r,rv,vs,rh,hs,method,stage,key):
       eligible[method,stage]=False;failures[method+'|'+stage]=dict(reason='FROZEN_NUMERICAL_GATE_FAILED',first_key=key)
      if stage=='D' and r['generation']['token_ids']!=ref['generation']['token_ids']:
       # Candidate cannot inherit old quality; keep diagnostics and stop formal entry.
       eligible[method,stage]=False;failures[method+'|'+stage]=dict(reason='TOKEN_CHANGE_REQUIRES_INDEPENDENT_QUALITY_NOT_HISTORICAL_BINDING',first_key=key)
      del vs,hs
     except (torch.OutOfMemoryError,AssertionError) as e:
      eligible[method,stage]=False;failures[method+'|'+stage]=dict(reason=repr(e),first_key=key,traceback=traceback.format_exc())
      if isinstance(e,torch.OutOfMemoryError):raise
      gc.collect();torch.cuda.empty_cache()
    del rv,rh
  save(G/'candidate_eligibility.json',dict(eligible={m:{s:eligible[m,s] for s in stages} for m in methods},failures=failures))
  save(G/'physical_audit.json',audit_snapshots)
  # Ordinary A/B/C(/D) timed ablation, not numerical/profiler time.
  candidates=[('Native','A')]+[(m,'A') for m in methods]+[(m,s) for m in methods for s in stages if eligible[m,s]]
  for i in range(12):request(smoke[i%len(smoke)],'Native','A','warmup')
  for m,s in candidates:
   for key in [smoke[0],smoke[-1]]:request(key,m,s,'warmup')
  for rep in range(3):
   for ki,key in enumerate(smoke):
    shift=(ki+rep)%len(candidates);order=candidates[shift:]+candidates[:shift]
    for oi,(m,s) in enumerate(order):request(key,m,s,'ablation',repeat=rep,order_in_input=oi)
  selected={}
  for method in methods:
   def mean(stage,metric):return statistics.mean(r['generation'][metric] for r in records if r['category']=='ablation' and r['method']==method and r['stage']==stage)
   chosen=None;decisions=[]
   for s in stages:
    if not eligible[method,s]:continue
    benefit={k:1-mean(s,k)/mean('A',k) for k in ['TTFT_ms','TPOT_ms','E2E_ms']}
    # Prespecified screening: some useful timing improvement >=1%, E2E no worse >0.5%.
    useful=max(benefit.values())>=.01 and benefit['E2E_ms']>=-.005
    if useful and (chosen is None or mean(s,'E2E_ms')<mean(chosen,'E2E_ms')):chosen=s
    decisions.append(dict(stage=s,relative_improvement=benefit,useful=useful))
   selected[method]=dict(stage=chosen,status='ELIGIBLE_FOR_FORMAL' if chosen else 'NO_MEASURED_USEFUL_GAIN',decisions=decisions)
  save(A/K/'selected_stages.json',selected);save(G/'selected_stages.json',selected)
  state.update(status='COMPLETE_SCREEN',eligible=selected)
 elif phase.startswith('session'):
  selected=read(A/K/'selected_stages.json')
  chosen={m:d['stage'] for m,d in selected.items() if d['stage']}
  candidates=[('Native','A')]+[(m,s) for m in chosen for s in ['A',chosen[m]]]
  assert chosen,'NO_AUDITED_BENEFICIAL_CANDIDATE'
  probes=[rows[0]['key'],rows[15]['key']]
  for i in range(12):request(probes[i%2],'Native','A','warmup')
  for m,s in candidates:
   for key in probes:request(key,m,s,'warmup')
  base=[]
  for rep in range(3):
   for key in probes:base.append(request(key,'Native','A','baseline',repeat=rep)[0])
  metrics=['E2E_ms','TPOT_ms'];bm={k:{m:statistics.mean(r['generation'][m] for r in base if r['key']==k) for m in metrics} for k in probes};save(G/'baseline.json',bm);drifts=[]
  for i,row in enumerate(rows):
   shift=(i+(0 if phase=='session1' else len(candidates)//2))%len(candidates)
   for oi,(m,s) in enumerate(candidates[shift:]+candidates[:shift]):request(row['key'],m,s,'formal',input_index=i+1,order_in_input=oi)
   if (i+1)%5==0:
    pp=[]
    for rep in range(2):
     for key in probes:pp.append(request(key,'Native','A','calibration',after_inputs=i+1,repeat=rep)[0])
    changes={k:{m:statistics.mean(r['generation'][m] for r in pp if r['key']==k)/bm[k][m]-1 for m in metrics} for k in probes}
    warn=any(abs(v)>.05 for vals in changes.values() for v in vals.values())
    drifts.append(dict(after_inputs=i+1,relative_change=changes,status='WARN' if warn else 'WITHIN_5PCT'));save(G/'drift_checks.json',drifts)
    if warn:print('DRIFT_WARNING_PRESERVED',K,phase,json.dumps(drifts[-1]),flush=True)
  assert state['formal_completed']==30*len(candidates)
  state.update(status='COMPLETE_WITH_WARNINGS' if any(x['status']=='WARN' for x in drifts) else 'COMPLETE',formal_expected=30*len(candidates))
 elif phase=='audit':
  selected=read(A/K/'screen/candidate_eligibility.json')['eligible']
  key=smoke[0];checks=[]
  for method in methods:
   ref,rv,rh=request(key,method,'A','audit_reference',True)
   for stage,ok in selected[method].items():
    if not ok:continue
    r,vs,hs=request(key,method,stage,'audit_mode',True,audit=True)
    exact=all(torch.equal(x,y) for x,y in zip(rv,vs))
    assert r['generation']['token_ids']==ref['generation']['token_ids'],'AUDIT_MODE_CHANGED_TOKENS'
    if stage in ['B','C']:assert exact,'AUDIT_MODE_CHANGED_LOGITS'
    checks.append(dict(method=method,stage=stage,full_tokens_equal=True,all32_logits_equal=exact,audit_enabled=True))
   del rv
  save(G/'audit_mode_validation.json',dict(status='PASS',checks=checks))
  state.update(status='COMPLETE_AUDIT_MODE')
 elif phase=='diagnosis':
  assert K in ['Qwen','InternVL']
  method='SVH' if K=='Qwen' else 'SVH-R';key=smoke[0]
  ref,rv,rh=request(key,method,'A','failure_reference',True,hidden=True)
  r,vs,hs=request(key,method,'D','failed_padding_diagnosis',True,hidden=True,forced=ref['generation']['token_ids'])
  compare(ref,r,rv,vs,rh,hs,method,'D',key)
  save(G/'physical_audit.json',audit_snapshots)
  state.update(status='COMPLETE_DIAGNOSIS',candidate_status='FAILED_RETAINED_NOT_FORMAL')
 elif phase=='profile':
  elig=read(A/K/'screen/candidate_eligibility.json')['eligible']
  candidates=[('Native','A')]+[(m,'A') for m in methods]+[(m,s) for m in methods for s,ok in elig[m].items() if ok]
  # Full32-token CPU/CUDA profiler independent of official timing.
  for i in range(6):request(smoke[i%len(smoke)],'Native','A','warmup')
  key=smoke[1];(G/'traces').mkdir()
  for method,stage in candidates:
   request(key,method,stage,'warmup')
   path=G/'traces'/(method.replace('+','_')+'_'+stage+'.json')
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=True,with_stack=False) as prof:
    request(key,method,stage,'profile',profile_mode=True)
   prof.export_chrome_trace(str(path))
   with path.open('rb') as src,gzip.open(str(path)+'.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst,1024*1024)
   path.unlink();del prof
  state.update(status='COMPLETE_PROFILE')
 for x in read(A/'runtime_source_guard.json'):assert sha(x['path'])==x['sha256']
 save(G/'completion.json',dict(state,source_unchanged=True,original_records_unchanged=True))
except BaseException as e:
 state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save(G/'failure.json',state);save(G/'completion.json',state)
 print(traceback.format_exc(),flush=True)
finally:
 (G/'monitor.stop').write_text('done')
 for p in mon:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
 if db:db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
 backend=a=None;gc.collect()
 if torch.cuda.is_initialized():torch.cuda.empty_cache()
 print('FINAL',json.dumps(state,ensure_ascii=False),flush=True)
raise SystemExit(0 if state['status'].startswith('COMPLETE') else 1)
