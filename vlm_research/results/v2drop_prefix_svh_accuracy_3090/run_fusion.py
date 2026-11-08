"""One GPU load; four gated physical smokes, then exactly 200 fusion answers."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
from datetime import datetime,timezone
import fcntl,gc,gzip,hashlib,json,shutil,sqlite3,subprocess,sys,traceback
import torch,transformers
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/v2drop_prefix_svh_accuracy_3090';V=R/'vlm_research/results/v2drop_quick_reproduction';M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
sys.path[:0]=[str(O),str(R)]
from fusion_engine import PrefixSVH,DenseOracle,greedy,SELECTOR_PROVENANCE
from vlm_research.c3v2_middle_reaccess.engine import MiddleWriter
from vlm_research.physical_handoff_fastpath.engine import Layout
from vlm_research.physical_visual_handoff.engine import equivalence
from vlm_research.information_horizon_alignment import engine as frozen,score as scorer
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.gqa_testdev_c3.common import prediction
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def cmd(*args):return subprocess.check_output(args,cwd=R,text=True).strip()
def save(name,x):
 p=O/name;t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def score(s,text):
 if s['benchmark']=='GQA':
  parsed=prediction(text);return dict(parsed=parsed,score=float(parsed==s['answer']))
 result=scorer.score(s,text);e=scorer.text_evaluator();p=e.answer_processor(text);answers=[e.answer_processor(x) for x in s['answer']]
 assert len(answers)==10;count=answers.count(p)
 assert abs(result['score']-sum(min((count-int(x==p))/3,1) for x in answers)/10)<1e-12
 return result

@contextmanager
def trace_execution(a,layout):
 handles=[];projected=[];last={};trace=dict(vision_calls=0,projector_calls=0,image_embedding_exact=False,
  norm_calls=[0]*32,post_norm_calls=[0]*32,mlp_calls=[0]*32,projection_calls={},projection_shapes=[],skip_residual_exact_checks=0)
 def vision(m,args,out):trace['vision_calls']+=1
 def projector(m,args,out):
  trace['projector_calls']+=1;assert list(out.shape)==[1,576,4096];projected.append(out.detach())
 def language(m,args,kw):
  if projected:
   assert torch.equal(kw['inputs_embeds'][0,list(layout.image)],projected.pop().reshape(576,4096))
   trace['image_embedding_exact']=True
 handles.append(a.model.vision_tower.register_forward_hook(vision));handles.append(a.model.multi_modal_projector.register_forward_hook(projector));handles.append(a.model.language_model.register_forward_pre_hook(language,with_kwargs=True))
 for i,layer in enumerate(a.layers):
  def before(m,args,kw,i=i):
   if i in (2,3,4,5):last[i]=args[0][:,-1].detach().clone()
  def norm(m,args,out,i=i):trace['norm_calls'][i]+=1
  def post(m,args,i=i):
   trace['post_norm_calls'][i]+=1
   if i in (2,3,4,5):assert torch.equal(args[0][:,-1],last.pop(i));trace['skip_residual_exact_checks']+=1
  def mlp(m,args,out,i=i):trace['mlp_calls'][i]+=1
  handles.extend([layer.register_forward_pre_hook(before,with_kwargs=True),layer.input_layernorm.register_forward_hook(norm),layer.post_attention_layernorm.register_forward_pre_hook(post),layer.mlp.register_forward_hook(mlp)])
  for op in ('q_proj','k_proj','v_proj','o_proj'):
   def projection(m,args,out,i=i,op=op):
    key=f'{i}:{op}';trace['projection_calls'][key]=trace['projection_calls'].get(key,0)+1
    if i in (2,3,4,5):trace['projection_shapes'].append(dict(layer=i,op=op,input_rows=args[0].shape[1]))
   handles.append(getattr(layer.self_attn,op).register_forward_hook(projection))
 try:yield trace
 finally:
  for h in handles:h.remove()
  projected.clear();last.clear()

def structural(control,layout,trace,steps=3):
 expected=[576]*3+[288]*14+[144]*3+[0]*12
 assert [r['visual_hidden'] for r in control.layer_events if r['prefill']]==expected
 assert [e['layer'] for e in control.selection_events]==[3,17]
 assert [e['visual_after'] for e in control.selection_events]==[288,144]
 assert trace['vision_calls']==trace['projector_calls']==1 and trace['image_embedding_exact']
 assert trace['norm_calls']==trace['post_norm_calls']==trace['mlp_calls']==[steps]*32
 assert trace['skip_residual_exact_checks']==4*steps
 for i in (2,3,4,5):
  assert all(trace['projection_calls'][f'{i}:{op}']==1 for op in ('q_proj','k_proj','v_proj','o_proj'))
  shapes={r['op']:r['input_rows'] for r in trace['projection_shapes'] if r['layer']==i}
  rows=len(layout.text)+expected[i]
  assert shapes==dict(q_proj=rows-1,k_proj=rows,v_proj=rows,o_proj=rows-1)
 assert all(r['answer_image_mass']==0 for r in control.visual_events)
 assert all(r['image_keys']==0 for r in control.visual_events if not r['prefill'] or r['layer']>=20)
 assert all(r['relay_image_mass']>0 for r in control.visual_events if r['prefill'] and r['layer']<20)
 kv=control.cache.audit();assert kv['retained_image_layers']==[] and kv['historical_cache_copies']==kv['append_concatenations']==0
 for r in kv['layers']:
  assert r['original_positions']==([] if r['layer'] in (2,3,4,5) else list(layout.text)+list(range(layout.length,layout.length+steps-1)))
 return kv

def main():
 guard=(O/'run.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'failure.json').exists(),'A previous critical failure must not be retried automatically'
 p=read(V/'protocol.json');audit=read(O/'baseline_audit.json');assert audit['status']=='PASS' and audit['archived_records_audited']==600
 assert read(O/'cpu_checks.json')['status']=='PASS'
 assert sha(O/'input_manifest.jsonl.gz')==p['manifest_sha256']
 for name,h in p['frozen_ATE_sources'].items():assert sha(R/name)==h,name
 assert cmd('git','rev-parse','HEAD')==p['code_commit'] and not cmd('git','status','--porcelain','--untracked-files=no')
 with gzip.open(O/'input_manifest.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
 bykey={s['key']:s for s in rows};assert len(rows)==len(bykey)==200
 root=O/'scorer_source/llava/eval';root.mkdir(parents=True,exist_ok=True)
 raw=(V/'frozen_scoring_sources/m4c_evaluator.py').read_bytes();dest=root/'m4c_evaluator.py'
 if dest.exists():assert dest.read_bytes()==raw
 else:dest.write_bytes(raw)
 scorer.AUTHOR=O/'scorer_source'
 if hasattr(scorer.text_evaluator,'cache_clear'):scorer.text_evaluator.cache_clear()
 scorer.text_evaluator()
 config=dict(condition='V2Drop-prefix+SVH',model=p['model'],revision=p['revision'],model_path=str(M),code_commit=p['code_commit'],
  dtype='float16',attention='eager',greedy=True,num_beams=1,batch_size=1,EOS=2,max_new_tokens=dict(GQA=32,TextVQA=128),
  manifest_sha256=p['manifest_sha256'],baseline_audit_sha256=sha(O/'baseline_audit.json'),
  sources={n:sha(O/n) for n in ['fusion_engine.py','run_fusion.py','cpu_checks.py']},frozen_sources=p['frozen_ATE_sources'],official_selector=SELECTOR_PROVENANCE,
  semantics=dict(pruning_layers=[3,17],visual_hidden_counts=[576,288,144,0],visual_exit_before=20,answer_image_access=[],answer_attention_skip=[2,3,4,5],persistent_original_visual_KV=0,third_pruning_L22=False),
  smoke_keys=p['smoke_keys'],scorer_source_sha256=sha(dest),strict_gate='unchanged physical_visual_handoff.engine.equivalence llava tolerances',no_efficiency_experiment=True)
 if (O/'protocol.json').exists():assert read(O/'protocol.json')==config
 else:save('protocol.json',config)
 binding=digest(config)
 db=sqlite3.connect(O/'predictions.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 db.execute('CREATE TABLE IF NOT EXISTS predictions(key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)');db.commit()
 existing={}
 for key,payload,h in db.execute('SELECT key,payload,sha256 FROM predictions'):
  r=json.loads(payload);assert digest(r)==h and r['binding']==binding and r['sample_sha256']==bykey[key]['sample_sha256'];existing[key]=r
 state=dict(status='RUNNING',phase='GPU_LOAD',at=now(),pid=os.getpid(),model_loads=1,completed=len(existing),expected=200,protocol_binding=binding,physical_smoke='NOT_VALIDATED')
 a=None;ctl=None;smokes=[]
 try:
  assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'),'GPU occupied'
  assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
  assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090' and torch.cuda.mem_get_info()[0]>20*2**30
  state['GPU']=cmd('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.free','--format=csv,noheader');save('progress.json',state)
  loader.MODELS=dict(loader.MODELS,llava=M);print('MODEL_LOAD_ONCE',flush=True);a=frozen.load('B')
  assert {str(v.device) for v in a.model.parameters()}=={'cuda:0'} and {v.dtype for v in a.model.parameters()}=={torch.float16}
  assert a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager'
  assert a.model.generation_config.eos_token_id==2
  state['environment']=dict(torch=torch.__version__,transformers=transformers.__version__,cuda_build=torch.version.cuda);state['phase']='PHYSICAL_SMOKE';save('progress.json',state)
  pixels={}
  def prepare(s):
   if s['image_sha256'] not in pixels:
    assert sha(s['image_path'])==s['image_sha256']
    pixels[s['image_sha256']]=frozen.paper_pixels(s['image_path'],a.image_processor).half()
   x=dict(input_ids=torch.tensor([s['input_ids']],dtype=torch.long),attention_mask=torch.ones((1,len(s['input_ids'])),dtype=torch.long),pixel_values=pixels[s['image_sha256']])
   assert {k:tensor_hash(v) for k,v in x.items()}==s['input_hashes'];a.inputs={k:v.to('cuda:0') for k,v in x.items()}
   image=tuple(i for i,t in enumerate(s['input_ids']) if t==32000);text=tuple(i for i,t in enumerate(s['input_ids']) if t!=32000);relay=tuple(range(image[-1]+1,len(s['input_ids'])-1))
   return Layout(len(s['input_ids']),image,text,relay,relay,20)
  for key in p['smoke_keys']:
   s=bykey[key];layout=prepare(s);case=dict(key=key,status='RUNNING',image_sha256=s['image_sha256']);smokes.append(case);save('physical_smoke.json',dict(status='RUNNING',cases=smokes))
   physical=PrefixSVH(a,layout,3,enabled=True,audit=True)
   with trace_execution(a,layout) as trace:pg,pl=greedy(a,layout,physical,3,True,True)
   kv=structural(physical,layout,trace);case.update(structural='PASS',generation=pg,selection_events=physical.selection_events,layer_events=physical.layer_events,visual_access=physical.visual_events,attention_audit=physical.events,KV_positions=kv,execution_trace=trace)
   dense=DenseOracle(a,layout,3,physical.selection_events);dg,dl=greedy(a,layout,dense,3,True,True)
   gate=equivalence('llava',dl,pl,dense.hidden,physical.hidden);case['strict_numerical_gate']=gate;case['dense_reference_token_ids']=dg['token_ids']
   save('physical_smoke.json',dict(status='RUNNING',cases=smokes))
   assert gate['passed'],'Frozen numerical gate failed: '+json.dumps(gate)
   assert pg['token_ids']==dg['token_ids'],'Short physical/dense generation differs'
   disabled=PrefixSVH(a,layout,3,enabled=False,audit=True);ug,ul=greedy(a,layout,disabled,3,True,True)
   reference=MiddleWriter(a,layout,(),3,True);rg,rl=greedy(a,layout,reference,3,True,True)
   assert ug['token_ids']==rg['token_ids'] and all(torch.equal(x,y) for x,y in zip(ul,rl)),'Disabled-pruning SVH logits differ'
   assert all(torch.equal(x,y) for x,y in zip(disabled.hidden,reference.hidden)),'Disabled-pruning SVH hidden differs'
   case.update(status='PASS',disabled_pruning_SVH_logits_and_hidden_exact=True,scoring=score(s,pg['text']),real_prefill_decode_steps=3)
   save('physical_smoke.json',dict(status='RUNNING',cases=smokes));print('PHYSICAL_SMOKE_PASS',key,gate,flush=True)
   del physical,dense,disabled,reference,pl,dl,ul,rl;gc.collect()
  assert len(smokes)==4
  save('physical_smoke.json',dict(status='PASS',cases=smokes,strict_original_thresholds_unchanged=True,only_short_dense_references=True))
  state['physical_smoke']='PASS';state['phase']='FUSION_ACCURACY';save('progress.json',state)
  for s in rows:
   if s['key'] in existing:continue
   assert not (O/'STOP_REQUESTED').exists(),'Explicit stop request'
   layout=prepare(s);ctl=PrefixSVH(a,layout,s['max_new_tokens'],enabled=True,audit=False)
   generation,_=greedy(a,layout,ctl,s['max_new_tokens']);scoring=score(s,generation['text'])
   record=dict(key=s['key'],benchmark=s['benchmark'],question_id=s['question_id'],condition=config['condition'],binding=binding,
    sample_sha256=s['sample_sha256'],image_sha256=s['image_sha256'],prompt_sha256=s['prompt_sha256'],input_hashes=s['input_hashes'],gold=s['answer'],generation=generation,scoring=scoring,
    pruning_events=ctl.selection_events,at=now())
   payload=json.dumps(record,ensure_ascii=False,separators=(',',':'))
   with db:db.execute('INSERT INTO predictions VALUES(?,?,?)',(s['key'],payload,digest(record)))
   existing[s['key']]=record;ctl=None;a.inputs={};state.update(completed=len(existing),last_key=s['key'],at=now());save('progress.json',state)
   if len(existing)%10==0:print('FUSION_ACCURACY',len(existing),'/200',flush=True)
  assert len(existing)==200
  state.update(status='COMPLETE',phase='ACCURACY_DONE',at=now());save('progress.json',state)
 except BaseException as e:
  state.update(status='BLOCKED' if state['phase']!='FUSION_ACCURACY' else 'PARTIAL',at=now(),error=repr(e))
  if state['phase']=='PHYSICAL_SMOKE':state['physical_smoke']='FAIL';save('physical_smoke.json',dict(status='FAIL',cases=smokes,error=repr(e),strict_original_thresholds_unchanged=True))
  save('failure.json',dict(phase=state['phase'],error=repr(e),traceback=traceback.format_exc()[-12000:],at=now()));print('STOPPED',state['phase'],traceback.format_exc(),flush=True)
 finally:
  with (O/'predictions.jsonl').open('w') as out:
   for s in rows:
    if s['key'] in existing:out.write(json.dumps(existing[s['key']],ensure_ascii=False,separators=(',',':'))+'\n')
  db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
  ctl=None;a=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  state['process_exit_releases_CUDA_context']=True;save('progress.json',state);print('FINAL_STATUS',state['status'],state['completed'],'/200',flush=True)

if __name__=='__main__':main()
