"""Natural-EOS accuracy only; inherited frozen HF greedy path and scorers."""
from s55_common import *
import gc,sqlite3,traceback,statistics,fcntl
from contextlib import contextmanager,ExitStack
import torch
from vlm_research.information_horizon_alignment import engine as q,score as scorer
from vlm_research.information_horizon_alignment.common import tensor_sha
from vlm_research.physical_visual_handoff.engine import equivalence
from s55_ablation_engine import generate

@contextmanager
def trace(a,layout):
 rows=[];handles=[];step=[-1]
 def first(m,args,kw):step[0]+=1
 handles.append(a.layers[0].register_forward_pre_hook(first,with_kwargs=True))
 for i,l in enumerate(a.layers):
  def layer(m,args,kw,i=i):
   pos=kw['position_ids'][0];rot=a.model.language_model.model.rotary_emb(args[0],kw['position_ids'])
   assert all(torch.equal(x,y) for x,y in zip(rot,kw['position_embeddings']))
   rows.append(dict(step=step[0],layer=i,op='layer',rows=int(args[0].shape[-2]),positions=pos.tolist(),rope_exact=True))
  handles.append(l.register_forward_pre_hook(layer,with_kwargs=True))
  for op in ('q','k','v','o'):
   handles.append(getattr(l.self_attn,op+'_proj').register_forward_pre_hook(lambda m,args,i=i,op=op:rows.append(dict(step=step[0],layer=i,op=op,rows=int(args[0].shape[-2])))))
 try:yield rows
 finally:
  for h in handles:h.remove()

def check_trace(rows,layout,condition,steps):
 layer=[r for r in rows if r['op']=='layer'];assert len(layer)==32*steps
 for r in layer:
  s,i=r['step'],r['layer'];n=layout.length
  expected=(list(range(n)) if condition in ('Native','wo_selective') or i<20 else list(layout.text)) if s==0 else [n+s-1]
  assert r['positions']==expected and r['rows']==len(expected),(condition,r)
  skip=condition in ('SVH','wo_selective') and i in (2,3,4,5)
  for op in ('q','k','v','o'):
   got=sum(x['rows'] for x in rows if x['step']==s and x['layer']==i and x['op']==op)
   want=(n-1 if op in ('q','o') else n) if skip and s==0 else 0 if skip else len(expected)
   assert got==want,(condition,s,i,op,got,want)

def smoke(a,inputs):
 details=[];gold=read(O/'ablation/answers.json');refs={r['key']:r for r in lines(O/'ablation/reused_svh.jsonl.gz')}
 for bench in ('ScienceQA-IMG','POPE'):
  s=next(x for x in inputs if x['benchmark']==bench);layout,binding=q.prepare(a,s)
  before={k:tensor_sha(v) for k,v in a.inputs.items()}
  native,nlog=generate(a,layout,'Native',4,capture=True)
  noop,zlog=generate(a,layout,'Native',4,capture=True,noop=True)
  assert native['token_ids']==noop['token_ids'] and len(nlog)==len(zlog)
  assert all(torch.equal(x,y) for x,y in zip(nlog,zlog)),'Same GPU Native/Noop logits must be exact'
  details.append(dict(key=s['key'],check='NativeNoop',status='PASS',token_ids=native['token_ids'],all_logits_bitwise_exact=True))
  for condition in ('SVH','wo_selective','wo_bypass'):
   with trace(a,layout) as tr:actual,al=generate(a,layout,condition,4,capture=True,audit=True)
   dense,dl=generate(a,layout,condition,4,capture=True,audit=True,dense=True)
   check_trace(tr,layout,condition,len(actual['token_ids']))
   assert actual['token_ids']==dense['token_ids'],(condition,'Physical/Dense tokens differ')
   gate=equivalence('llava',dl,al,dense.pop('_hidden'),actual.pop('_hidden'));assert gate['passed'],gate
   ar=actual.pop('_rotary');dr=dense.pop('_rotary')
   # Original position IDs after early exit differ in row count by construction;
   # retained text RoPE and hidden states are independently checked above.
   kv=actual['kv_audit'];n=layout.length;steps=len(actual['token_ids'])
   for row in kv['layers']:
    absent=condition!='wo_bypass' and row['layer'] in (2,3,4,5)
    assert row['used_tokens']==(0 if absent else n-576+steps-1)
    if not absent:assert row['original_positions']==list(layout.text)+list(range(n,n+steps-1))
   assert kv['live_bytes']<=kv['storage_bytes']
   assert all(x['blocked_probability_max']==0 for x in actual['attention_audit'])
   assert all(x['answer_image_mass']==0 for x in actual['visual_audit'])
   if condition=='SVH':
    assert actual['token_ids']==refs[s['key']]['generation']['token_ids'][:steps],'Reused SVH smoke output mismatch'
   details.append(dict(key=s['key'],condition=condition,status='PASS',token_ids=actual['token_ids'],kv_audit=kv,physical_trace=tr,visual_audit=actual['visual_audit'],numerical_equivalence=gate))
   del actual,dense,al,dl,ar,dr
  assert before=={k:tensor_sha(v) for k,v in a.inputs.items()}
  a.inputs={};gc.collect();torch.cuda.empty_cache()
 save(O/'ablation/smoke_validation.json',dict(status='PASS',cases=details))
 print('ABLATION_PHYSICAL_NUMERICAL_NOOP_SMOKE_PASS',flush=True)

def main():
 root=O/'ablation';lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (root/'checkpoint.sqlite').exists(),'No automatic new run after a failed run'
 a=None;db=None;state=dict(status='RUNNING',phase='PRECHECK',pid=os.getpid(),completed=0,expected=43708,new_completed=0,new_expected=32781,started_ns=time.time_ns())
 inputs=lines(root/'inputs.jsonl.gz');gold=read(root/'answers.json');bindings={r['key']:r for r in lines(root/'input_verification.jsonl.gz')}
 def status():save(root/'progress.json',dict(state,updated_ns=time.time_ns()))
 try:
  status();a=load_model();state['phase']='SMOKE';status();smoke(a,inputs);verify_sources()
  save(root/'environment.json',dict(GPU=gpu(),GPU_detail=command(['nvidia-smi','-q']),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,transformers=__import__('transformers').__version__,CPU_affinity=sorted(os.sched_getaffinity(0)),intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads(),model_receipt_sha256=sha(M/'.ate_revision.json'),protocol_sha256=sha(O/'protocol.json')))
  db=sqlite3.connect(root/'checkpoint.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE predictions(key TEXT,condition TEXT,payload TEXT,sha256 TEXT,PRIMARY KEY(key,condition))')
  def store(r):
   with db:db.execute('INSERT INTO predictions VALUES(?,?,?,?)',(r['key'],r['condition'],json.dumps(r,ensure_ascii=False),digest(r)))
   state['completed']+=1
  for r in lines(root/'reused_svh.jsonl.gz'):store(r)
  state['phase']='FORMAL';status();start=time.monotonic()
  with (root/'telemetry.jsonl').open('x') as telemetry:
   for index,s in enumerate(inputs):
    layout,binding=q.prepare(a,s);ref=bindings[s['key']]
    assert digest(binding['input_ids'])==ref['input_ids_sha256'] and digest(binding['prompt'])==ref['prompt_sha256'] and binding['pixel_values_sha256']==ref['pixel_values_sha256']
    methods=['Native','wo_selective','wo_bypass'];offset=index%3;methods=methods[offset:]+methods[:offset]
    for condition in methods:
     if state['new_completed']%30==0:
      telemetry.write(json.dumps(dict(at_ns=time.time_ns(),completed=state['new_completed'],GPU=guard()))+'\n');telemetry.flush()
     g,_=generate(a,layout,condition,s['max_new_tokens'])
     assert g['all_effective_logits_finite'] and g['independent_cache'] and g['logits_to_keep']==1
     sample=dict(s,answer=gold[s['key']]);scoring=scorer.score(sample,g['text'])
     r=dict(key=s['key'],benchmark=s['benchmark'],condition=condition,question_id=s['question_id'],strategy=s.get('strategy'),image_sha256=s['image_sha256'],input_binding=ref,generation=g,scoring=scoring,source='NEW_GPU_NATURAL_EOS',at_ns=time.time_ns())
     store(r);state['new_completed']+=1
    a.inputs={}
    if (index+1)%50==0 or index+1==len(inputs):
     status();print('ABLATION_PROGRESS',state['new_completed'],'/',state['new_expected'],s['key'],'elapsed_s',round(time.monotonic()-start,1),flush=True)
    if (index+1)%500==0:verify_sources()
  assert state['completed']==43708 and state['new_completed']==32781
  assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok';db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
  state.update(status='GENERATIONS_COMPLETE_PENDING_CPU_ANALYSIS',phase='COMPLETE',finished_ns=time.time_ns());status()
 except BaseException as e:
  state.update(status='BLOCKED' if state['phase']!='FORMAL' else 'PARTIAL_CORRECTNESS_OR_RUNTIME_FAILURE',error=repr(e),traceback=traceback.format_exc(),failed_ns=time.time_ns());status();print(traceback.format_exc(),flush=True);raise
 finally:
  if db:db.close()
  a=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  save(root/'resource_release.json',dict(at_ns=time.time_ns(),process=os.getpid(),allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0,context_released_on_process_exit=True))
if __name__=='__main__':main()
