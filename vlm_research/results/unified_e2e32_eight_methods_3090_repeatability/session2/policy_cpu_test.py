"""CPU-only policy and AST regression checks before the one authorized model load."""
from pathlib import Path
import ast,copy,hashlib,json,gzip,sqlite3
from history_policy import compare_history
P=Path(__file__).parent;R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/unified_e2e32_eight_methods_3090';U=O.parent/'unified_e2e32_4090'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
old=(O/'code/run_3090.py').read_text();new=(P/'run_3090.py').read_text()
old_request=next(n for n in ast.walk(ast.parse(old)) if isinstance(n,ast.FunctionDef) and n.name=='request')
new_request=next(n for n in ast.walk(ast.parse(new)) if isinstance(n,ast.FunctionDef) and n.name=='request')
def prefix(fn):
 out=[]
 for n in copy.deepcopy(fn.body):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ref' for t in n.targets):break
  # Session identity changes metadata only, after the frozen timed call returns.
  for x in ast.walk(n):
   if isinstance(x,ast.keyword) and x.arg=='session':x.value=ast.Constant(1)
  out.append(ast.dump(n,include_attributes=False))
 return out
assert prefix(old_request)==prefix(new_request),'Timed call, CPU preparation or measurement/finite/layout check altered'
for name in ['clean']:
 a=next(n for n in ast.walk(ast.parse(old)) if isinstance(n,ast.FunctionDef) and n.name==name)
 b=next(n for n in ast.walk(ast.parse(new)) if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
assert "for i in range(16)" in new and "range(120)" not in new
assert "Native calibration drift stop" not in new and "Independent baseline left admitted state" not in new and "Abrupt within-request Native decode transition" not in new
saved=read(O/'protocol.json');assert sha(O/'code/run_3090.py')==saved['entry_source_SHA256']['run_3090.py']
assert sha(U/'code/e2e32_runtime.py')=='513a122ff5f4da008b74e4d5560b0b6ef50baf8ccf7feb2fa766a0cdcdc51a3b'
assert read('/home/featurize/ate-local-models/llava-1.5-7b-hf/generation_config.json')['eos_token_id']==2
cases=[]
def case(label,method,c,r,ok):
 before=(list(c),list(r));result=compare_history(method,c,r,2);assert result['accepted'] is ok,label;assert (c,r)==before,'Policy mutated generation'
 cases.append(dict(name=label,accepted=result['accepted'],rule=result['rule']))
base=[1939,2]+[13]*30
case('exact FastV','FastV',base,base,True)
post=base.copy();post[5]=99
case('post EOS FastV allowed','FastV',post,base,True)
case('Native post EOS still rejected','Native',post,base,False)
pre=base.copy();pre[0]=99
case('pre EOS rejected','FastV',pre,base,False)
eos=base.copy();eos[1]=99;eos[2]=2
case('first EOS displaced rejected','FastV',eos,base,False)
case('one EOS missing rejected','FastV',[13]*32,base,False)
case('no EOS exact accepted','FastV',[13]*32,[13]*32,True)
no=[13]*32;no[31]=99
case('no EOS difference rejected','FastV',no,[13]*32,False)
case('other historical differences recorded only','SVH',post,base,True)
case('first token EOS tail allowed','FastV',[2]+[99]*31,[2]+[13]*31,True)
db=sqlite3.connect('file:'+str(O/'checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True)
records=[json.loads(p) for (p,) in db.execute('SELECT payload FROM records ORDER BY id')];db.close()
failed=records[-1];assert failed['formal_index']==31 and failed['key']=='GQA:201759069' and failed['method']=='FastV'
completion=read(O/'completion.json');assert completion['failure']=="AssertionError('Historical token mismatch: GQA:201759069 FastV')"
assert "assert exact is not False" in completion['traceback']
with gzip.open(U/'formal_requests.jsonl.gz','rt') as f:history=next(r for r in map(json.loads,f) if r['key']==failed['key'] and r['method']=='FastV')
g=failed['generation'];h=history['generation'];evidence=compare_history('FastV',g['token_ids'],h['token_ids'],2)
assert evidence['accepted'] and evidence['post_EOS_only_difference'] and evidence['first_difference_0based']==5 and evidence['current_EOS_positions_0based'][0]==1
fields=['layer','used_tokens','capacity_tokens','visual_tokens','live_bytes','storage_bytes','K_shape','V_shape','dtype']
assert all({k:x.get(k) for k in fields}=={k:y.get(k) for k in fields} for x,y in zip(g['cache']['layers'],h['cache']['layers']))
assert g['pruning_events']==h['pruning_events']
assert failed['input_hashes']==history['input_hashes'] and failed['sample_sha256']==history['sample_sha256']
report=dict(status='PASS',GPU_requests=0,test_cases=cases,old_stop_only_full32_history_assertion=True,old_failed_request_policy=evidence,original_completion_SHA256=sha(O/'completion.json'),original_checkpoint_SHA256=sha(O/'checkpoint.sqlite'),old_entry_SHA256=sha(O/'code/run_3090.py'),new_entry_SHA256=sha(P/'run_3090.py'),frozen_runtime_SHA256=sha(U/'code/e2e32_runtime.py'),request_prefix_AST_unchanged=True,clean_AST_unchanged=True,performance_gates_warn_only=True,measurements_and_generation_not_mutated=True,patch_scope='Output acceptance after generate returned, plus Session2 identity and one extra correctness Smoke. Frozen algorithms/timing unchanged.',old_prepared_proposal_replaced='Latest authorization: all pure performance gates warn; Native full32, FastV first-EOS and physical/No-op safety gates hard.',FastV_previous_KV_layout_and_pruning_exact=True)
(P/'patch_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='PASS',tests=len(cases),old_failed_request_now_accepted_only_after_EOS=True,first_EOS_0based=1,first_difference_0based=5,timed_call_AST_unchanged=True)))
