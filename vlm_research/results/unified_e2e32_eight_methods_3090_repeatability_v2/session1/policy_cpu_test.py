"""CPU regressions for Native first-EOS acceptance; immutable prior records."""
from pathlib import Path
import ast,hashlib,json,gzip,sqlite3,difflib
from history_policy import compare_history
P=Path(__file__).parent;R=Path('/home/featurize/work/ATE');OLD=R/'vlm_research/results/unified_e2e32_eight_methods_3090_repeatability';U=OLD.parent/'unified_e2e32_4090'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=(OLD/'session1/code/run_3090.py').read_text();new=(P/'run_3090.py').read_text()
def function(text,name):return next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name)
for name in ['request','clean','guard','verify_sources']:
 assert ast.dump(function(old,name),include_attributes=False)==ast.dump(function(new,name),include_attributes=False),name
def nooploops(text):
 return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.For) and any(isinstance(x,ast.Constant) and x.value in ('BridgeNative','FastVNoop') for x in ast.walk(n)) and not any(isinstance(x,ast.FunctionDef) for x in ast.walk(n))]
assert nooploops(old)==nooploops(new),'Same-GPU No-op loops changed'
assert 'range(40)' in new and 'range(16)' not in new and 'range(120)' not in new
assert 'Native calibration drift stop' not in new and 'Abrupt within-request Native decode transition' not in new
assert sha(U/'code/e2e32_runtime.py')=='513a122ff5f4da008b74e4d5560b0b6ef50baf8ccf7feb2fa766a0cdcdc51a3b'
cases=[]
def check(name,method,current,ref,accepted):
 before=(list(current),None if ref is None else list(ref));v=compare_history(method,current,ref,2)
 assert v['accepted'] is accepted,(name,v);assert (current,ref)==before,'Policy mutated generation'
 cases.append(dict(name=name,accepted=v['accepted'],rule=v['rule']))
for method in ['Native','FastV']:
 base=[1939,2]+[13]*30;post=base.copy();post[5]=99;pre=base.copy();pre[0]=99;moved=base.copy();moved[1]=99;moved[2]=2;no=[13]*32;nodiff=no.copy();nodiff[31]=99
 check(method+' exact',method,base,base,True)
 check(method+' only post EOS different',method,post,base,True)
 check(method+' pre EOS different rejected',method,pre,base,False)
 check(method+' EOS displaced rejected',method,moved,base,False)
 check(method+' one EOS missing rejected',method,no,base,False)
 check(method+' neither EOS exact',method,no,no,True)
 check(method+' neither EOS differing rejected',method,nodiff,no,False)
 check(method+' first token EOS',method,[2]+[99]*31,[2]+[13]*31,True)
 check(method+' missing history rejected',method,no,None,False)
check('Other historical output diagnostic only','SVH',post,base,True)
check('BridgeNative has strict manual same-GPU checks','BridgeNative',base,None,True)
db=sqlite3.connect('file:'+str(OLD/'session1/checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True)
failed=json.loads(db.execute('SELECT payload FROM records WHERE category="formal" ORDER BY id DESC LIMIT 1').fetchone()[0]);db.close()
assert failed['formal_index']==151 and failed['key']=='TextVQA:38508' and failed['method']=='Native'
assert read(OLD/'session1/completion.json')['status']=='PARTIAL' and not failed['history_acceptance']['accepted']
with gzip.open(U/'formal_requests.jsonl.gz','rt') as f:history=next(r for r in map(json.loads,f) if r['key']==failed['key'] and r['method']=='Native')
evidence=compare_history('Native',failed['generation']['token_ids'],history['generation']['token_ids'],2)
assert evidence['accepted'] and evidence['post_EOS_only_difference'] and evidence['first_difference_0based']==11 and evidence['current_EOS_positions_0based'][0]==evidence['historical_EOS_positions_0based'][0]==3
assert failed['input_hashes']==history['input_hashes'] and failed['sample_sha256']==history['sample_sha256']
fields=['layer','used_tokens','capacity_tokens','visual_tokens','live_bytes','storage_bytes','K_shape','V_shape','dtype','original_positions_sha256']
assert all({k:x.get(k) for k in fields}=={k:y.get(k) for k in fields} for x,y in zip(failed['generation']['cache']['layers'],history['generation']['cache']['layers']))
saved=read(OLD/'session1/protocol.json');assert sha(OLD/'session1/code/run_3090.py')==saved['entry_source_SHA256']['run_3090.py']
report=dict(status='PASS',GPU_requests=0,test_cases=cases,request_entire_AST_identical=True,same_GPU_noop_AST_identical=True,cleanup_and_safety_AST_identical=True,old151_record_remains_PARTIAL=True,old151_revised_policy_evidence=evidence,old_checkpoint_SHA256=sha(OLD/'session1/checkpoint.sqlite'),old_entry_SHA256=sha(OLD/'session1/code/run_3090.py'),new_entry_SHA256=sha(P/'run_3090.py'),frozen_runtime_SHA256=sha(U/'code/e2e32_runtime.py'),new_policy_SHA256=sha(P/'history_policy.py'),fixed_Native_warmup=40,other_warmup_each=2,performance_warn_only=True,scope='Only cross-GPU Native historical acceptance extends the existing FastV first-EOS rule. Same request function, metrics, generation, physical gates and same-GPU strict No-op loops. Fixed40 warmup and one extra Native correctness Smoke are declared outside the request function.')
(P/'patch_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='PASS',test_cases=len(cases),old151_accepted_only_after_EOS=True,entire_request_AST_identical=True,same_GPU_noop_AST_identical=True,frozen_runtime_unchanged=True)))
