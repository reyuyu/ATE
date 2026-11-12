from pathlib import Path
import ast,json,hashlib,types
import torch
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113');B=A/'opt_b';C=A/'opt_c'
assert not torch.cuda.is_initialized()
tests=[]
def tree_defs(path,names,env):
 t=ast.parse(path.read_text());nodes=[n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 exec(compile(ast.Module(nodes,type_ignores=[]),str(path),'exec'),env);return env
# Actual B mask routine extracted without importing model modules.
env={'torch':torch,'original':types.SimpleNamespace(SKIP=tuple(range(1,7)))}
t=ast.parse((B/'qwen3vl_svhr_search/physical.py').read_text());cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Physical')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='masks_for')
exec(compile(ast.Module([fn],type_ignores=[]),'mask_actual','exec'),env)
checks=0
for n,v,left in [(12,4,2),(226,192,4),(709,672,4),(1298,1024,4)]:
 image=list(range(left,left+v));text=[p for p in range(n) if p not in set(image)]
 f=types.SimpleNamespace(n=n,I=image,_text_cpu=text,read_layers=frozenset(range(19,36)),_masks={},_decode_masks={})
 f.cache=types.SimpleNamespace(positions=[list(range(n)) if i in f.read_layers else text for i in range(36)])
 for step in [0,1,8,31]:
  f.step=step
  for i in range(36):
   got=env['masks_for'](f,torch.empty(1,dtype=torch.bfloat16),step==0,i)
   if step and i in range(1,7):assert got is None;continue
   qp=(text if i>=22 else list(range(n))) if not step else [n+step-1]
   kp=(list(range(n)) if i<22 or i in f.read_layers else text) if not step else f.cache.positions[i]+list(range(n,n+step))
   q=torch.tensor(qp);k=torch.tensor(kp)
   ref=torch.zeros(1,1,len(qp),len(kp),dtype=torch.bfloat16);ref.masked_fill_((k[None,:]>q[:,None])[None,None],torch.finfo(torch.bfloat16).min)
   ik=torch.tensor([p in set(image) for p in kp])
   if i not in f.read_layers:ref[0,0,(q>=n-1)[:,None]&ik[None,:]]=-torch.inf
   if step==0 and i>=22:ref[0,0,((q>image[-1])&(q<n-1))[:,None]&ik[None,:]]=-torch.inf
   assert torch.equal(got,ref),(n,step,i)
   assert torch.isfinite(got).any(-1).all()
   checks+=1
tests.append(dict(name='Qwen_B_full_allowed_mask_value_equivalence',checks=checks,status='PASS'))
# Qwen real C cache methods against independent growing dense reference.
qenv={'torch':torch};tree_defs(C/'qwen3vl_c3v2_physical/engine.py',['CompactKV'],qenv)
qenv['original']=types.SimpleNamespace(CompactKV=qenv['CompactKV'],SKIP=tuple(range(1,7)))
tree_defs(C/'qwen3vl_svhr_search/physical.py',['SelectedKV'],qenv)
n=12;text=torch.tensor([0,1,6,7,8,9,10,11]);im=torch.tensor([2,3,4,5]);selected=set(range(19,36))
late={i:(torch.randn(1,8,4,4),torch.randn(1,8,4,4)) for i in range(22,36)}
expected_late={i:tuple(x.clone() for x in p) for i,p in late.items()}
cache=qenv['SelectedKV'](n,text,31,selected,late,text_cpu=text.tolist());refs={};ptrs={}
for i in range(36):
 if i in range(1,7):continue
 length=len(text) if i>=22 else n
 k,v=[torch.randn(1,8,length,4) for _ in range(2)]
 ret=cache.update(k,v,i)
 if i in selected and i>=22:
  full=[torch.empty(1,8,n,4) for _ in range(2)]
  for out,txt,pic in zip(full,(k,v),expected_late[i]):out[:,:,text]=txt;out[:,:,im]=pic
  assert all(torch.equal(a,b) for a,b in zip(ret,full));refs[i]=full
 else:
  assert torch.equal(ret[0],k) and torch.equal(ret[1],v)
  refs[i]=[x.clone() if i in selected else x.index_select(-2,text) if length==n else x.clone() for x in (k,v)]
 ptrs[i]=[x.data_ptr() for x in cache.entries[i]]
assert not cache.late
cache.prefill=False
for step in range(1,32):
 for i in refs:
  k,v=[torch.randn(1,8,1,4) for _ in range(2)];ret=cache.update(k,v,i)
  refs[i]=[torch.cat((x,y),-2) for x,y in zip(refs[i],(k,v))]
  assert all(torch.equal(a,b) for a,b in zip(ret,refs[i]))
  assert [x.data_ptr() for x in cache.entries[i]]==ptrs[i]
  assert cache.used[i]==refs[i][0].shape[-2] and cache.used[i]<=cache.capacities[i]
try:cache.update(k,v,0);raise AssertionError('capacity overflow accepted')
except AssertionError as e:
 if str(e)=='capacity overflow accepted':raise
tests.append(dict(name='Qwen_C_real_KV_values_growth_capacity_pointers_late_consumption',layers=len(refs),decode_steps=31,status='PASS'))
# InternVL C use actual methods with a CPU stub base Cache.
class Stub:
 def __init__(self):pass
ienv={'torch':torch,'DynamicCache':Stub,'SKIP':set(range(2,6))}
tree_defs(C/'internvl3_svh_physical_d_pilot/engine.py',['text_positions','CompactCache'],ienv)
roles={'n':270,'image':list(range(4,260))}
cache=ienv['CompactCache'](roles,device='cpu');ref={};ptrs={}
for i in range(28):
 length=270 if i<20 else 14;k,v=[torch.randn(1,2,length,4) for _ in range(2)]
 returned=cache.update(k,v,i);assert torch.equal(returned[0],k)
 if i in range(2,6):assert cache.key_cache[i] is None;continue
 ref[i]=[x.index_select(-2,cache.keep) if i<20 else x for x in (k,v)];ptrs[i]=[x.data_ptr() for x in cache.buffers[i]]
cache.prefill=False
for step in range(1,32):
 for i in ref:
  k,v=[torch.randn(1,2,1,4) for _ in range(2)];ret=cache.update(k,v,i)
  ref[i]=[torch.cat((x,y),-2) for x,y in zip(ref[i],(k,v))]
  assert all(torch.equal(a,b) for a,b in zip(ret,ref[i]))
  assert ptrs[i]==[x.data_ptr() for x in cache.buffers[i]]
tests.append(dict(name='InternVL_C_actual_buffer_values_vs_cat_and_pointer_stability',layers=len(ref),decode_steps=31,status='PASS'))
assert not torch.cuda.is_initialized()
out=dict(status='PASS_CPU_BEFORE_GPU',tests=tests,CUDA_initialized=False,
 scope='Exact production mask/KV functions extracted via AST; mock tensors verify semantics independently of GPU timing; model numerical checks still required.')
(A/'cpu_tests.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
