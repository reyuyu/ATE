from pathlib import Path
import ast,json,hashlib,shutil,torch
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
assert not (A/'opt_e').exists()
shutil.copytree(A/'opt_b',A/'opt_e',ignore=shutil.ignore_patterns('__pycache__'))
p=A/'opt_e/qwen3vl_c3v2_numerical_repair2/repair.py';s=p.read_text()
i=s.index('def restore_dead_image_columns(');j=s.index('\nclass Repair2Execution',i)
s=s[:i]+'''def restore_dead_image_columns(k, v, text_positions, n, step, cached=None):
    if cached is None:
        positions=list(text_positions)+list(range(n,n+step))
        ix=torch.tensor(positions,device=k.device)
        mask=k.new_full((1,1,1,n+step),-torch.inf)
        mask.index_fill_(-1,ix,0.0)
    else:
        ix,mask=cached
    assert ix.numel()==k.shape[-2] and mask.shape[-1]==n+step
    full_k=k.new_zeros((*k.shape[:-2],n+step,k.shape[-1]))
    full_v=v.new_zeros((*v.shape[:-2],n+step,v.shape[-1]))
    full_k.index_copy_(-2,ix,k);full_v.index_copy_(-2,ix,v)
    return full_k,full_v,mask,ix
''' +s[j:]
s=s.replace('self.decode_padding = []','self.decode_padding = [];self.decode_column_metadata = {}')
s=s.replace('kp, vp, mp, ix = restore_dead_image_columns(k, v, self._text_cpu, self.n, self.step)','kp, vp, mp, ix = restore_dead_image_columns(k, v, self._text_cpu, self.n, self.step,self.decode_column_metadata.get(self.step))\n                if self.step not in self.decode_column_metadata:self.decode_column_metadata[self.step]=(ix,mp)')
p.write_text(s);compile(s,str(p),'exec')
# Corrected C implementation is separate from failed opt_c; no previous source modified.
p=A/'opt_e/internvl3_svh_physical_d_pilot/engine.py';s=p.read_text()
orig=(A/'opt_c/internvl3_svh_physical_d_pilot/engine.py').read_text();tree=ast.parse(orig);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CompactCache')
cls=ast.get_source_segment(orig,node)
old="""self.key_cache[i]=torch.cat((self.key_cache[i],k),dim=-2)
  self.value_cache[i]=torch.cat((self.value_cache[i],v),dim=-2)
  return self.key_cache[i],self.value_cache[i]"""
assert old in cls;cls=cls.replace(old,"return self.allocate_update(k,v,i,(k,v))")
tree=ast.parse(s);n=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CompactCache')
lines=s.splitlines(True);s=''.join(lines[:n.lineno-1])+cls+'\n'+''.join(lines[n.end_lineno:]);p.write_text(s);compile(s,str(p),'exec')
# CPU-only production-function checks; no model import or CUDA initialization.
source=(A/'opt_e/qwen3vl_c3v2_numerical_repair2/repair.py').read_text();tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='restore_dead_image_columns')
g={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'production_E_mask','exec'),g);new=g[fn.name]
source=(A/'opt_b/qwen3vl_c3v2_numerical_repair2/repair.py').read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='restore_dead_image_columns')
g={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_B_mask','exec'),g);oldfn=g[fn.name]
checks=0
for n,im in [(8,2),(17,5),(36,19),(96,40)]:
 text=[i for i in range(n) if i not in range(1,1+im)]
 for step in range(1,32):
  k=torch.randn(1,2,len(text)+step,4);v=torch.randn_like(k)
  ref=oldfn(k,v,text,n,step);first=new(k,v,text,n,step);cached=new(k,v,text,n,step,(first[3],first[2]))
  assert all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(ref,first,cached));assert cached[2] is first[2] and cached[3] is first[3];checks+=1
# C2 real cache methods vs independent concatenation, all28 layers.
source=(A/'opt_e/internvl3_svh_physical_d_pilot/engine.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CompactCache')
roles={'n':7,'image':[1,2]};skip=set(range(2,6));text=[0,3,4,5,6]
g={'torch':torch,'DynamicCache':object,'SKIP':skip,'text_positions':lambda roles,device:torch.tensor(text,device=device)}
exec(compile(ast.Module(body=[node],type_ignores=[]),'production_E_cache','exec'),g);cache=g['CompactCache'](roles,'cpu');refs={};ptr={}
for i in range(28):
 if i in skip:continue
 k=torch.randn(1,2,7 if i<20 else 5,4);v=torch.randn_like(k);cache.update(k,v,i)
 refs[i]=tuple(x.index_select(-2,torch.tensor(text)) if i<20 else x for x in (k,v));ptr[i]=cache.buffers[i][0].data_ptr()
cache.prefill=False
for step in range(1,32):
 for i in refs:
  k=torch.randn(1,2,1,4);v=torch.randn_like(k);pair=cache.update(k,v,i);refs[i]=tuple(torch.cat((a,b),-2) for a,b in zip(refs[i],(k,v)))
  assert all(torch.equal(a,b) for a,b in zip(pair,refs[i]));assert cache.used[i]==5+step and ptr[i]==cache.buffers[i][0].data_ptr()
assert not torch.cuda.is_initialized()
(A/'refinement_cpu_tests.json').write_text(json.dumps(dict(status='PASS_BEFORE_REFINEMENT_GPU',Qwen_actual_mask_value_and_reuse_cases=checks,Intern_actual_cache_layers=24,decode_steps=31,CUDA_initialized=False,failed_C_preserved=True),indent=2)+'\n')
(A/'protocol_refinement_addendum.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_REFINEMENT_GPU',reason='Profiler revealed repeated same-step Qwen SVH dead-column metadata H2D; original Intern SVH C rejected by used-length guard because Decode still cat. Independent E revision fixes these implementation paths only.',
 model_and_layer_parameters_unchanged=True,methods=['Qwen SVH','InternVL SVH'],stage='E(B2 or C2)',reference_stages=['A','B'],numerical_gate='All30 full32 logits torch.equal and full tokens exact, plus actual KV lifetimes and Audit Mode. No threshold relaxation.',
 candidate_attention_padding='Retained unchanged; failed D still excluded',screen='Same3 frozen length representatives,3 ordinary repeats; A/B/E with Native; use predeclared1percent usefulness rule and preserve all stages.',
 formal='Only if E benefits vs B; original30,2 independent sessions, Native/A/E same-input interleaved, no historical Native times',no_full_quality_or_git=True),indent=2)+'\n')
manifest=json.loads((A/'optimized_source_manifest.json').read_text())
for p in sorted((A/'opt_e').rglob('*.py')):manifest.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(A/'optimized_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(status='E_BUILT_CPU_PASS',mask_cases=checks,cache_layers=24,steps=31,source_files=len(list((A/'opt_e').rglob('*.py'))))))
