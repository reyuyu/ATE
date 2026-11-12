from pathlib import Path
import json,ast,shutil
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113');B=A/'opt_b';C=A/'opt_c';D=A/'opt_d'
helper=(A/'code/build_b.py').read_text().split('# Qwen fixed CPU layout')[0]
(A/'code/ast_helpers.py').write_text(helper)
exec(compile(helper,str(A/'code/ast_helpers.py'),'exec'))
# ApET only caches per-stage identical geometry; official FPS/merging/ranking unchanged.
p=A/'opt_b/apet_fusion.py'
p.write_text((Path('/home/featurize/ate-experiments/rtx3090-18conditions-20261113/vlm_research/results/rtx3090_unified_efficiency_18conditions/code/apet_fusion.py')).read_text())
s=p.read_text().replace('visual=288;pos=None;rope=None;mask=None','visual=288;pos=None;rope=None;mask=None;decode_geometry={}')
s=s.replace('self.cache.text_index=torch.cat(', 'if i in (0,16,20):self.cache.text_index=torch.cat(')
s=s.replace('pos=torch.tensor([logical],device=h.device);rope=decoder.rotary_emb(h,pos[None]);mask=None',
 'if logical not in decode_geometry:\n     pos=torch.tensor([logical],device=h.device);decode_geometry[logical]=(pos,decoder.rotary_emb(h,pos[None]))\n    pos,rope=decode_geometry[logical];mask=None')
s=s.replace('assert hh.shape[1]==h.shape[1]-192 and torch.equal(pp,torch.arange(hh.shape[1],device=h.device)[None])',
 'assert hh.shape[1]==h.shape[1]-192\n    if self.trace:assert torch.equal(pp,torch.arange(hh.shape[1],device=h.device)[None])')
p.write_text(s)
assert not C.exists();shutil.copytree(B,C)
# C Qwen: gather directly into final buffer; selected late KV direct assembly.
p=C/'qwen3vl_c3v2_physical/engine.py';s=p.read_text()
s=s.replace("stored=[t.index_select(-2,self.text) if t.shape[-2]==self.length else t for t in (k,v)]\n            n=stored[0].shape[-2]\n            pair=[t.new_empty((*t.shape[:-2],self.capacity,t.shape[-1])) for t in stored]\n            for out,t in zip(pair,stored):out[...,:n,:].copy_(t)",
 "n=len(self.text) if k.shape[-2]==self.length else k.shape[-2]\n            pair=[t.new_empty((*t.shape[:-2],self.capacity,t.shape[-1])) for t in (k,v)]\n            for out,t in zip(pair,(k,v)):\n                if t.shape[-2]==self.length:torch.index_select(t,-2,self.text,out=out[...,:n,:])\n                else:out[...,:n,:].copy_(t)")
p.write_text(s)
p=C/'qwen3vl_svhr_search/physical.py'
replace_method(p,'SelectedKV','update',r"""
def update(self,k,v,layer_idx,cache_kwargs=None):
    i=layer_idx
    assert i not in self.absent and k.shape==v.shape
    self.updates[i]+=1
    if self.prefill:
        n=len(self.positions[i]);capacity=self.capacities[i]
        pair=[t.new_empty((*t.shape[:-2],capacity,t.shape[-1])) for t in (k,v)]
        if i in self.selected and i>=22:
            assert k.shape[-2]==len(self.text)
            ik,iv=self.late.pop(i)
            image_ix=torch.tensor(self.image,device=k.device)
            for destination,now,visual in zip(pair,(k,v),(ik,iv)):
                view=destination[...,:n,:]
                view.index_copy_(-2,self.text,now);view.index_copy_(-2,image_ix,visual)
            returned=tuple(t[...,:n,:] for t in pair)
        else:
            returned=(k,v)
            for destination,now in zip(pair,(k,v)):
                view=destination[...,:n,:]
                if i in self.selected or now.shape[-2]!=self.length:view.copy_(now)
                else:torch.index_select(now,-2,self.text,out=view)
        self.entries[i]=pair;self.used[i]=n
        return returned
    assert k.shape[-2]==1 and self.entries[i] is not None
    n=self.used[i];assert n+1<=self.capacities[i]
    for destination,now in zip(self.entries[i],(k,v)):destination[...,n:n+1,:].copy_(now)
    self.used[i]+=1
    return tuple(t[...,:n+1,:] for t in self.entries[i])
""")
# C InternVL: fixed buffers, preserve original attention return geometry and scratch.
# Preallocated storage separate from valid prefix views; used/capacity never confused.
new=r"""
def allocate_update(self,k,v,i,stored):
    n=stored[0].shape[-2]
    if self.buffers[i] is None:
        cap=n+31
        pair=[x.new_empty((*x.shape[:-2],cap,x.shape[-1])) for x in stored]
        for out,x in zip(pair,stored):out[...,:n,:].copy_(x)
        self.buffers[i]=pair;self.used[i]=n;self.capacities[i]=cap
    else:
        n=self.used[i];m=k.shape[-2];assert n+m<=self.capacities[i]
        for out,x in zip(self.buffers[i],(k,v)):out[...,n:n+m,:].copy_(x)
        self.used[i]+=m
    n=self.used[i]
    self.key_cache[i]=self.buffers[i][0][...,:n,:]
    self.value_cache[i]=self.buffers[i][1][...,:n,:]
    return self.key_cache[i],self.value_cache[i]
"""
p=C/'internvl3_svh_physical_d_pilot/engine.py';s=p.read_text()
s=s.replace('self.prefill=True\n', 'self.prefill=True;self.buffers=[None]*28;self.used=[0]*28;self.capacities=[0]*28\n',1)
s=s.replace("self.key_cache[i]=k.index_select(-2,self.keep) if i<20 else k.contiguous()\n    self.value_cache[i]=v.index_select(-2,self.keep) if i<20 else v.contiguous()",
 "stored=tuple(x.index_select(-2,self.keep) if i<20 else x for x in (k,v))\n    self.allocate_update(k,v,i,stored)")
s=s.replace("self.key_cache[i]=torch.cat((self.key_cache[i],k),dim=-2)\n   self.value_cache[i]=torch.cat((self.value_cache[i],v),dim=-2)\n   return self.key_cache[i],self.value_cache[i]",
 "return self.allocate_update(k,v,i,(k,v))")
p.write_text(s);append_methods(p,'CompactCache',new)
p=C/'internvl3_svh_r_terminal_windows_textvqa/engine.py';s=p.read_text()
s=s.replace('self.key_cache=[None]*28;self.value_cache=[None]*28;self.prefill=True',
 'self.key_cache=[None]*28;self.value_cache=[None]*28;self.prefill=True;self.buffers=[None]*28;self.used=[0]*28;self.capacities=[0]*28')
s=s.replace("self.key_cache[i]=k.contiguous() if i in self.K or i>=self.cutoff else k.index_select(-2,self.keep)\n   self.value_cache[i]=v.contiguous() if i in self.K or i>=self.cutoff else v.index_select(-2,self.keep)",
 "stored=tuple(x if i in self.K or i>=self.cutoff else x.index_select(-2,self.keep) for x in (k,v))\n   self.allocate_update(k,v,i,stored)")
s=s.replace("self.key_cache[i]=torch.cat((self.key_cache[i],k),dim=-2)\n  self.value_cache[i]=torch.cat((self.value_cache[i],v),dim=-2)\n  pair=(self.key_cache[i],self.value_cache[i])",
 "pair=self.allocate_update(k,v,i,(k,v))")
p.write_text(s);append_methods(p,'CompactCache',new)
# D separate high risk code: keep all projection algorithms, remove only dead padding.
assert not D.exists();shutil.copytree(C,D)
for rel in ['qwen3vl_c3v2_numerical_debug/repair.py','qwen3vl_c3v2_numerical_repair2/repair.py',
            'qwen3vl_svh_numerical_repair/engine.py','qwen3vl_svh_l28_repair/engine.py','qwen3vl_svh_qk_repair/engine.py']:
 p=D/rel;t=ast.parse(p.read_text())
 class RemoveAlignment(ast.NodeTransformer):
  def visit_FunctionDef(self,n):
   if n.name=='aligned':
    # qk/l28 nested function uses **kwargs, others **kw.
    var=n.args.kwarg.arg
    n.body=ast.parse('return eager(module,q,k,v,attention_mask,**'+var+')').body
    return n
   return self.generic_visit(n)
 t=RemoveAlignment().visit(t);ast.fix_missing_locations(t);p.write_text(ast.unparse(t)+'\n')
p=D/'internvl3_svh_r_terminal_windows_textvqa/engine.py';s=p.read_text()
s=s.replace('return torch.cat((k,z),dim=-2),torch.cat((v,torch.zeros_like(z)),dim=-2)','return k,v')
s=s.replace('return self.zero_image_scratch(k,v) if i>=self.cutoff else (k,v)','return k,v')
s=s.replace('return pair if i in self.K else self.zero_image_scratch(*pair)','return pair')
s=s.replace('deep=late_blocked.index_select(-2,text)','deep=late_blocked.index_select(-2,text).index_select(-1,text)')
s=s.replace('compact=self.decode_blocked[...,:self.roles["n"]+self.step]','compact=self.decode_text[...,:self.roles["n"]-256+self.step]')
s=s.replace('attention_mask=m[:,:,:-1,:],position_ids=pos[:-1]', 'attention_mask=m[:,:,:-1,:-1],position_ids=pos[:-1]')
p.write_text(s)
for root in [B,C,D]:
 for p in root.rglob('*.py'):compile(p.read_text(),str(p),'exec')
print('STAGES_B_C_D_READY')
