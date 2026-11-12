from pathlib import Path
import ast,json,shutil
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113');B=A/'opt_b'
def replace_method(p,cls,name,body):
 s=p.read_text();ls=s.splitlines(True);t=ast.parse(s)
 c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls)
 n=next(n for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
 indent=' '*n.col_offset
 text='\n'.join(indent+l if l else '' for l in body.strip('\n').splitlines())+'\n'
 ls[n.lineno-1:n.end_lineno]=[text];p.write_text(''.join(ls))
def append_methods(p,cls,body):
 s=p.read_text();ls=s.splitlines(True);c=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name==cls)
 ls.insert(c.end_lineno,'\n'+'\n'.join('    '+l if l else '' for l in body.strip('\n').splitlines())+'\n');p.write_text(''.join(ls))
# Qwen fixed CPU layout, no .tolist in normal forward.
p=B/'qwen3vl_c3v2_physical/engine.py';s=p.read_text()
s=s.replace("self.text=torch.tensor([i for i in range(self.n) if i not in set(self.I)],device=self.device)",
 "image_set=set(self.I); self._text_cpu=[i for i in range(self.n) if i not in image_set]\n        self.text=torch.tensor(self._text_cpu,device=self.device)")
s=s.replace("assert cp.tolist()==(list(range(self.n)) if pre else [self.n+self.step-1])",
 "if self.audit_enabled: assert cp.tolist()==(list(range(self.n)) if pre else [self.n+self.step-1])")
s=s.replace("if pre and self.physical and i>=22:mask=mask.index_select(-2,self.text).index_select(-1,self.text)",
 "if pre and self.physical and i>=22:\n                if i==22: compact_late=mask.index_select(-2,self.text).index_select(-1,self.text)\n                mask=compact_late")
p.write_text(s)
# Same fixed LT/projection/softmax/padding kernels in B; move only diagnostics off timed path.
class AuditOnly(ast.NodeTransformer):
 def visit_Assert(self,n):
  z=ast.unparse(n.test)
  if any(v in z for v in ['float(', 'bool(', 'torch.equal(', 'torch.isfinite(', '.any(', '.all(']):
   return ast.If(test=ast.Attribute(ast.Name('self',ast.Load()),'audit_enabled',ast.Load()),body=[n],orelse=[])
  return n
 def visit_Call(self,n):
  n=self.generic_visit(n)
  if isinstance(n.func,ast.Attribute) and n.func.attr=='tolist' and ast.unparse(n.func.value)=='self.text':
   return ast.copy_location(ast.Attribute(ast.Name('self',ast.Load()),'_text_cpu',ast.Load()),n)
  return n
 def visit_Expr(self,n):
  n=self.generic_visit(n)
  if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='append':
   z=ast.unparse(n.value.func.value)
   if z in ['self.linear_audit','self.early_linears','self.column_audit','self.qk_row_alignment','self.av_row_alignment','self.decode_padding','self.padding_records']:
    return ast.If(ast.Attribute(ast.Name('self',ast.Load()),'audit_enabled',ast.Load()),[n],[])
  return n
for name in ['qwen3vl_svh_numerical_repair/engine.py','qwen3vl_c3v2_numerical_repair2/repair.py','qwen3vl_svh_qk_repair/engine.py','qwen3vl_svh_l28_repair/engine.py','qwen3vl_svh_sqa7908_repair/engine.py','qwen3vl_c3v2_numerical_debug/repair.py']:
 p=B/name;t=ast.parse(p.read_text())
 # Helpers have no self; transform only class bodies.
 for n in t.body:
  if isinstance(n,ast.ClassDef):AuditOnly().visit(n)
 ast.fix_missing_locations(t);p.write_text(ast.unparse(t)+'\n')
p=B/'qwen3vl_svhr_search/physical.py';s=p.read_text().replace('def __init__(self, n, text, capacity_extra, layers, late):','def __init__(self, n, text, capacity_extra, layers, late, text_cpu=None):')
s=s.replace('self.image=[p for p in range(n) if p not in set(text.tolist())]',
 'text_cpu=text.tolist() if text_cpu is None else text_cpu\n        text_set=set(text_cpu)\n        self.image=[p for p in range(n) if p not in text_set]')
s=s.replace('else text.tolist()','else list(text_cpu)')
s=s.replace('layers,{})','layers,{},text_cpu=self._text_cpu)')
s=s.replace('self.builder_layers=[]','self.builder_layers=[]\n        self._masks={}\n        self._decode_masks={}')
p.write_text(s)
append_methods(p,'Physical',r"""
def masks_for(self,h,pre,i):
    selected=i in self.read_layers
    if not pre:
        if i in original.SKIP:
            return None
        if selected not in self._decode_masks:
            length=(self.n if selected else len(self._text_cpu))+31
            self._decode_masks[selected]=h.new_zeros((1,1,1,length))
        return self._decode_masks[selected][...,:len(self.cache.positions[i])+self.step]
    key=(i>=22,selected)
    if key not in self._masks:
        q_cpu=self._text_cpu if i>=22 else list(range(self.n))
        k_cpu=list(range(self.n)) if i<22 or selected else self._text_cpu
        q=torch.tensor(q_cpu,dtype=torch.long)
        k=torch.tensor(k_cpu,dtype=torch.long)
        cpu=torch.zeros((1,1,len(q_cpu),len(k_cpu)),dtype=h.dtype)
        cpu.masked_fill_((k[None,:]>q[:,None])[None,None],torch.finfo(h.dtype).min)
        image_keys=torch.tensor([p in set(self.I) for p in k_cpu])
        if image_keys.any():
            if not selected:cpu[0,0,(q>=self.n-1)[:,None]&image_keys[None,:]]=-torch.inf
            if i>=22:
                relay=(q>self.I[-1])&(q<self.n-1)
                cpu[0,0,relay[:,None]&image_keys[None,:]]=-torch.inf
        assert bool(torch.isfinite(cpu).any(-1).all())
        self._masks[key]=cpu.to(h.device)
    return self._masks[key]
""")
replace_method(p,'Physical','forward',r"""
def forward(self,input_ids=None,inputs_embeds=None,position_ids=None,cache_position=None,
            attention_mask=None,visual_pos_masks=None,deepstack_visual_embeds=None,**kwargs):
    h=inputs_embeds if inputs_embeds is not None else self.d.embed_tokens(input_ids)
    pre=self.step==0;self.cache.prefill=pre;cp=cache_position
    if self.audit_enabled:assert cp.tolist()==(list(range(self.n)) if pre else [self.n+self.step-1])
    textpos=position_ids[0]
    if position_ids.shape[0]==4:position_ids=position_ids[1:]
    rotary=self.d.rotary_emb(h,position_ids)
    for i,layer in enumerate(self.d.layers):
        if pre and i==22:
            self.construct_late_KV(h,rotary,textpos,cp)
            h=h.index_select(1,self.text);cp=cp.index_select(0,self.text)
            textpos=textpos.index_select(-1,self.text)
            rotary=tuple(v.index_select(1,self.text) for v in rotary)
        mask=self.masks_for(h,pre,i)
        if self.audit_enabled:
            qpos=cp
            kpos=torch.arange(self.n,device=h.device) if pre and (i<22 or i in self.read_layers) else self.text if pre else torch.tensor(self.cache.positions[i]+list(range(self.n,self.n+self.step)),device=h.device)
            reference=h.new_zeros((1,1,len(qpos),len(kpos)))
            reference.masked_fill_((kpos[None,:]>qpos[:,None])[None,None],torch.finfo(h.dtype).min)
            image_keys=torch.isin(kpos,self.image)
            if image_keys.any():
                if i not in self.read_layers:reference[0,0,(qpos>=self.n-1)[:,None]&image_keys[None,:]]=-torch.inf
                if pre and i>=22:
                    relay=(qpos>self.I[-1])&(qpos<self.n-1)
                    reference[0,0,relay[:,None]&image_keys[None,:]]=-torch.inf
            if mask is not None:assert torch.equal(mask,reference),'MASK_CHANGED'
            self.layers.append(dict(step=self.step,layer=i,hidden_rows=h.shape[1],logical_positions=cp.tolist(),persistent_image_keys=len(self.I) if i in self.read_layers else 0))
        self.current=(i,pre,mask)
        h=layer(h,attention_mask=mask,position_ids=textpos,cache_position=cp,
                position_embeddings=rotary,past_key_values=self.cache,use_cache=True)
        if deepstack_visual_embeds is not None and i<len(deepstack_visual_embeds):
            assert pre and i<3 and h.shape[1]==self.n
            h=self.d._deepstack_process(h,visual_pos_masks,deepstack_visual_embeds[i])
            self.deepstack.append(i)
    self.cache.logical_length=self.n+self.step;self.step+=1
    return BaseModelOutputWithPast(last_hidden_state=self.d.norm(h),past_key_values=self.cache)
""")
# InternVL mask values reused; independent audit can compare old generator masks.
p=B/'internvl3_svh_physical_d_pilot/engine.py';s=p.read_text()
s=s.replace('self.positions=[];self.skips=[];self.exits=[];self.finite=True;self.attention_calls=[]',
 'self.positions=[];self.skips=[];self.exits=[];self.finite=True;self.attention_calls=[]\n  n=roles["n"]; ref=next(model.parameters()); self.decode_full=ref.new_zeros((1,1,1,n+31)); self.decode_text=ref.new_zeros((1,1,1,n-256+31)); self.decode_blocked=self.decode_full.clone(); self.decode_blocked[...,roles["image"]]=-torch.inf')
s=s.replace('mask=decoder._update_causal_mask(attention_mask,h,pos,cache,False)',
 'mask=decoder._update_causal_mask(attention_mask,h,pos,cache,False) if self.prefill else self.decode_full[...,:self.roles["n"]+self.step]')
s=s.replace("compact_decode=mask.index_select(-1,torch.cat((text,torch.arange(self.roles['n'],self.roles['n']+self.step,device=h.device)))) if not self.prefill else None",
 "compact_decode=self.decode_text[...,:self.roles['n']-256+self.step] if not self.prefill else None")
p.write_text(s)
p=B/'internvl3_svh_r_terminal_windows_textvqa/engine.py';s=p.read_text()
s=s.replace('mask=decoder._update_causal_mask(attention_mask,h,pos,cache,False)',
 'mask=decoder._update_causal_mask(attention_mask,h,pos,cache,False) if self.prefill else self.decode_full[...,:self.roles["n"]+self.step]')
s=s.replace('keys=torch.cat((text,torch.arange(self.roles["n"],self.roles["n"]+self.step,device=h.device)))\n   compact=edit_mask(mask,self.roles,False,0,set())',
 'compact=self.decode_blocked[...,:self.roles["n"]+self.step]')
p.write_text(s)
# V2Drop stage-specific local text metadata copied ONCE per stage, not per layer.
p=B/'prefix_fusion.py';s=p.read_text()
s=s.replace("text=tuple(self.local_text.tolist())","text=self.local_text_cpu")
s=s.replace('self.cache.active_positions=self.active;self.cache.local_text=self.local_text',
 'self.cache.active_positions=self.active;self.cache.local_text=self.local_text\n        self.cache.local_text_cpu=tuple(self.local_text.tolist())')
s=s.replace("start=int(self.local_image[0])","start=self.layout.image[0]")
s=s.replace("assert torch.equal(full_ids[0],original),'Unexpected RoPE input positions'","if self.audit_enabled:assert torch.equal(full_ids[0],original),'Unexpected RoPE input positions'")
s=s.replace("assert torch.equal(self.local_image,torch.arange(start,start+amount,device=self.device))","if self.audit_enabled:assert torch.equal(self.local_image,torch.arange(start,start+amount,device=self.device))")
s=s.replace("retained_original_image_positions=self.active.index_select(0,self.local_image).tolist(),","retained_original_image_positions=self.active.index_select(0,self.local_image).tolist() if self.audit_enabled else None,")
# Reuse same gather result once at each stage, including masks/rotary/ids.
s=s.replace("if pre:\n                if self.enabled", "if pre:\n                stage_changed=(i==0 or self.enabled and i in (3,17) or i==20)\n                if self.enabled")
s=s.replace("if i>=20:mask=self.prefill_late\n                elif self.active.numel()==n:mask=self.prefill_early\n                else:mask=self.prefill_early.index_select(-2,self.active).index_select(-1,self.active)\n                ids=full_ids if self.active.numel()==n else full_ids.index_select(-1,self.active)\n                rotary=full_rotary if self.active.numel()==n else tuple(t.index_select(1,self.active) for t in full_rotary)",
 "if stage_changed:\n                    if i>=20:stage_mask=self.prefill_late\n                    elif self.active.numel()==n:stage_mask=self.prefill_early\n                    else:stage_mask=self.prefill_early.index_select(-2,self.active).index_select(-1,self.active)\n                    stage_ids=full_ids if self.active.numel()==n else full_ids.index_select(-1,self.active)\n                    stage_rope=full_rotary if self.active.numel()==n else tuple(t.index_select(1,self.active) for t in full_rotary)\n                mask,ids,rotary=stage_mask,stage_ids,stage_rope")
p.write_text(s)
# Fast and Sparse: Gate diagnostics, never change ranking/recycling math or RNG seed.
for rel in ['fastv_svh/engine.py','sparsevlm_svh/engine.py']:
 p=B/rel;s=p.read_text();t=ast.parse(s)
 class Gate(ast.NodeTransformer):
  def visit_Assert(self,n):
   x=ast.unparse(n.test)
   if any(v in x for v in ['torch.equal(', 'torch.isfinite(', 'torch.count_nonzero(', '.tolist()', '.any(', '.all(']):
    return ast.If(ast.Attribute(ast.Name('self',ast.Load()),'audit_enabled',ast.Load()),[n],[])
   return n
  def visit_Expr(self,n):
   n=self.generic_visit(n)
   # structural SHA is diagnostic only; selections needed by Sparse bookkeeping remain.
   if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self.structure.append':
    return ast.If(ast.Attribute(ast.Name('self',ast.Load()),'audit_enabled',ast.Load()),[n],[])
   return n
 # only classes, avoid standalone indices_sha using no self
 for n in t.body:
  if isinstance(n,ast.ClassDef):Gate().visit(n)
 ast.fix_missing_locations(t);s=ast.unparse(t)+'\n'
 # event timings are diagnostic; event.synchronize removed only outside audit.
 s=s.replace('end.synchronize()','end.synchronize() if self.audit_enabled else None')
 s=s.replace('start.elapsed_time(end) if start is not None else None','start.elapsed_time(end) if start is not None and self.audit_enabled else None')
 s=s.replace('begin.elapsed_time(end) if begin is not None else None','begin.elapsed_time(end) if begin is not None and self.audit_enabled else None')
 # Fast selector outputs are not algorithmic inputs to the next stage except tensors self.keep/top.
 if rel.startswith('fastv'):
  s=s.replace('selected_visual_indices=top.tolist()', 'selected_visual_indices=top.tolist() if self.audit_enabled else None')
  s=s.replace('selected_visual_indices_sha256=indices_sha(top)', 'selected_visual_indices_sha256=indices_sha(top) if self.audit_enabled else None')
  s=s.replace('physical_keep_positions_sha256=indices_sha(keep)', 'physical_keep_positions_sha256=indices_sha(keep) if self.audit_enabled else None')
 # Sparse labels/anchors affect only provenance; keep their CPU list transfers until a separate proven change.
 p.write_text(s)
# Syntax/AST validation without importing torch or touching CUDA.
checked=0
for p in B.rglob('*.py'):compile(p.read_text(),str(p),'exec');checked+=1
(A/'code/ast_helpers.py').write_text(Path(__file__).read_text().split('# Qwen fixed CPU layout')[0])
print('B_READY',checked)
