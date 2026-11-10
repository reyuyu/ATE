"""Read-only, pinned official FastV selector bridged to HF 4.49 DynamicCache.

The original cached FastV algorithm is retained: select once before zero-based
L3 using L2 last-query head-mean attention; retain original RoPE positions.
No installed Transformers or official source file is patched.
"""
import ast,hashlib
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
import torch

SOURCE_SHA='48dfd9351aafede82ccc15bbb772571293a0669e0b7051572dc344c8a40e0726'

@lru_cache(None)
def selector(path):
 raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
 tree=ast.parse(raw)
 blocks=[n for n in ast.walk(tree) if isinstance(n,ast.If) and n.lineno==1038]
 assert len(blocks)==1 and ast.unparse(blocks[0].test)=='layer_idx == FASTV_k'
 nodes=[]
 for n in blocks[0].body:
  nodes.append(n)
  if ast.unparse(n)=='keep_indexs = keep_indexs.sort().values':break
 assert ast.unparse(nodes[-1])=='keep_indexs = keep_indexs.sort().values'
 body=ast.unparse(ast.Module(body=nodes,type_ignores=[]))
 code='def select(layer_outputs, FASTV_image_token_start_index, FASTV_image_token_length, FASTV_r, seq_length_with_past, device):\n'
 code+='\n'.join('    '+l for l in body.splitlines())
 code+='\n    return keep_indexs, top_attention_rank_index, last_layer_attention_avg_last_tok_image\n'
 ns={'torch':torch};exec(compile(code,str(path),'exec'),ns)
 return ns['select'],{'source_sha256':SOURCE_SHA,'selection_AST_sha256':hashlib.sha256(body.encode()).hexdigest(),
  'selection_AST_changes':0,'official_lines':[1040,1052],'prune_before_zero_based_layer':3,'ranking_layer':2,
  'ranking':'torch.mean(attention, dim=1)[0][-1][dynamic_image_span].topk(round(576*(1-2/3)))',
  'cache_semantics':'Official use_cache=True: frozen visual selection after initial prefill; original re-prunes each full forward'}

class FastV:
 def __init__(self,model,source,image,n,enabled=True,trace=False):
  self.model=model;self.lm=model.language_model.model;self.layers=self.lm.layers
  self.select,self.provenance=selector(str(source));self.image=tuple(image);self.start=self.image[0];self.n=n
  assert len(image)==576 and self.image==tuple(range(self.start,self.start+576))
  self.enabled=enabled;self.trace=trace;self.step=-1;self.active=None;self.previous=None;self.events=[];self.rows=[]
  self.prefill_rows={};self.original_positions={}

 def before(self,index,module,args,kw):
  h=args[0];prefill=h.shape[1]>1
  if index==0:
   self.step+=1
   if prefill:self.active=torch.arange(self.n,device=h.device)
  if self.enabled:
   if prefill:
    if index==2:kw=dict(kw,output_attentions=True)
    if index==3:
     assert self.previous is not None
     keep,top,scores=self.select((None,self.previous),self.start,576,2/3,self.n,h.device)
     assert keep.numel()==self.n-384 and top.numel()==192
     if self.trace:
      # Independent expression checks wrapper arguments; original AST executes unchanged.
      ref=self.previous.mean(dim=1)[0,-1,self.start:self.start+576].topk(192).indices+self.start
      assert torch.equal(ref,top)
     self.active=keep;h=h.index_select(1,keep)
     self.events.append({'prune_before_layer':3,'ranking_layer':2,'visual_before':576,'visual_after':192,'physically_removed':384,
                         'hidden_before':self.n,'hidden_after':int(h.shape[1])})
     self.previous=None
    if index>=3:
     kw=dict(kw)
     kw['position_ids']=kw['position_ids'].index_select(-1,self.active)
     kw['position_embeddings']=tuple(v.index_select(1,self.active) for v in kw['position_embeddings'])
     # Match official prefix cache_position. DynamicCache appends and ignores these
     # slots; RoPE uses selected ORIGINAL position_ids, never compressed indices.
     kw['cache_position']=kw['cache_position'][:len(self.active)]
     mask=kw.get('attention_mask')
     if mask is not None:kw['attention_mask']=mask.index_select(-2,self.active).index_select(-1,self.active)
   elif index>=3:
    kw=dict(kw);mask=kw.get('attention_mask')
    size=kw['past_key_value'].get_seq_length(index)+1
    if mask is not None:kw['attention_mask']=mask[...,:size]
  if prefill:
   self.prefill_rows[index]=h.shape[1]
   if self.trace:self.original_positions[index]=self.active.tolist()
  elif self.trace:self.original_positions[index].append(self.n+self.step-1)
  if self.trace:
   pos=kw['position_ids'][0];rope=kw['position_embeddings'];mask=kw.get('attention_mask')
   expected=self.lm.rotary_emb(h,kw['position_ids'])
   assert all(torch.equal(x,y) for x,y in zip(expected,rope)),'RoPE mismatch'
   if prefill:
    assert pos.tolist()==self.active.tolist()
    assert [p for p in pos.tolist() if p not in self.image]==[p for p in range(self.n) if p not in self.image]
    if mask is not None:
     assert mask.shape[-2:]==(h.shape[1],h.shape[1])
     assert torch.equal(mask[0,0]<0,torch.triu(torch.ones_like(mask[0,0],dtype=torch.bool),diagonal=1))
   else:
    assert pos.item()==self.n+self.step-1
    size=kw['past_key_value'].get_seq_length(index)+1
    assert mask is None or torch.count_nonzero(mask[...,:size]).item()==0
   self.rows.append({'step':self.step,'layer':index,'hidden':list(h.shape),'mask':None if mask is None else list(mask.shape),
                     'position_ids':pos.tolist(),'cache_position':kw['cache_position'].tolist(),
                     'RoPE':[list(x.shape) for x in rope],'KV_before':kw['past_key_value'].get_seq_length(index),
                     'RoPE_values_exact':True,'causal_mask_verified':True})
  return (h,*args[1:]),kw

 def after(self,index,module,args,kw,out):
  if self.enabled and index==2 and args[0].shape[1]>1:
   assert len(out)==2 and out[1] is not None
   self.previous=out[1]
  if self.trace:
   c=kw['past_key_value'];r=self.rows[-1]
   assert r['layer']==index
   r['K_after']=list(c.key_cache[index].shape);r['V_after']=list(c.value_cache[index].shape)
   assert r['K_after'][-2]==self.prefill_rows[index]+self.step

 @contextmanager
 def context(self):
  handles=[]
  try:
   for i,l in enumerate(self.layers):
    handles.append(l.register_forward_pre_hook(lambda m,a,k,i=i:self.before(i,m,a,k),with_kwargs=True))
    if self.trace or (self.enabled and i==2):
     handles.append(l.register_forward_hook(lambda m,a,k,o,i=i:self.after(i,m,a,k,o),with_kwargs=True))
   yield self
  finally:
   for h in handles:h.remove()
   self.previous=None

def cache_snapshot(cache,ctrl,n,steps,method):
 rows=[];storages={}
 for i,(k,v) in enumerate(zip(cache.key_cache,cache.value_cache)):
  assert k.is_cuda and v.is_cuda and k.dtype==v.dtype==torch.float16
  tokens=int(k.shape[-2]);visual=192 if method=='FastV' and i>=3 else 576
  assert tokens==n-576+visual+steps-1
  live=sum(x.numel()*x.element_size() for x in (k,v))
  layerstorage={}
  for x in (k,v):
   s=x.untyped_storage();layerstorage[s.data_ptr()]=s.nbytes();storages[s.data_ptr()]=s.nbytes()
  rows.append({'layer':i,'K_shape':list(k.shape),'V_shape':list(v.shape),'dtype':str(k.dtype),'device':str(k.device),
               'used_tokens':tokens,'visual_tokens':visual,'live_bytes':live,'storage_bytes':sum(layerstorage.values()),
               'original_positions':ctrl.original_positions.get(i) if ctrl else None})
 return {'layers':rows,'live_bytes':sum(r['live_bytes'] for r in rows),'storage_bytes':sum(storages.values()),
         'live_MiB':sum(r['live_bytes'] for r in rows)/2**20,'generated_tokens':steps,'cached_decode_inputs':steps-1,
         'layout':'native HF DynamicCache, actual CUDA K/V tensors, unequal layer lengths'}
