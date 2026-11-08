from contextlib import contextmanager
import torch,json,hashlib
from vlm_research.physical_handoff_fastpath.engine import Layout
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def hashes(x):return {k:tensor_hash(v.half() if v.is_floating_point() else v) for k,v in x.items()}

def layout_from(ids,question,cutoff=20):
 image=tuple(i for i,t in enumerate(ids) if t==32000);text=tuple(i for i,t in enumerate(ids) if t!=32000);relay=tuple(range(image[-1]+1,len(ids)-1))
 assert len(image)==576 and question and set(question)<=set(relay)
 return Layout(len(ids),image,text,relay,tuple(question),cutoff)

@contextmanager
def structural_trace(a,l):
 out={'vision_calls':0,'projector_calls':0,'image_embedding_exact':False,'layer_events':[],
  'norm_calls':[0]*32,'post_norm_calls':[0]*32,'mlp_calls':[0]*32,'skip_residual_checks':0}
 handles=[];projected=[];before={};n=l.length;image_mask=a.inputs['input_ids'][0]==32000
 def vision(m,args,v):out['vision_calls']+=1;assert v.last_hidden_state.is_cuda
 def project(m,args,v):out['projector_calls']+=1;assert list(v.shape)==[1,576,4096];projected.append(v.detach())
 def language(m,args,kw):
  if projected:assert torch.equal(kw['inputs_embeds'][0,image_mask],projected.pop().reshape(576,4096));out['image_embedding_exact']=True
 try:
  handles.extend([a.model.vision_tower.register_forward_hook(vision),a.model.multi_modal_projector.register_forward_hook(project),a.model.language_model.register_forward_pre_hook(language,with_kwargs=True)])
  for i,layer in enumerate(a.layers):
   def pre(m,args,kw,i=i):
    step=sum(e['layer']==i for e in out['layer_events']);pos=list(range(n)) if step==0 and i<l.cutoff else list(l.text) if step==0 else [n+step-1]
    assert args[0].shape[1]==len(pos) and kw['cache_position'].tolist()==pos and kw['position_ids'][0].tolist()==pos
    if i in (2,3,4,5):before[i]=args[0].detach().clone()
    out['layer_events'].append({'layer':i,'step':step,'rows':len(pos),'original_positions_sha256':digest(pos)})
   def norm(m,args,v,i=i):out['norm_calls'][i]+=1
   def post(m,args,i=i):
    out['post_norm_calls'][i]+=1
    if i in (2,3,4,5):assert torch.equal(args[0][:,-1],before.pop(i)[:,-1]);out['skip_residual_checks']+=1
   def mlp(m,args,v,i=i):out['mlp_calls'][i]+=1
   handles.extend([layer.register_forward_pre_hook(pre,with_kwargs=True),layer.input_layernorm.register_forward_hook(norm),layer.post_attention_layernorm.register_forward_pre_hook(post),layer.mlp.register_forward_hook(mlp)])
  yield out
 finally:
  for h in handles:h.remove()
  projected.clear();before.clear()
