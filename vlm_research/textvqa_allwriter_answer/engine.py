"""Native dense LLaVA generation with answer-only source masks and output zeroing."""
from contextlib import ExitStack
from unittest.mock import patch
import torch
from .common import tensor_hash

def footprint(model):return [(id(m),len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]

def source_mask(old,image,n,past,condition,noop=False):
 assert condition in ['N','A','B']
 if condition=='N' or noop:return old
 assert old is not None and old.shape[:2]==(1,1) and old.shape[2:]==(1 if past else n,past+1 if past else n)
 changed=old.clone();row=0 if past else n-1;changed[0,0,row,image]=-torch.inf
 expected=torch.zeros_like(old,dtype=torch.bool);expected[0,0,row,image]=True
 assert torch.equal(old[~expected],changed[~expected]) and (old[expected]==0).all() and torch.isneginf(changed[expected]).all()
 return changed

def zero_answer(output,n,prefill):
 assert output.shape[:2]==(1,n if prefill else 1)
 changed=output.clone();row=n-1 if prefill else 0;changed[:,row]=0
 assert torch.count_nonzero(changed[:,row])==0
 if prefill:assert torch.equal(changed[:,:n-1],output[:,:n-1])
 return changed

class Control:
 def __init__(self,model,mapping,condition,noop=False,audit=False):
  self.model,self.mapping,self.condition,self.noop,self.audit=model,mapping,condition,noop,audit
  self.layers=model.language_model.model.layers;self.n=len(mapping['prompt_ids']);self.handles=[];self.steps=[];self.events=[];self.zero_events=[];self.prefix_hashes={};self.rope=[]
  self.counts={name:[0]*32 for name in ['MLP','input_norm','post_norm','block']}
 def __enter__(self):
  try:
   def begin(module,args,kwargs):
    t=len(self.steps);n=self.n;expected=n if t==0 else 1;ids=kwargs.get('inputs_embeds',kwargs.get('input_ids'));assert ids.shape[:2]==(1,expected)
    cache=kwargs.get('past_key_values');assert (cache.get_seq_length() if cache is not None else 0)==(n+t-1 if t else 0)
    cp=kwargs['cache_position'];assert cp.tolist()==(list(range(n)) if t==0 else [n+t-1]);self.steps.append(dict(step=t,query_length=expected,cache_position=cp.tolist() if self.audit else [int(cp[0]),int(cp[-1])]))
   self.handles.append(self.model.register_forward_pre_hook(begin,with_kwargs=True))
   def rotary(module,args,kwargs,out):
    if self.audit:self.rope.append([tensor_hash(v) for v in out])
   self.handles.append(self.model.language_model.model.rotary_emb.register_forward_hook(rotary,with_kwargs=True))
   for i,layer in enumerate(self.layers):
    def before(module,args,kwargs,i=i):
     t=len(self.steps)-1;n=self.n;h=kwargs['hidden_states'];assert h.shape[:2]==(1,1 if t else n)
     past=kwargs['past_key_value'].get_seq_length(i);assert past==(n+t-1 if t else 0)
     pos=kwargs['position_ids']
     target=list(range(n)) if t==0 else [n+t-1];assert pos.tolist()==[target] and kwargs['cache_position'].tolist()==target
     mask=source_mask(kwargs['attention_mask'],self.mapping['image'],n,past,self.condition,self.noop)
     return args,dict(kwargs,attention_mask=mask)
    def after(module,args,kwargs,out,i=i):
     t=len(self.steps)-1;weights=out[1];assert weights is not None and weights.shape[:2]==(1,32) and weights.shape[2:]==(1 if t else self.n,self.n+t)
     rows=weights[0,:,([0] if t else self.mapping['R']+[self.n-1]),:].float();assert torch.isfinite(rows).all() and (rows>=0).all();error=float((rows.sum(-1)-1).abs().max());assert error<=torch.finfo(weights.dtype).eps
     blocked=0.
     if self.condition!='N' and not self.noop:
      blocked=float(weights[0,:,0 if t else self.n-1,self.mapping['image']].abs().max());assert blocked==0
     if not t:
      r=torch.tensor(self.mapping['R'],device=weights.device);im=torch.tensor(self.mapping['image'],device=weights.device);assert float(weights[0,:,r[:,None],im[None,:]].max())>0
     self.events.append(dict(layer=i,step=t,blocked_probability_max=blocked,row_sum_max_error=error,blocked_connections_per_head=len(self.mapping['image']) if self.condition!='N' and not self.noop else 0,query_length=weights.shape[2],key_length=weights.shape[3]))
    self.handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True));self.handles.append(layer.self_attn.register_forward_hook(after,with_kwargs=True))
    if i in [2,3,4,5]:
     def zero(module,args,out,i=i):
      t=len(self.steps)-1;active=self.condition=='B' and not self.noop
      self.zero_events.append(dict(layer=i,step=t,active=active,original_query=self.n-1 if t==0 else self.n+t-1,only_answer_query=True))
      return zero_answer(out,self.n,t==0) if active else out
     self.handles.append(layer.self_attn.o_proj.register_forward_hook(zero))
    for name,module in [('MLP',layer.mlp),('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm),('block',layer)]:
     def count(module,args,out,i=i,name=name):
      self.counts[name][i]+=1
      if self.audit and name=='block' and len(self.steps)==1:self.prefix_hashes[str(i)]=tensor_hash(out[0][:,:self.n-1])
     self.handles.append(module.register_forward_hook(count))
   return self
  except BaseException:
   self.__exit__();raise
 def __exit__(self,*args):
  for h in self.handles:h.remove()
  self.handles.clear()

@torch.inference_mode()
def generate(model,processor,inputs,mapping,condition,audit=False,noop=False):
 assert condition in ['N','A','B'];before=footprint(model);n=inputs['input_ids'].shape[1];assert n==len(mapping['prompt_ids'])
 x={k:v.to('cuda:0') for k,v in inputs.items()};input_hashes={k:tensor_hash(v) for k,v in x.items()};ctl=Control(model,mapping,condition,noop,audit);scores=[];hashes=[];ids=[];cache=None;visual=[]
 original=model.get_image_features
 def vision(*args,**kwargs):
  out=original(*args,**kwargs)
  if audit:visual.append(tensor_hash(out))
  return out
 try:
  eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
  positions=torch.arange(n+31,device='cuda:0');attention=torch.ones((1,n+31),dtype=torch.long,device='cuda:0')
  with ctl,patch.object(model,'get_image_features',vision):
   for step in range(32):
    kw=dict(x) if step==0 else dict(input_ids=token.reshape(1,1),attention_mask=attention[:,:n+step])
    kw.update(use_cache=True,return_dict=True,past_key_values=cache,cache_position=positions[:n] if step==0 else positions[n+step-1:n+step],logits_to_keep=1,output_attentions=False,output_hidden_states=False)
    out=model(**kw);assert out.logits.shape==(1,1,model.config.text_config.vocab_size) and torch.isfinite(out.logits).all();last=out.logits[:,-1];token=last.argmax(-1);ids.append(int(token));cache=out.past_key_values
    assert all(cache.get_seq_length(i)==n+step for i in range(32));v=last.detach().float().cpu().clone();hashes.append(tensor_hash(v))
    if audit or step==0:scores.append(v)
    del out
    if ids[-1] in eos:break
  assert len(ctl.steps)==len(ids) and len(ctl.events)==32*len(ids)
  assert all(v==[len(ids)]*32 for v in ctl.counts.values())
  assert sum(e['active'] for e in ctl.zero_events)==(4*len(ids) if condition=='B' and not noop else 0)
  assert input_hashes=={k:tensor_hash(v) for k,v in x.items()}
  text=processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
  g=dict(condition=condition,text=text,token_ids=ids,first_token=ids[0],first_logits_sha256=hashes[0],all_step_logits_sha256=hashes,eos=ids[-1] in eos,truncated=len(ids)==32 and ids[-1] not in eos,empty=not text,repeated=any(ids[i:i+3]==ids[j:j+3] for i in range(len(ids)-2) for j in range(i+3,len(ids)-2)),generated_tokens=len(ids),all_logits_finite=True,dense_cache_lengths=[cache.get_seq_length(i) for i in range(32)],audit=dict(blocked_probability_max=max(e['blocked_probability_max'] for e in ctl.events),row_sum_max_error=max(e['row_sum_max_error'] for e in ctl.events),layer_events=ctl.events,answer_zero_events=ctl.zero_events,all_blocks_MLP_norm_counts=ctl.counts,original_positions=ctl.steps,all32_R_image_open=True,all_non_target_masks_exact=True,all_non_target_o_proj_outputs_exact=True,all_image_KV_retained=True,input_hashes_unchanged=True,hooks_clean=True,noop=noop),prefix_hidden_sha256=ctl.prefix_hashes,rotary_sha256=ctl.rope,visual_feature_sha256=visual)
  return g,scores,cache if audit else None
 finally:
  for h in list(ctl.handles):h.remove()
  ctl.handles.clear();assert footprint(model)==before
