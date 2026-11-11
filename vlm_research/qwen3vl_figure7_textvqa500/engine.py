"""Figure7 Qwen: native prefill/P-1; only true decode image-key hard mask."""
from pathlib import Path
import sys, gc, contextlib, hashlib
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, DynamicCache
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
from common import *

class DecodeImagePermission:
 def __init__(self,model,image_positions,keep,audit=False):
  self.model=model;self.ids={id(layer.self_attn):i for i,layer in enumerate(model.model.language_model.layers)}
  assert len(self.ids)==N
  self.image_positions=torch.tensor(image_positions,device='cuda',dtype=torch.long)
  self.keep=set(keep);self.audit=audit;self.step=0;self.events=[];self.original=None
 def __enter__(self):
  assert self.original is None
  self.original=hf.eager_attention_forward
  original=self.original
  def wrapped(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
   layer=self.ids.get(id(module))
   blocked=layer is not None and self.step>0 and layer not in self.keep
   if blocked:
    assert query.shape[-2]==1,('decode_query_length',query.shape)
    assert int(self.image_positions[-1])<key.shape[-2]
    mask=(attention_mask[...,:key.shape[-2]].clone() if attention_mask is not None
          else torch.zeros((query.shape[0],1,query.shape[-2],key.shape[-2]),device=query.device,dtype=query.dtype))
    mask.index_fill_(-1,self.image_positions,float('-inf'))
    result=original(module,query,key,value,mask,scaling,dropout=dropout,**kwargs)
   else:
    result=original(module,query,key,value,attention_mask,scaling,dropout=dropout,**kwargs)
   if self.audit and layer is not None:
    p=float(result[1].index_select(-1,self.image_positions).abs().max().item()) if blocked else None
    if blocked:assert p==0.,('forbidden_probability',layer,p)
    self.events.append(dict(step=self.step,layer=layer,blocked=blocked,q=list(query.shape),k=list(key.shape),v=list(value.shape),image_probability_max=p,prefill_native=self.step==0))
   return result
  hf.eager_attention_forward=wrapped;return self
 def __exit__(self,*args):
  hf.eager_attention_forward=self.original;self.original=None

def load_model():
 assert torch.__version__=='2.6.0+cu124'
 import transformers
 assert transformers.__version__=='4.57.6'
 torch.set_num_threads(4)
 torch.manual_seed(0)
 model=Qwen3VLForConditionalGeneration.from_pretrained(S['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True).eval()
 assert len(model.model.language_model.layers)==N
 assert model.config.text_config._attn_implementation=='eager'
 assert model.config.vision_config._attn_implementation=='eager'
 return model

def prepared(key,device='cuda'):
 row=read(OUT/'sample_manifest.json')[key];binding=read(OUT/'input_bindings.json')[key]
 p=RUN/'prepared'/(row['question_id']+'.inputs.pt')
 assert sha(p)==binding['input_file_sha256']
 x=torch.load(p,map_location='cpu',weights_only=True)
 pp=RUN/'prepared'/binding['pixels']['file'];assert sha(pp)==binding['pixels']['sha256']
 x['pixel_values']=torch.load(pp,map_location='cpu',weights_only=True)
 for name,t in x.items():
  expected=binding['pixels']['tensor_sha256'] if name=='pixel_values' else binding['tensors'][name]['sha256']
  assert th(t)==expected,('TENSOR_BINDING',key,name)
 return row,{k:v.to(device) for k,v in x.items()}

def pairs(cache):
 return [(layer.keys,layer.values) for layer in cache.layers]

def clone_cache(model,cache):
 return DynamicCache(ddp_cache_data=[(k.clone(),v.clone()) for k,v in pairs(cache)],config=model.config.text_config)

def layout(cache,image_tokens):
 out=[]
 for i,(k,v) in enumerate(pairs(cache)):
  assert k is not None and v is not None
  out.append(dict(layer=i,key_shape=list(k.shape),value_shape=list(v.shape),key_dtype=str(k.dtype),value_dtype=str(v.dtype),key_tokens=k.shape[-2],visual_key_tokens=image_tokens,live_bytes=k.numel()*k.element_size()+v.numel()*v.element_size()))
 assert len(out)==N
 return out

def prefill(model,inputs):
 o=model(**inputs,use_cache=True,logits_to_keep=1,return_dict=True)
 assert torch.isfinite(o.logits).all()
 return o.logits[:,-1,:].clone(),o.past_key_values,model.model.rope_deltas.clone()

def decode(model,inputs,base_logits,cache,rope,permission,processor,cap=32,ignore_eos=False,logits_record=False):
 model.model.rope_deltas=rope.clone()
 logits=base_logits
 token=logits.argmax(-1,keepdim=True)
 ids=[];hashes=[];raw=[];eos=model.generation_config.eos_token_id
 eos={eos} if isinstance(eos,int) else set(eos)
 prompt_len=inputs['input_ids'].shape[1]
 for t in range(cap):
  assert torch.isfinite(logits).all(),('nonfinite',t)
  if logits_record:hashes.append(th(logits));raw.append(logits.detach().cpu())
  value=int(token.item());ids.append(value)
  if value in eos and not ignore_eos:break
  if t==cap-1:break
  if permission is not None:permission.step=t+1
  output=model(input_ids=token,attention_mask=torch.ones((1,prompt_len+t+1),device='cuda',dtype=inputs['attention_mask'].dtype),
     cache_position=torch.tensor([prompt_len+t],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
  logits=output.logits[:,-1,:];cache=output.past_key_values;token=logits.argmax(-1,keepdim=True)
 result=dict(token_ids=ids,text=processor.tokenizer.decode(ids,skip_special_tokens=True),generated_tokens=len(ids),eos=bool(ids[-1] in eos),truncated=bool(len(ids)==cap and ids[-1] not in eos),empty=not bool(processor.tokenizer.decode(ids,skip_special_tokens=True).strip()),prefill_first_token=int(base_logits.argmax(-1).item()),prefill_logits_sha256=th(base_logits),decode_steps=len(ids)-1,logit_hashes=hashes)
 return result,cache,raw
