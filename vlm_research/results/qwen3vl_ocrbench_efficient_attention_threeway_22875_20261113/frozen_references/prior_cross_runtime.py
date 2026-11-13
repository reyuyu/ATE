"""Frozen cross-backbone physical engines under the common synchronized32-token loop."""
from pathlib import Path
from contextlib import nullcontext
import importlib.util,time,hashlib,json
import torch
from transformers.cache_utils import DynamicCache
from PIL import Image
MODEL_ROOT=Path('/home/featurize/ate-local-models')
def th(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def content(s):
 # Preserve full question, original Paper-OCR text and answer instruction; never gold.
 return s['prompt'].split('USER: <image>\n',1)[1].rsplit(' ASSISTANT:',1)[0]
class Backend:
 def __init__(self,kind,O):
  self.kind=kind;self.O=O
  if kind=='Qwen':
   from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
   from vlm_research.qwen3vl_c3v2_physical import engine as base
   from vlm_research.qwen3vl_svhr_search.physical import Physical
   from vlm_research.qwen3vl_svh_sqa7908_repair.engine import EarlyAlignedExecution
   from vlm_research.qwen3vl_c3v2_l22_kernel_diagnosis import candidates
   candidates.PRIVATE=O/'resources'
   self.base=base;self.R=Physical;self.S=EarlyAlignedExecution
   M=MODEL_ROOT/'Qwen3-VL-8B-Instruct';self.processor=AutoProcessor.from_pretrained(M,local_files_only=True)
   self.model=Qwen3VLForConditionalGeneration.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval();self.tok=self.processor.tokenizer
   self.layers=self.model.model.language_model.layers;self.skip=set(range(1,7));self.keep=set(range(19,36));self.cutoff=22
  else:
   from transformers import AutoModel,AutoTokenizer
   from vlm_research.internvl3_svh_physical_d_pilot import engine as base
   from vlm_research.internvl3_svh_r_terminal_windows_textvqa import engine as late
   self.base=base;self.R=late;M=MODEL_ROOT/'InternVL3-8B'
   self.tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True,use_fast=False,local_files_only=True)
   self.model=AutoModel.from_pretrained(M,trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,device_map={'':'cuda:0'},use_safetensors=True,local_files_only=True).eval()
   self.model.img_context_token_id=self.tok.convert_tokens_to_ids('<IMG_CONTEXT>');self.loader=base.frozen.official_preprocessor(M)
   self.layers=self.model.language_model.model.layers;self.skip=set(range(2,6));self.keep=set(range(19,28));self.cutoff=20
  assert not self.model.training and all(p.device.type=='cuda' and (not p.is_floating_point() or p.dtype==torch.bfloat16) for p in self.model.parameters())
 def prepare(self,s,path):
  text=content(s)
  if self.kind=='Qwen':
   with Image.open(path) as handle:im=handle.convert('RGB')
   messages=[{'role':'user','content':[{'type':'image','image':path},{'type':'text','text':text}]}]
   prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
   inputs=dict(self.processor(text=[prompt],images=[im],return_tensors='pt'));ids=inputs['input_ids'][0].tolist();image_id=self.model.config.image_token_id;I=[i for i,t in enumerate(ids) if t==image_id]
   layout=dict(input_length=len(ids),image_positions=I,image_tokens=len(I),p_minus_1=len(ids)-1,question_positions=[],image_grid_thw=inputs['image_grid_thw'].tolist(),input_tensor_hashes={k:th(v) for k,v in inputs.items()})
  else:
   request='<image>\n'+text;c=self.model.conv_template.copy();c.system_message=self.model.system_message;c.append_message(c.roles[0],request);c.append_message(c.roles[1],None)
   prompt=c.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*256+'</img>',1);ids=self.tok(prompt,return_tensors='pt')['input_ids'];roles=self.base.frozen.role_layout(ids[0].tolist(),self.model.img_context_token_id)
   inputs=dict(ids=ids,pixels=self.loader(path,max_num=1).to(dtype=torch.bfloat16));layout=roles
  return inputs,layout,dict(key=s['key'],prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),question=s['question'],image_sha256=s['image_sha256'],input_hashes={k:th(v) for k,v in inputs.items()},input_shapes={k:list(v.shape) for k,v in inputs.items()},layout=layout,Gold_passed_to_model=False)
 def kv(self,cache,layout,method):
  N=layout.get('input_length',layout.get('n'));V=layout.get('image_tokens',256);rows=[];storages={}
  if hasattr(cache,'entries'):pairs=cache.entries
  elif hasattr(cache,'key_cache'):pairs=list(zip(cache.key_cache,cache.value_cache))
  else:pairs=[(l.keys,l.values) for l in cache.layers]
  for i,pair in enumerate(pairs):
   absent=method not in ('Native','Noop') and i in self.skip
   visual=V if method in ('Native','Noop') or method=='SVH-R' and i in self.keep else 0
   if absent:
    assert pair is None or all(t is None for t in pair);rows.append(dict(layer=i,used_tokens=0,capacity_tokens=0,visual_tokens=0,live_bytes=0,storage_bytes=0,K_shape=None,V_shape=None,dtype=None));continue
   k,v=pair;assert k.dtype==v.dtype==torch.bfloat16 and k.shape==v.shape
   used=cache.used[i] if hasattr(cache,'used') else k.shape[-2];expected=N-V+visual+31
   assert used==expected,(self.kind,method,i,used,expected)
   live=sum(t.shape[0]*t.shape[1]*used*t.shape[-1]*t.element_size() for t in pair);ss={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in pair};storages.update(ss)
   rows.append(dict(layer=i,used_tokens=used,capacity_tokens=k.shape[-2],visual_tokens=visual,live_bytes=live,storage_bytes=sum(ss.values()),K_shape=list(k.shape),V_shape=list(v.shape),dtype=str(k.dtype)))
  assert len(rows)==len(self.layers)
  return dict(layers=rows,live_bytes=sum(r['live_bytes'] for r in rows),storage_bytes=sum(storages.values()),cached_decode_inputs=31,cache_class=type(cache).__name__)
 @torch.inference_mode()
 def generate(self,inputs,layout,method,smoke=False):
  # Inputs have already been copied to GPU. No image reading, processor or H2D here.
  ev=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(32)]
  tokens=[];finite=[];logits=[];cache=None;control=None;out=None
  N=layout.get('input_length',layout.get('n'));device=next(self.model.parameters()).device
  torch.cuda.synchronize();start=time.perf_counter()
  try:
   if self.kind=='Qwen':
    if method=='SVH':control=self.S(self.model,layout,'C',32,audit=smoke)
    elif method=='SVH-R':control=self.R(self.model,layout,'C',32,audit=smoke,layers=tuple(sorted(self.keep)))
    elif method=='Noop':control=self.base.Execution(self.model,layout,'N',32,audit=False,noop=True)
    context=control.context() if control else nullcontext()
   else:
    if method=='SVH-R':cache=self.R.CompactCache(layout,self.keep,device,cutoff=28,audit=smoke);control=self.R.Control(self.model,True,layout,cache,self.keep,audit=smoke,cutoff=28)
    else:
     m={'Native':'Native','Noop':'No-op','SVH':'Physical-D'}[method];cache=self.base.CompactCache(layout) if method=='SVH' else DynamicCache();control=self.base.Control(self.model,m,layout,cache,smoke)
    context=control
   pos=torch.arange(N+31,device=device);mask=torch.ones((1,N+31),device=device,dtype=torch.long)
   with context:
    embed=self.base.embeddings(self.model,dict(inputs,roles=layout)) if self.kind!='Qwen' else None
    for step in range(32):
     if self.kind!='Qwen':
      control.step=step;control.prefill=step==0;control.masks={}
      if hasattr(cache,'prefill'):cache.prefill=step==0
      kw=dict(inputs_embeds=embed) if step==0 else dict(input_ids=tokens[-1].reshape(1,1))
      kw.update(position_ids=(pos[:N] if step==0 else pos[N+step-1:N+step])[None],attention_mask=mask[:,:N+step]);model=self.model.language_model
     else:
      kw=dict(inputs) if step==0 else dict(input_ids=tokens[-1].reshape(1,1),attention_mask=mask[:,:N+step]);model=self.model
     kw.update(use_cache=True,past_key_values=cache,cache_position=pos[:N] if step==0 else pos[N+step-1:N+step],logits_to_keep=1,return_dict=True)
     ev[step][0].record();out=model(**kw);last=out.logits[:,-1];tokens.append(last.argmax(-1));finite.append(torch.isfinite(last).all());ev[step][1].record();cache=out.past_key_values
     if step==0:torch.cuda.synchronize();first=time.perf_counter()
     if smoke:logits.append(last.detach().float().cpu())
     del out,last;out=None
    del embed
   torch.cuda.synchronize();end=time.perf_counter()
   assert torch.stack(finite).all().item(),'NONFINITE'
   ids=torch.cat(tokens).tolist();kv=self.kv(cache,layout,method);events=[a.elapsed_time(b) for a,b in ev]
   result=dict(method=method,token_ids=ids,output_tokens=32,real_decode_calls=31,prefill_calls=1,vision_forward_calls=1,all_logits_finite=True,TTFT_ms=(first-start)*1000,E2E_ms=(end-start)*1000,TPOT_ms=(end-first)*1000/31,CUDA_Event_Decode_TPOT_ms=sum(events[1:])/31,step_event_ms=events,cache=kv,logits_to_keep=1,use_cache=True,smoke=smoke,formal_layer_observer_hooks=smoke)
   if smoke:
    result['effective_logits_sha256']=[th(x) for x in logits]
    if self.kind=='Qwen':
     if control and method!='Noop':
      if method=='SVH-R':assert not control.cache.late
      result['execution_audit']=control.audit()
    else:result['execution_audit']=control.validate(32)
   return result,logits
  finally:del cache,control,out,tokens,finite
