"""Audit-only tensor capture and exact native Qwen2 operator replay."""
from .common import *
import torch,math,gc
from vlm_research.internvl3_svh_r_middle_reaccess_search import engine as e
from transformers.models.qwen2 import modeling_qwen2 as qwen
from .adapter import ProjectionRepair
def tensor_sha(t):
 a=t.detach().cpu().contiguous().view(torch.uint8)
 return hashlib.sha256(a.numpy().tobytes()).hexdigest()
def info(t):
 return {"shape":list(t.shape),"stride":list(t.stride()),"contiguous":t.is_contiguous(),"dtype":str(t.dtype),"device":str(t.device),"sha256":tensor_sha(t),"data_ptr":t.data_ptr(),"storage_offset":t.storage_offset()}
def metric(a,b):
 assert a.shape==b.shape,(a.shape,b.shape)
 # torch.equal acts on original BF16 tensors; cast only for reporting difference.
 d=a.double()-b.double();den=a.double().square().mean().sqrt().clamp_min(1e-30)
 return dict(torch_equal=bool(torch.equal(a,b)),max_abs=float(d.abs().max()) if d.numel() else 0.0,mean_abs=float(d.abs().mean()) if d.numel() else 0.0,relative_RMS=float(d.square().mean().sqrt()/den) if d.numel() else 0.0,different_elements=int(torch.count_nonzero(d)),elements=d.numel(),a_shape=list(a.shape),b_shape=list(b.shape),a_dtype=str(a.dtype),b_dtype=str(b.dtype))
def save_raw(name,data,origin):
 p=RUN/"raw"/name;p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+"."+uuid.uuid4().hex+".tmp")
 try:
  with tmp.open("xb") as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
  os.link(tmp,p)
 finally:tmp.unlink(missing_ok=True)
 def describe(v):
  if isinstance(v,torch.Tensor):return info(v)
  if isinstance(v,dict):return {str(k):describe(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [describe(x) for x in v]
  return v
 receipt={"file":str(p),"sha256":sha(p),"bytes":p.stat().st_size,"origin":origin,"tensors":describe(data)}
 writej(p.with_suffix(p.suffix+".json"),receipt);return receipt
def capture(model,tok,ready,physical,spec=None,layers=range(20,28),first_only=False,forced=None):
 data={};metadata={};handles=[];step={};before=e.frozen.hooks_count(model)
 def put(i,name,value):
  k=(i,step[i])
  if first_only and step[i]!=0:return
  data.setdefault(k,{})[name]=value.detach().cpu().clone()
  metadata.setdefault(str(k),{})[name]=info(value)
 for i in layers:
  l=model.language_model.model.layers[i]
  def before_norm(m,args,i=i):step[i]=step.get(i,-1)+1;put(i,"hidden_input",args[0])
  handles.append(l.input_layernorm.register_forward_pre_hook(before_norm))
  def attn_pre(m,args,kw,i=i):
   for name,v in kw.items():
    if name=="position_embeddings":
     put(i,"rope_cos",v[0]);put(i,"rope_sin",v[1])
    elif name in ["attention_mask","position_ids","hidden_states"]:put(i,name,v)
  handles.append(l.self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True))
  for name,m in [("input_norm",l.input_layernorm),("q_proj",l.self_attn.q_proj),("k_proj",l.self_attn.k_proj),("v_proj",l.self_attn.v_proj),("o_proj",l.self_attn.o_proj),("post_norm",l.post_attention_layernorm),("gate_proj",l.mlp.gate_proj),("up_proj",l.mlp.up_proj),("down_proj",l.mlp.down_proj)]:
   def h(mod,args,out,i=i,name=name):put(i,name,out);put(i,name+"_input",args[0])
   handles.append(m.register_forward_hook(h))
 try:
  with ProjectionRepair(model,ready,spec or {}) as repair:
   result,scores,control=e.generate(model,tok,ready,"R0",[],max_tokens=16 if forced is None else len(forced),fixed=forced is not None,audit=True,physical=physical,forced=forced)
  for (step_i,i),(pos,h) in control.hidden.items():
   if i in layers and (not first_only or step_i==0):data[(i,step_i)]["layer_hidden"]=h
  events=repair.events
 finally:
  for h in handles:h.remove()
 assert before==e.frozen.hooks_count(model),"HOOK_LEAK_CAPTURE"
 return result,scores,control,data,metadata,events
@torch.inference_mode()
def replay(layer,d,physical,keep,n,repair_names=()):
 device=layer.input_layernorm.weight.device
 h=d["hidden_input"].to(device);norm=layer.input_layernorm(h);a=layer.self_attn
 out={"hidden_input":h,"input_norm":norm};raw={}
 for name in ["q_proj","k_proj","v_proj"]:
  inp=norm
  if name in repair_names:
   from .adapter import pad_text
   inp=pad_text(norm,n,keep)
  y=getattr(a,name)(inp)
  if name in repair_names:y=y.index_select(1,keep)
  raw[name]=y;out[name]=y
 shape=(*h.shape[:-1],-1,a.head_dim)
 q=raw["q_proj"].view(shape).transpose(1,2);k=raw["k_proj"].view(shape).transpose(1,2);v=raw["v_proj"].view(shape).transpose(1,2)
 cos=d["rope_cos"].to(device);sin=d["rope_sin"].to(device)
 q,k=qwen.apply_rotary_pos_emb(q,k,cos,sin);out.update(rope_q=q,rope_k=k)
 if physical:
  shape=(*k.shape[:-2],n,k.shape[-1]);zp=k.new_zeros(shape);zv=v.new_zeros(shape);zp.index_copy_(-2,keep,k);zv.index_copy_(-2,keep,v);k,v=zp,zv
 k=qwen.repeat_kv(k,a.num_key_value_groups);v=qwen.repeat_kv(v,a.num_key_value_groups)
 scores=torch.matmul(q,k.transpose(2,3))*a.scaling;mask=d["attention_mask"].to(device)
 masked=scores+mask[:,:,:,:k.shape[-2]]
 prob=torch.nn.functional.softmax(masked,dim=-1,dtype=torch.float32).to(q.dtype)
 av=torch.matmul(prob,v);flat=av.transpose(1,2).contiguous().reshape(*h.shape[:-1],-1)
 o=a.o_proj(flat);res=h+o;post=layer.post_attention_layernorm(res)
 gate=layer.mlp.gate_proj(post);up=layer.mlp.up_proj(post);product=layer.mlp.act_fn(gate)*up;down=layer.mlp.down_proj(product);final=res+down
 out.update(attention_scores=scores,masked_scores=masked,softmax=prob,AV=av,o_proj_input=flat,o_proj=o,attention_residual=res,post_norm=post,gate_proj=gate,up_proj=up,mlp_product=product,down_proj=down,mlp_residual=final,layer_hidden=final)
 return {k:v.detach().cpu() for k,v in out.items()}
def compare_operator_traces(a,b,keep,mask,version):
 rows=[];n=len(keep)
 for name,x in a.items():
  y=b[name]
  if name in ["attention_scores","masked_scores","softmax"]:
   x=x.index_select(-2,keep).index_select(-1,keep);y=y.index_select(-1,keep)
   legal=(mask.index_select(-2,keep).index_select(-1,keep)==0).expand_as(x)
   x=x[legal];y=y[legal]
  elif name in ["rope_q","AV"]:x=x.index_select(-2,keep)
  elif name=="rope_k":x=x.index_select(-2,keep)
  else:x=x.index_select(1,keep)
  rows.append(dict(version=version,operator=name,**metric(x,y),comparison_scope="original logical shared text; Attention restricted to common legal keys"))
 return rows
