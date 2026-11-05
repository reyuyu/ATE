"""Predetermined same-input operator diagnostics, not model/layer search."""
import csv,torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from .common import OUT
from .probe import metrics
def operators(model,probe,rank):
 rows=[];raw={};extra=[]
 with torch.inference_mode():
  for layer,x in probe.attention_shadow.items():
   a=model.model.language_model.layers[layer].self_attn
   q,k,v=[x[n].to('cuda:0') for n in ['query','key','value']];mask=x['mask'].to('cuda:0')
   ids=list(range(probe.n-1)) if layer==1 else probe.text;index=torch.tensor(ids,device=q.device)
   qr=q.index_select(-2,index);ks=k.index_select(-2,index);vs=v.index_select(-2,index);ms=mask.index_select(-2,index).index_select(-1,index)
   full=qm.eager_attention_forward(a,q,k,v,mask,scaling=a.scaling)[0].index_select(1,index)
   sparse=qm.eager_attention_forward(a,qr,ks,vs,ms,scaling=a.scaling)[0]
   mm=metrics(full,sparse);extra.append(dict(layer=layer,operator='native_eager_all_components_same_inputs',variant='physical_subset',**mm))
   raw[f'attention/{layer}/full']=full.cpu();raw[f'attention/{layer}/sparse']=sparse.cpu()
   if layer==1:
    # One zero, always causally forbidden key avoids computing P-1 Q/K/V.
    kp=torch.cat([ks,torch.zeros_like(k[...,-1:,:])],dim=-2);vp=torch.cat([vs,torch.zeros_like(v[...,-1:,:])],dim=-2)
    padded=qm.eager_attention_forward(a,qr,kp,vp,mask.index_select(-2,index),scaling=a.scaling)[0]
    extra.append(dict(layer=layer,operator='native_eager_all_components_same_inputs',variant='one_causally_dead_zero_key',**metrics(full,padded)));raw['attention/1/padded']=padded.cpu()
   # FP32 remains a local diagnostic, never modifies the BF16 model path.
   fp=qm.eager_attention_forward(a,q.float(),k.float(),v.float(),mask.float(),scaling=a.scaling)[0].index_select(1,index)
   fps=qm.eager_attention_forward(a,qr.float(),ks.float(),vs.float(),ms.float(),scaling=a.scaling)[0]
   extra.append(dict(layer=layer,operator='native_eager_all_components_same_inputs',variant='FP32_diagnostic_only',**metrics(fp,fps)))
  for key,x_cpu in probe.shadow_inputs.items():
   layer,name=key.split('/');layer=int(layer)
   if layer not in [22,23] or name not in ['k_proj','v_proj','gate_proj','up_proj']:continue
   x=x_cpu.to('cuda:0');idx=torch.tensor(probe.text,device=x.device);short=x.index_select(1,idx)
   block=model.model.language_model.layers[layer];mod=getattr(block.self_attn if name in ['k_proj','v_proj'] else block.mlp,name)
   full=mod(x).index_select(1,idx)
   variants={
    'transposed_gemm':lambda:torch.matmul(mod.weight,short.transpose(-2,-1)).transpose(-2,-1)+(mod.bias if mod.bias is not None else 0),
    'column_major_input':lambda:torch.nn.functional.linear(short.transpose(1,2).contiguous().transpose(1,2),mod.weight,mod.bias),
   }
   for size in [32,64]:
    def padded(size=size):
     n=short.shape[1];pad=torch.cat((short,short.new_zeros((1,max(0,size-n),short.shape[-1]))),dim=1)
     return mod(pad)[:,:n]
    variants[f'BF16_tile_{size}']=padded
   for variant,fn in variants.items():
    result=fn();m=metrics(full,result);rows.append(dict(layer=layer,operator=name,variant=variant,full_rows=x.shape[1],retained_rows=short.shape[1],**m));raw[f'linear/{key}/{variant}']=result.cpu()
 if rows:
  with (OUT/f'shard{rank}/operator_shape_variants.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 if extra:
  with (OUT/f'shard{rank}/attention_shape_sensitivity.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(extra[0]));writer.writeheader();writer.writerows(extra)
 return dict(linear=rows,attention=extra),raw
