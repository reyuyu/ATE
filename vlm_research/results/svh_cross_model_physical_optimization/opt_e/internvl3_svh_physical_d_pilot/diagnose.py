"""Failure diagnosis only. No quality pilot or timed benchmark. Gates and runtime unchanged."""
import gc,json
import torch
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb,repeat_kv
from .common import *
from .run import load
from . import engine as p
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as b

@torch.inference_mode()
def main():
 assert read(RUN/'smoke_validation.json')['status']=='FAILED'
 assert not (RUN/'quality_checkpoints').exists() and not (RUN/'efficiency_checkpoints').exists()
 assert not command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader')
 assert command('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==GPU
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 # Keep model parameters as normal versioned tensors as in the frozen historical loader.
 with torch.inference_mode(False):model=load()
 tok=AutoTokenizer.from_pretrained(MODEL,use_fast=False,trust_remote_code=True,local_files_only=True)
 loader=b.official_preprocessor(MODEL)
 samples=read(OLD/'sample_manifest.json');samples=samples['samples'] if isinstance(samples,dict) else samples
 s=samples[525];prep=b.prepare(model,tok,loader,s);ready=p.gpu_ready(model,tok,loader,s)
 layer=model.language_model.model.layers[2];attn=layer.self_attn;capture={}
 def before(module,args,kw):
  if kw['hidden_states'].shape[1]>1:
   for key in ['hidden_states','attention_mask','position_ids','cache_position']:
    capture[key]=kw[key].detach().clone()
   capture['position_embeddings']=tuple(x.detach().clone() for x in kw['position_embeddings'])
 def o_input(module,args):
  if args[0].shape[1]>1:capture['o_input']=args[0].detach().clone()
 handles=[attn.register_forward_pre_hook(before,with_kwargs=True),attn.o_proj.register_forward_pre_hook(o_input)]
 try:dense,ds,dc=p.generate(model,tok,ready,'Dense-D',fixed=True,max_tokens=9,audit=True,prepared=prep)
 finally:
  for h in handles:h.remove()
 physical,ps,pc=p.generate(model,tok,ready,'Physical-D',fixed=True,max_tokens=9,audit=True,prepared=prep)
 rows=[]
 def compare(name,a,c):
  x=a.float();y=c.float();d=(x-y).abs()
  row=dict(operator=name,equal=torch.equal(a,c),max_abs=float(d.max()),mean_abs=float(d.mean()),
   changed_elements=int((a!=c).sum()),elements=a.numel(),full_shape=list(a.shape),short_shape=list(c.shape))
  rows.append(row);return row
 x=capture['hidden_states'];rope=capture['position_embeddings'];mask=capture['attention_mask']
 full={};short={}
 for name in ['q_proj','k_proj','v_proj']:
  module=getattr(attn,name);full[name]=module(x);short[name]=module(x[:,:-1,:])
  compare(name+'_same_input_prefix_changed_matrix_rows',full[name][:,:-1,:],short[name])
 compare('o_proj_same_context_prefix_changed_matrix_rows',attn.o_proj(capture['o_input'])[:,:-1,:],attn.o_proj(capture['o_input'][:,:-1,:]))
 # Same numerical input, separately test shape changes through official attention primitives.
 def qkv(values,n,rop):
  q=values['q_proj'].view(1,n,-1,attn.head_dim).transpose(1,2)
  k=values['k_proj'].view(1,n,-1,attn.head_dim).transpose(1,2)
  v=values['v_proj'].view(1,n,-1,attn.head_dim).transpose(1,2)
  q,k=apply_rotary_pos_emb(q,k,*rop)
  return q,repeat_kv(k,attn.num_key_value_groups),repeat_kv(v,attn.num_key_value_groups)
 n=x.shape[1];q,k,v=qkv(full,n,rope);qs,ks,vs=qkv(short,n-1,tuple(z[:,:-1,:] for z in rope))
 compare('RoPE_query_prefix',q[:,:,:-1,:],qs);compare('RoPE_key_prefix',k[:,:,:-1,:],ks)
 qk=torch.matmul(q,k.transpose(-1,-2))*attn.scaling
 qks=torch.matmul(qs,ks.transpose(-1,-2))*attn.scaling
 compare('QK_prefix',qk[:,:,:-1,:-1],qks)
 prob=torch.softmax(qk+mask,dim=-1,dtype=torch.float32).to(q.dtype)
 probs=torch.softmax(qks+mask[:,:,:-1,:-1],dim=-1,dtype=torch.float32).to(q.dtype)
 compare('Softmax_prefix',prob[:,:,:-1,:-1],probs)
 av=torch.matmul(prob,v);avs=torch.matmul(probs,vs)
 compare('AV_prefix',av[:,:,:-1,:],avs)
 # Every non-answer dense query has zero causal probability on the last P-1 key.
 assert float(prob[:,:,:-1,-1].abs().max())==0
 out=attn.o_proj(av.transpose(1,2).contiguous().reshape(1,n,-1))
 outs=attn.o_proj(avs.transpose(1,2).contiguous().reshape(1,n-1,-1))
 compare('complete_attention_non_answer_prefix_same_layer_input',out[:,:-1,:],outs)
 # CPU float64 operator reference excludes GPU shape-dependent BF16 arithmetic.
 # Only first differing primitive is reproduced; no model precision/backend change.
 first=next((r for r in rows if not r['equal']),None)
 first_hidden=[]
 for key,(pos,value) in pc.hidden.items():
  dpos,dvalue=dc.hidden[key];lookup={int(z):i for i,z in enumerate(dpos)}
  aligned=dvalue[:,[lookup[int(z)] for z in pos],:]
  if not torch.equal(aligned,value):
   first_hidden.append(dict(step=key[0],layer=key[1],max_abs=float((aligned.float()-value.float()).abs().max())))
 first_hidden.sort(key=lambda r:(r['step'],r['layer']))
 # Explicit logical position / RoPE source checks, rather than interpreting agreement of answers as parity.
 for entry in pc.positions:
  step=entry['step'];i=entry['layer']
  expected=(list(range(n-1)) if i in range(2,6) else [z for z in range(n) if not(i>=20 and z in set(ready['roles']['image']))]) if step==0 else [n+step-1]
  assert entry['query_positions']==expected,(step,i)
 assert all(row['visual_kv']==0 for row in physical['kv_layout'])
 assert all(row['physical_length']==(0 if row['layer'] in range(2,6) else n-256+8) for row in physical['kv_layout'])
 save(RUN/'diagnostics/operator_diagnosis.json',dict(status='DIAGNOSTIC_ONLY',question_id=s['question_id'],
  exact_same_L2_input=True,first_changed_operator=first,first_changed_hidden=first_hidden[:5],
  causal_Pminus_key_unused_by_earlier_queries=True,mask_and_original_logical_positions_checked=True,
  original_RoPE_used_without_reindex=True,physical_decode_steps=8,
  dense_tokens=dense['token_ids'],physical_tokens=physical['token_ids'],
  tokens_equal=dense['token_ids']==physical['token_ids'],
  numerical_gates_unchanged=True,formal_quality='NOT_RUN',formal_efficiency='NOT_RUN',
  conclusion='Direct same-input operator isolation identifies a shape-dependent BF16 arithmetic difference. This is a localized observation, not permission to dismiss or waive failed numerical gates.',
  operators=rows))
 csvwrite(RUN/'diagnostics/operator_comparison.csv',rows,['operator','equal','max_abs','mean_abs','changed_elements','elements','full_shape','short_shape'])
 save(RUN/'diagnostics/decode8_physical_audit.json',physical)
 save(RUN/'diagnostics/decode8_dense_audit.json',dense)
 print(json.dumps(dict(first_changed_operator=first,first_hidden=first_hidden[:1],tokens_equal=dense['token_ids']==physical['token_ids'],real_decode=8)),flush=True)
 # Explicitly remove all GPU references before process exit.
 del model,prep,ready,dc,pc,ds,ps,capture,x,rope,mask,full,short,q,k,v,qs,ks,vs,qk,qks,prob,probs,av,avs,out,outs
 gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
