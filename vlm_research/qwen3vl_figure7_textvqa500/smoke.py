import sys,traceback,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common import *
from engine import *
def once(model,processor,inputs,row,keep=None,audit=False,cap=8,ignore=True):
 permission=DecodeImagePermission(model,row['image_positions'],keep,audit=audit) if keep is not None else None
 with permission if permission is not None else contextlib.nullcontext():
  first,cache,rope=prefill(model,inputs)
  result,cache,raw=decode(model,inputs,first,cache,rope,permission,processor,cap=cap,ignore_eos=ignore,logits_record=True)
  shapes=layout(cache,row['image_tokens'])
 return result,raw,shapes,permission.events if permission else []
def run(model,processor):
 rows=read(OUT/'sample_manifest.json');ordered=sorted(rows,key=lambda k:rows[k]['input_tokens'])
 candidates=[ordered[0],ordered[len(ordered)//2]]+list(reversed(ordered))
 selected=[];seen=set();records=[];blocked=[];all_events=[]
 old=read(RUN/'resources/eager_capacity_smoke.json')
 assert old['status']=='BLOCKED_EAGER_OOM'
 blocked.append(dict(key=old['sample_key'],image_sha256=rows[old['sample_key']]['image_sha256'],status='BLOCKED_OOM',stage='Native full-resolution all-Eager capacity smoke',evidence_path='resources/eager_capacity_smoke.json',evidence_sha256=sha(RUN/'resources/eager_capacity_smoke.json'),error=old['error']))
 blocked_images={blocked[0]['image_sha256']}
 for key in candidates:
  if key in seen or rows[key]['image_sha256'] in blocked_images:continue
  seen.add(key);print('SMOKE_SAMPLE',key,rows[key]['input_tokens'],flush=True)
  inputs=None
  try:
   row,inputs=prepared(key)
   native,nlog,nshape,_=once(model,processor,inputs,row)
  except torch.OutOfMemoryError as exc:
   blocked.append(dict(key=key,image_sha256=rows[key]['image_sha256'],status='BLOCKED_OOM',stage='Native capacity preflight',error=str(exc),traceback=traceback.format_exc()))
   blocked_images.add(rows[key]['image_sha256']);write(OUT/'smoke_capacity_blocked.json',blocked)
   inputs=None;gc.collect();torch.cuda.empty_cache();continue
  noop,zlog,zshape,events=once(model,processor,inputs,row,list(range(N)),audit=True)
  assert native['token_ids']==noop['token_ids'] and all(torch.equal(a,b) for a,b in zip(nlog,zlog)) and nshape==zshape
  assert all(not e['blocked'] for e in events)
  all_events += [dict(key=key,condition='NoOp',**e) for e in events]
  # Every QKV/Norm/MLP remains the original native forward implementation.
  module_calls={};handles=[]
  for i,l in enumerate(model.model.language_model.layers):
   for name in ['q_proj','k_proj','v_proj','o_proj']:
    tag=f'L{i}.{name}';module_calls[tag]=0
    def hook(m,a,o,tag=tag):module_calls[tag]+=1
    handles.append(getattr(l.self_attn,name).register_forward_hook(hook))
   for name in ['input_layernorm','post_attention_layernorm','mlp']:
    tag=f'L{i}.{name}';module_calls[tag]=0
    def hook(m,a,o,tag=tag):module_calls[tag]+=1
    handles.append(getattr(l,name).register_forward_hook(hook))
  first,base,rope=prefill(model,inputs)
  originals=[(k.clone(),v.clone()) for k,v in pairs(base)]
  condition_results={}
  try:
   for name,keep in [('NoImage',[]),('Random18',configs()['Random18'])]:
    with DecodeImagePermission(model,row['image_positions'],keep,audit=True) as permission:
     fork=clone_cache(model,base)
     got,fork,glog=decode(model,inputs,first,fork,rope,permission,processor,cap=8,ignore_eos=True,logits_record=True)
     assert len(got['token_ids'])==8 and got['decode_steps']==7
     assert got['token_ids'][0]==native['token_ids'][0] and got['prefill_logits_sha256']==native['prefill_logits_sha256']
     for (bk,bv),(ok,ov) in zip(pairs(base),originals):assert torch.equal(bk,ok) and torch.equal(bv,ov)
     fshape=layout(fork,row['image_tokens']);assert fshape==nshape
     all_events += [dict(key=key,condition=name,**e) for e in permission.events]
     condition_results[name]=dict(token_ids=got['token_ids'],fork_base_unchanged=True,blocked_probabilities_zero=True,kv_layout_native=True)
     del fork,glog
    fresh,flog,_,events=once(model,processor,inputs,row,keep,audit=True)
    # Raw logit SHA covers every valid output step; equality is bitwise.
    assert fresh['logit_hashes']==got['logit_hashes'] and fresh['token_ids']==got['token_ids']
  finally:
   for h in handles:h.remove()
  assert all(v>0 for v in module_calls.values()),module_calls
  # Official HF generation and manual native cache decode match to natural EOS.
  full,_,_,_=once(model,processor,inputs,row,None,cap=32,ignore=False)
  generated=model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True,logits_to_keep=1)
  assert generated[:,inputs['input_ids'].shape[1]:].tolist()[0]==full['token_ids']
  records.append(dict(key=key,image_tokens=row['image_tokens'],input_tokens=row['input_tokens'],native_noop_logits_torch_equal=True,first_token_native_for_all=True,independent_cache_fork_vs_fresh_bitwise=True,HF_generate_manual_full_tokens_equal=True,native_full=full,conditions=condition_results,kv_layout=nshape,native_module_calls=module_calls))
  selected.append(key)
  write(OUT/'smoke_capacity_blocked.json',blocked)
  write(OUT/'smoke_progress.json',dict(passed=selected,blocked=blocked))
  del inputs,first,base,rope,originals,nlog,zlog;gc.collect();torch.cuda.empty_cache()
  if len(selected)==3:break
 assert len(selected)==3
 write(OUT/'smoke_attention_audit.json',all_events)
 result=dict(status='PASS',samples=records,capacity_blocked=blocked,real_decode_steps_each=7,mask_only_true_decode=True,native_prefill_Pminus1=True,all36_dense_native_QKV_Norm_MLP_RoPE_Cache=True,native_noop_bitwise=True,cache_fork_vs_fresh_bitwise=True,all_actual_forbidden_softmax_probabilities_zero=True,no_physical_compression_or_SVH_skip=True,model_loads_this_process=1)
 write(OUT/'smoke_summary.json',result)
 print('SMOKE_PASS',selected,flush=True);return result
def main():
 gpu();model=load_model();processor=AutoProcessor.from_pretrained(S['model'],local_files_only=True)
 with torch.inference_mode():run(model,processor)
if __name__=='__main__':main()
