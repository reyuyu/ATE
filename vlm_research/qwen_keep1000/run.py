"""Historical Keep/D equivalence gate, then only the two requested quality schemes."""
import os,sys,gc,traceback,shutil
from .common import *
def main(rank,stage):
 import torch,transformers
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from ..qwen3vl_early_windows.experiment import prepare,hook_counts
 from ..qwen_recovery.engine import generate as latest_generate
 from .engine import generate
 assert stage in ['smoke','accuracy'] and os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[rank]
 assert read(OUT/'gpu_authorization.json')['authorized_this_round'];assert transformers.__version__=='4.57.6';torch.set_num_threads(1)
 source=check_sources();m=read(OUT/'sample_manifest.json');specs=read(OUT/'input_specs.json');p=read(OUT/'protocol.json')
 assert sha(OUT/'sample_manifest.json')==p['sample_manifest_sha256'] and sha(OUT/'input_specs.json')==p['input_specs_sha256']
 for f,h in read(OUT/'model_manifest.json')['files'].items():assert sha(MODEL/f)==h,f
 if stage=='accuracy':assert read(OUT/'equivalence_validation.json')['runtime_smoke_passed']
 tag=dict(rank=rank,stage=stage,uuid=UUIDS[rank],source_manifest_sha256=source,protocol_sha256=sha(OUT/'protocol.json'),model_manifest_sha256=sha(OUT/'model_manifest.json'),sample_manifest_sha256=sha(OUT/'sample_manifest.json'))
 write(OUT/f'{stage}/rank{rank}/environment.json',dict(**tag,at=now(),pid=os.getpid(),torch=torch.__version__,transformers=transformers.__version__,accuracy_only=True))
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
 assert torch.cuda.device_count()==1 and all(v.dtype==torch.bfloat16 for v in model.parameters());hc=hook_counts(model)
 samples=[s for s in m['samples'] if s['rank']==rank and (stage=='accuracy' or s['question_id'] in m['smoke_question_ids'])]
 def hashes(v):return [__import__('hashlib').sha256(x.numpy().tobytes()).hexdigest() for x in v]
 cases=[]
 for j,s in enumerate(samples):
  assert shutil.disk_usage('/data').free>20*2**30
  q=s['question_id'];x,l=prepare(processor,s);assert l==m['input_bindings'][q];spec=specs[q];assert spec['layout']==l
  bind=dict(**tag,question_id=q,input_sha256=digest(l),image_sha256=s['image_sha256'])
  if stage=='smoke':
   out=OUT/f'smoke/rank{rank}/{q}.json'
   if out.exists():d=read(out);assert d['binding']==bind;cases.append(d['comparison']['strict_equal']);continue
   dr,dv=latest_generate(model,processor,x,l,'D',audit=True,capture=True)
   assert dr['token_ids']==read(baseline_path(s,'D'))['result']['token_ids']
   old=read(RECOVERY/f'smoke/rank{rank}/{q}.json');assert hashes(dv)==old['valid_logit_sha256']['D'],'Latest D historical logits changed'
   kr,kv,kcache=generate(model,processor,x,spec,'Keep-L0-L21',audit=True)
   assert hook_counts(model)==hc
   nr,nv=latest_generate(model,processor,x,l,'Native',audit=True,capture=True)
   noop,npv,ncache=generate(model,processor,x,spec,'Keep-L0-L21',audit=True,noop=True)
   assert nr['token_ids']==noop['token_ids'] and len(nv)==len(npv) and all(torch.equal(a,b) for a,b in zip(nv,npv)),'Historical no-op differs from latest Native'
   ar,av,acache=generate(model,processor,x,spec,'All-Writer',audit=True)
   assert hook_counts(model)==hc
   assert all(a.keys.data_ptr()!=b.keys.data_ptr() and a.values.data_ptr()!=b.values.data_ptr() for a,b in zip(kcache.layers,acache.layers))
   strict=kr['token_ids']==dr['token_ids'] and len(kv)==len(dv) and all(torch.equal(a,b) for a,b in zip(kv,dv));steps=[]
   for t,(a,b) in enumerate(zip(kv,dv)):
    difference=(a-b).abs();steps.append(dict(step=t,torch_equal=bool(torch.equal(a,b)),max_abs=float(difference.max()),mean_abs=float(difference.mean()),top1_equal=int(a.argmax())==int(b.argmax())))
   write(out,dict(binding=bind,comparison=dict(strict_equal=strict,complete_token_equal=kr['token_ids']==dr['token_ids'],all_valid_logits_equal=len(kv)==len(dv) and all(torch.equal(a,b) for a,b in zip(kv,dv)),steps=steps),keep=kr,D=dr,All_Writer=ar,logit_sha256=dict(Keep=hashes(kv),D=hashes(dv),All_Writer=hashes(av)),checks=dict(native_noop_strict=True,D_matches_historical_tokens_logits_strict=True,prohibited_probability_zero=True,non_target_mask_exact=True,native_full_MLP_DeepStack_mrope_via_latest_D_audit=True,early_attention_not_zeroed=True,independent_cache_storage=True,hooks_clean=True)))
   cases.append(strict);del kcache,ncache,acache,dv,kv,nv,npv,av
  else:
   conditions=['All-Writer'] if read(OUT/'equivalence_validation.json')['reuse_D_for_Keep'] else ['Keep-L0-L21','All-Writer']
   for c in conditions:
    out=OUT/f'checkpoints/rank{rank}/{q}__{c}.json'
    if out.exists():assert read(out)['binding']==bind;continue
    r,v,cache=generate(model,processor,x,spec,c,audit=False);assert hook_counts(model)==hc
    r.update(gold=s['gold'],correct=r['prediction']==s['gold'],question=s['question'],task=s['task']);write(out,dict(binding=bind,result=r,at=now()))
  write(OUT/f'{stage}/rank{rank}/progress.json',dict(**tag,completed_questions=j+1,total_questions=len(samples),last_question=q,at=now()));print(stage,rank,j+1,'/',len(samples),q,flush=True);del x;gc.collect()
 write(OUT/f'{stage}/rank{rank}/completion.json',dict(**tag,status='PASSED' if stage=='smoke' else 'COMPLETE',questions=len(samples),all_Keep_D_strict=all(cases) if stage=='smoke' else None,at=now(),accuracy_only=True))
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 rank=int(sys.argv[1]);stage=sys.argv[2]
 try:main(rank,stage)
 except BaseException as e:write(OUT/f'{stage}/rank{rank}/failure.json',dict(status='STOPPED',at=now(),error=repr(e),traceback=traceback.format_exc()));raise
