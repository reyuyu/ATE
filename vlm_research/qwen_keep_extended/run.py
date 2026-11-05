"""Two fixed cutoff conditions, no timing or additional search."""
import sys,os,gc,traceback,shutil
from .common import *

def main(rank,stage):
 import torch,transformers,hashlib
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from ..qwen3vl_early_windows.experiment import prepare,hook_counts
 from ..qwen_keep1000.engine import generate as old_generate
 from ..qwen3vl_query_role.engine import generate as native_generate
 from .engine import generate
 assert stage in ['smoke','accuracy'] and os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[rank] and transformers.__version__=='4.57.6'
 torch.set_num_threads(1);source=check_sources();p=read(OUT/'protocol.json');m=read(OUT/'sample_manifest.json');specs=read(OUT/'input_specs.json')
 assert sha(OUT/'sample_manifest.json')==p['sample_manifest_sha256'] and sha(OUT/'input_specs.json')==p['input_specs_sha256']
 for path,h in read(OUT/'model_manifest.json')['files'].items():assert sha(MODEL/path)==h
 if stage=='accuracy':assert read(OUT/'gpu_smoke_validation.json')['status']=='PASSED'
 tag=dict(rank=rank,stage=stage,uuid=UUIDS[rank],source_manifest_sha256=source,protocol_sha256=sha(OUT/'protocol.json'),model_manifest_sha256=sha(OUT/'model_manifest.json'),sample_manifest_sha256=sha(OUT/'sample_manifest.json'))
 write(OUT/f'{stage}/rank{rank}/environment.json',dict(**tag,pid=os.getpid(),torch=torch.__version__,transformers=transformers.__version__,at=now()))
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval();assert torch.cuda.device_count()==1;hc=hook_counts(model)
 samples=[s for s in m['samples'] if s['rank']==rank and (stage=='accuracy' or s['question_id'] in p['smoke_ids'])]
 for j,s in enumerate(samples):
  assert shutil.disk_usage('/data').free>20*2**30
  q=s['question_id'];x,layout=prepare(processor,s);assert layout==m['input_bindings'][q]==specs[q]['layout'];spec=specs[q];bind=dict(**tag,question_id=q,image_sha256=s['image_sha256'],input_sha256=digest(layout))
  if stage=='smoke':
   old,ov,oc=old_generate(model,processor,x,spec,'Keep-L0-L21',audit=True);new,nv,nc=generate(model,processor,x,spec,21,audit=True)
   assert old['token_ids']==new['token_ids']==read(reference_path(s,'Keep-L0-L21'))['result']['token_ids'] and len(ov)==len(nv) and all(torch.equal(a,b) for a,b in zip(ov,nv))
   previous=read(PREVIOUS/f'smoke/rank{rank}/{q}.json');assert [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in nv]==previous['logit_sha256']['Keep']
   native,pv,observer,pc=native_generate(model,processor,{k:v.to('cuda:0') for k,v in x.items()},spec['roles'],native=True,audit=True,trace=False)
   noop,zv,zc=generate(model,processor,x,spec,26,noop=True,audit=True);assert native['token_ids']==noop['token_ids'] and len(pv)==len(zv) and all(torch.equal(a,b) for a,b in zip(pv,zv))
   results={};caches=[]
   for c,end in CONDITIONS.items():
    r,v,cache=generate(model,processor,x,spec,end,audit=True);results[c]=r;caches.append(cache)
   assert all(a.keys.data_ptr()!=b.keys.data_ptr() and a.values.data_ptr()!=b.values.data_ptr() for a,b in zip(caches[0].layers,caches[1].layers));assert hook_counts(model)==hc
   write(OUT/f'smoke/rank{rank}/{q}.json',dict(binding=bind,checks=dict(Keep21_matches_history_logits_tokens_strict=True,native_noop_all_valid_logits_strict=True,prohibited_probability_zero=True,non_target_mask_exact=True,independent_cache_storage=True,hooks_clean=True),results=results,Keep21_logit_vectors=len(nv)));del oc,nc,pc,zc,caches,cache,ov,nv,pv,zv
  else:
   for c,end in CONDITIONS.items():
    path=OUT/f'checkpoints/rank{rank}/{q}__{c}.json'
    if path.exists():assert read(path)['binding']==bind;continue
    r,v,cache=generate(model,processor,x,spec,end);assert hook_counts(model)==hc;r.update(gold=s['gold'],correct=r['prediction']==s['gold'],task=s['task'],question=s['question']);write(path,dict(binding=bind,result=r,at=now()))
  write(OUT/f'{stage}/rank{rank}/progress.json',dict(completed_questions=j+1,total_questions=len(samples),at=now()));print(stage,rank,j+1,'/',len(samples),q,flush=True);del x;gc.collect()
 write(OUT/f'{stage}/rank{rank}/completion.json',dict(**tag,status='PASSED' if stage=='smoke' else 'COMPLETE',questions=len(samples),at=now()));del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
 rank=int(sys.argv[1]);stage=sys.argv[2]
 try:main(rank,stage)
 except BaseException as e:write(OUT/f'{stage}/rank{rank}/failure.json',dict(status='STOPPED',error=repr(e),traceback=traceback.format_exc(),at=now()));raise
