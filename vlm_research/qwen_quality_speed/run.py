import os,sys,gc,traceback,time,subprocess,shutil
from .common import *

def main(rank,stage):
 import torch,transformers
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from ..qwen3vl_early_windows.experiment import prepare,hook_counts,gqa_prediction
 from .generate import generate
 torch.set_num_threads(1)
 assert os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[rank]
 assert transformers.__version__=='4.57.6'
 binding=check_sources();manifest=read(OUT/'sample_manifest.json');protocol=read(OUT/'protocol.json')
 for f,h in read(OUT/'model_manifest.json')['files'].items():assert sha(MODEL/f)==h,f
 assert sha(OUT/'sample_manifest.json')==protocol['sample_manifest_sha256']
 if stage!='smoke':
  assert all(read(OUT/f'smoke/rank{i}/completion.json')['status']=='PASSED' for i in range(2))
 samples=[r for r in manifest['samples'] if r['rank']==rank]
 ids=manifest['smoke_question_ids'] if stage=='smoke' else manifest['benchmark_question_ids'] if stage=='benchmark' else [r['question_id'] for r in samples]
 samples=[r for r in samples if r['question_id'] in ids]
 tag=dict(rank=rank,stage=stage,source_manifest_sha256=binding,protocol_sha256=sha(OUT/'protocol.json'),model_manifest_sha256=sha(OUT/'model_manifest.json'),sample_manifest_sha256=sha(OUT/'sample_manifest.json'),uuid=UUIDS[rank])
 write(OUT/f'{stage}/rank{rank}/environment.json',dict(**tag,at=now(),pid=os.getpid(),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0)))
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
 assert all(p.dtype==torch.bfloat16 for p in model.parameters())
 hooks=hook_counts(model);started=time.time()
 def call(x,l,c,**kw):
  result,vectors=generate(model,processor,x,l,c,**kw)
  assert hook_counts(model)==hooks,'Hook leak'
  return result,vectors
 def same(a,b,av,bv):
  assert a['token_ids']==b['token_ids'] and len(av)==len(bv) and all(torch.equal(x,y) for x,y in zip(av,bv)),'Probe removal / no-op changed valid logits'
 for j,s in enumerate(samples):
  assert shutil.disk_usage('/data').free>20*2**30
  q=s['question_id'];x,l=prepare(processor,s);assert l==manifest['input_bindings'][q],q
  sb=dict(**tag,question_id=q,input_sha256=digest(l),image_sha256=s['image_sha256'])
  if stage=='smoke':
   p=OUT/f'smoke/rank{rank}/{q}.json'
   if p.exists():assert read(p)['binding']==sb;continue
   results={};vectors={}
   for c in CONDITIONS:
    r,v=call(x,l,c,audit=True,capture=True);results[c]=r;vectors[c]=v
   nr,nv=call(x,l,'Native',audit=True,capture=True,noop=True)
   same(results['Native'],nr,vectors['Native'],nv)
   # Audit-disabled loop must be bitwise identical to audited repaired math.
   fr,fv=call(x,l,'Physical-Repaired',audit=False,capture=True)
   same(results['Physical-Repaired'],fr,vectors['Physical-Repaired'],fv)
   frozen=False
   if j in [0,len(samples)-1]:
    dr,dv=call(x,l,'Physical-Repaired',audit=True,capture=True,frozen_debug=True)
    same(results['Physical-Repaired'],dr,vectors['Physical-Repaired'],dv);frozen=True
   for c in ['Dense-F','Physical-Original','Physical-Repaired']:
    a=results[c]['audit'];n=len(results[c]['token_ids'])
    assert len([r for r in a['attention_events'] if r.get('mlp_and_norm_executed')])==36*n
    if c!='Dense-F':
     assert all(r['hidden_rows']==(l['input_length'] if r['layer']<22 else l['input_length']-l['image_tokens']) for r in a['layers'] if r['step']==0)
     assert all(results[c]['kv']['layers'][i]['live_bytes']==0 for i in range(1,7))
     assert a['deepstack']==[0,1,2]
   write(p,dict(binding=sb,results=results,checks=dict(native_noop_all_valid_logits_strict=True,repaired_audit_on_off_all_valid_logits_strict=True,frozen_repaired_debug_strict=frozen,all36_mlp_norm_residual=True,physical_exit_and_compact_cache=True,independent_execution_cache=True,hooks_clean=True),logit_sha256={c:[__import__('hashlib').sha256(v.numpy().tobytes()).hexdigest() for v in vv] for c,vv in vectors.items()}))
   del vectors,nv,fv
   if frozen:del dv
  elif stage=='quality':
   for c in CONDITIONS:
    p=OUT/f'checkpoints/rank{rank}/{q}__{c}.json'
    if p.exists():assert read(p)['binding']==sb;continue
    r,v=call(x,l,c,audit=False,check_finite=True)
    pred=gqa_prediction(r['text']);r.update(prediction=pred,gold=s['gold'],correct=pred==s['gold'],question=s['question'],task=s['task'])
    write(p,dict(binding=sb,result=r,at=now()))
  else:
   # Both references are made outside timing. Fixed16 ignores EOS only for speed control.
   refs={}
   for fixed in [False,True]:
    for c in CONDITIONS:
     if not fixed:
      refs[fixed,c]=read(OUT/f'checkpoints/rank{rank}/{q}__{c}.json')['result']['token_ids']
     else:
      rr,_=call(x,l,c,audit=False,fixed16=True,check_finite=True);refs[fixed,c]=rr['token_ids']
   rotation=int(digest(q)[:8],16)%4
   for fixed in [False,True]:
    regime='fixed16' if fixed else 'natural'
    for rep in range(-2,5):
     order=CONDITIONS[(rotation+rep)%4:]+CONDITIONS[:(rotation+rep)%4]
     for c in order:
      p=OUT/f'benchmark/rank{rank}/raw/{q}__{regime}__{c}__{rep}.json'
      if rep>=0 and p.exists():assert read(p)['binding']==sb;continue
      r,_=call(x,l,c,audit=False,fixed16=fixed,check_finite=False)
      assert r['token_ids']==refs[fixed,c],('Non-repeatable benchmark tokens',q,c,fixed,rep)
      if rep>=0:write(p,dict(binding=sb,regime=regime,repeat=rep,condition_order=order,result=r,at=now(),reference_finite_checked=True))
  write(OUT/f'{stage}/rank{rank}/progress.json',dict(**tag,completed_questions=j+1,total_questions=len(samples),last_question=q,elapsed_seconds=time.time()-started,at=now()))
  print(stage,rank,j+1,'/',len(samples),q,'seconds',round(time.time()-started,1),flush=True)
  del x;gc.collect()
 write(OUT/f'{stage}/rank{rank}/completion.json',dict(**tag,status='PASSED' if stage=='smoke' else 'COMPLETE',questions=len(samples),elapsed_seconds=time.time()-started,at=now()))
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 rank=int(sys.argv[1]);stage=sys.argv[2]
 try:main(rank,stage)
 except BaseException as exc:
  write(OUT/f'{stage}/rank{rank}/failure.json',dict(status='STOPPED',error=repr(exc),traceback=traceback.format_exc(),at=now()));raise
