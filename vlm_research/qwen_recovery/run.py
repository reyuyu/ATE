"""Only Smoke and answer-accuracy collection. No performance stage exists."""
import os,sys,gc,traceback,shutil
from .common import *
def main(rank,stage):
 import torch,transformers
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from ..qwen3vl_early_windows.experiment import prepare,hook_counts,gqa_prediction
 from .engine import generate
 assert stage in ['smoke','accuracy'];assert os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[rank]
 assert read(OUT/'gpu_authorization.json')['authorized_this_round']
 assert transformers.__version__=='4.57.6';torch.set_num_threads(1)
 source=check_sources();m=read(OUT/'sample_manifest.json');protocol=read(OUT/'protocol.json');assert sha(OUT/'sample_manifest.json')==protocol['sample_manifest_sha256']
 for f,h in read(OUT/'model_manifest.json')['files'].items():assert sha(MODEL/f)==h,f
 if stage=='accuracy':assert all(read(OUT/f'smoke/rank{i}/completion.json')['status']=='PASSED' for i in range(2))
 tag=dict(rank=rank,stage=stage,uuid=UUIDS[rank],source_manifest_sha256=source,protocol_sha256=sha(OUT/'protocol.json'),model_manifest_sha256=sha(OUT/'model_manifest.json'),sample_manifest_sha256=sha(OUT/'sample_manifest.json'))
 write(OUT/f'{stage}/rank{rank}/environment.json',dict(**tag,at=now(),pid=os.getpid(),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),accuracy_only=True))
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
 assert torch.cuda.device_count()==1 and all(p.dtype==torch.bfloat16 for p in model.parameters())
 hooks=hook_counts(model)
 samples=[s for s in m['samples'] if s['rank']==rank and (stage=='accuracy' or s['question_id'] in m['smoke_question_ids'])]
 def call(x,l,c,**kw):
  r,v=generate(model,processor,x,l,c,**kw);assert hook_counts(model)==hooks,'Hook leak';return r,v
 def strict(a,b,av,bv):assert a['token_ids']==b['token_ids'] and len(av)==len(bv) and all(torch.equal(x,y) for x,y in zip(av,bv)),'Valid logits or tokens changed'
 def hashes(vv):return [__import__('hashlib').sha256(v.numpy().tobytes()).hexdigest() for v in vv]
 for j,s in enumerate(samples):
  assert shutil.disk_usage('/data').free>20*2**30
  q=s['question_id'];x,l=prepare(processor,s);assert l==m['input_bindings'][q]
  binding=dict(**tag,question_id=q,image_sha256=s['image_sha256'],input_sha256=digest(l))
  if stage=='smoke':
   p=OUT/f'smoke/rank{rank}/{q}.json'
   if p.exists():assert read(p)['binding']==binding;continue
   nr,nv=call(x,l,'Native',audit=True,capture=True);noop,nopv=call(x,l,'Native',audit=True,noop=True,capture=True);strict(nr,noop,nv,nopv)
   old=read(HIST/f'smoke/rank{rank}/{q}.json');assert hashes(nv)==old['logit_sha256']['Native']
   fr,fv=call(x,l,'F',audit=True,capture=True);lr,lv=call(x,l,'F',legacy=True,capture=True);strict(fr,lr,fv,lv)
   assert hashes(fv)==old['logit_sha256']['Dense-F']
   results={};allhash={};toggle_checks={};a_first=None
   for c in CONDITIONS:
    r,v=call(x,l,c,audit=True,capture=True);assert r['audit']['visual_features_sha256']==nr['audit']['visual_features_sha256'];results[c]=r;allhash[c]=hashes(v)
    if c=='A':a_first=v[0]
    if c=='B':assert torch.equal(v[0],fv[0]),'Decode-only restoration changed Prefill logits'
    if c=='C':assert torch.equal(v[0],a_first),'A/C Prefill logits differ'
    toggle_checks[c]='NOT_RUN'
    if j in [0,len(samples)-1]:
     rr,vv=call(x,l,c,audit=False,capture=True);strict(r,rr,v,vv);toggle_checks[c]='PASSED'
    # Only declared visual access changes; early-zero or restored output is validated inside hooks.
    scopes=[r0 for r0 in r['audit']['records'] if 'layer' in r0]
    assert len(scopes)==36*len(r['token_ids']) and all(r0['non_target_mask_exact'] for r0 in scopes)
    for row in scopes:
     spec=SPECS[c];allow=(spec['p1'] if row['step']==0 else spec['decode']) and row['layer'] not in SKIP
     assert row['p1_or_decode_image_allowed']==allow
   write(p,dict(binding=binding,results=results,valid_logit_sha256=allhash,checks=dict(native_noop_strict=True,historical_native_logits_strict=True,new_F_legacy_F_logits_strict=True,historical_F_logits_strict=True,dense_cache_full_image_hidden_all_layers=True,mrope_absolute_positions_correct=True,deepstack_visual_features_strict=True,non_target_mask_strict=True,forbidden_probability_zero=True,all36_mlp_norm_residual=True,independent_cache=True,hooks_clean=True,audit_toggle=toggle_checks)))
   del nv,nopv,fv,lv
  else:
   for c in CONDITIONS:
    p=OUT/f'checkpoints/rank{rank}/{q}__{c}.json'
    if p.exists():assert read(p)['binding']==binding;continue
    r,_=call(x,l,c,audit=False);prediction=gqa_prediction(r['text']);r.update(prediction=prediction,gold=s['gold'],correct=prediction==s['gold'],question=s['question'],task=s['task'])
    if c=='B':assert r['first_token']==read(HIST/f'checkpoints/rank{rank}/{q}__Dense-F.json')['result']['first_token']
    if c=='C':assert r['first_token']==read(OUT/f'checkpoints/rank{rank}/{q}__A.json')['result']['first_token']
    write(p,dict(binding=binding,result=r,at=now()))
  write(OUT/f'{stage}/rank{rank}/progress.json',dict(**tag,completed_questions=j+1,total_questions=len(samples),last_question=q,at=now()))
  print(stage,rank,j+1,'/',len(samples),q,flush=True);del x;gc.collect()
 write(OUT/f'{stage}/rank{rank}/completion.json',dict(**tag,status='PASSED' if stage=='smoke' else 'COMPLETE',questions=len(samples),at=now(),accuracy_only=True,no_performance_tests=True))
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 rank=int(sys.argv[1]);stage=sys.argv[2]
 try:main(rank,stage)
 except BaseException as e:write(OUT/f'{stage}/rank{rank}/failure.json',dict(status='STOPPED',at=now(),error=repr(e),traceback=traceback.format_exc()));raise
