"""Quality inference only; no timers, events or performance measurements."""
import os,sys,gc,shutil,traceback
from .common import *

def load_model():
 import torch,transformers
 from transformers import LlavaForConditionalGeneration
 assert transformers.__version__=='4.49.0' and torch.cuda.device_count()==1
 torch.set_num_threads(1);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 processor=processor_cpu();model=LlavaForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16,attn_implementation='eager',device_map={'':'cuda:0'},low_cpu_mem_usage=True,use_safetensors=True).eval()
 assert all(p.dtype==torch.float16 and p.device.type=='cuda' for p in model.parameters());assert len(model.language_model.model.layers)==32
 assert model.config.text_config._attn_implementation==model.config.vision_config._attn_implementation=='eager'
 model.generation_config.do_sample=False;model.generation_config.temperature=1.;model.generation_config.top_p=1.;model.generation_config.top_k=50
 return model,processor

def smoke(model,processor,x,sample,binding,key,tag):
 import torch
 from .engine import generate,footprint
 m=sample['input_mapping'];native,nv,nc=generate(model,processor,x,m,'N',audit=True);noop,zv,zc=generate(model,processor,x,m,'B',audit=True,noop=True)
 assert native['token_ids']==noop['token_ids'] and len(nv)==len(zv) and all(torch.equal(a,b) for a,b in zip(nv,zv))
 old=read(OUT/f'baselines/{key}/N.json');assert old['generation']['token_ids']==native['token_ids'] and old['generation']['first_logits_sha256']==native['first_logits_sha256']
 hf_scores=[]
 def observe(module,args,out):hf_scores.append(out.logits[:,-1].detach().float().cpu().clone())
 handle=model.register_forward_hook(observe)
 try:
  with torch.inference_mode():hf=model.generate(**{k:v.to('cuda:0') for k,v in x.items()},do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,logits_to_keep=1)
 finally:handle.remove()
 assert hf[0,len(m['prompt_ids']):].tolist()==native['token_ids'] and len(hf_scores)==len(nv) and all(torch.equal(a,b) for a,b in zip(hf_scores,nv))
 ar,av,ac=generate(model,processor,x,m,'A',audit=True);br,bv,bc=generate(model,processor,x,m,'B',audit=True)
 assert native['prefix_hidden_sha256']==ar['prefix_hidden_sha256']==br['prefix_hidden_sha256'] and len(native['prefix_hidden_sha256'])==32
 assert native['visual_feature_sha256']==ar['visual_feature_sha256']==br['visual_feature_sha256']==noop['visual_feature_sha256'] and len(native['visual_feature_sha256'])==1
 for r in [ar,br]:
  for a,b in zip(native['rotary_sha256'],r['rotary_sha256']):assert a==b
 assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(ac.key_cache,bc.key_cache)) and all(a.data_ptr()!=b.data_ptr() for a,b in zip(ac.value_cache,bc.value_cache))
 checks=dict(native_noop_all_valid_logits_torch_equal=True,native_HF_all_valid_logits_torch_equal=True,historical_native_tokens_first_logits_strict=True,all32_prefill_before_P1_hidden_N_A_B_strict=True,visual_features_identical=True,RoPE_original=True,all32_R_visual_writing_preserved=True,actual_forbidden_probability_zero=True,non_target_connections_and_outputs_exact=True,MLP_norm_and_full_blocks_preserved=True,independent_dense_KV_storage=True,hook_cleanup=True)
 atomic(OUT/f'smoke/rank{tag["rank"]}/{key}.json',dict(binding=tag,key=key,checks=checks,native_valid_logit_vectors=len(nv),Native=native,Noop=noop,A=ar,B=br))

def main(rank,stage):
 import torch,transformers
 from .engine import generate,footprint
 assert stage in ['smoke','accuracy'] and os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[rank]
 assert read(OUT/'gpu_authorization.json')['authorized_this_round'];source=verify_sources();p=read(OUT/'protocol.json');samples=read(OUT/'sample_manifest.json');part=read(OUT/'partition.json');bindings=read(OUT/'input_bindings.json')
 for name in ['sample_manifest','input_bindings','model_manifest']:assert sha(OUT/(name+'.json'))==p[name+'_sha256']
 for path,h in read(OUT/'model_manifest.json')['files'].items():assert sha(path)==h,path
 if stage=='accuracy':assert read(OUT/'gpu_smoke_validation.json')['status']=='PASSED'
 tag=dict(rank=rank,stage=stage,uuid=UUIDS[rank],source_manifest_sha256=source,protocol_sha256=sha(OUT/'protocol.json'),model_manifest_sha256=sha(OUT/'model_manifest.json'),sample_manifest_sha256=sha(OUT/'sample_manifest.json'))
 atomic(OUT/f'{stage}/rank{rank}/environment.json',dict(**tag,pid=os.getpid(),at=now(),torch=torch.__version__,transformers=transformers.__version__))
 model,processor=load_model();versions=[(v._version,v.data_ptr()) for v in model.parameters()];hooks=footprint(model)
 keys=sorted([k for k,s in samples.items() if part['image_shard'][s['image_sha256']]==rank and (stage=='accuracy' or k in p['smoke_ids'])],key=lambda k:(samples[k]['image_sha256'],k))
 for j,key in enumerate(keys):
  assert not list(OUT.glob('*/rank*/failure.json')),'A task worker failed; stop this task';assert shutil.disk_usage('/data').free>20*2**30
  s=samples[key];x,binding=prepare_input(processor,s);assert binding==bindings[key];bind=dict(**tag,key=key,input_sha256=digest(binding),image_sha256=s['image_sha256'],sample_sha256=digest(s))
  if stage=='smoke':smoke(model,processor,x,s,binding,key,bind)
  else:
   for c in ['A','B']:
    path=OUT/f'checkpoints/{key}/{c}.json'
    if path.exists():assert read(path)['binding']==bind;continue
    r,_,_=generate(model,processor,x,s['input_mapping'],c);atomic(path,dict(binding=bind,generation=r,score=float(textvqa_score(r['text'],s['answer'])),at=now()))
  assert hooks==footprint(model) and versions==[(v._version,v.data_ptr()) for v in model.parameters()]
  atomic(OUT/f'{stage}/rank{rank}/progress.json',dict(questions=j+1,total=len(keys),at=now()));print(stage,rank,j+1,'/',len(keys),key,flush=True);del x;gc.collect()
 atomic(OUT/f'{stage}/rank{rank}/completion.json',dict(**tag,status='PASSED' if stage=='smoke' else 'COMPLETE',questions=len(keys),at=now()));del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
 rank=int(sys.argv[1]);stage=sys.argv[2]
 try:main(rank,stage)
 except BaseException as e:atomic(OUT/f'{stage}/rank{rank}/failure.json',dict(status='STOPPED',at=now(),error=repr(e),traceback=traceback.format_exc()));raise
