from pathlib import Path
import os,json,sys,traceback,gc,hashlib
R=Path('/home/featurize/ate-runs/qwen3vl-figure7-textvqa500');sys.path.insert(0,str(R/'code'))
from common import *
import torch
from transformers import Qwen3VLForConditionalGeneration
def main():
 g=gpu();samples=read(OUT/'sample_manifest.json');key=max(samples,key=lambda k:samples[k]['image_tokens']);s=samples[key];b=read(OUT/'input_bindings.json')[key];p=RUN/'prepared'/(s['question_id']+'.inputs.pt');assert sha(p)==b['input_file_sha256']
 x=torch.load(p,map_location='cpu',weights_only=True);pp=RUN/'prepared'/b['pixels']['file'];assert sha(pp)==b['pixels']['sha256'];x['pixel_values']=torch.load(pp,map_location='cpu',weights_only=True)
 result=dict(status='RUNNING',purpose='Native full-resolution Eager capacity smoke only; no formal quality predictions',sample_key=key,question_id=s['question_id'],input_tokens=s['input_tokens'],visual_tokens=s['image_tokens'],GPU=g,source_sha256=sha(Path(__file__)),input_binding=s['input_binding'],backend=dict(text='eager',vision='eager'),dtype='bfloat16',source_figure7_commit=S['figure7_commit'])
 model=None
 try:
  model=Qwen3VLForConditionalGeneration.from_pretrained(S['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True).eval()
  torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():
   x={k:v.to('cuda:0') for k,v in x.items()}
   out=model(**x,use_cache=True,logits_to_keep=1,return_dict=True);assert torch.isfinite(out.logits).all();result.update(status='PASS',first_token=int(out.logits[:,-1].argmax(-1)),cache_layers=len(out.past_key_values.layers));del out
 except torch.cuda.OutOfMemoryError as e:
  result.update(status='BLOCKED_EAGER_OOM',error=str(e),traceback=traceback.format_exc(),no_smaller_image_or_backend_fallback=True)
 finally:
  result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  x=None;model=None;gc.collect();torch.cuda.empty_cache()
  result.update(released_allocated_bytes=torch.cuda.memory_allocated(),released_reserved_bytes=torch.cuda.memory_reserved())
  write(RUN/'resources/eager_capacity_smoke.json',result)
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
