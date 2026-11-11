from common import *
import gzip,torch
from PIL import Image
from transformers import AutoProcessor
def main():
 assert not torch.cuda.is_initialized()
 assert read(RUN/'resources/environment_status.json')['status']=='PASS' and read(RUN/'resources/model_status.json')['status']=='VERIFIED'
 ref=RUN/'references/figure7';original=json.loads(gzip.decompress((ref/'sample_manifest.json.gz').read_bytes()));old=read(ref/'protocol.json');keys=list(original)
 assert len(keys)==500 and keys==sorted(keys) and hashlib.sha256(json.dumps(keys,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==old['order_sha256']
 assert len({s['image_sha256'] for s in original.values()})==309
 bundle=Path('/home/featurize/ate-local-data/bundles/ate3090-textvqa-ocr');inventory=read(bundle/'image_sha256.json')
 proc=AutoProcessor.from_pretrained(S['model'],local_files_only=True);modelconfig=read(Path(S['model'])/'config.json');assert modelconfig['text_config']['num_hidden_layers']==36
 files={};samples={};bindings={};imagefiles={};verified={};source_annotations=list((bundle/'source_annotations').rglob('*.json'))
 goldrows=None
 for path in source_annotations:
  if 'textvqa' in path.name.lower() and ('val' in path.name.lower()):
   d=read(path)
   if isinstance(d,dict) and 'data' in d and len(d['data'])==5000:goldrows={str(r['question_id']):r for r in d['data']};break
 assert goldrows is not None,source_annotations
 for i,(key,s) in enumerate(original.items()):
  q=str(s['question_id']);g=goldrows[q];assert g['question']==s['question'] and g['image_id']==s['image_id'] and g['answers']==s['answer'] and len(s['answer'])==10
  assert s['prompt_text']==s['question']+'\nAnswer the question using a single word or phrase.'
  path=bundle/'images/textvqa'/(s['image_id']+'.jpg');h=s['image_sha256']
  if h not in verified:
   assert sha(path)==h;verified[h]=dict(path=str(path),bytes=path.stat().st_size,sha256=h)
  chat=proc.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=s['prompt_text'])])],tokenize=False,add_generation_prompt=True)
  with Image.open(path) as im:x=proc(images=im.convert('RGB'),text=chat,return_tensors='pt')
  assert set(x)=={'input_ids','attention_mask','pixel_values','image_grid_thw'},list(x)
  x['pixel_values']=x['pixel_values'].to(torch.bfloat16)
  pixels=x.pop('pixel_values');vis=(x['input_ids'][0]==modelconfig['image_token_id']).nonzero().flatten().tolist()
  assert len(vis)==int(x['image_grid_thw'].prod())//4 and len(vis)>0
  if h not in imagefiles:
   target=RUN/'prepared'/(h+'.pixels.pt');torch.save(pixels,target);imagefiles[h]=dict(file=target.name,sha256=sha(target),tensor_sha256=th(pixels),shape=list(pixels.shape))
  else:assert th(pixels)==imagefiles[h]['tensor_sha256']
  target=RUN/'prepared'/(q+'.inputs.pt');torch.save(dict(x),target)
  b=dict(question_id=q,image_sha256=h,rendered_prompt_sha256=hashlib.sha256(chat.encode()).hexdigest(),prompt_text_sha256=hashlib.sha256(s['prompt_text'].encode()).hexdigest(),tensors={k:dict(sha256=th(v),shape=list(v.shape),dtype=str(v.dtype)) for k,v in x.items()},pixels=imagefiles[h],input_file_sha256=sha(target))
  bindings[key]=b
  samples[key]=dict(question_id=q,image_id=s['image_id'],image_sha256=h,question=s['question'],gold=s['answer'],prompt_text=s['prompt_text'],rendered_prompt=chat,input_tokens=x['input_ids'].shape[1],image_positions=vis,image_tokens=len(vis),input_binding=digest(b),source_figure7_manifest_sha256=sha(ref/'sample_manifest.json.gz'))
  if (i+1)%50==0:write(OUT/'cpu_progress.json',dict(status='PREPARING_INPUTS',completed=i+1,total=500));print('INPUTS',i+1,flush=True)
 write(OUT/'sample_manifest.json',samples);write(OUT/'input_bindings.json',bindings);write(OUT/'image_inventory.json',verified)
 write(OUT/'model_manifest.json',read(Path(S['model'])/'.ate_revision.json'))
 write(OUT/'input_validation.json',dict(status='PASS',N=500,image_clusters=309,question_image_SHA_Gold_order_matches_Figure7=True,all309_image_hashes_verified=True,all500_model_inputs_bound=True,Gold_separate_from_prepared=True,original_question_suffix=True,native_Qwen_processor=True,visual_tokens_min=min(s['image_tokens'] for s in samples.values()),visual_tokens_max=max(s['image_tokens'] for s in samples.values()),source_figure7_commit=S['figure7_commit'],source_figure7_manifest_sha256=sha(ref/'sample_manifest.json.gz'),CUDA_initialized=False))
 print(json.dumps(read(OUT/'input_validation.json')),flush=True)
if __name__=='__main__':main()
