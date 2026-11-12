import sys,json,time
from run_common import *
sys.path[:0]=[str(L/'sources/llava'),'/home/featurize/work/ATE']
import torch
from transformers import AutoProcessor
from vlm_research.information_horizon_alignment import engine as f
from vlm_research.information_horizon_alignment.common import digest as olddigest,tensor_sha
torch.set_num_threads(1)
O=L/'results/llava';O.mkdir(exist_ok=True)
P=json.loads((L/'history/llava_paper_protocol.json').read_text());sources={}
for p,h in P['sources'].items():assert sha(L/'sources/llava'/p)==h,p;sources[p]=h
proc=AutoProcessor.from_pretrained('/home/featurize/ate-local-models/llava-1.5-7b-hf',local_files_only=True)
rows=json.loads((L/'history/llava_inputs_1000.json').read_text());old={r['question_id']:r for r in json.loads((L/'history/llava_svhr_1000.json').read_text())}
cpu={};images={};outrows=[]
for i,s in enumerate(rows):
 q=s['question_id'];assert q==f'{i:04d}' and s['image_id']==RAW[i]['image_path'] and s['answer']==RAW[i]['answers'] and s['question']==RAW[i]['question']
 x=dict(s,image_path=str(D/'OCRBench_Images'/s['image_id']));h=x['image_sha256']
 if h not in images:assert sha(x['image_path'])==h;images[h]=True
 pix=f.paper_pixels(x['image_path'],proc.image_processor);ids=[]
 for t in f.image_ids(x['prompt'],proc.tokenizer,-200):ids.extend([32000]*576 if t==-200 else [t])
 assert ids.count(32000)==576
 a=dict(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones(1,len(ids),dtype=torch.long),pixel_values=pix.half())
 expected=old[q];assert olddigest(ids)==expected['input_ids_sha256'] and tensor_sha(pix)==expected['pixel_values_sha256']
 hashes={k:__import__('hashlib').sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in a.items()}
 cpu[q]=dict(question_id=q,prompt=x['prompt'],image_sha256=h,input_ids=ids,input_hashes=hashes,reference_pixel_FP32_sha256=tensor_sha(pix),input_tokens=len(ids),max_new_tokens=128)
 outrows.append(x)
assert len(images)==930
samples={r['question_id']:r for r in json.loads((L/'history/llava_native20_samples.json').read_text())}
records=json.loads((L/'history/llava_native20_formal.json').read_text());reused={'Native':{},'Physical-C3':{}}
rp=json.loads((L/'history/llava_native20_protocol.json').read_text());assert rp['revision']=='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
assert rp['sources']['/home/featurize/work/ATE/vlm_research/c3v2_middle_reaccess/engine.py']==sources['vlm_research/c3v2_middle_reaccess/engine.py']
for r in records:
 if r['method'] not in ['Native','SVH']:continue
 q=r['key'].split(':')[-1];s=samples[q];v=cpu[q]
 assert s['input_ids']==v['input_ids'] and s['image_sha256']==v['image_sha256'] and s['prompt']==v['prompt']
 assert s['input_hashes']==r['input_hashes']==v['input_hashes']
 assert r['generation']['max_new_tokens']==128 and r['generation']['all_logits_finite']
 assert r['protocol_sha256']==digest(rp)
 text=proc.tokenizer.decode(r['output_ids'],skip_special_tokens=True).strip();assert text==r['raw_answer']
 method='Native' if r['method']=='Native' else 'Physical-C3'
 reused[method][q]=dict(model='LLaVA-1.5-7B',method=method,question_id=q,generation=dict(text=text,token_ids=r['output_ids'],eos=r['generation']['stop_reason']=='EOS',truncated=r['generation']['stop_reason']=='MAX_TOKENS',empty=not text),input_hashes=v['input_hashes'],image_sha256=v['image_sha256'],**score(q,text),origin='REUSED_VERIFIED_NATURAL_EOS_FORMAL',original_record=r,original_record_sha256=digest(r))
assert len(reused['Native'])==len(reused['Physical-C3'])==10
atomic(O/'cpu_bindings.json',cpu);atomic(O/'rows.json',outrows);atomic(O/'verified_reuse.json',reused)
missing=[s['question_id'] for s in rows if s['question_id'] not in reused['Native']];smoke=[missing[0],missing[len(missing)//3],missing[2*len(missing)//3],missing[-1]]
atomic(O/'protocol.json',dict(model='llava-hf/llava-1.5-7b-hf',revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',source_commit='0414e553ba6fd1e21626ecf741489d225b44279c',dtype='float16',attention='eager',max_new_tokens=128,conditions=['Native','Physical-C3'],existing_SVHR=281,source_SHA256=sources,reused_each=10,new_each=990,smoke_ids=smoke,smoke_reused_as_formal=True,wrapper_SHA256=sha(L/'llava_run.py'),all_1000_images_inputs_gold_match=True,processor_SHA256=sha(Path('/home/featurize/ate-local-models/llava-1.5-7b-hf/preprocessor_config.json'))))
print('LLAVA_CPU_PASS_1000_930_REUSE10_EACH_NEW990_EACH',flush=True)
