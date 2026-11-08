"""Freeze official-aligned inputs without loading model weights or using CUDA."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import sys,json,gzip,hashlib,csv,subprocess
import torch
from transformers import AutoProcessor,AutoConfig,LlamaTokenizer
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/featurize_4090_gqa_official_aligned_threeway'
A=O.parent/'featurize_4090_gqa_native_protocol_audit';D=Path('/home/featurize/data/ate-recovery');M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
sys.path[:0]=[str(A),str(R),str(O.parent/'featurize_new3090_recovery/scripts')]
from protocol_helpers import official_functions,official_prompt,official_ids,configure
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from vlm_research.physical_visual_handoff.common import TOLERANCES
import ate_data

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cmd(*args):return subprocess.check_output(args,cwd=R,text=True).strip()
assert not (O/'protocol.json').exists(),'Use existing frozen protocol rather than recreating it'
assert cmd('git','rev-parse','HEAD')=='e281754ef17b3accc09d6938b5b9f020048b2aec'
assert not cmd('git','status','--porcelain','--untracked-files=no') and not torch.cuda.is_initialized()
assert read(A/'audit_conclusion.json')['audit_status']=='COMPLETE'
H=O.parent/'featurize_new3090_recovery/history/c3v2-gqa-testdev-balanced-12578/vlm_research/results/c3v2_gqa_testdev_balanced_12578'
with gzip.open(H/'sample_manifest.jsonl.gz','rt') as f:history=[json.loads(l) for l in f]
old={s['question_id']:s for s in history};hf=read(H/'execution_freeze.json');sources={};unavailable={}
for name,h in hf['sources'].items():
 p=R/'vlm_research'/name.split('/vlm_research/',1)[1]
 if p.exists():assert sha(p)==h,('Frozen source changed',str(p));sources[str(p.relative_to(R))]=h
 else:unavailable[name]=h
rawpath=D/'restored_bundles/ate3090-gqa-testdev-balanced/source_annotations/gqa/testdev_balanced_questions.json';raw=read(rawpath)
assert sha(rawpath)==read(A/'cpu_audit_summary.json')['raw_official_annotation_sha256']
officialpath=D/'audit_support/gqa_native_protocol/official_llava_gqa_testdev_balanced.jsonl'
assert sha(officialpath)==read(A/'official_prepared_input_audit.json')['sha256']
official=[json.loads(l) for l in officialpath.read_text().splitlines()]
current={s['question_id']:s for s in ate_data.iter_inputs('GQA-testdev-balanced')};answers=ate_data.load_answers('GQA-testdev-balanced')
assert len(old)==len(official)==len(raw)==len(current)==len(answers)==12578
assert set(old)==set(raw)==set(current)==set(answers)=={str(s['question_id']) for s in official}
proc=configure(AutoProcessor.from_pretrained(M,local_files_only=True),AutoConfig.from_pretrained(M,local_files_only=True))
tok=LlamaTokenizer.from_pretrained(D/'audit_support/gqa_native_protocol/official_tokenizer',local_files_only=True,legacy=True)
conv,functions=official_functions()
pixels={};records=[];seen=set()
with (A/'image_protocol_audit.csv').open() as f:imageaudit={s['image_id']:s for s in csv.DictReader(f)}
from PIL import Image
from types import SimpleNamespace
for index,item in enumerate(official):
 q=str(item['question_id']);s=old[q];g=raw[q];c=current[q];image_id=s['image_id']
 assert q not in seen;seen.add(q)
 assert item['text']==s['prompt_text'] and Path(item['image']).stem==image_id
 assert s['question']==g['question']==c['question'] and s['answer']==g['answer']==answers[q]
 assert image_id==g['imageId']==c['image_id'] and g['isBalanced'] and s['split']=='testdev_balanced'
 if image_id not in pixels:
  with Image.open(c['image_path']) as im:p=functions.process_images([im.convert('RGB')],proc.image_processor,SimpleNamespace(image_aspect_ratio='pad')).half()
  pixels[image_id]=tensor_hash(p);assert pixels[image_id]==imageaudit[image_id]['pad_fp16_sha256']
 prompt=official_prompt(s,conv);ids=official_ids(prompt,tok,functions)
 compact=proc.tokenizer(prompt,return_offsets_mapping=True);expanded=[];question=[];begin=prompt.index(s['question']);end=begin+len(s['question'])
 for token,(a,b) in zip(compact['input_ids'],compact['offset_mapping']):
  pos=len(expanded);expanded.extend([token]*(576 if token==32000 else 1))
  if b>begin and a<end:question.extend(range(pos,len(expanded)))
 assert expanded==ids and ids.count(32000)==576
 image=tuple(i for i,v in enumerate(ids) if v==32000);relay=tuple(range(image[-1]+1,len(ids)-1))
 assert question and set(question)<=set(relay)
 input_hashes={'input_ids':tensor_hash(torch.tensor([ids],dtype=torch.long)),
  'attention_mask':tensor_hash(torch.ones((1,len(ids)),dtype=torch.long)),'pixel_values':pixels[image_id]}
 record={'index':index,'question_id':q,'image_id':image_id,'image_sha256':s['image_sha256'],'image_path':c['image_path'],
  'question':s['question'],'prompt_text':s['prompt_text'],'prompt':prompt,'answer':s['answer'],'split':'testdev_balanced','types':g['types'],
  'input_ids':ids,'question_positions':question,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_hashes':input_hashes}
 record['sample_binding']=digest(record);records.append(record)
assert len(pixels)==398
with gzip.open(O/'input_manifest.jsonl.gz','wt',compresslevel=6) as f:
 for s in records:f.write(json.dumps(s,ensure_ascii=False,separators=(',',':'))+'\n')
with (O/'question_order.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['index','question_id','image_id','image_sha256','structural']);w.writeheader()
 for s in records:w.writerow({k:s[k] for k in ['index','question_id','image_id','image_sha256']}|{'structural':s['types']['structural']})
protocol={'status':'FROZEN','created_at':datetime.now(timezone.utc).isoformat(),'code_commit':cmd('git','rev-parse','HEAD'),
 'audit_commit':'25989d8c0e00cf53a980c726be94d84feaabbc14','model':'llava-hf/llava-1.5-7b-hf','revision':'b234b804b114d9e37bb655e11cbbb5f5e971b7a9','model_path':str(M),
 'data_path':str(D.resolve()),'conditions':['Native','C3-V2','C3-V2+L11-L24'],'questions_per_condition':12578,'expected_formal_generations':37734,
 'prompt':'Pinned official Vicuna-v1 system + <image> + original question + single word or phrase suffix',
 'preprocessing':'RGB; official CLIP-mean square-pad; CLIP336 bicubic/crop/normalize',
 'generation':{'dtype':'float16','attention':'eager','greedy':True,'batch_size':1,'max_new_tokens':32,'eos_token_id':2,'bos_token_id':1,'image_token_id':32000,'image_positions':576},
 'scorer':"Official LLaVA convert: text.rstrip('.').lower(); exact match to official answer; denominator12578",
 'order':'Original official eval.zip gqa/llava_gqa_testdev_balanced.jsonl order; same unchanged order for each of three complete conditions',
 'official_order_equals_old_manifest_order':[s['question_id'] for s in records]==[s['question_id'] for s in history],
 'manifest_sha256':sha(O/'input_manifest.jsonl.gz'),'question_order_sha256':sha(O/'question_order.csv'),
 'official_prepared_sha256':sha(officialpath),'official_Gold_sha256':sha(rawpath),'historical_manifest_sha256':sha(H/'sample_manifest.jsonl.gz'),
 'historical_controls_sha256':sha(O.parent/'featurize_new3090_c3v2_reaccess_sevenbench/gqa_historical_controls.jsonl.gz'),
 'frozen_sources':sources,'historical_external_sources_to_check_on_load':unavailable,
 'audit_helper_sha256':sha(A/'protocol_helpers.py'),'audit_source_manifest_sha256':sha(A/'official_sources/source_manifest.json'),
 'strict_numerical_tolerances_unchanged':TOLERANCES['llava'],
 'physical_smoke_question_ids':['201030233','201030272','20886973'],'cross_hardware_old_protocol_question_ids':['201030233','201030272'],
 'structural_counts':dict(Counter(s['types']['structural'] for s in records)),
 'image_sha_clusters':len({s['image_sha256'] for s in records}),'CUDA_initialized':torch.cuda.is_initialized(),
 'no_repeated_downloads':True,'no_repeat_all_image_SHA':True,'manifest_uses_accepted_image_SHA_and_verified_pad_pixels':True}
save('protocol.json',protocol)
save('progress.json',{'status':'PREPARED','completed':0,'expected':37734,'counts':{k:0 for k in protocol['conditions']},'protocol_sha256':sha(O/'protocol.json')})
print('OFFICIAL_THREEWAY_INPUTS_FROZEN',len(records),len(pixels),sha(O/'protocol.json'),flush=True)
