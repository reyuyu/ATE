from pathlib import Path
import sys,json,gzip,csv,importlib.util,types,time
from run_common import *
sys.path.insert(0,str(L/'late9_sources'))
import torch
from transformers import AutoTokenizer
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as b
from vlm_research.internvl3_svh_seven_benchmarks.common import request_text,tensor_sha
torch.set_num_threads(1)
M=Path('/home/featurize/ate-local-models/InternVL3-8B')
manifest=json.loads((A/'historical/late9/source_manifest.json').read_text())
for name,h in manifest['runtime_source_sha256'].items():assert sha(L/'late9_sources'/name)==h,name
rows=[json.loads(s) for s in gzip.open(A/'historical/native_stopv2/manifests/OCRBench.jsonl.gz','rt')]
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True,use_fast=False,local_files_only=True)
spec=importlib.util.spec_from_file_location('pinned_old_conversation',M/'conversation.py');convmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=convmod;spec.loader.exec_module(convmod)
loader=b.official_preprocessor(M);pixels={};image_paths={};bindings=json.loads((L/'late9_reference/processor_bindings.json').read_text());fresh=[]
old=list(csv.DictReader(gzip.open(A/'historical/native_stopv2/OCRBench/per_question.csv.gz','rt')))
old_native={r['question_id']:r for r in old if r['method']=='native'};assert len(old_native)==1000
for i,r in enumerate(rows):
 q=f'{i:04d}';assert r['question_id']==q and r['gold']==RAW[i]['answers'] and r['image_id']==RAW[i]['image_path'] and r['question']==RAW[i]['question']
 path=D/'OCRBench_Images'/r['image_id'];h=r['image_sha256']
 if h not in pixels:
  assert sha(path)==h
  px=loader(str(path),max_num=1).to(torch.bfloat16);assert list(px.shape)==[1,3,448,448]
  pixels[h]=tensor_sha(px);image_paths[h]=str(path)
 if h in bindings:assert pixels[h]==bindings[h]['pixels_bf16_sha256']
 assert pixels[h]==json.loads(old_native[q]['input_binding_details'])['pixels_bf16_sha256']
 req=request_text('OCRBench',r);assert req==r['request']
 conv=convmod.get_conv_template('internvl2_5');conv.append_message(conv.roles[0],req);conv.append_message(conv.roles[1],None)
 prompt=conv.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*256+'</img>',1);ids=tok(prompt)['input_ids']
 assert prompt==r['prompt'] and ids==r['input_ids'] and __import__('hashlib').sha256(prompt.encode()).hexdigest()==r['prompt_sha256']
 fresh.append(dict(question_id=q,image_sha256=h,pixels_bf16_sha256=pixels[h],input_ids_sha256=tensor_sha(torch.tensor([ids])),prompt_sha256=r['prompt_sha256'],single_tile=True,visual_tokens=256))
assert len(image_paths)==930 and len(rows)==1000
hist=list(csv.DictReader((A/'historical/late9/ocrbench_partial_per_question.csv').open()));assert {r['question_id'] for r in hist}=={'0053','0963','0988'}
O=L/'results/internvl3';O.mkdir(exist_ok=True)
atomic(O/'inputs_verified.json',dict(status='PASS',questions=1000,unique_images=930,all_image_SHA256=True,all_old_pixel_hashes=True,all_prompt_ids_gold=True,bindings=fresh))
atomic(O/'input_paths.json',image_paths);atomic(O/'frozen_rows.json',rows)
atomic(O/'historical_late9_3.json',hist)
atomic(O/'protocol.json',dict(model='OpenGVLab/InternVL3-8B',revision='853e3a797a661694b1b8ece0cb72dc2b23e3dac9',source_commit='c60a9919112a3cf5bec8c494a533cba56a97f29b',frozen_protocol= json.loads((A/'historical/late9/protocol.json').read_text()),new_method='late9',new_question_count=997,never_generate_ids=['0053','0963','0988'],single_tile=True,max_new_tokens=128,smoke_ids=['0000','0333','0666','0999'],smoke_reused_as_formal=True,source_SHA256=manifest['runtime_source_sha256'],new_wrapper_SHA256=sha(L/'internvl_run.py')))
print('INTERNVL_CPU_PASS_1000_930_NEW997',flush=True)
