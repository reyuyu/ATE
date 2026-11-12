import sys,json,gzip,time,collections
from run_common import *
sys.path[:0]=[str(L/'sources/qwen'),str(L)]
import torch
from transformers import AutoProcessor
from qwen_frozen_benchmark_inputs import prepare,digest as frozen_digest
torch.set_num_threads(1)
O=L/'results/qwen';O.mkdir(exist_ok=True)
allrows=json.loads((L/'history/qwen_inputs_1000.json').read_text());assert len(allrows)==2000
groups=collections.defaultdict(list)
for r in allrows:groups[r['question_id']].append(r)
assert len(groups)==1000 and all(len({r['request_SHA256'] for r in v})==1 for v in groups.values())
rows=sorted([r for r in allrows if r['protocol_label']=='ATE_PAPER_PROTOCOL'],key=lambda r:r['question_id'])
proc=AutoProcessor.from_pretrained('/home/featurize/ate-local-models/Qwen3-VL-8B-Instruct',local_files_only=True)
reused={m:{} for m in ['Native','Physical-Original']};layouts={};newrows=[]
hist=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
for i,s in enumerate(rows):
 q=s['question_id'];assert q==f'{i:04d}' and s['image_id']==RAW[i]['image_path'] and s['answer']==RAW[i]['answers'] and s['question']==RAW[i]['question']
 s=dict(s,image_path=str(D/'OCRBench_Images'/s['image_id']))
 x,l=prepare(proc,s);layouts[q]=l;newrows.append(s)
 for m in reused:
  p=hist/'request_predictions'/s['request_SHA256'][:2]/(s['request_SHA256']+'__'+m+'.json')
  if p.exists():
   old=json.loads(p.read_text());assert old['binding']['input_SHA256']==frozen_digest(l) and old['binding']['request_SHA256']==s['request_SHA256']
   reused[m][q]=dict(model='Qwen3-VL-8B',method=m,question_id=q,generation=old['result'],input_layout=l,**score(q,old['result']['text']),origin='REUSED_VERIFIED_REQUEST_CACHE',original_record=old,original_record_sha256=digest(old))
 if i%100==0:print('QWEN_CPU_LAYOUT',i,'tokens',l['input_length'],'visual',l['image_tokens'],flush=True)
 del x
atomic(O/'rows.json',newrows);atomic(O/'verified_reuse.json',reused)
with gzip.open(O/'cpu_layouts.json.gz','wt') as f:json.dump(layouts,f,ensure_ascii=False)
ordered=sorted(newrows,key=lambda s:(layouts[s['question_id']]['input_length'],s['question_id']));smoke=[ordered[0]['question_id'],ordered[len(ordered)//3]['question_id'],ordered[2*len(ordered)//3]['question_id'],ordered[-1]['question_id']]
sources={str(p.relative_to(L/'sources/qwen')):sha(p) for p in (L/'sources/qwen').rglob('*.py')}
atomic(O/'protocol.json',dict(model='Qwen/Qwen3-VL-8B-Instruct',revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',source_commit='7e18895a6ccd57c4cb3aaedd5d512789e7467b59',frozen_protocol=json.loads((L/'history/qwen_protocol.json').read_text()),max_new_tokens=128,dtype='bfloat16',attention='eager',conditions=['Native','Physical-Original'],source_SHA256=sources,wrapper_SHA256=sha(L/'qwen_run.py'),reuse_counts={m:len(r) for m,r in reused.items()},smoke_ids=smoke,smoke_input_lengths={q:layouts[q]['input_length'] for q in smoke},official_and_ATE_views_identical_requests=1000,all_layout_hashes={q:frozen_digest(x) for q,x in layouts.items()},same_default_RGB_processor=True,all_1000_questions_930_images=True))
print('QWEN_CPU_PASS',json.dumps(dict(smoke=smoke,lengths=[layouts[q]['input_length'] for q in smoke],max_image_tokens=max(l['image_tokens'] for l in layouts.values()),reused={m:len(r) for m,r in reused.items()})),flush=True)
