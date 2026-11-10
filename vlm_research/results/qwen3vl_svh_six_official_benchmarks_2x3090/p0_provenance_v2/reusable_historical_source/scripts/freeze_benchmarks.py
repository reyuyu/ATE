"""Freeze original questions, authors' gold and explicit protocol variants."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
def read(p):return json.loads(p.read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
data=read(O/'data_restore.json');assert data['status']=='PASS'
views=data['views'];hashes=data['image_SHA256_manifest'];allrows=[];counts={};differences=[]
def view(name):
 v=views[name];rows=[json.loads(l) for l in Path(v['path']).read_text().splitlines()];gold=read(Path(v['answers']));return v,rows,gold
def add(b,name,label,cap,makebody=None):
 v,rows,gold=view(name);selected=[]
 for r in rows:
  path=str(Path(v['root'])/r['image_path']);qid=str(r['question_id']);answer=gold[qid]
  s=dict(key=b+':'+label+':'+qid,benchmark=b,question_id=qid,image_id=r['image_id'],image_path=path,image_sha256=hashes[path],question=r['question'],body=makebody(r) if makebody else r['user_text'],answer=answer,input_view=name,protocol_label=label,max_new_tokens=cap,source_row_SHA256=digest(r),source_inputs_SHA256=v['inputs_SHA256'],source_Gold_SHA256=v['Gold_SHA256'])
  for k in ['strategy','subtask','choices','hint','official_question_ids','author_question_id']:
   if k in r:s[k]=r[k]
  if b=='MMBench':s['circular_group']=str(int(qid)%1000000)
  if b=='MME':s['pair_id']=r['image_id'];s['task']=r['subtask']
  if b=='TextVQA':assert len(answer)==10;s['paper_ocr']=True
  selected.append(s)
 allrows.extend(selected);counts[b+':'+label]=len(selected);return selected
add('GQA','GQA-testdev-balanced','OFFICIAL_PROTOCOL',32,lambda r:r['question']+'\nAnswer using a single word or short phrase.')
add('ScienceQA-IMG','ScienceQA-IMG-paper','ATE_PAPER_PROTOCOL_AND_OFFICIAL_IMG_SCORING',1024)
add('TextVQA','TextVQA-OCR','ATE_PAPER_OCR_AND_OFFICIAL_SOFT_SCORING',128)
add('POPE','POPE-paper','ATE_PAPER_PROTOCOL',128)
add('POPE','POPE','OFFICIAL_PROTOCOL',128,lambda r:r['question'])
add('MME','MME-paper','ATE_PAPER_PROTOCOL',128)
add('MME','MME','OFFICIAL_PROTOCOL',128,lambda r:r['question'])
add('MMBench','MMBench-paper','ATE_PAPER_INPUT_OFFICIAL_CIRCULAR_EXACT_MATCHING',1024)
v=views['OCRBench-v1'];anns=read(Path(v['annotations']));rows=[json.loads(l) for l in Path(v['path']).read_text().splitlines()]
ocr=[]
for i,r in enumerate(rows):
 a=anns[i];assert r['official_annotation_index']==i and r['official_id']==a['id'] and r['answer']==a['answers'] and r['question']==a['question']
 base=dict(r);base['image_path']=str(Path(v['root'])/'OCRBench_Images'/r['image_id']);assert Path(base['image_path']).is_file()
 for label,body in [('ATE_PAPER_PROTOCOL',r['user_text']),('OFFICIAL_PROTOCOL',r['question']+'\nAnswer the question using a single word or phrase.')]:
  s=dict(base,key='OCRBench-v1:'+label+':'+r['question_id'],body=body,protocol_label=label,max_new_tokens=128,source_row_SHA256=digest(r),source_inputs_SHA256=v['inputs_SHA256'],source_Gold_SHA256=v['Gold_SHA256']);ocr.append(s)
allrows.extend(ocr);counts['OCRBench-v1:ATE_PAPER_PROTOCOL']=counts['OCRBench-v1:OFFICIAL_PROTOCOL']=1000
# Share newly generated outputs only for byte-identical prompt/image tensors/caps.
representative={}
for s in allrows:
 request=digest([s['image_sha256'],s['body'],s['max_new_tokens']]);s['request_SHA256']=request
 s['generation_key']=representative.setdefault(request,s['key'])
assert len({s['key'] for s in allrows})==len(allrows)
for b,n in [('GQA',12578),('ScienceQA-IMG',2017),('TextVQA',5000),('MMBench',4377)]:assert sum(s['benchmark']==b for s in allrows)==n
groups=defaultdict(list)
for s in allrows:
 if s['benchmark']=='MMBench':groups[s['circular_group']].append(s)
assert len(groups)==1176 and sum(map(len,groups.values()))==4377
result=dict(status='FROZEN_BEFORE_SEVEN_BENCHMARK_PREDICTIONS',rows=allrows,counts=counts,total_protocol_rows=len(allrows),unique_generation_requests=len(representative),protocol_differences=['POPE: author 9000 original questions vs ATE 8910 with short-answer instruction; both generated separately','MME: author yes/no instruction retained vs ATE short phrase replacement; both generated separately','OCRBench: restored ATE user_text contains literal backslash-n; official author script uses a real newline; preserve both, no historical mutation','ScienceQA: image-bearing test subset 2017 with Hint and choices preserved; report IMG Accuracy, not overall4241','TextVQA: Paper-OCR kept explicit, official ten-answer VQA soft metric; no claim of image-only official model setting','MMBench: complete 1176 Circular groups, authors exact_matching; any required unpaid unavailable judge marked awaiting, no random answer fallback','Qwen model author sampled vLLM reproduction settings differ from mandated BF16/Eager greedy; benchmark official scoring does not imply Qwen model-zoo reproduction'],no_old_predictions_reused=True)
p=O/'input_manifest.json'
if p.exists():assert read(p)==result
else:p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('SEVEN_INPUT_PROTOCOLS_FROZEN',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
