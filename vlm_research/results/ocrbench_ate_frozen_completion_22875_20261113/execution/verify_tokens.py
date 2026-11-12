import sys,json,csv,gzip
from run_common import *
from transformers import AutoTokenizer,AutoProcessor
name=sys.argv[1];checks=[]
if name=='internvl3':
 tok=AutoTokenizer.from_pretrained('/home/featurize/ate-local-models/InternVL3-8B',use_fast=False,trust_remote_code=True,local_files_only=True)
 for r in csv.DictReader(gzip.open(A/'historical/native_stopv2/OCRBench/per_question.csv.gz','rt')):checks.append((r['method'],r['question_id'],json.loads(r['token_ids']),r['text']))
 for r in json.loads((L/'results/internvl3/historical_late9_3.json').read_text()):checks.append(('Late9',r['question_id'],json.loads(r['late9_token_ids']),r['late9_text']))
 for p in (L/'results/internvl3/predictions').glob('*.json'):
  r=json.loads(p.read_text());checks.append(('Late9',r['question_id'],r['generation']['token_ids'],r['generation']['text']))
elif name=='llava':
 tok=AutoProcessor.from_pretrained('/home/featurize/ate-local-models/llava-1.5-7b-hf',local_files_only=True).tokenizer
 for r in json.loads((L/'history/llava_svhr_1000.json').read_text()):checks.append(('SVHR',r['question_id'],r['generation']['token_ids'],r['generation']['text']))
 for m,rr in json.loads((L/'results/llava/verified_reuse.json').read_text()).items():
  if m=='Native':continue
  for q,r in rr.items():checks.append((m,q,r['generation']['token_ids'],r['generation']['text']))
 for q,r in json.loads((L/'results/llava/verified_full_native_reuse.json').read_text()).items():checks.append(('Native',q,r['generation']['token_ids'],r['generation']['text']))
 for p in (L/'results/llava/predictions').glob('*/*.json'):
  r=json.loads(p.read_text());checks.append((r['method'],r['question_id'],r['generation']['token_ids'],r['generation']['text']))
elif name=='qwen':
 tok=AutoProcessor.from_pretrained('/home/featurize/ate-local-models/Qwen3-VL-8B-Instruct',local_files_only=True).tokenizer
 for p in (L/'results/qwen/predictions').glob('*/*.json'):
  r=json.loads(p.read_text());g=r['generation'];checks.append((r['method'],r['question_id'],g['token_ids'],g['text']))
else:raise ValueError(name)
fail=[]
for m,q,ids,text in checks:
 decoded=tok.decode(ids,skip_special_tokens=True)
 if name!='qwen':decoded=decoded.strip()
 if decoded!=text:fail.append(dict(method=m,question_id=q,decoded=decoded,stored=text))
atomic(L/'results'/name/'token_decode_audit.json',dict(status='PASS' if not fail else 'FAIL',checked=len(checks),failures=fail));print(name,'TOKEN_DECODE_CHECK',len(checks),'mismatches',len(fail));assert not fail,fail[:2]
