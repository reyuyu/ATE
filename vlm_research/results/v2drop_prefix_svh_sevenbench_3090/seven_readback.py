"""CPU-only early readback of COMPLETE benchmarks; no files overwritten."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1')
from seven_common import *
import sqlite3
from collections import defaultdict
from vlm_research.information_horizon_alignment import analyze
audit=read(O/'input_audit.json');samples=gz(O/'input_manifest.jsonl.gz');bench=sys.argv[1];by={s['key']:s for s in samples if s['benchmark']==bench};records={}
with sqlite3.connect('file:'+str(O/'predictions.sqlite')+'?mode=ro',uri=True) as db:
 for k,p,h in db.execute('SELECT key,payload,sha256 FROM predictions WHERE key LIKE ?', (bench+':%',)):
  r=json.loads(p);s=by[k];assert digest(r)==h and r['sample_sha256']==digest(s) and r['gold']==s['answer'] and r['question_id']==s['question_id'] and r['image_sha256']==s['image_sha256'];assert score(s,r['generation']['text'])==r['scoring'];records[k]=r
if len(records)!=audit['expected'][bench]:print('STILL_RUNNING',bench,len(records),'/',audit['expected'][bench]);raise SystemExit(0)
ss=list(by.values());values=[records[s['key']]['scoring']['score'] for s in ss]
if bench=='MMBench':
 groups,correct=analyze.circular_rows(ss,values);metric=100*float(correct.mean());extra=dict(groups=len(groups),correct_groups=int(correct.sum()))
elif bench=='MME':metric=analyze.mme_score(ss,values)[0];extra={}
elif bench=='OCRBench-v1':metric=sum(values);extra=dict(unit='official /1000')
else:metric=100*sum(values)/len(ss);extra=dict(score_sum=sum(values))
print('COMPLETE_BENCHMARK_CPU_READBACK',json.dumps(dict(benchmark=bench,questions=len(ss),metric=metric,empty=sum(r['generation']['empty'] for r in records.values()),truncated=sum(r['generation']['truncated'] for r in records.values()),Gold_QID_record_SHA_and_score_check='PASS',**extra),ensure_ascii=False))
