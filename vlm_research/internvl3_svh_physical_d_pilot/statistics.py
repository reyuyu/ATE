"""Independent CPU helpers for a fully gated successful pilot; never estimate missing rows."""
import statistics
from collections import defaultdict
import numpy as np
from .common import *
def bootstrap(samples,values):
 groups=sorted({s['image_sha256'] for s in samples});ids={g:i for i,g in enumerate(groups)}
 sums=np.zeros(len(groups));counts=np.zeros(len(groups))
 for s,v in zip(samples,values):j=ids[s['image_sha256']];sums[j]+=v;counts[j]+=1
 rng=np.random.default_rng(20261110);out=[]
 for start in range(0,20000,500):
  draw=rng.integers(0,len(groups),(500,len(groups)))
  out.extend((100*sums[draw].sum(1)/counts[draw].sum(1)).tolist())
 return np.percentile(out,[2.5,97.5]).tolist()
def analyze_complete():
 assert read(RUN/'smoke_validation.json')['operational_correctness']=='PASS'
 assert read(RUN/'execution_status.json')['status']=='COMPLETE_WITH_NUMERICAL_WAIVER'
 samples=read(RUN/'sample_manifest.json')['samples'];records=[];rows=[]
 scorer_path=FIG/'references/qwen/vlm_research/results/qwen3vl_visual_writing_necessity_3000/official_gqa_prediction.json'
 scorer_record=read(scorer_path);scope={}
 exec(compile(scorer_record['function_source'],str(scorer_path),'exec'),scope)
 official_prediction=scope['gqa_prediction']
 save(RUN/'official_scorer_reference.json',dict(source_file=str(scorer_path),source_sha256=sha(scorer_path),frozen_record=scorer_record))
 for s in samples:
  r=read(RUN/'quality_checkpoints'/(s['task']+'__'+s['question_id']+'.json'))
  assert r['complete'] and r['sample_sha256']==digest(s) and r['protocol_sha256']==sha(RUN/'protocol.json')
  assert set(r['conditions'])==set(METHODS)
  for method in METHODS:
   q=r['conditions'][method];assert int(official_prediction(q['text'].strip())==s['answer'].lower())==q['correct']
   rows.append(dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],gold=s['answer'],
    method=method,text=q['text'],token_ids=q['token_ids'],correct=q['correct'],eos=q['eos'],truncated=q['truncated'],empty=q['empty']))
  records.append(r)
 assert len(rows)==360
 csvwrite(RUN/'per_question.csv',rows,list(rows[0]))
 summary=[];tasks=[];pairs=[];harm=[]
 for method in METHODS:
  part=[r for r in rows if r['method']==method]
  summary.append(dict(method=method,status='PASS',n=120,correct=sum(r['correct'] for r in part),accuracy_percent=100*statistics.mean(r['correct'] for r in part)))
  for task in TASKS:
   p=[r for r in part if r['task']==task];assert len(p)==20
   tasks.append(dict(task=task,method=method,status='PASS',n=20,correct=sum(r['correct'] for r in p),accuracy_percent=100*statistics.mean(r['correct'] for r in p)))
 for a,c in [('Dense-D','Physical-D'),('Native','Physical-D')]:
  delta=[r['conditions'][c]['correct']-r['conditions'][a]['correct'] for r in records]
  lo,hi=bootstrap(samples,delta)
  pair=dict(comparison=c+' - '+a,status='PASS',n=120,delta_pp=100*statistics.mean(delta),CI95_low=lo,CI95_high=hi,correct_to_wrong=sum(d<0 for d in delta),wrong_to_correct=sum(d>0 for d in delta),
   first_token_flip=sum(r['conditions'][a]['token_ids'][0]!=r['conditions'][c]['token_ids'][0] for r in records),
   text_change=sum(r['conditions'][a]['text']!=r['conditions'][c]['text'] for r in records))
  pairs.append(pair)
  for s,r,d in zip(samples,records,delta):
   if d<0:harm.append(dict(comparison=pair['comparison'],task=s['task'],question_id=s['question_id'],gold=s['answer'],before=r['conditions'][a]['text'],after=r['conditions'][c]['text']))
 taskpairs=[]
 for a,c in [('Dense-D','Physical-D'),('Native','Physical-D')]:
  for task in TASKS:
   sub=[(s,r) for s,r in zip(samples,records) if s['task']==task]
   values=[r['conditions'][c]['correct']-r['conditions'][a]['correct'] for s,r in sub]
   lo,hi=bootstrap([s for s,r in sub],values)
   taskpairs.append(dict(comparison=c+' - '+a,task=task,n=20,delta_pp=100*statistics.mean(values),CI95_low=lo,CI95_high=hi,
    correct_to_wrong=sum(v<0 for v in values),wrong_to_correct=sum(v>0 for v in values)))
 csvwrite(RUN/'task_paired_comparisons.csv',taskpairs,list(taskpairs[0]))
 csvwrite(RUN/'accuracy_summary.csv',summary,list(summary[0]))
 csvwrite(RUN/'task_accuracy.csv',tasks,list(tasks[0]))
 csvwrite(RUN/'paired_comparisons.csv',pairs,list(pairs[0]))
 csvwrite(RUN/'harm_cases.csv',harm,['comparison','task','question_id','gold','before','after'])
 efficiency=[read(p) for p in sorted((RUN/'efficiency_checkpoints').glob('*.json'))]
 assert len(efficiency)==450
 fields=['question_id','task','image_sha256','method','repeat','input_tensor_binding','KV_live_MiB','KV_capacity_MiB','E2E_ms','TTFT_ms','TPOT_ms','vision_encoder_ms','language_prefill_ms','incremental_decode_ms','peak_allocated_MiB','peak_reserved_MiB']
 for r in efficiency:
  assert r['generated_tokens']==32 and r['protocol_sha256']==sha(RUN/'protocol.json')
  assert abs((r['E2E_ms']-r['TTFT_ms'])/31-r['TPOT_ms'])<1e-9
 csvwrite(RUN/'efficiency_per_input.csv',efficiency,fields)
 perinput=[]
 for method in METHODS:
  for qid in read(RUN/'sample_manifest.json')['efficiency_question_ids']:
   part=[r for r in efficiency if r['method']==method and r['question_id']==qid]
   assert len(part)==5 and len({r['input_tensor_binding'] for r in part})==1
   row=dict(method=method,question_id=qid,task=part[0]['task'],repeats=5,input_tensor_binding=part[0]['input_tensor_binding'])
   for field in fields[6:]:row[field]=statistics.mean(r[field] for r in part)
   perinput.append(row)
 for qid in read(RUN/'sample_manifest.json')['efficiency_question_ids']:
  assert len({r['input_tensor_binding'] for r in perinput if r['question_id']==qid})==1
 csvwrite(RUN/'efficiency_per_input_summary.csv',perinput,list(perinput[0]))
 es=[]
 for method in METHODS:
  part=[r for r in efficiency if r['method']==method];assert len(part)==150
  row=dict(method=method,status='PASS',inputs=30,repeats=5,requests=150)
  meanrows=[r for r in perinput if r['method']==method]
  for field in fields[6:]:
   row[field+'_mean']=statistics.mean(r[field] for r in meanrows);row[field+'_median']=statistics.median(r[field] for r in meanrows)
   row[field+'_all_repeats_median']=statistics.median(r[field] for r in part)
  es.append(row)
 csvwrite(RUN/'efficiency_summary.csv',es,list(es[0]))
 return summary,tasks,pairs,es
