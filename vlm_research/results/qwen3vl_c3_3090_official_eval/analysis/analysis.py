"""CPU-only independent scoring and paired statistics. Never changes raw records."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import csv,gzip,hashlib,json,os,statistics,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
sys.path.insert(0,str(O/'scripts'))
import numpy as np
from scorers_v2 import score
from scorers import text_evaluator
SEED=20261106;REPS=20000;CONDITIONS=['Native','Dense-F','Physical-Original']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def table(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r));t=p.with_suffix(p.suffix+'.tmp')
 opener=gzip.open if p.name.endswith('.gz') else open
 with opener(t,'wt',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 t.replace(p)
def bootstrap(values,weights=None,scale=1.):
 """Question weighted resampling of whole image clusters, not question rows."""
 values=np.asarray(values,dtype=np.float64);weights=np.ones(len(values)) if weights is None else np.asarray(weights,dtype=np.float64)
 assert len(values)==len(weights) and len(values)>0
 rng=np.random.default_rng(SEED);out=[];n=len(values)
 for first in range(0,REPS,128):
  ix=rng.integers(0,n,size=(min(128,REPS-first),n));den=weights[ix].sum(1);out.extend((values[ix].sum(1)/den*scale).tolist())
 return [float(x) for x in np.quantile(out,[.025,.975])]
def independent(s,text,parsed):
 b=s['benchmark'];gold=s['answer']
 if b=='GQA':return float(text.rstrip('.').lower()==gold)
 if b=='TextVQA':
  norm=text_evaluator().answer_processor;matches=sum(norm(a)==norm(text) for a in gold)
  # Independent count formula for ten-answer leave-one-out (not min(count/3,1)).
  return [0.,.3,.6,.9,1.][min(matches,4)]
 if b=='POPE':
  tokens=text.partition('.')[0].replace(',','').split(' ');negative=('No' in tokens or 'not' in tokens or 'no' in tokens)
  return float(('no' if negative else 'yes')==gold)
 if b=='MME':
  prefix=text.lower()[:4];v='yes' if 'yes' in prefix else 'no' if 'no' in prefix else 'other';return float(v==gold)
 if b=='ScienceQA-IMG':
  # Independently verify the integer-Gold bridge; extraction itself is frozen author matching.
  return float(isinstance(parsed,str) and len(parsed)==1 and parsed in 'ABCDE' and ord(parsed)==65+gold)
 if b=='MMBench':return float(parsed==gold)
 if b=='OCRBench-v1':
  ans=gold if isinstance(gold,list) else [gold]
  if s['dataset_name']=='HME100k':return float(any(str(a).strip().replace('\n',' ').replace(' ','') in text.strip().replace('\n',' ').replace(' ','') for a in ans))
  return float(any(str(a).lower().strip().replace('\n',' ') in text.lower().strip().replace('\n',' ') for a in ans))
 raise ValueError(b)
def paired(rows,comparison,benchmark,label,metric='Accuracy',scale=100.):
 left,right=comparison;by=defaultdict(dict)
 for r in rows:by[r['key']][r['condition']]=r
 pairs=[v for v in by.values() if left in v and right in v]
 if not pairs:return None
 clusters=defaultdict(lambda:[0.,0]);harm=fix=equal=first_change=answer_change=0
 for p in pairs:
  a,b=p[left],p[right];assert a['image_sha256']==b['image_sha256'];d=a['score']-b['score']
  clusters[a['image_sha256']][0]+=d;clusters[a['image_sha256']][1]+=1
  harm+=d<0;fix+=d>0;equal+=d==0;first_change+=a.get('first_token')!=b.get('first_token');answer_change+=a['parsed']!=b['parsed']
 vals=list(clusters.values());ci=bootstrap([v[0] for v in vals],[v[1] for v in vals],scale)
 return dict(benchmark=benchmark,protocol_label=label,metric=metric,condition=left,reference=right,paired_questions=len(pairs),image_clusters=len(clusters),delta=sum(v[0] for v in vals)/len(pairs)*scale,CI95_low=ci[0],CI95_high=ci[1],score_decreases=harm,score_increases=fix,score_ties=equal,first_token_changed=first_change,parsed_answer_changed=answer_change,bootstrap_replicates=REPS,seed=SEED,unit='Image SHA256 cluster',soft_score=b'VQA' in benchmark.encode() and benchmark=='TextVQA')
def paired_cases(rows,comparisons):
 by=defaultdict(dict)
 for r in rows:by[r['benchmark'],r['protocol_label'],r['key']][r['condition']]=r
 cases=[]
 for (b,label,key),preds in by.items():
  for condition,reference in comparisons:
   if condition not in preds or reference not in preds:continue
   c,n=preds[condition],preds[reference];delta=c['score']-n['score']
   cases.append(dict(benchmark=b,protocol_label=label,key=key,image_sha256=n['image_sha256'],condition=condition,reference=reference,reference_score=n['score'],condition_score=c['score'],delta=delta,change='score_decrease' if delta<0 else 'score_increase' if delta>0 else 'tie',reference_raw_text=n['raw_text'],condition_raw_text=c['raw_text'],reference_parsed=n['parsed'],condition_parsed=c['parsed'],first_token_changed=n['first_token']!=c['first_token'],complete_token_ids_changed=n['token_ids']!=c['token_ids']))
 return cases
def mme_metrics(rows):
 tasks=defaultdict(list)
 for r in rows:tasks[r['task']].append(r)
 result={}
 for task,rr in tasks.items():
  pairs=defaultdict(list)
  for r in rr:pairs[r['pair_id']].append(r)
  assert all(len(p)==2 for p in pairs.values()),('MME incomplete pair',task)
  acc=sum(r['score'] for r in rr)/len(rr);ap=sum(all(r['score']==1 for r in p) for p in pairs.values())/len(pairs)
  result[task]=dict(questions=len(rr),pairs=len(pairs),Accuracy=acc*100,AccuracyPlus=ap*100,Total=(acc+ap)*100)
 perception=['existence','count','position','color','posters','celebrity','scene','landmark','artwork','OCR']
 return dict(tasks=result,Perception=sum(v['Total'] for k,v in result.items() if k in perception),Cognition=sum(v['Total'] for k,v in result.items() if k not in perception),Total=sum(v['Total'] for v in result.values()))
def mme_paired(rows,label):
 by=defaultdict(dict)
 for r in rows:by[r['key']][r['condition']]=r
 assert all(set(p)=={'Native','Physical-Original'} for p in by.values())
 # A cluster carries its question and double-question pair contributions in every task.
 tasks=sorted({r['task'] for r in rows});images=sorted({r['image_sha256'] for r in rows});ii={v:i for i,v in enumerate(images)}
 num=np.zeros((len(images),len(tasks),2));den=np.zeros_like(num);image_pairs=defaultdict(list)
 for p in by.values():
  n,c=p['Native'],p['Physical-Original'];i=ii[n['image_sha256']];j=tasks.index(n['task']);num[i,j,0]+=c['score']-n['score'];den[i,j,0]+=1;image_pairs[n['task'],n['pair_id']].append(p)
 for (task,_),pair in image_pairs.items():
  assert len(pair)==2;i=ii[pair[0]['Native']['image_sha256']];j=tasks.index(task)
  num[i,j,1]+=float(all(p['Physical-Original']['score']==1 for p in pair))-float(all(p['Native']['score']==1 for p in pair));den[i,j,1]+=1
 def total(a,b):return np.divide(a,b,out=np.zeros_like(a),where=b>0).sum(axis=(-1,-2))*100
 rng=np.random.default_rng(SEED);out=[]
 for start in range(0,REPS,64):
  ix=rng.integers(0,len(images),size=(min(64,REPS-start),len(images)));out.extend(total(num[ix].sum(1),den[ix].sum(1)).tolist())
 ci=np.quantile(out,[.025,.975]);return dict(benchmark='MME',protocol_label=label,metric='Official task-summed Acc+AccPlus points',condition='Physical-Original',reference='Native',paired_questions=len(by),image_clusters=len(images),delta=float(total(num.sum(0),den.sum(0))),CI95_low=float(ci[0]),CI95_high=float(ci[1]),bootstrap_replicates=REPS,seed=SEED,unit='Image SHA256 cluster across all tasks')
def summarize(rows,benchmark,label,expected,complete):
 summary=[];details={};conds=sorted({r['condition'] for r in rows},key=lambda c:CONDITIONS.index(c))
 for c in conds:
  rr=[r for r in rows if r['condition']==c];full=complete and len(rr)==expected;formal=None;metric='Accuracy %';status='COMPLETE' if full else 'PARTIAL_NOT_A_FORMAL_SCORE'
  if full:
   if benchmark=='MME':details[c]=mme_metrics(rr);formal=details[c]['Total'];metric='Official Total points (0-2800)'
   elif benchmark=='MMBench':
    groups=defaultdict(list)
    for r in rr:groups[r['circular_group']].append(r)
    assert len(groups)==1176 and sum(map(len,groups.values()))==4377
    formal=100*sum(all(r['score']==1 for r in g) for g in groups.values())/len(groups);metric='Circular exact-matching Accuracy %'
    if any(r['parse_failed'] for r in rr):status='EXACT_MATCHING_COMPLETE_AWAITING_OFFICIAL_JUDGE_FOR_UNRESOLVED'
    details[c]=dict(groups=1176,all_correct=sum(all(r['score']==1 for r in g) for g in groups.values()),unresolved_rows=sum(r['parse_failed'] for r in rr),official_judge_status='AWAITING_OFFICIAL_SCORING' if any(r['parse_failed'] for r in rr) else 'NOT_REQUIRED_ALL_OPTIONS_PARSED')
   elif benchmark=='OCRBench-v1':formal=sum(r['score'] for r in rr);metric='Official raw points (0-1000)'
   else:formal=sum(r['score'] for r in rr)/expected*100
  if benchmark=='POPE':
   tp=sum(r['parsed']=='yes' and r['gold']=='yes' for r in rr);fp=sum(r['parsed']=='yes' and r['gold']=='no' for r in rr);fn=sum(r['parsed']=='no' and r['gold']=='yes' for r in rr);tn=sum(r['parsed']=='no' and r['gold']=='no' for r in rr)
   precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
   details[c]=dict(TP=tp,FP=fp,FN=fn,TN=tn,Accuracy=(tp+tn)/len(rr)*100,Precision=precision*100,Recall=recall*100,F1=(2*precision*recall/(precision+recall)*100 if precision+recall else 0.),Yes_ratio=(tp+fp)/len(rr)*100)
  summary.append(dict(benchmark=benchmark,protocol_label=label,condition=c,status=status,completed=len(rr),expected=expected,formal_score=formal,metric=metric,observed_subset_mean_score=sum(r['score'] for r in rr)/len(rr)*100,empty=sum(r['empty'] for r in rr),truncated=sum(r['truncated'] for r in rr),parse_failed=sum(r['parse_failed'] for r in rr),duplicate_raw_outputs=len(rr)-len({r['raw_text'] for r in rr}),generated_tokens=sum(r['generated_tokens'] for r in rr),notes='Repeated yes/no/letter answers are legitimate; duplicate_raw_outputs is descriptive, not a data integrity error.'))
 return summary,details
def analyze_historical():
 manifest=read(O/'historical_inputs.json');rows=[];mismatches=[];expected_binding={k:sha(O/f'{k}.json') for k in ['protocol','source_manifest','model_manifest']}
 for s in manifest['rows']:
  for c in CONDITIONS:
   p=O/'historical_predictions'/f"{s['question_id']}__{c}.json"
   if not p.exists():continue
   record=read(p);b=record['binding'];r=record['result'];assert b['question_id']==s['question_id'] and b['sample_SHA256']==digest(s) and b['input_SHA256']==digest(manifest['input_bindings'][s['question_id']])
   for k,v in expected_binding.items():assert b[k+'_SHA256']==v
   parsed=r['text'].rstrip('.').lower();value=float(parsed==s['gold'])
   if value!=r['score'] or parsed!=r['prediction']:mismatches.append(str(p))
   rows.append(dict(benchmark='HISTORICAL_GQA1000_REPRODUCTION',protocol_label='HISTORICAL_FROZEN_PROTOCOL',key=s['question_id'],question_id=s['question_id'],image_sha256=s['image_sha256'],condition=c,raw_text=r['text'],parsed=parsed,gold=s['gold'],score=value,initial_score=r['score'],empty=r['empty'],truncated=r['truncated'],parse_failed=False,generated_tokens=r['generated_tokens'],first_token=r['first_token'],token_ids=json.dumps(r['token_ids']),eos=r['eos'],task=s['task'],types=json.dumps(s['types'],sort_keys=True)))
 assert not mismatches,('Historical independent scoring mismatch',mismatches)
 complete=len(rows)==3000 and (O/'historical_completion.json').exists();summary,_=summarize(rows,'HISTORICAL_GQA1000_REPRODUCTION','HISTORICAL_FROZEN_PROTOCOL',1000,complete)
 comparisons=[]
 if complete:
  for comp in [('Dense-F','Native'),('Physical-Original','Native'),('Physical-Original','Dense-F')]:comparisons.append(paired(rows,comp,'HISTORICAL_GQA1000_REPRODUCTION','HISTORICAL_FROZEN_PROTOCOL'))
 table(O/'historical_per_question.csv.gz',rows);table(O/'historical_accuracy_summary.csv',summary);table(O/'historical_paired_comparisons.csv',comparisons)
 table(O/'historical_paired_cases.csv.gz',paired_cases(rows,[('Dense-F','Native'),('Physical-Original','Native'),('Physical-Original','Dense-F')]))
 strata=[]
 if complete:
  for dim in ['task','structural','semantic']:
   values=defaultdict(list)
   for r in rows:
    val=r['task'] if dim=='task' else json.loads(r['types']).get(dim,'UNKNOWN');values[str(val)].append(r)
   for v,rr in values.items():
    for c in CONDITIONS:
     cc=[r for r in rr if r['condition']==c];strata.append(dict(benchmark='HISTORICAL_GQA1000_REPRODUCTION',dimension=dim,stratum=v,condition=c,n=len(cc),mean_score=sum(r['score'] for r in cc)/len(cc)*100))
 table(O/'historical_stratified.csv',strata);write(O/'historical_scoring_audit.json',dict(status='PASS' if complete else 'PARTIAL',checked_predictions=len(rows),independent_full_answer_exact_match=True,original_raw_scores_match=True,complete=complete,comparison_to_old_reference_only={'Native':69.4,'Dense-F':58.8,'Physical-Original':59.0}))
 return summary,comparisons
def analyze_seven():
 manifest=read(O/'input_manifest.json');manifest_sha=sha(O/'input_manifest.json');grouped=defaultdict(list)
 for s in manifest['rows']:grouped[s['benchmark'],s['protocol_label']].append(s)
 summary=[];comparisons=[];allrows=[];allstrata=[];details={};corrections=[];mismatches=[]
 source=read(O/'data_restore.json');v=source['views']['GQA-testdev-balanced'];gqa_annotations=read(Path(v['root'])/'source_annotations/gqa/testdev_balanced_questions.json');v=source['views']['ScienceQA-IMG-paper'];sqa_annotations=read(Path(v['root'])/'source_annotations/scienceqa/problems.json')
 expected_binding={k+'_SHA256':sha(O/f'{k}.json') for k in ['protocol','source_manifest','model_manifest']}
 for (benchmark,label),samples in grouped.items():
  rows=[];folder=O/'benchmark_predictions'/benchmark/label;request_cache={}
  for s in samples:
   for c in ['Native','Physical-Original']:
    p=folder/(hashlib.sha256(s['key'].encode()).hexdigest()+'__'+c+'.json')
    if not p.exists():continue
    record=read(p);assert record['binding']['sample_SHA256']==digest(s) and record['binding']['input_manifest_SHA256']==manifest_sha and record['gold']==s['answer'] and record['image_SHA256']==s['image_sha256']
    req=record['request_path']
    if req not in request_cache:request_cache[req]=read(O/req)
    raw=request_cache[req];assert digest(raw)==record['generation_SHA256'];assert digest(raw['input_layout'])==record['binding']['input_SHA256']
    for name,h in expected_binding.items():assert raw['binding'][name]==record['binding'][name]==h
    r=raw['result'];sc=score(s,r['text']);ind=independent(s,r['text'],sc['parsed']);assert abs(sc['score']-ind)<1e-10,(benchmark,s['key'],sc,ind)
    initial=record['scoring']
    if abs(initial['score']-sc['score'])>1e-10:
     item=dict(benchmark=benchmark,key=s['key'],condition=c,initial_score=initial['score'],correct_score=sc['score'],gold=s['answer'],parsed=sc['parsed'],raw_text=r['text'])
     if benchmark=='ScienceQA-IMG':corrections.append(item)
     else:mismatches.append(item)
    row=dict(benchmark=benchmark,protocol_label=label,key=s['key'],question_id=s['question_id'],image_sha256=s['image_sha256'],condition=c,raw_text=r['text'],parsed=sc['parsed'],gold=json.dumps(s['answer'],ensure_ascii=False),score=sc['score'],initial_score=initial['score'],empty=r['empty'],truncated=r['truncated'],parse_failed=sc['parse_failed'],generated_tokens=r['generated_tokens'],first_token=r['first_token'],token_ids=json.dumps(r['token_ids']),eos=r['eos'],request_SHA256=s['request_SHA256'])
    if benchmark in ['POPE','MME']:row['gold']=s['answer']
    for k in ['task','subtask','strategy','pair_id','circular_group','dataset_name','type','category']:
     if k in s:row[k]=s[k]
    if benchmark=='GQA':
     types=gqa_annotations[s['question_id']].get('types',{})
     for dim in ['semantic','structural','detailed']:
      if dim in types:row['question_type_'+dim]=types[dim]
    if benchmark=='ScienceQA-IMG':
     original=sqa_annotations[s['question_id']]
     for dim in ['subject','topic','grade','skill']:
      if dim in original:row['question_type_'+dim]=original[dim]
    rows.append(row)
  assert not mismatches,('Unexpected raw scorer mismatch',mismatches)
  completion=read(folder/'completion.json') if (folder/'completion.json').exists() else {};complete=completion.get('status')=='COMPLETE' and len(rows)==2*len(samples)
  ss,dd=summarize(rows,benchmark,label,len(samples),complete);summary.extend(ss);details[benchmark+':'+label]=dd;allrows.extend(rows)
  if complete:
   if benchmark=='MME':comparisons.append(mme_paired(rows,label))
   elif benchmark=='MMBench':
    gr=defaultdict(list)
    for r in rows:gr[r['circular_group'],r['condition']].append(r)
    rr=[dict(key=g,condition=c,image_sha256=rs[0]['image_sha256'],score=float(all(r['score']==1 for r in rs)),parsed=str(all(r['score']==1 for r in rs))) for (g,c),rs in gr.items()]
    comparisons.append(paired(rr,('Physical-Original','Native'),benchmark,label,metric='Circular Accuracy pp'))
   else:comparisons.append(paired(rows,('Physical-Original','Native'),benchmark,label,metric='Official raw points' if benchmark=='OCRBench-v1' else 'Accuracy pp',scale=1000. if benchmark=='OCRBench-v1' else 100.))
   strata=defaultdict(list)
   for r in rows:
    for dim in ['task','strategy','dataset_name','type','category','question_type_semantic','question_type_structural','question_type_detailed','question_type_subject','question_type_topic','question_type_grade','question_type_skill']:
     if dim in r:strata[dim,str(r[dim]),r['condition']].append(r)
   for (dim,v,c),rr in strata.items():allstrata.append(dict(benchmark=benchmark,protocol_label=label,dimension=dim,stratum=v,condition=c,n=len(rr),mean_score=sum(r['score'] for r in rr)/len(rr)*100))
  print('CPU_SCORING_AUDITED',benchmark,label,len(rows),flush=True)
 table(O/'per_question.csv.gz',allrows);table(O/'accuracy_summary.csv',summary);table(O/'paired_comparisons.csv',comparisons);table(O/'stratified_results.csv',allstrata);table(O/'scienceqa_offline_score_corrections.csv',corrections)
 table(O/'paired_cases.csv.gz',paired_cases(allrows,[('Physical-Original','Native')]))
 write(O/'scoring_details.json',details);write(O/'independent_scoring_audit.json',dict(status='PASS' if (O/'seven_completion.json').exists() else 'PARTIAL',checked_predictions=len(allrows),non_ScienceQA_disagreements=mismatches,ScienceQA_integer_Gold_bridge_corrections=len(corrections),ScienceQA_initial_scores_preserved=True,independent_logic='Exact full answer; ten-answer count formula; explicit POPE first-sentence word logic; explicit MME first-four-character and double-question arithmetic; explicit OCR substring normalization; option index and complete Circular aggregation checked independently. Official option extraction and TextVQA normalization source are shared, not claimed as independent implementations.',raw_outputs_and_frozen_input_unchanged=True))
 return summary,comparisons
def analyze_efficiency():
 manifest=read(O/'historical_inputs.json');qids=manifest['benchmark_question_ids'];records=defaultdict(list);rawrows=[];layouts={}
 for q in qids:
  for regime in ['natural','fixed16','fixed32']:
   for c in CONDITIONS:
    for rep in range(5):
     p=O/'efficiency_raw'/f'{q}__{regime}__{c}__{rep}.json'
     if not p.exists():continue
     record=read(p);r=record['result'];assert r['condition']==c
     if regime!='natural':assert r['generated_tokens']==int(regime[5:]) and r['incremental_decode_steps']==int(regime[5:])-1
     if c=='Physical-Original':assert all(x['live_tokens']==0 for x in r['kv']['layers'] if x['layer'] in range(1,7))
     v=dict(question_id=q,image_sha256=record['image_sha256'],regime=regime,condition=c,repeat=rep,condition_order=json.dumps(record['condition_order']),generated_tokens=r['generated_tokens'],incremental_decode_steps=r['incremental_decode_steps'],KV_Live_MiB=r['kv']['live_bytes']/2**20,KV_Capacity_MiB=r['kv']['capacity_bytes']/2**20,Peak_Allocated_MiB=r['memory']['peak_allocated']/2**20,Peak_Reserved_MiB=r['memory']['peak_reserved']/2**20,**r['timing'])
     records[q,regime,c].append(v);rawrows.append(v)
     if rep==0:layouts[q,regime,c]=r['kv']
 per=[];metrics=['generated_tokens','incremental_decode_steps','KV_Live_MiB','KV_Capacity_MiB','Peak_Allocated_MiB','Peak_Reserved_MiB','vision_ms','prefill_language_ms','ttft_gpu_ms','ttft_wall_ms','decode_gpu_ms','decode_gpu_ms_per_token','e2e_wall_ms']
 for (q,regime,c),rr in records.items():
  if len(rr)!=5:continue
  per.append(dict(question_id=q,image_sha256=rr[0]['image_sha256'],regime=regime,condition=c,measured_repeats=5,**{m:statistics.median(r[m] for r in rr) for m in metrics}))
 summary=[];comparisons=[]
 for regime in ['natural','fixed16','fixed32']:
  for c in CONDITIONS:
   rr=[r for r in per if r['regime']==regime and r['condition']==c]
   if not rr:continue
   full=len(rr)==48;summary.append(dict(regime=regime,condition=c,inputs=len(rr),status='COMPLETE' if full else 'PARTIAL',**{m:statistics.mean(r[m] for r in rr) for m in metrics}))
  for c in CONDITIONS[1:]:
   by=defaultdict(dict)
   for r in per:
    if r['regime']==regime:by[r['question_id']][r['condition']]=r
   pairs=[p for p in by.values() if c in p and 'Native' in p]
   if len(pairs)!=48:continue
   native=np.array([p['Native']['e2e_wall_ms'] for p in pairs]);cond=np.array([p[c]['e2e_wall_ms'] for p in pairs]);rng=np.random.default_rng(SEED);speed=[]
   for start in range(0,REPS,128):
    ix=rng.integers(0,48,size=(min(128,REPS-start),48));speed.extend((native[ix].sum(1)/cond[ix].sum(1)).tolist())
   ci=np.quantile(speed,[.025,.975]);kvn=np.array([p['Native']['KV_Live_MiB'] for p in pairs]);kvc=np.array([p[c]['KV_Live_MiB'] for p in pairs]);capn=np.array([p['Native']['KV_Capacity_MiB'] for p in pairs]);capc=np.array([p[c]['KV_Capacity_MiB'] for p in pairs])
   comparisons.append(dict(regime=regime,condition=c,reference='Native',inputs=48,metric='E2E speedup ratio of summed input repeat medians',speedup=float(native.sum()/cond.sum()),CI95_low=float(ci[0]),CI95_high=float(ci[1]),KV_live_saving_percent=float((1-kvc.sum()/kvn.sum())*100),KV_capacity_saving_percent=float((1-capc.sum()/capn.sum())*100),bootstrap_replicates=REPS,seed=SEED))
 table(O/'efficiency_measurements.csv.gz',rawrows);table(O/'efficiency_per_input.csv',per);table(O/'efficiency_summary.csv',summary);table(O/'efficiency_paired_comparisons.csv',comparisons)
 write(O/'kv_layout.json',dict(status='COMPLETE' if len(layouts)==48*3*3 else 'PARTIAL',measurements='actual tensor live elements and untyped storage bytes; sum includes both K/V. Prefill generates first token; fixed32 cache has input+31 processed positions.',layouts=[dict(question_id=q,regime=regime,condition=c,kv=kv) for (q,regime,c),kv in layouts.items()]))
 return summary,comparisons
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['historical','seven','efficiency','all'],default='all');a=p.parse_args()
 result={}
 if a.stage in ['historical','all']:result['historical']=analyze_historical()
 if a.stage in ['seven','all']:result['seven']=analyze_seven()
 if a.stage in ['efficiency','all']:result['efficiency']=analyze_efficiency()
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
