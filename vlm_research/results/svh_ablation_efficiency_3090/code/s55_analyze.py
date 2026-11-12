"""Independent CPU completeness, score recomputation and paired arithmetic."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from s55_common import *
import argparse,sqlite3,csv,collections,statistics,math,bisect
from vlm_research.information_horizon_alignment import score as scorer

METHODS=['Native','wo_selective','wo_bypass','SVH']
LABELS={'Native':'Native Full','wo_selective':'SVH w/o Selective Transfer','wo_bypass':'SVH w/o Attention Bypass','SVH':'SVH (Full)','SVH-R':'SVH-R'}
def csvwrite(p,rows):
 rows=list(rows);assert rows,p;fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def stats(xs):return dict(mean=statistics.mean(xs),median=statistics.median(xs),std=statistics.stdev(xs) if len(xs)>1 else 0,min=min(xs),max=max(xs),n=len(xs))
def quality():
 root=O/'ablation';state=read(root/'progress.json');assert state['status']=='GENERATIONS_COMPLETE_PENDING_CPU_ANALYSIS'
 inputs={s['key']:s for s in lines(root/'inputs.jsonl.gz')};gold=read(root/'answers.json');records=[];seen=set();by=collections.defaultdict(list)
 db=sqlite3.connect('file:'+str(root/'checkpoint.sqlite')+'?mode=ro',uri=True)
 assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
 for key,condition,payload,h in db.execute('SELECT key,condition,payload,sha256 FROM predictions ORDER BY key,condition'):
  r=json.loads(payload);assert digest(r)==h and (key,condition) not in seen;seen.add((key,condition));s=inputs[key];g=r['generation'];text=g['text']
  check=scorer.score(dict(s,answer=gold[key]),text);assert check==r['scoring'],key
  assert g['eos'] or g['truncated']
  assert len(g['token_ids'])<=s['max_new_tokens'] and (g['token_ids'][-1]==2 if g['eos'] else len(g['token_ids'])==s['max_new_tokens'])
  # Independently reimplement the simple official lexical arithmetic as a check.
  if s['benchmark']=='POPE':
   tokens=text.split('.')[0].replace(',','').split(' ');parsed='no' if any(w in tokens for w in ('No','not','no')) else 'yes';independent=float(parsed==gold[key])
  else:
   import re
   option=text if len(text)==1 and text in 'ABCDE' else text[0] if len(text)>=3 and text[0] in 'ABCDE' and text[1:3]=='. ' else None
   if option is None:
    matches=re.findall(r'The answer is ([A-Z]).',text);option=matches[0] if len(matches)==1 else None
   parsed='ABCDE'.index(option) if option in s['options'] else -1;independent=float(parsed==gold[key])
  assert independent==check['score']
  row=dict(key=key,benchmark=s['benchmark'],condition=condition,strategy=s.get('strategy'),question_id=s['question_id'],prediction=text,parsed=check['parsed'],gold=gold[key],correct=int(check['score']),output_tokens=len(g['token_ids']),eos=g['eos'],truncated=g['truncated'],empty=g['empty'],parse_failed=check.get('parse_failed',False),provenance='REUSED_VERIFIED_FULL_SVH' if condition=='SVH' else 'NEW_GPU_NATURAL_EOS')
  records.append(row);by[(s['benchmark'],condition)].append(row)
 assert len(records)==43708 and len(seen)==len(inputs)*4
 for key in inputs:assert all((key,m) in seen for m in METHODS)
 full={r['key']:r['correct'] for r in records if r['condition']=='SVH'}
 for r in records:r['correct_minus_full_SVH']=r['correct']-full[r['key']]
 summary=[]
 for condition in METHODS:
  for bench in ('ScienceQA-IMG','POPE'):
   rs=by[(bench,condition)];assert len(rs)==(2017 if bench=='ScienceQA-IMG' else 8910)
   correct=sum(r['correct'] for r in rs);ref=sum(full[r['key']] for r in rs)
   summary.append(dict(condition=condition,label=LABELS[condition],benchmark=bench,questions=len(rs),correct=correct,accuracy_percent=correct/len(rs)*100,delta_vs_full_SVH_pp=(correct-ref)/len(rs)*100,empty=sum(r['empty'] for r in rs),truncated=sum(r['truncated'] for r in rs),parse_failures=sum(r['parse_failed'] for r in rs),full_SVH_correct_method_wrong=sum(full[r['key']] and not r['correct'] for r in rs),method_correct_full_SVH_wrong=sum(r['correct'] and not full[r['key']] for r in rs)))
 subsets=[]
 for condition in METHODS:
  for strategy in ('random','popular','adversarial'):
   rs=[r for r in by[('POPE',condition)] if r['strategy']==strategy];assert len(rs)==(2910 if strategy=='random' else 3000)
   tp=sum(r['parsed']=='yes' and r['gold']=='yes' for r in rs);tn=sum(r['parsed']=='no' and r['gold']=='no' for r in rs);fp=sum(r['parsed']=='yes' and r['gold']=='no' for r in rs);fn=len(rs)-tp-tn-fp
   subsets.append(dict(condition=condition,strategy=strategy,questions=len(rs),correct=tp+tn,accuracy_percent=(tp+tn)/len(rs)*100,tp=tp,tn=tn,fp=fp,fn=fn))
 csvwrite(root/'summary.csv',summary);csvwrite(root/'pope_subsets.csv',subsets);csvwrite(root/'sqa_results.csv',[r for r in records if r['benchmark']=='ScienceQA-IMG']);csvwrite(root/'pope_results.csv',[r for r in records if r['benchmark']=='POPE'])
 with (root/'predictions.jsonl').open('w') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 validation=dict(status='PASS',records=43708,new_generations=32781,reused_full_SVH=10927,all_row_hashes_valid=True,all_scores_recomputed_and_independent_lexical_check=True,all_four_conditions_same_inputs=True,missing=[],duplicate=[],summaries=summary,pope_subsets=subsets)
 save(root/'validation.json',validation);verify_sources();print('QUALITY_CPU_FINAL_PASS',json.dumps(summary,ensure_ascii=False),flush=True);return validation

def length():
 root=O/'generation_length';formal=[];summary=[];paired=[];session_status={};drift=[];telemetry_stats=[]
 expected_inputs={s['key']:s['input_hashes'] for s in lines(root/'input_manifest.jsonl.gz')}
 assert len(expected_inputs)==30
 expected_pairs={(key,method,n) for key in expected_inputs for method in ('Native','SVH','SVH-R') for n in (16,32,64,128)}
 for session in (1,2):
  p=root/('session'+str(session));state=read(p/'progress.json');assert state['status'] in ('COMPLETE','COMPLETE_WITH_PERFORMANCE_WARNINGS');session_status[str(session)]=state
  assert read(p/'smoke_validation.json')['status']=='PASS'
  gpu_rows=[]
  with (p/'gpu_telemetry.jsonl').open() as f:
   for l in f:
    row=json.loads(l)
    if 'SM_MHz' in row:gpu_rows.append(row)
  timestamps=[r['at_ns'] for r in gpu_rows];assert timestamps==sorted(timestamps) and gpu_rows
  telemetry_stats.append(dict(session=session,samples=len(gpu_rows),temperature=stats([r['temperature_C'] for r in gpu_rows]),SM_MHz=stats([r['SM_MHz'] for r in gpu_rows]),power_mW=stats([r['power_mW'] for r in gpu_rows]),throttle_masks=dict(collections.Counter(str(r['throttle_mask']) for r in gpu_rows))))
  db=sqlite3.connect('file:'+str(p/'checkpoint.sqlite')+'?mode=ro',uri=True);seen=set();current=[]
  assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
  for rid,category,key,method,n,payload,h in db.execute('SELECT * FROM requests ORDER BY id'):
   r=json.loads(payload);assert digest(r)==h and r['id']==rid;g=r['generation'];assert g['output_tokens']==n and len(g['token_ids'])==n and g['real_decode_calls']==n-1
   assert abs(g['E2E_ms']-g['TTFT_ms']-(n-1)*g['TPOT_ms'])<1e-6
   assert g['all_logits_finite'] and g['cache']['storage_bytes']==g['cache']['live_bytes']
   if category!='formal':continue
   assert key in expected_inputs and r['input_hashes']==expected_inputs[key] and r['GPU_UUID']==UUID
   assert (key,method,n) not in seen;seen.add((key,method,n));assert method in ('Native','SVH','SVH-R')
   ix=bisect.bisect_right(timestamps,r['begin_ns'])-1;gpu_state=gpu_rows[max(0,ix)]
   within=[x for x in gpu_rows[max(0,ix):] if x['at_ns']<=r['end_ns']];within=within or [gpu_state]
   row=dict(session=session,GPU_UUID=r['GPU_UUID'],request_id=rid,input_id=key,method=method,output_tokens=n,TTFT_ms=g['TTFT_ms'],TPOT_ms=g['TPOT_ms'],CUDA_Event_Decode_TPOT_ms=g['CUDA_Event_Decode_TPOT_ms'],E2E_ms=g['E2E_ms'],KV_MiB=g['cache']['live_bytes']/2**20,begin_ns=r['begin_ns'],end_ns=r['end_ns'],SM_MHz_at_start=gpu_state['SM_MHz'],SM_MHz_min=min(x['SM_MHz'] for x in within),SM_MHz_max=max(x['SM_MHz'] for x in within),temperature_C_max=max(x['temperature_C'] for x in within),power_W_mean=statistics.mean(x['power_mW'] for x in within)/1000,Pstates=sorted(set(x['pstate'] for x in within)),throttle_masks=sorted(set(x['throttle_mask'] for x in within)),thread_CPU_ms=r['thread_CPU_ms'],thread_runqueue_ms=r['thread_runqueue_ms'],early_vs_late_decode_change=r['early_vs_late_decode_change'])
   current.append(row);formal.append(row)
  assert len(current)==360 and seen==expected_pairs
  assert collections.Counter(r['method'] for r in current)=={'Native':120,'SVH':120,'SVH-R':120}
  csvwrite(p/'per_request.csv',current)
  for n in (16,32,64,128):
   native={r['input_id']:r for r in current if r['method']=='Native' and r['output_tokens']==n}
   for method in ('Native','SVH','SVH-R'):
    rows=[r for r in current if r['method']==method and r['output_tokens']==n];assert len(rows)==30
    result=dict(session=session,output_tokens=n,method=method,requests=30)
    for metric in ('TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','E2E_ms','KV_MiB'):
     st=stats([r[metric] for r in rows])
     for stat,v in st.items():result[metric+'_'+stat]=v
     native_mean=statistics.mean(r[metric] for r in native.values());result[metric+'_reduction_percent']=(1-st['mean']/native_mean)*100
     differences=[native[r['input_id']][metric]-r[metric] for r in rows];diff=stats(differences);ci=2.045229642*diff['std']/math.sqrt(30)
     paired.append(dict(session=session,length=n,method=method,metric=metric,n=30,native_minus_method_mean=diff['mean'],median=diff['median'],std=diff['std'],exploratory_CI95_low=diff['mean']-ci,exploratory_CI95_high=diff['mean']+ci,method_better_inputs=sum(x>0 for x in differences),method_worse_inputs=sum(x<0 for x in differences),note='Paired input t interval; exploratory, correlated timing/system drift; no multiple-testing correction'))
    result['E2E_speedup']=statistics.mean(r['E2E_ms'] for r in native.values())/result['E2E_ms_mean'];summary.append(result)
  checks=read(p/'calibration.json')['checks'];assert len(checks)==12
  assert collections.Counter(x['window'] for x in checks)=={w:2 for w in range(1,7)}
  drift.extend([dict(session=session,**x) for x in checks])
 csvwrite(root/'per_request.csv',formal);csvwrite(root/'latency_summary.csv',summary);csvwrite(root/'paired_statistics.csv',paired)
 means=[]
 for n in (16,32,64,128):
  for method in ('Native','SVH','SVH-R'):
   rows=[r for r in summary if r['output_tokens']==n and r['method']==method];assert len(rows)==2
   result=dict(output_tokens=n,method=method,sessions=2,requests=60)
   for metric in ('TTFT_ms','TPOT_ms','E2E_ms','KV_MiB','CUDA_Event_Decode_TPOT_ms'):
    values=[r[metric+'_mean'] for r in rows];result[metric]=statistics.mean(values);result[metric+'_session_mean_std']=statistics.stdev(values);result[metric+'_session1']=values[0];result[metric+'_session2']=values[1];result[metric+'_session2_vs_session1_percent']=(values[1]/values[0]-1)*100
   result['E2E_speedup_mean_within_session']=statistics.mean(r['E2E_speedup'] for r in rows);result['E2E_reduction_percent_mean_within_session']=statistics.mean(r['E2E_ms_reduction_percent'] for r in rows);means.append(result)
 csvwrite(root/'latency_two_sessions.csv',means);save(root/'calibration.json',dict(checks=drift,max_abs_E2E_drift=max(abs(x['relative_drift']['E2E_ms']) for x in drift),max_abs_TPOT_drift=max(abs(x['relative_drift']['TPOT_ms']) for x in drift),warn_checks=sum(x['warn'] for x in drift),no_baseline_resets=True));save(root/'GPU_telemetry_summary.json',telemetry_stats)
 validation=dict(status='COMPLETE_WITH_PERFORMANCE_WARNINGS' if any(s['status']=='COMPLETE_WITH_PERFORMANCE_WARNINGS' for s in session_status.values()) else 'COMPLETE',execution_correctness='PASS',sessions=2,formal_requests=720,all_30_ids_and_tensor_hashes_exact=True,physical_KV_and_RoPE_smoke='PASS',same_GPU_Native_noop='PASS',request_hashes_and_timing_identity='PASS',session_status=session_status,summary=means,paired_statistics_rows=len(paired))
 save(root/'validation.json',validation);verify_sources();print('LENGTH_CPU_FINAL_PASS',validation['status'],flush=True);return validation

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--quality-only',action='store_true');ap.add_argument('--length-only',action='store_true');args=ap.parse_args()
 if not args.length_only:quality()
 if not args.quality_only:length()
if __name__=='__main__':main()
