from pathlib import Path
import json,csv,statistics,bisect,collections
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
def percentile(xs,p):
 xs=sorted(xs);i=(len(xs)-1)*p;lo=int(i);hi=min(lo+1,len(xs)-1);return xs[lo]+(xs[hi]-xs[lo])*(i-lo)
summary=[];issues=[]
for G in [p for p in A.glob('*/*') if p.name.startswith('session') or p.name.startswith('refine_session')]:
 if not (G/'completion.json').exists():continue
 done=json.loads((G/'completion.json').read_text())
 if not done['status'].startswith('COMPLETE'):continue
 rows=[json.loads(l) for l in (G/'requests.jsonl').read_text().splitlines()]
 formal=[r for r in rows if r['category']=='formal']
 if not formal:continue
 telemetry=[json.loads(l) for l in (G/'gpu_telemetry.jsonl').read_text().splitlines()]
 stamps=[r['at_ns'] for r in telemetry]
 groups=collections.defaultdict(list)
 for r in formal:
  samples=telemetry[bisect.bisect_left(stamps,r['begin_ns']):bisect.bisect_right(stamps,r['end_ns'])]
  groups[r['method'],r['stage']].extend(samples)
 for (method,stage),rs in groups.items():
  out=dict(kind=G.parent.name,session=G.name,method=method,stage=stage,samples=len(rs))
  for field in ['SM_MHz','memory_MHz','temperature_C','power_mW','GPU_util','memory_used']:
   xs=[r[field] for r in rs if isinstance(r.get(field),(float,int))]
   if not xs:continue
   out[field+'_min']=min(xs);out[field+'_median']=statistics.median(xs);out[field+'_p95']=percentile(xs,.95);out[field+'_max']=max(xs)
  out['power_limits_mW']=json.dumps(sorted(set(r['power_limit_mW'] for r in rs)))
  out['throttle_masks_raw']=json.dumps(dict(collections.Counter(r['throttle_mask'] for r in rs)))
  summary.append(out)
 for l in (G/'processes_telemetry.jsonl').read_text().splitlines():
  r=json.loads(l)
  if r.get('other_processes'):issues.append(dict(kind=G.parent.name,session=G.name,**r))
fields=list(dict.fromkeys(k for r in summary for k in r))
with (A/'gpu_environment_per_method.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(summary)
(A/'process_conflict_validation.json').write_text(json.dumps(dict(status='PASS' if not issues else 'FAILED',other_compute_process_records=issues,completed_sessions=len(set((x['kind'],x['session']) for x in summary))),indent=2)+'\n')
print(json.dumps(dict(summary_rows=len(summary),other_process_records=len(issues))))
