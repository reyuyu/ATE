from pathlib import Path
import json,csv,collections,statistics,hashlib,sqlite3,math,argparse,gzip,os,subprocess,difflib
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
R=Path('/home/featurize/work/ATE')
D=R/'vlm_research/results/rtx3090_unified_efficiency_18conditions'
W=Path('/home/featurize/ate-experiments/rtx3090-18conditions-20261113')
MiB=1048576
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def csvsave(name,rows,fields=None):
 p=A/name
 if not fields:fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v) for k,v in r.items()} for r in rows)
def mean(v):return statistics.mean(v) if v else None
def cache(r):return r['generation']['cache']
def flat(r):
 g=r['generation'];c=cache(r)
 return dict(kind=r['kind'],session=r['session'],campaign='refinement_E' if r['session'].startswith('refine_') else 'primary',category=r['category'],key=r['key'],method=r['method'],stage=r['stage'],request_id=r['id'],
  input_index=r.get('input_index'),order_in_input=r.get('order_in_input'),repeat=r.get('repeat'),GPU_UUID=r['GPU_UUID'],
  TTFT_ms=g['TTFT_ms'],TPOT_ms=g['TPOT_ms'],E2E_ms=g['E2E_ms'],CUDA_Event_Decode_TPOT_ms=g['CUDA_Event_Decode_TPOT_ms'],
  KV_Live_MiB=c['live_bytes']/MiB,KV_Storage_MiB=c['storage_bytes']/MiB,
  KV_Capacity_Tokens=sum(x.get('capacity_tokens',0) for x in c['layers']),
  peak_allocated_MiB=r['peak_allocated_bytes']/MiB,peak_reserved_MiB=r['peak_reserved_bytes']/MiB,
  output_tokens=g['output_tokens'],real_decode_calls=g['real_decode_calls'],instrumentation_perturbed=r['instrumentation_perturbed'],
  teacher_forced_reference=r['teacher_forced_reference'],input_hashes=r['input_hashes'],prompt_sha256=r['prompt_sha256'],
  image_sha256=r['image_sha256'],token_ids=g['token_ids'],begin_ns=r['begin_ns'],end_ns=r['end_ns'])
def aggregate(rows,keys):
 groups=collections.defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in keys)].append(r)
 out=[]
 metrics=['TTFT_ms','TPOT_ms','E2E_ms','CUDA_Event_Decode_TPOT_ms','KV_Live_MiB','KV_Storage_MiB','peak_allocated_MiB','peak_reserved_MiB']
 for key,rs in sorted(groups.items()):
  row=dict(zip(keys,key));row.update(requests=len(rs),distinct_inputs=len(set(r['key'] for r in rs)))
  for m in metrics:
   values=[r[m] for r in rs];row[m+'_mean']=mean(values);row[m+'_median']=statistics.median(values)
  out.append(row)
 return out
def collect():
 records=[];validation=[];phases=[];warnings=[]
 for kind in ['Qwen','InternVL','LLaVA']:
  for G in sorted((A/kind).glob('*')) if (A/kind).exists() else []:
   if not G.is_dir() or not (G/'requests.jsonl').exists():continue
   rows=[json.loads(x) for x in (G/'requests.jsonl').read_text().splitlines() if x.strip()]
   complete=read(G/'completion.json') if (G/'completion.json').exists() else read(G/'progress.json')
   phases.append(dict(complete,kind=kind,phase=G.name))
   if complete['status'].startswith('COMPLETE'):
    con=sqlite3.connect('file:'+str(G/'checkpoint.sqlite')+'?mode=ro',uri=True)
    sql=con.execute('select id,payload,sha256 from requests order by id').fetchall();con.close()
    assert len(sql)==len(rows),(G,len(sql),len(rows))
    for (i,payload,digest),r in zip(sql,rows):
     assert hashlib.sha256(payload.encode()).hexdigest()==digest and json.loads(payload)==r and i==r['id'],('CHECKPOINT_INTEGRITY',G,i)
    validation.append(dict(check='SQLite_SHA256_JSONL_exact',kind=kind,phase=G.name,count=len(rows),status='PASS'))
   if (G/'drift_checks.json').exists():
    for d in read(G/'drift_checks.json'):
     if d['status']=='WARN':warnings.append(dict(kind=kind,session=G.name,type='NATIVE_DRIFT_GT5PCT',**d,policy='Preserved all requests; no rebaselining/retry/deletion'))
   for r in rows:
    g=r['generation'];c=cache(r);assert r['GPU_UUID']=='GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6'
    assert g['output_tokens']==32 and len(g['token_ids'])==32 and g['real_decode_calls']==31 and g['prefill_calls']==1 and g['use_cache'] and g['logits_to_keep']==1 and g['all_logits_finite']
    assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6
    assert sum(x['live_bytes'] for x in c['layers'])==c['live_bytes']
    assert sum(x['storage_bytes'] for x in c['layers'])==c['storage_bytes']
    for x in c['layers']:
     assert x['capacity_tokens']>=x['used_tokens']>=0 and x['storage_bytes']>=x['live_bytes']
     if x['K_shape'] is not None:
      assert x['K_shape']==x['V_shape'];assert x['live_bytes']==2*math.prod(x['K_shape'])*2
    if r['category'] in ['formal','ablation']:assert not r['instrumentation_perturbed'] and not r['teacher_forced_reference']
    if r['category']=='formal' and r['method']!='Native':
     skip=set(range(1,7)) if kind=='Qwen' else set(range(2,6))
     for x in c['layers']:
      assert (x['used_tokens']==0)==(x['layer'] in skip),('SKIP_KV',kind,r['method'],x)
      if r['method']=='SVH' or kind=='LLaVA':assert x['visual_tokens']==0,('UNEXPECTED_VISUAL_KV',kind,r['method'],x)
      elif r['method']=='SVH-R':assert (x['visual_tokens']>0)==(x['layer']>=19)
   records.extend(rows)
  for phase in ['screen','audit','session1','session2','refine_screen','refine_audit','refine_session1','refine_session2']:
   G=A/kind/phase
   if not (G/'completion.json').exists():continue
   if phase.startswith('session') or phase.startswith('refine_session'):
    done=read(G/'completion.json')
    if not done['status'].startswith('COMPLETE'):continue
    rs=[r for r in records if r['kind']==kind and r['session']==phase and r['category']=='formal']
    assert len(rs)==done['formal_expected']==done['formal_completed']
    grouping=collections.defaultdict(list)
    for r in rs:grouping[r['method'],r['stage']].append(r)
    expected_keys={r['key'] for r in rs if r['method']=='Native'}
    assert len(expected_keys)==30
    for (method,stage),sub in grouping.items():
     assert len(sub)==30 and {r['key'] for r in sub}==expected_keys
    validation.append(dict(check='30_same_inputs_each_condition',kind=kind,phase=phase,methods=len(grouping),status='PASS'))
  if (A/kind/'screen/native_noop.json').exists():assert read(A/kind/'screen/native_noop.json')['status']=='PASS'
 allflat=[flat(r) for r in records]
 formal=[r for r in allflat if r['category']=='formal'];ablation=[r for r in allflat if r['category']=='ablation']
 csvsave('efficiency_per_request.csv',formal)
 csvsave('ablation_per_request.csv',ablation)
 csvsave('ablation_efficiency.csv',aggregate(ablation,['kind','campaign','method','stage']))
 summary=aggregate(formal,['kind','session','method','stage'])
 csvsave('optimized_efficiency_32tokens.csv',summary)
 csvsave('pooled_efficiency_32tokens.csv',aggregate(formal,['kind','campaign','method','stage']))
 # Same-session paired comparisons only; improve percent is mean of per-input fractional reductions.
 lookup={(r['kind'],r['session'],r['key'],r['method'],r['stage']):r for r in formal}
 paired=[];regression=[]
 for r in formal:
  if r['method']=='Native':continue
  refs=[('same_session_Native','Native','A')]
  if r['stage']!='A':refs.append(('same_method_frozen',r['method'],'A'))
  for refname,method,stage in refs:
   if (r['kind'],r['session'],r['key'],method,stage) not in lookup:continue
   ref=lookup[r['kind'],r['session'],r['key'],method,stage]
   assert r['input_hashes']==ref['input_hashes'] and r['prompt_sha256']==ref['prompt_sha256'] and r['image_sha256']==ref['image_sha256']
   item={k:r[k] for k in ['kind','session','key','method','stage']}
   item.update(reference=refname,reference_method=method,reference_stage=stage)
   for m in ['TTFT_ms','TPOT_ms','E2E_ms','KV_Live_MiB','KV_Storage_MiB']:
    item[m+'_delta']=r[m]-ref[m];item[m+'_improvement_pct']=(1-r[m]/ref[m])*100
   paired.append(item)
  if r['stage']!='A':
   if (r['kind'],r['session'],r['key'],r['method'],'A') not in lookup:continue
   ref=lookup[r['kind'],r['session'],r['key'],r['method'],'A']
   exact=r['token_ids']==ref['token_ids'];assert exact,('FORMAL_TOKEN_REGRESSION',r['kind'],r['method'],r['key'])
   regression.append(dict(kind=r['kind'],session=r['session'],key=r['key'],method=r['method'],stage=r['stage'],all32_tokens_exact=exact,quality_scope='Matched fixed32-token regression; no full quality benchmark and no changed attention arithmetic'))
 csvsave('paired_per_input.csv',paired);csvsave('quality_output_regression.csv',regression)
 groups=collections.defaultdict(list)
 for r in paired:groups[tuple(r[k] for k in ['kind','session','method','stage','reference'])].append(r)
 pair_summary=[]
 for k,rs in sorted(groups.items()):
  row=dict(zip(['kind','session','method','stage','reference'],k));row['pairs']=len(rs)
  for m in ['TTFT_ms','TPOT_ms','E2E_ms','KV_Live_MiB','KV_Storage_MiB']:
   for suffix in ['delta','improvement_pct']:
    values=[r[m+'_'+suffix] for r in rs];row[m+'_'+suffix+'_mean']=mean(values);row[m+'_'+suffix+'_median']=statistics.median(values)
  pair_summary.append(row)
 csvsave('paired_comparisons.csv',pair_summary)
 kv_rows=[]
 for r in records:
  if r['category'] not in ['formal','ablation']:continue
  for layer in cache(r)['layers']:
   kv_rows.append(dict(kind=r['kind'],session=r['session'],campaign='refinement_E' if r['session'].startswith('refine_') else 'primary',category=r['category'],key=r['key'],method=r['method'],stage=r['stage'],request_id=r['id'],**layer))
 csvsave('kv_physical_audit.csv',kv_rows)
 snaprows=[]
 for G in A.glob('*/*'):
  p=G/'physical_audit.json'
  if not p.exists():continue
  for snap in read(p):
   for layer in snap['layers']:
    snaprows.append(dict(kind=G.parent.name,phase=G.name,**{k:v for k,v in snap.items() if k!='layers'},**layer))
 csvsave('kv_lifecycle_snapshots.csv',snaprows)
 numerical=[];layerwise=[];logits=[]
 for G in A.glob('*/*'):
  p=G/'numerical_validation.json'
  if not p.exists():continue
  for v in read(p):
   meta=dict(kind=G.parent.name,phase=G.name,key=v['key'],method=v['method'],stage=v['stage'])
   numerical.append(dict(meta,**{k:x for k,x in v.items() if k not in ['layerwise_differences','logits_per_step']}))
   for x in v.get('layerwise_differences',[]):layerwise.append(dict(meta,**x))
   for x in v.get('logits_per_step',[]):logits.append(dict(meta,**x))
 csvsave('numerical_summary.csv',numerical);csvsave('numerical_per_layer.csv',layerwise);csvsave('numerical_per_step.csv',logits)
 # Explicit limitations are not drift exclusions.
 warnings+= [
  dict(type='FAILED_PADDING_CANDIDATES',scope='Qwen SVH and any failed InternVL D',policy='Failure diagnostics retained; no formal D timing or historical Accuracy binding'),
  dict(type='QWEN_AUDIT_ORDER_DEVIATION',scope='Session1',policy='Full30 exact numerical screen preceded Session1; actual Audit Mode scheduled after Session1 and before Session2. Session1 provisional until that Audit Mode passes. No inference process terminated.'),
  dict(type='PROFILE_NOT_OFFICIAL_TIMING',policy='Profiler samples excluded; one fixed median input per stage, no latency claim from profiled wall time'),
  dict(type='QUALITY_SCOPE',policy='No full quality benchmark. B/C validated full32 effective logits and tokens on original30; fusion historical Accuracy provenance remains as before. D cannot inherit historical Accuracy.')
 ]
 save(A/'performance_warnings.json',dict(warnings=warnings,no_samples_deleted=True,no_fastest_session_selection=True))
 # Protected existing artifacts, original runtime and optimized packages: content check only, never git.
 protected=[]
 for manifest in ['historical_protected.json','runtime_source_guard.json','optimized_source_manifest.json']:
  items=read(A/manifest)
  for x in items:
   actual=sha(x['path']);assert actual==x['sha256'],('SOURCE_OR_HISTORY_CHANGED',manifest,x['path'])
  protected.append(dict(manifest=manifest,count=len(items),status='PASS'))
 save(A/'validation_incremental.json',dict(status=('PASS_DERIVED_DATA_CHECKS' if read(A/'coordinator_status.json')['status']=='COMPLETE_ALL_PHASES' and read(A/'refinement_coordinator_status.json')['status']=='COMPLETE' else 'IN_PROGRESS'),formal_requests=len(formal),all_requests=len(records),phases=phases,checks=validation,protected=protected,numerical_records=len(numerical),GPU_UUIDs=sorted(set(r['GPU_UUID'] for r in records))))
 print(json.dumps(dict(status='INCREMENTAL_ANALYSIS_COMPLETE',formal=len(formal),all_requests=len(records),phases=[(p['kind'],p['phase'],p['status']) for p in phases],warnings=sum(w.get('type')=='NATIVE_DRIFT_GT5PCT' for w in warnings)),ensure_ascii=False))
if __name__=='__main__':collect()
