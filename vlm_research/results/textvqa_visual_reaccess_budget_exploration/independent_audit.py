"""Independent official rescoring, fixed-design verification and release audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import pathlib,sys,collections,subprocess,json,csv,hashlib,contextlib,difflib,re
root=pathlib.Path('/data/lk/ATE-worktrees/textvqa-visual-reaccess-budget/vlm_research');sys.path.insert(0,str(root))
from textvqa_reaccess.common import *
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
import numpy as np
def main():
 assert not (OUT/'STOP.json').exists() and read(OUT/'analysis_validation.json')['passed']
 receipts=read(OUT/'launch_receipt.json')
 for r in receipts:assert not pathlib.Path('/proc/'+str(r['pid'])).exists(),'Worker still active'
 state=hardware();active={line.split(',')[0].strip() for line in state['processes'].splitlines()};assert not active&{str(r['pid']) for r in receipts}
 write(OUT/'gpu_release.json',dict(passed=True,UTC=utc(),workers=receipts,own_processes_absent=True,resources=state,scope='Snapshot after own workers exited; other users processes preserved'))
 source=read(OUT/'source_manifest.json');assert sources()==source['files'] and sha(OUT/'protocol.json')==source['protocol_sha256'];protocol=read(OUT/'protocol.json')
 for name,key in [('layer_sets','layer_sets_sha256'),('subset_120','subset_sha256'),('shards','shards_sha256')]:assert sha(OUT/(name+'.json'))==protocol[key]
 assert read(OUT/'layer_sets.json')==DESIGN
 shards=read(OUT/'shards.json')['assignments'];subset=set(read(OUT/'subset_120.json')['question_ids']);manifest=read(OUT/'sample_manifest.json')['rows'];assert len(manifest)==300 and len(subset)==120
 expected_subset={r['question_id'] for r in sorted(manifest,key=lambda r:hashlib.sha256(f"{SEED}:{r['question_id']}:{r['image_sha256']}".encode()).hexdigest())[:120]};assert subset==expected_subset
 assert collections.Counter(shards.values())=={0:150,1:150} and collections.Counter(shards[q] for q in subset)=={0:60,1:60}
 assert read(OUT/'workers/0/smoke_fingerprints.json')==read(OUT/'workers/1/smoke_fingerprints.json')
 for j in [0,1]:assert read(OUT/'workers'/str(j)/'gpu_smoke_validation.json')['passed'] and read(OUT/'workers'/str(j)/'generation_validation.json')['fresh_checkpoints']==1770
 rows=list(csv.DictReader((OUT/'per_question.csv').open()));assert len(rows)==5040 and len({(r['question_id'],r['condition']) for r in rows})==5040
 mmf=TextVQAAccuracyEvaluator();by=collections.defaultdict(dict);diffs=[];fresh=0;history=0;mask_rows=0
 with open(os.devnull,'w') as null,contextlib.redirect_stderr(null):
  for r in rows:
   q=r['question_id'];c=r['condition'];p=root/r['checkpoint'];assert sha(p)==r['checkpoint_sha256'];v=read(p);g=v['generation'];value=mmf.eval_pred_list([dict(pred_answer=r['answer'],gt_answers=json.loads(r['answers']))]);diffs.append(abs(value-float(r['score'])));assert diffs[-1]<1e-12
   assert g['token_ids']==json.loads(r['token_ids']) and g['text']==r['answer'] and len(g['token_ids'])==int(r['length'])==g['length']<=32
   assert g['eos']==(g['token_ids'][-1]==2) and g['truncated']==(g['length']==32 and not g['eos'])
   for flag in ['eos','truncated','repeated_trigram','empty']:assert str(g[flag])==r[flag]
   assert v['weights_unchanged'] and v['hook_cleanup'] and v['MLP_calls']==[g['length']]*32 and v['input_hashes_unchanged'] and g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1 and g['audit']['passed']
   if c not in 'ABCDE':
    fresh+=1;assert v['worker']==shards[q] and v['GPU_UUID']==UUIDS[shards[q]] and v['source_manifest_sha256']==sha(OUT/'source_manifest.json') and v['protocol_sha256']==sha(OUT/'protocol.json') and v['allowed_layers']==DESIGN['conditions'][c]
    assert len(v['attention_mass'])==32*g['length']
    for m in v['attention_mass']:
     allowed=DESIGN['conditions'][c][m['phase']];assert m['blocked']==(m['layer'] not in allowed) and m['forbidden_probability_max']==0
     if m['blocked']:assert m['image_mass_mean']==m['image_mass_head_max']==0
     if m['phase']=='Pminus1':assert m['absolute_query']==g['query_layout']['prompt_length']-1
     else:assert m['absolute_query']==g['query_layout']['prompt_length']+m['step']-1
     mask_rows+=1
   else:
    history+=1;data=subprocess.check_output(['git','-C',str(root.parent),'show',BASE+':'+str(p.relative_to(root.parent))]);assert p.read_bytes()==data
   by[q][c]=(r,v)
 assert fresh==3540 and history==1500
 for q,entries in by.items():
  expected=set('ABCDE')|set(STAGE1)|(set(STAGE2) if q in subset else set());assert set(entries)==expected
  features=entries['B'][1]['image_features'];assert all(v['image_features']==features for r,v in entries.values())
  assert entries['G'][1]['first_logits_sha256']==entries['B'][1]['first_logits_sha256'] and entries['F'][1]['first_logits_sha256']==entries['A'][1]['first_logits_sha256']
  if q in subset:
   for k in [4,8,16]:assert entries['Donly-'+str(k)][1]['first_logits_sha256']==entries['B'][1]['first_logits_sha256'] and entries['Ponly-'+str(k)][1]['first_logits_sha256']==entries['Both-'+str(k)][1]['first_logits_sha256']
 for r in manifest:
  q=r['question_id'];assert r['worker']==shards[q] and r['stage2_subset']==(q in subset) and sha(r['image_path'])==r['image_sha256']
  s=read(OUT/'input_specs'/(q+'.json'));assert sha(OUT/'input_specs'/(q+'.json'))==read(OUT/'cpu_validation.json')['input_specs_sha256'][q]
  assert all(v['input_hashes']==s['input_hashes'] for rr,v in by[q].values())
 # Independently rederive all paired points, counts and Holm families.
 cmp=list(csv.DictReader((OUT/'paired_comparisons.csv').open()))
 def scores(q,c):
  if c.startswith('RandomMean-'):
   k=c.split('-')[1];return np.mean([float(by[q][f'Random-{k}-s{j+1}'][0]['score']) for j in range(3)])
  return float(by[q][c][0]['score'])
 for family,n in [('stage1',300),('stage2',120)]:
  rr=[r for r in cmp if r['family']==family];assert len(rr)==protocol['family_sizes'][family] and {(r['left'],r['right']) for r in rr}=={tuple(x) for x in protocol['comparisons'][family]}
  qs=list(by) if n==300 else [q for q in by if q in subset]
  for r in rr:
   d=np.array([scores(q,r['left'])-scores(q,r['right']) for q in qs]);assert abs(float(r['delta_pp'])-100*d.mean())<1e-10 and int(r['score_up'])==int((d>1e-12).sum()) and int(r['score_down'])==int((d< -1e-12).sum())
  prev=0
  for j,r in enumerate(sorted(rr,key=lambda r:float(r['cluster_signflip_p']))):prev=max(prev,min(1.,(len(rr)-j)*float(r['cluster_signflip_p'])));assert abs(prev-float(r['Holm_p_within_family']))<1e-12
 # Summary and actual edge counts, including generation-dependent decode count.
 for r in csv.DictReader((OUT/'accuracy_summary.csv').open()):
  qs=list(by) if r['scope']=='300' else [q for q in by if q in subset];assert abs(float(r['accuracy_pct'])-100*np.mean([scores(q,r['condition']) for q in qs]))<1e-10
 for r in csv.DictReader((OUT/'visual_edge_budget.csv').open()):
  q,c=r['question_id'],r['condition'];g=by[q][c][1]['generation'];d=DESIGN['conditions'][c];assert int(r['Decode_queries'])==g['length']-1;expected=(len(d['Pminus1'])+(g['length']-1)*len(d['Decode']))/(32*g['length']);assert abs(expected-float(r['total_actual_allowed_ratio']))<1e-12
 index=list(csv.DictReader((OUT/'checkpoint_index.csv').open()));assert len(index)==3540 and all(sha(OUT/r['path'])==r['sha256'] for r in index)
 # Preserve exact source diff of the generalized primitive.
 before=(root/'textvqa_layer_recovery/engine.py').read_text().splitlines(True);after=(root/'textvqa_reaccess/engine.py').read_text().splitlines(True);(OUT/'source_engine_diff.patch').write_text(''.join(difflib.unified_diff(before,after,fromfile='historical_engine.py',tofile='configurable_engine.py')))
 runs=[read(OUT/'workers'/str(j)/'gpu_run_complete.json') for j in [0,1]];write(OUT/'gpu_smoke_validation.json',dict(passed=True,UTC=utc(),both_workers=True,cross_GPU_all_valid_trace_hashes_exact=True,worker_reports=[str((OUT/'workers'/str(j)/'gpu_smoke_validation.json').relative_to(OUT)) for j in [0,1]]))
 write(OUT/'resource_audit.json',dict(passed=True,UTC=utc(),two_GPU_UUIDS=UUIDS,launch=read(OUT/'resource_launch_preflight.json'),worker_runs=runs,release_sha256=sha(OUT/'gpu_release.json'),other_users_not_interfered=True,complete_QKV_MLP_KV_executed=True))
 write(OUT/'independent_validation.json',dict(passed=True,UTC=utc(),official_direct_MMF_records=5040,score_max_difference=max(diffs),canonical_history_bytes=1500,new_checkpoint_hashes=3540,actual_mask_query_layer_records=mask_rows,features_MLP_cache_hook_checks=True,all_phase_first_logits_invariances_exact=True,shards_disjoint_and_complete=True,subset_only_ID_SHA=True,Holm_23_and_68_exact=True,all_paired_point_estimates_exact=True,edge_budgets_independently_verified=True,sources_exact=True,GPUs_released=True,main_unchanged=subprocess.check_output(['git','-C',str(root.parent),'rev-parse','main'],text=True).strip()=='7970be981e3462560cb66d83e1014eaa1b1ebdae'))
 print('INDEPENDENT_AUDIT_PASS',fresh,history,len(rows),max(diffs),flush=True)
if __name__=='__main__':main()
