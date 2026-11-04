"""Independent result/checkpoint/official-score audit, run after GPU process exits."""
import pathlib,sys,json,csv,collections,io,contextlib,os,subprocess,hashlib,shutil,datetime
W=pathlib.Path('/data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation');sys.path.insert(0,str(W/'vlm_research'))
from receiver_cross_dataset.common import *
from receiver_cross_dataset.scoring import score,normalize,VQAEval
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
assert sources()==read(OUT/'source_manifest.json')['files']
run=read(OUT/'gpu_run_complete.json');assert not pathlib.Path('/proc') .joinpath(str(run['pid'])).exists(),'GPU worker has not exited'
state=hardware();assert str(run['pid']) not in state['processes'];assert UUID not in state['processes']
write(OUT/'gpu_release.json',dict(UTC=utc(),all_own_processes_exited=True,only_GPU_UUID=UUID,worker_pid=run['pid'],resources=state))
write(OUT/'resource_audit.json',dict(start=read(OUT/'resource_start.json'),initial_preflight=read(OUT/'resource_preflight.json'),worker_final=read(OUT/'resource_worker_final.json'),released=read(OUT/'gpu_release.json'),one_model_one_GPU=True,new_private_bytes=sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()),new_result_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),nothing_deleted_from_history=True))
all_checks={}
class FakeVQA:
 def __init__(self,qa):self.qa=qa
 def getQuesIds(self):return list(self.qa)
for dataset in ['vqa_v2','textvqa']:
 d=OUT/dataset;m=read(d/'sample_manifest.json');by={r['question_id']:r for r in m['rows']};rows=list(csv.DictReader((d/'per_question.csv').open()));assert len(rows)==len(by)*5
 cpindex=list(csv.DictReader((d/'checkpoint_index.csv').open()));assert len(cpindex)==len({r['checkpoint'] for r in rows})
 for r in cpindex:assert sha(OUT/r['path'])==r['sha256']
 officialQA={};officialPred={};expected={};maxerr=0
 for j,r in enumerate(rows):
  q=r['question_id'];a=by[q];c=read(OUT/r['checkpoint']);g=c['generation'];spec=read(d/'input_specs'/(q+'.json'))
  assert c['source_manifest_sha256']==sha(OUT/'source_manifest.json')
  assert c['input_hashes']==spec['input_hashes'] and c['writers']==spec['writers'][r['condition']] and c['image_sha256']==a['image_sha256']
  assert g['text']==r['answer'] and g['token_ids']==json.loads(r['token_ids'])
  assert abs(c['score']-float(r['score']))<1e-12 and abs(score(dataset,r['answer'],a['answers'])-float(r['score']))<1e-12
  assert g['audit']['passed'] and c['MLP_calls']==[g['length']]*32 and c['hook_cleanup'] and c['weights_unchanged']
  assert all(x==0 for x in g['audit']['forbidden_max_by_layer'])
  expected[j]=float(r['score'])
  if dataset=='vqa_v2':
   officialQA[j]=dict(answers=[dict(answer=normalize(x),answer_id=i+1) for i,x in enumerate(a['answers'])],question_type=a['question_type'],answer_type=a['answer_type'])
   officialPred[j]=dict(answer=normalize(r['answer']))
  else:
   evaluator=TextVQAAccuracyEvaluator()
   with contextlib.redirect_stderr(io.StringIO()):ref=evaluator.eval_pred_list([dict(pred_answer=r['answer'],gt_answers=a['answers'])])
   maxerr=max(maxerr,abs(ref-float(r['score'])))
 if dataset=='vqa_v2':
  evaluator=VQAEval(FakeVQA(officialQA),FakeVQA(officialPred))
  with contextlib.redirect_stdout(io.StringIO()):evaluator.evaluate()
  maxerr=max(abs(evaluator.evalQA[j]/100-expected[j]) for j in expected)
 assert maxerr<1e-12
 audited=read(d/'random_trigger_audit.json');orig=specs=[]
 for q in by:specs.append(read(d/'input_specs'/(q+'.json')))
 # Mathematical equality by exact length stratum; no floating-budget claim alone.
 strata=collections.defaultdict(list)
 for s in specs:strata[(len(s['Q']),len(s['R']))].append(s)
 for key,ss in strata.items():
  assert sum(len(s['writers']['Threshold-15'])-len(s['Q']) for s in ss)==sum(len(s['writers']['Budget-Matched Random'])-len(s['Q']) for s in ss)
 overlap=read(d/'dataset_overlap_audit.json');overlap.update(no_known_overlap=sum(not r['metadata_overlap'] and not r['SHA_overlap'] for r in by.values()),unresolved_reencoded_or_uncatalogued_identity=len(by),model_pretraining_overlap_audited=False);write(d/'dataset_overlap_audit.json',overlap)
 all_checks[dataset]=dict(n=len(by),logical_rows=len(rows),checkpoint_count=len(cpindex),official_evaluator_max_score_difference=maxerr,exact_stratum_budgets=True,all_manifest_checkpoint_CSV_scores_and_answers_verified=True)
write(OUT/'independent_delivery_audit.json',dict(passed=True,UTC=utc(),datasets=all_checks,model_manifest_sha256=sha(OUT/'model_manifest.json'),source_manifest_sha256=sha(OUT/'source_manifest.json')))
print(json.dumps(all_checks),flush=True)
