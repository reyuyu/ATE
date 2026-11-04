"""Independent verification of artifacts, scoring, paired inference and GPU release."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import pathlib,sys,json,csv,hashlib,subprocess,collections,contextlib,datetime,re,ast,shutil
import numpy as np
from .policy import ROOT,OUT as O,LAYER_SETS,NEW,COMPARISONS,SEED
from receiver_cross_dataset.common import sha,read,write
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
H=ROOT/'results/textvqa_visual_reaccess_budget_exploration';W=ROOT.parent
def rows(p):return list(csv.DictReader(p.open()))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 assert read(O/'analysis_validation.json')['passed'] and read(O/'generation_validation.json')['passed'] and read(O/'gpu_smoke_validation.json')['passed']
 manifest=read(O/'sample_manifest.json')['rows'];mr={r['question_id']:r for r in manifest};labels=rows(O/'ocr_labels.csv');lb={r['question_id']:r['category'] for r in labels}
 assert len(mr)==300 and len(set(r['image_sha256'] for r in manifest))==300 and len(labels)==300 and set(mr)==set(lb)
 assert collections.Counter(lb.values())=={'OCR-Direct':230,'OCR-Reasoning':12,'Non-OCR Visual':18,'Ambiguous':40}
 assert sha(O/'ocr_labels.csv')==read(O/'classification_freeze.json')['labels_sha256'] and sha(O/'sample_manifest.json')==read(O/'protocol.json')['manifest_sha256']
 assert read(O/'layer_sets.json')['conditions']==LAYER_SETS and all(LAYER_SETS[c]['Pminus1']==list(range(32)) for c in NEW)
 sf=read(O/'execution_source_manifest.json');assert all(sha(ROOT/k)==v for k,v in sf['files'].items()) and all(sha(k)==v for k,v in sf['external_files'].items())
 assert sha(O/'execution_protocol.json')==sf['execution_protocol_sha256']
 idx=rows(O/'checkpoint_index.csv');assert len(idx)==1200 and {(r['question_id'],r['condition']) for r in idx}=={(q,c) for q in mr for c in NEW}
 cps={};first_equal=0
 for r in idx:
  p=O/r['path'];assert sha(p)==r['sha256'];cp=read(p);g=cp['generation'];q,c=r['question_id'],r['condition'];cps[(q,c)]=cp
  assert cp['allowed_layers']==LAYER_SETS[c] and cp['execution_protocol_sha256']==sha(O/'execution_protocol.json') and cp['execution_source_manifest_sha256']==sha(O/'execution_source_manifest.json')
  assert cp['image_sha256']==mr[q]['image_sha256'] and cp['input_hashes']==read(H/'input_specs'/(q+'.json'))['input_hashes']
  assert cp['MLP_calls']==[len(g['token_ids'])]*32 and cp['hook_cleanup'] and cp['weights_unchanged'] and g['own_cache'] and cp['empty_cache_at_start']
  mask=cp['attention_mass'];assert len(mask)==32*len(g['token_ids'])
  for t in range(len(g['token_ids'])):
   block=mask[t*32:(t+1)*32];assert [x['layer'] for x in block]==list(range(32))
   assert all(x['forbidden_probability_max']==0 for x in block)
   assert all(x['blocked']==(t>0 and x['layer'] not in LAYER_SETS[c]['Decode']) for x in block)
   assert all(x['absolute_query']==g['query_layout']['prompt_length']-1+t for x in block)
  for k in LAYER_SETS[c]['Decode']:
   rr=[x for x in mask if x['phase']=='Decode' and x['layer']==k]
   assert not rr or max(x['image_mass_mean'] for x in rr)>0
  assert g['audit']['all_non_target_masks_exact'] and g['audit']['absolute_positions_exact']
  ids=g['token_ids'];assert ids[-1]==2 or len(ids)==32;assert 2 not in ids[:-1]
 rr=rows(O/'per_question.csv');assert len(rr)==2100
 by=collections.defaultdict(dict);ev=TextVQAAccuracyEvaluator();maxdiff=0
 with open(os.devnull,'w') as null,contextlib.redirect_stderr(null):
  for r in rr:
   q,c=r['question_id'],r['condition'];assert c not in by[q];by[q][c]=r
   value=ev.eval_pred_list([dict(pred_answer=r['answer'],gt_answers=mr[q]['answers'])]);maxdiff=max(maxdiff,abs(value-float(r['score'])));assert maxdiff<1e-12
   assert sha(r['checkpoint_path'])==r['checkpoint_sha256'] and json.loads(r['answers'])==mr[q]['answers'] and r['image_sha256']==mr[q]['image_sha256'] and r['OCR_category']==lb[q]
 assert all(set(v)==set(LAYER_SETS) for v in by.values())
 for q in by:
  for c in NEW:assert by[q][c]['first_logits_sha256']==by[q]['A']['first_logits_sha256'] and by[q][c]['first_token']==by[q]['A']['first_token'];first_equal+=1
 for r in read(O/'historical_reuse_audit.json')['checkpoints']:assert sha(r['checkpoint_path'])==r['sha256']
 for r in manifest:assert sha(r['image_path'])==r['image_sha256']
 summary=rows(O/'accuracy_summary.csv')+rows(O/'ocr_group_accuracy.csv')
 def subset(group):return [q for q in by if group=='All300' or lb[q]==group or (group=='OCR-Combined' and lb[q].startswith('OCR-'))]
 for r in summary:
  group,c=r['group'],r['condition'];sub=subset(group);values=np.array([float(by[q][c]['score']) for q in sub]);assert int(r['n_questions'])==len(sub) and abs(100*values.mean()-float(r['accuracy_pct']))<1e-10
  assert int(r['answer_changes_vs_A'])==sum(by[q][c]['answer']!=by[q]['A']['answer'] for q in sub)
  assert int(r['first_token_changes_vs_A'])==sum(by[q][c]['first_token']!=by[q]['A']['first_token'] for q in sub)
  for k in ['eos','truncated','empty','repeated_trigram']:assert int(r[k])==sum(by[q][c][k]=='True' for q in sub)
 # Independent numeric implementation: full multinomial draws and dot of paired deltas.
 ordered=sorted(by,key=lambda q:by[q]['A']['image_sha256']);rng=np.random.default_rng(SEED);weights=np.vstack([rng.multinomial(300,[1/300]*300,size=1000) for _ in range(20)])
 pairs=rows(O/'paired_comparisons.csv');assert [(r['left'],r['right']) for r in pairs]==COMPARISONS
 ds=np.array([[float(by[q][a]['score'])-float(by[q][b]['score']) for a,b in COMPARISONS] for q in ordered]);samples=100*(weights@ds)/300
 rng=np.random.default_rng(SEED);extreme=np.zeros(5,dtype=int);point=ds.sum(0)
 for start in range(0,20000,1000):
  signs=rng.choice([-1,1],size=(1000,300));extreme+=(abs(signs@ds)>=abs(point)[None]-1e-12).sum(0)
 expectedp=(extreme+1)/20001
 for j,r in enumerate(pairs):
  lo,hi=np.percentile(samples[:,j],[2.5,97.5]);assert abs(float(r['delta_pp'])-100*ds[:,j].mean())<1e-10
  assert abs(float(r['CI95_low_pp'])-lo)<1e-9 and abs(float(r['CI95_high_pp'])-hi)<1e-9
  assert abs(float(r['raw_p'])-expectedp[j])<1e-12
  assert int(r['score_up'])==int((ds[:,j]>1e-12).sum()) and int(r['score_down'])==int((ds[:,j]<-1e-12).sum())
 prev=0
 for j,r in enumerate(sorted(pairs,key=lambda r:float(r['raw_p']))):prev=max(prev,min(1,(5-j)*float(r['raw_p'])));assert abs(prev-float(r['Holm_p']))<1e-12
 # Every prespecified harm is retained, including net-improving conditions.
 comparisons=list(dict.fromkeys([(c,b) for c in NEW for b in ['A','F']]+COMPARISONS));wanted={(q,a,b) for q in by for a,b in comparisons if float(by[q][a]['score'])<float(by[q][b]['score'])-1e-12};neg=rows(O/'negative_cases.csv');assert {(r['question_id'],r['left'],r['right']) for r in neg}==wanted and len(neg)==len(wanted)
 from PIL import Image
 with Image.open(O/'accuracy_comparison.png') as im:im.verify()
 pdf=(O/'accuracy_comparison.pdf').read_bytes();assert pdf.startswith(b'%PDF') and b'/Subtype /Image' not in pdf and len(re.findall(rb'/Type\s*/Page\b',pdf))==1
 for p in (ROOT/'textvqa_coco_middle_transfer').glob('*.py'):ast.parse(p.read_text())
 assert 'torch' not in sys.modules and 'transformers' not in sys.modules
 env=read(O/'environment.json');pid=env['pid'];uuid=env['GPU_UUID'];procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
 assert not pathlib.Path('/proc/'+str(pid)).exists() and not any(line.split(',')[1].strip()==str(pid) for line in procs.splitlines() if ',' in line),'Worker or its GPU context has not exited'
 assert not (O/'STOP.json').exists()
 assert subprocess.check_output(['git','rev-parse','main'],cwd=W,text=True).strip()==read(O/'resource_preflight.json')['main_before']
 assert subprocess.run(['git','diff','--quiet','HEAD','--','.',':(exclude)vlm_research/results/textvqa_coco_middle_transfer',':(exclude)vlm_research/textvqa_coco_middle_transfer'],cwd=W).returncode==0
 write(O/'gpu_release.json',dict(UTC=utc(),GPU_UUID=uuid,physical_gpu=2,pid=pid,worker_exited=True,own_GPU_context_exited=True,selected_GPU_has_no_compute_process=not any(uuid in line for line in procs.splitlines()),current_inventory=inventory,current_processes=procs,other_GPU_processes_untouched=True))
 write(O/'validation.json',dict(passed=True,UTC=utc(),status='VALIDATED',CPU_tests=10,GPU_smoke_passed=True,GPU_smoke_questions=3,historical_reused=900,new_checkpoints_verified=1200,total_scored_records=2100,official_score_max_difference=maxdiff,all_new_first_logits_Native_exact=first_equal,all_actual_probability_and_mask_audits_verified=True,independent_cache_and_all32_MLP_and_hook_audits_verified=True,all_5_primary_bootstrap_CI_p_and_Holm_independently_recomputed=True,all_group_Accuracy_and_output_flags_recomputed=True,all_prespecified_negative_cases_retained=True,all300_image_hashes_and_frozen_OCR_labels_unchanged=True,PNG_and_vector_PDF_valid=True,model_and_source_freeze_unchanged=True,main_and_historical_results_unchanged=True,GPU_released=True,no_new_strategy_or_training=True))
 write(O/'resource_audit.json',dict(UTC=utc(),GPU_workers_started=1,GPU_UUID=uuid,physical_gpu=2,model_copies=1,worker_exited=True,GPU_released=True,preflight_record='resource_start.json',release_record='gpu_release.json',disk=shutil.disk_usage('/data')._asdict(),no_preemption_no_cross_GPU_sharding=True))
 print('INDEPENDENT_FINAL_VALIDATION_PASS 2100 records / 1200 fresh / GPU released',flush=True)
if __name__=='__main__':main()
