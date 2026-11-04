"""Independent CPU rescoring, resource release and final delivery verification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,pathlib,json,csv,hashlib,subprocess,datetime,collections,math
root=pathlib.Path('/data/lk/ATE-worktrees/textvqa-partial-visual-layer-recovery/vlm_research');sys.path.insert(0,str(root))
from textvqa_layer_recovery.common import *
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
import numpy as np
def main():
 assert not (OUT/'STOP.json').exists() and read(OUT/'generation_validation.json')['passed']
 receipt=read(OUT/'launch_receipt.json');pid=receipt['pid'];assert not pathlib.Path('/proc/'+str(pid)).exists(),'Worker has not exited'
 state=hardware();assert all(line.split(',')[0].strip()!=str(pid) for line in state['processes'].splitlines());write(OUT/'gpu_release.json',dict(passed=True,UTC=utc(),worker_pid=pid,worker_exited=True,own_GPU_process_absent=True,chosen_GPU_currently_free=UUID not in state['processes'],resources=state,scope='Immediate post-run observation; not a reservation of future GPU availability'))
 rr=list(csv.DictReader((OUT/'per_question.csv').open()));assert len(rr)==1500 and len({(r['question_id'],r['condition']) for r in rr})==1500
 mmf=TextVQAAccuracyEvaluator();by=collections.defaultdict(dict);discrepancy=[];score_differences=[]
 for r in rr:
  # Independent entry point: official evaluator directly, not the collection scorer.
  value=mmf.eval_pred_list([dict(pred_answer=r['answer'],gt_answers=json.loads(r['answers']))])
  score_differences.append(abs(value-float(r['score'])))
  if abs(value-float(r['score']))>1e-12:discrepancy.append(dict(question_id=r['question_id'],condition=r['condition'],stored=r['score'],official=value))
  cp=root/r['checkpoint'];assert sha(cp)==r['checkpoint_sha256'];g=read(cp)['generation'];assert g['token_ids']==json.loads(r['token_ids']) and g['text']==r['answer']
  assert str(g['eos'])==r['eos'] and str(g['truncated'])==r['truncated'] and str(g['repeated_trigram'])==r['repeated_trigram']
  assert len(g['token_ids'])==int(r['length']) and g['length']<=32 and g['eos']==(g['token_ids'][-1]==2)
  assert not g['truncated'] or g['length']==32
  by[r['question_id']][r['condition']]=r
 assert not discrepancy and len(by)==300 and all(set(v)==set(CONDITIONS) for v in by.values())
 for q,v in by.items():
  cps={c:read(root/r['checkpoint']) for c,r in v.items()}
  assert all(cps[c]['image_features']==cps['B']['image_features'] for c in CONDITIONS),'Projected image features changed'
  assert cps['B']['first_logits_sha256']==cps['D']['first_logits_sha256'],'Decode-only must leave first-token logits unchanged'
  assert cps['C']['first_logits_sha256']==cps['E']['first_logits_sha256'],'Same prefill masks must give equal first-token logits'
 summary=list(csv.DictReader((OUT/'accuracy_summary.csv').open()))
 for r in summary:assert abs(float(r['accuracy_pct'])-100*np.mean([float(v[r['condition']]['score']) for v in by.values()]))<1e-10
 cpidx=list(csv.DictReader((OUT/'checkpoint_index.csv').open()));assert len(cpidx)==1200
 assert all(sha(OUT/r['path'])==r['sha256'] for r in cpidx)
 for r in csv.DictReader((OUT/'baseline_reuse_index.csv').open()):
  p=root/r['path'];assert sha(p)==r['sha256'];data=subprocess.check_output(['git','-C',str(root.parent),'show',BASE+':'+str(p.relative_to(root.parent))]);assert data==p.read_bytes()
 src=read(OUT/'source_manifest.json');assert sources()==src['files'] and sha(OUT/'protocol.json')==src['protocol_sha256']
 manifest=read(OUT/'sample_manifest.json');assert len(manifest['rows'])==300
 for r in manifest['rows']:
  assert sha(r['image_path'])==r['image_sha256'] and sha(OUT/'input_specs'/(r['question_id']+'.json'))==r['input_spec_sha256']
 # Verify Holm independently, using the prespecified 8-member family.
 cmp=list(csv.DictReader((OUT/'paired_comparisons.csv').open()));inter=list(csv.DictReader((OUT/'interaction.csv').open()));family=cmp+inter;assert len(cmp)==7 and len(family)==8
 previous=0
 for j,r in enumerate(sorted(family,key=lambda x:float(x['cluster_signflip_p']))):
  previous=max(previous,min(1.,(8-j)*float(r['cluster_signflip_p'])));assert abs(previous-float(r['Holm_p_8']))<1e-12
 # Independent cluster bootstrap (separate RNG stream) checks numerical stability;
 # it is an audit, and never replaces the preregistered estimates or seed.
 images=sorted({v['B']['image_sha256'] for v in by.values()});clusters={s:[] for s in images}
 for v in by.values():clusters[v['B']['image_sha256']].append(v)
 rng=np.random.default_rng(991104);bootstrap=[]
 for c in cmp:
  a,b=c['left'],c['right'];total=np.array([sum(float(v[a]['score'])-float(v[b]['score']) for v in clusters[s]) for s in images]);counts=np.array([len(clusters[s]) for s in images]);samples=rng.integers(0,len(images),size=(20000,len(images)));bs=100*total[samples].sum(1)/counts[samples].sum(1);lo,hi=np.percentile(bs,[2.5,97.5]);point=100*sum(total)/sum(counts)
  assert abs(point-float(c['delta_pp']))<1e-10
  assert abs(lo-float(c['CI95_low_pp']))<.8 and abs(hi-float(c['CI95_high_pp']))<.8
  bootstrap.append(dict(left=a,right=b,audit_seed=991104,independent_CI_low=float(lo),independent_CI_high=float(hi),point_exact=True,CI_within_0_8pp=True))
 write(OUT/'independent_statistics_audit.json',dict(passed=True,official_MMf_entry_point_records=1500,score_max_difference=max(score_differences),Holm_eight_exact=True,first_token_phase_causal_invariances_exact_300=True,independent_bootstrap_checks=bootstrap))
 main_ref=subprocess.check_output(['git','-C',str(root.parent),'rev-parse','refs/heads/main'],text=True).strip();assert main_ref=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
 branch=subprocess.check_output(['git','-C',str(root.parent),'branch','--show-current'],text=True).strip();assert branch=='codex/textvqa-partial-visual-layer-recovery'
 for name in ['figure_accuracy','figure_paired_comparisons','figure_actual_image_mass','figure_functional_budget']:
  assert (OUT/(name+'.png')).stat().st_size>1000 and (OUT/(name+'.pdf')).read_bytes().startswith(b'%PDF')
 write(OUT/'validation.json',dict(passed=True,UTC=utc(),official_independent_scoring_records=1500,score_recomputation_difference=max(score_differences),fresh_checkpoints=1200,historical_reuse=300,all_inputs_images_source_protocol_hashes_verified=True,model_files_rehashed_before_run=True,CPU_tests=True,GPU_smoke=True,masks_and_positions=True,Native_Noop_strict_torch_equal=True,B_historical_token_ids_and_first_logits_sha_exact=True,B_old_engine_all_valid_logits_strict=True,cache_and_hook_cleanup=True,all_anomalies_retained=True,Holm_eight_family_verified=True,GPU_released=True,main_unchanged=main_ref,branch=branch))
 write(OUT/'completion.json',dict(status='COMPLETE',UTC=utc(),n_questions=300,logical_answers=1500,new_unique_answers=1200,reused_historical_B=300,all_statistics_and_files_verified=True,resource_released=True))
 entries=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='artifact_index.csv' and not p.name.endswith('.tmp') and '__pycache__' not in p.parts:entries.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
 csvwrite(OUT/'artifact_index.csv',entries)
 print(json.dumps(dict(status='AUDIT_PASS',artifact_files=len(entries),main_unchanged=main_ref,branch=branch)))
if __name__=='__main__':main()
