"""Package this experiment only; raw predictions stay out of GitHub."""
from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/featurize_4090_gqa_official_aligned_threeway'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert read(O/'run_metadata.json')['status']=='COMPLETE' and read(O/'independent_scoring_audit.json')['status']=='PASS'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sources={}
for name in ['engine.py','common.py','__init__.py']:
 original=R/'vlm_research/gqa_testdev_c3'/name;target=O/'required_frozen_sources/gqa_testdev_c3'/name;target.parent.mkdir(parents=True,exist_ok=True)
 assert sha(original)==read(O/'protocol.json')['frozen_sources'][str(original.relative_to(R))]
 if target.exists():assert sha(target)==sha(original)
 else:shutil.copy2(original,target)
 sources[str(original.relative_to(R))]={'artifact':str(target.relative_to(O)),'sha256':sha(target),'bytes':target.stat().st_size,'reason':'Frozen GQA module recovered locally; preserve unchanged required inference/scorer source when absent from publication base.'}
(O/'source_publish_map.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
names=['report_zh.md','run_metadata.json','protocol.json','execution_config.json','progress.json','scores.csv','scores_by_type.csv','paired_differences.csv',
 'independent_scoring_audit.json','completed_native_early_audit.json','completed_c3_early_audit.json','physical_validation_summary.json','physical_smoke.json','native_smoke.json','formal_release.json','prompt_examples.json',
 'input_manifest.jsonl.gz','question_order.csv','per_question_score_audit.csv.gz','predictions_Native.jsonl.gz','predictions_C3-V2.jsonl.gz','predictions_C3-V2_L11-L24.jsonl.gz',
 'official_score_Native.log','official_score_C3-V2.log','official_score_C3-V2_L11-L24.log','official_score_Native.json','official_score_C3-V2.json','official_score_C3-V2_L11-L24.json','prepare.py','run.py','score_and_report.py','package_evidence.py','runner.log','launch_receipt.json','source_publish_map.json']
names+=['attempt1_entry_lookup_failure/'+n for n in ['run.py','progress.json','runner.log','execution_config.json','launch_receipt.json']]
names+=[x['artifact'] for x in sources.values()]
files={n:{'bytes':(O/n).stat().st_size,'sha256':sha(O/n)} for n in names}
manifest={'status':'COMPLETE','files':files,'full_predictions_compressed':True,'model_and_dataset_images_included':False,'database_retained_server_only':str(O/'results.sqlite'),
 'GitHub_exclude':['input_manifest.jsonl.gz','question_order.csv','per_question_score_audit.csv.gz','predictions_Native.jsonl.gz','predictions_C3-V2.jsonl.gz','predictions_C3-V2_L11-L24.jsonl.gz','physical_smoke.json'],
 'immutable_old_result_paths_preserved':True}
(O/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('EVIDENCE_PACKAGED',len(files),sum(x['bytes'] for x in files.values()),flush=True)
