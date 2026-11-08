"""Explicit task-owned archive; no model, data images or unrelated results."""
from pathlib import Path
import json,hashlib
O=Path('/home/featurize/work/ATE/vlm_research/results/v2drop_quick_reproduction')
meta=json.loads((O/'run_metadata.json').read_text());assert meta['REPRODUCTION_STATUS'] in ('PASS','PARTIAL','BLOCKED')
names=['report_zh.md','run_metadata.json','protocol.json','execution_config.json','progress.json','selector_cpu_checks.json','smoke_summary.json','smoke_full.json',
 'input_manifest.jsonl.gz','inputs_light.jsonl','predictions_light.jsonl','predictions_full.jsonl.gz','accuracy_summary.csv','efficiency_summary.csv','efficiency_per_input.csv',
 'raw_timings.jsonl.gz','cache_layouts_and_shapes.jsonl.gz','operator_counts_full.jsonl.gz','per_layer_summary.csv','independent_audit.json','independent_FLOPs_audit.csv','scorer_provenance.json',
 'run.py','runtime.py','hf_adapter.py','prepare.py','finalize.py','package.py','runner.log','launch_receipt.json']
names+=[str(p.relative_to(O)) for p in sorted((O/'official_sources').glob('*')) if p.is_file()]
names+=[str(p.relative_to(O)) for p in sorted((O/'frozen_scoring_sources').glob('*')) if p.is_file()]
files={n:{'bytes':(O/n).stat().st_size,'sha256':hashlib.sha256((O/n).read_bytes()).hexdigest()} for n in names if (O/n).is_file()}
manifest={'status':meta['REPRODUCTION_STATUS'],'files':files,'questions':200,'normal_generations':meta['execution']['accuracy_completed'],
 'database_retained_server_only':str(O/'results.sqlite'),'GitHub_exclude':['input_manifest.jsonl.gz','predictions_full.jsonl.gz','cache_layouts_and_shapes.jsonl.gz','operator_counts_full.jsonl.gz','smoke_full.json'],
 'no_model_dataset_images_or_unrelated_results':True}
(O/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PACKAGED',len(files),sum(r['bytes'] for r in files.values()),flush=True)
