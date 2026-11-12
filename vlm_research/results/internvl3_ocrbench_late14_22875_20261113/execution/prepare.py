"""Add only Late14 to the previous immutable OCRBench100; reuse every completed output."""
import sys,json,shutil,gzip
from pathlib import Path
OLD=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
W=Path('/home/featurize/ate-experiments/internvl3-ocrbench-wider-reaccess-22875-20261113')
O=Path('/home/featurize/ate-experiments/internvl3-ocrbench-late14-22875-20261113')
REL='vlm_research/results/internvl3_ocrbench_wider_reaccess_22875_20261113'
sys.path.insert(0,str(OLD));from run_common import sha,atomic
assert not (O/'protocol.json').exists(), 'Never overwrite an existing protocol'
prior=W/'publication_final'/REL
integrity=json.loads((prior/'integrity_manifest.json').read_text())
for x in integrity:assert sha(prior/x['path'])==x['sha256'],x['path']
P=json.loads((W/'protocol.json').read_text());rows=json.loads((W/'selected_rows.json').read_text())
assert len(rows)==100 and [x['question_id'] for x in rows]==P['ids']
for f in ['selected_rows.json','input_paths.json','expected_bindings.json','gold.json']:
 shutil.copy2(W/f,O/f)
baseline={};sources={}
for f in sorted((prior/'predictions').glob('*.jsonl.gz')):
 method=f.name.removesuffix('.jsonl.gz');rs=[json.loads(s) for s in gzip.open(f,'rt')]
 assert len(rs)==100 and {x['question_id'] for x in rs}==set(P['ids'])
 baseline[method]=rs;sources[method]=dict(path=str(f),sha256=sha(f),commit='d47fc0d20d26df01d42b89754f50df3d9ea5dd62')
assert set(baseline)=={'Native','Physical_D','Late9','Late16_L12_27','Late20_L8_27','All24_ExceptL2_5'}
atomic(O/'baselines.json',baseline)
P.update(configs={'Late14_L14_27':list(range(14,28))},baseline_commit='d47fc0d20d26df01d42b89754f50df3d9ea5dd62',baselines=sources,
 expected_new_quality_requests=100,baseline_regeneration=0,previous_run_protocol_sha256=sha(W/'protocol.json'),baselines_sha256=sha(O/'baselines.json'),
 purpose='User requested Late14: exactlyL14-L27, one additional window on the unchanged previous100. No new selection, no baseline regeneration, no additional configurations or efficiency.',
 interpretation='14 terminal reaccess layers, inclusive L14 through L27, 0-based',
 adaptivity='Sequential exploratory follow-up after viewing previous100 scores; same sample is reused and is NOT independent validation.')
assert sha(O/'selected_rows.json')==P['selected_rows_sha256']
for p,h in P['source_SHA256'].items():assert sha(OLD/'late9_sources'/p)==h,p
atomic(O/'protocol.json',P)
runner=(W/'run.py').read_text()
assert runner.count(str(W))==1
runner=runner.replace(str(W),str(O)).replace('Three fixed wider windows;','One additional fixed Late14 window;').replace('quality_generations=12','quality_generations=4').replace('ALL300_COMPLETE','LATE14_ALL100_COMPLETE')
(O/'run.py').write_text(runner)
atomic(O/'cpu_audit.json',dict(status='PASS',input_selection='Unchanged previous100 IDs, Gold, images, Prompt, tensors and order',selected_rows_sha256=sha(O/'selected_rows.json'),baselines_reused=600,prior_integrity_files=len(integrity),frozen_source_files_verified=len(P['source_SHA256']),new_condition_only='Late14_L14_27',new_runner_sha256=sha(O/'run.py'),source_core_modified=False))
print('CPU_PREPARE_PASS',json.dumps(dict(selected=100,new_configs=P['configs'],baseline_scores={m:sum(r['official_score'] for r in rs) for m,rs in baseline.items()})))
