"""Final CPU-only validation. Keep all source experiment files unchanged."""
from pathlib import Path
import json,csv,sqlite3,hashlib,ast,os
from datetime import datetime,timezone
os.environ['CUDA_VISIBLE_DEVICES']=''
A=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_stability_audit_20261110');O=A.parent/'native_e2e32_3090_recheck'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((A/'original_files_before.json').read_text())
after={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()}
assert before==after
db=sqlite3.connect('file:'+str(O/'checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True)
counts={t:db.execute('select count(*) from '+t).fetchone()[0] for t in ['formal','smoke','warmup','probes','failures']};assert counts==dict(formal=15,smoke=8,warmup=44,probes=18,failures=1);db.close()
rows=list(csv.DictReader((A/'request_timeline.csv').open()));assert len(rows)==85
assert len(list(csv.DictReader((A/'per_step_event_timeline.csv').open())))==2720
assert len(list(csv.DictReader((A/'fixed_probe_pair_cycles.csv').open())))==31
tele=list(csv.DictReader((A/'telemetry_normalized.csv').open()));assert len(tele)==133
for row in rows:assert int(row['output_tokens'])==32 and int(row['real_decode_calls'])==31
for file in ['audit_native_stability.py','enrich_audit.py','write_audit_documents.py','validate_audit.py']:
 tree=ast.parse((A/file).read_text())
 imports=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):imports += [a.name for a in n.names]
  if isinstance(n,ast.ImportFrom):imports.append(n.module or '')
 assert not any(n=='torch' or n.startswith('transformers') for n in imports)
assert json.loads((A/'historical_experiment_status.json').read_text())['reference_eligible']==False
assert json.loads((A/'protocol_proposal.json').read_text())['status']=='PROPOSED_NOT_EXECUTED'
for n in ['report_zh.md','protocol_proposal_zh.md','historical_timeline.png','historical_timeline.svg']:assert (A/n).stat().st_size>100
v=dict(status='PASS',checked_at=datetime.now(timezone.utc).isoformat(),original_files_verified_unchanged=len(after),formal_results_preserved=15,checkpoint_counts=counts,request_timeline_rows=85,step_event_rows=2720,fixed_pair_cycle_rows=31,telemetry_rows=133,all85_recorded_output_tokens_match_frozen_reference=True,source_experiment_original_CSV_sha256=sha(O/'native_per_input.csv'),source_experiment_original_Checkpoint_sha256=sha(O/'checkpoint.sqlite'),GPU_generation_started_in_audit=False,frozen_sources_modified=False,GitHub_push_performed=False,proposal_executed=False,root_cause_uniquely_identified=False,notes='No model/runtime imports or GPU commands; CPU analysis and current read-only mount observation only.')
(A/'validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
(A/'original_files_final.json').write_text(json.dumps(after,ensure_ascii=False,indent=2)+'\n')
inventory={str(p.relative_to(A)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(A.rglob('*')) if p.is_file() and p.name!='artifact_SHA256.json'}
(A/'artifact_SHA256.json').write_text(json.dumps(dict(scope=str(A),created_at=datetime.now(timezone.utc).isoformat(),files=inventory),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(v,ensure_ascii=False,indent=2))
