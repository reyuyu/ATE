"""Final CPU/provenance validation and read-only GPU release check."""
from pathlib import Path
import os,json,csv,sqlite3,hashlib,subprocess
from datetime import datetime,timezone
os.environ['CUDA_VISIBLE_DEVICES']=''
D=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_rootcause_diagnostic');O=D.parent/'native_e2e32_3090_recheck';A=D.parent/'native_e2e32_3090_stability_audit_20261110'
def read(n):return json.loads((D/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=read('protected_files_before.json');current={str(p):sha(p) for root in [O,A] for p in sorted(root.rglob('*')) if p.is_file()};assert protected==current,'Historical experiment/audit content changed'
(D/'protected_files_after.json').write_text(json.dumps(current,indent=2)+'\n')
plan=read('protocol.json');completion=read('request_completion.json');rs=[json.loads(l) for l in (D/'requests.jsonl').read_text().splitlines()];inp=read('input_verification.json');old=json.loads((O/'protocol.json').read_text())
assert len(rs)==completion['completed']<=plan['planned_requests']<=plan['hard_request_limit']==100
assert len({r['request'] for r in rs})==len(rs) and [r['request'] for r in rs]==list(range(1,len(rs)+1))
for r in rs:
 g=r['generation'];assert r['input_hashes']==inp['input_hashes'] and g['token_ids']==inp['historical32_token_ids']
 assert r['GPU_UUID']==plan['GPU_UUID'] and g['method']=='Native' and g['output_tokens']==32 and g['prefill_calls']==g['vision_forward_calls']==1 and g['real_decode_calls']==31 and g['all_logits_finite']
 assert g['cache']['live_bytes']==g['cache']['storage_bytes'] and len(g['cache']['layers'])==32
 assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6 and abs(sum(g['step_event_ms'][1:])/31-g['CUDA_Event_Decode_TPOT_ms'])<1e-6
 assert r['runtime_SHA256']==old['sources'][str(D.parent/'unified_e2e32_4090/code/e2e32_runtime.py')]
for path,h in old['sources'].items():assert sha(Path(path))==h,path
for n,h in read('diagnostic_source_SHA256.json').items():assert sha(D/n)==h,n
db=sqlite3.connect('file:'+str(O/'checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True);assert db.execute('select count(*) from formal').fetchone()[0]==15;db.close()
required=['protocol.json','environment.json','request_timeline.csv','gpu_telemetry.csv','cpu_telemetry.csv','continuous_vs_idle.csv','profiler_summary.json','rootcause_evidence.json','recommended_protocol.json','report_zh.md','runner.log','gpu_release.json']
for n in required:assert (D/n).stat().st_size>0,n
assert len(list(csv.DictReader((D/'request_timeline.csv').open())))==len(rs)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip();gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,power.limit','--format=csv,noheader,nounits'],text=True).strip()
release=dict(at=datetime.now(timezone.utc).isoformat(),GPU_query=gpu,compute_processes=apps,no_compute_processes=not bool(apps),server_left_running=True,GPU_settings_modified=False)
(D/'gpu_release_final.json').write_text(json.dumps(release,indent=2)+'\n');assert not apps,'GPU compute resources not released'
v=dict(status='PASS',at=datetime.now(timezone.utc).isoformat(),diagnostic_execution_status=completion['status'],completed_native32=len(rs),attempted_native32=completion['attempted'],planned=84,hard_limit=100,profiled_requests=sum(r['profiled'] for r in rs),formal_requests_added=0,original_formal_count=15,protected_original_files_unchanged=len(current),frozen_source_SHA256_entries_verified=len(old['sources']),input_output_KV_and_timing_validation=True,trace_files=len(list((D/'profiler').glob('*.json'))),GPU_released=True,GPU_final=gpu,other_algorithms_run=False,GitHub_push=False,system_GPU_settings_modified=False,root_cause_uniquely_identified=read('rootcause_evidence.json')['root_cause_uniquely_identified'],stable_environment_fix_validated=read('rootcause_evidence.json')['stable_environment_fix_validated'])
(D/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
inventory={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='artifact_SHA256.json'}
(D/'artifact_SHA256.json').write_text(json.dumps(dict(scope=str(D),created_at=datetime.now(timezone.utc).isoformat(),files=inventory),indent=2)+'\n')
print(json.dumps(v,ensure_ascii=False,indent=2))
