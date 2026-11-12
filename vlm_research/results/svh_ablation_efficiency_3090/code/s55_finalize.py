"""Read-only completion checks and SHA inventory of this new result directory."""
from s55_common import *
import csv,hashlib,zipfile

def streamsha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 v=read(O/'validation.json');length=read(O/'generation_length/validation.json')
 assert v['quality']=='PASS' and v['efficiency_execution']=='PASS'
 assert length['formal_requests']==720 and length['sessions']==2
 assert read(O/'execution_progress.json')['status']=='ALL_MEASUREMENTS_AND_CPU_VALIDATION_COMPLETE'
 verify_sources()
 releases=[]
 for session in (1,2):
  p=O/f'generation_length/session{session}'
  release=read(p/'resource_release.json');pid=read(p/'progress.json')['pid']
  assert release['context_released_on_exit'] and not Path(f'/proc/{pid}').exists()
  releases.append(dict(session=session,pid=pid,pre_exit_allocated_bytes=release['allocated_bytes'],process_exited=True))
  assert not (p/'STOP.flag').exists()
 assert command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])=='', 'GPU still has a compute process'
 plot=read(O/'generation_length/latency_vs_tokens.plot_receipt.json')
 assert plot['csv_sha256']==streamsha(O/'generation_length/latency_two_sessions.csv')
 assert plot['plot_source_sha256']==streamsha(O/'code/s55_plot.py')
 assert plot['formal_requests']==720 and plot['only_measured_data']
 assert (O/'generation_length/latency_vs_tokens.pdf').stat().st_size>1000
 save(O/'final_resource_status.json',dict(at_ns=time.time_ns(),GPU=gpu(),compute_processes=[],session_processes=releases,release_basis='Both worker processes have exited; NVML has no compute process. Pre-exit allocated bytes are retained as historical measurements, not overwritten.',server_left_running=True,tracked_git_clean=True,source_integrity='PASS',model_integrity_basis='Previously verified complete fixed-revision SHA receipt, no redundant weight rehash',all_720_formal_requests_retained=True))
 entries=[]
 for p in sorted(O.rglob('*')):
  if not p.is_file() or p.name in ('artifact_manifest.json','artifact_manifest.json.tmp') or p.suffix=='.tmp':continue
  entries.append(dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=streamsha(p)))
 save(O/'artifact_manifest.json',dict(at_ns=time.time_ns(),root=str(O),files=entries,file_count=len(entries),bytes=sum(x['bytes'] for x in entries),scope='Only this new experiment directory; frozen source hashes in source_manifest.json; no model/data rescan'))
 # Lightweight delivery only; complete raw predictions/SQLite/telemetry stay on persistent work storage.
 archive=Path('/tmp/ate_s55_core_evidence.zip')
 keep=[]
 for p in O.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(O)
  if p.name=='figure_validation.json':keep.append(p);continue
  if 'code' in rel.parts or len(rel.parts)==1 or p.name in ('summary.csv','pope_subsets.csv','validation.json','smoke_validation.json','environment.json','warmup_baseline.json','calibration.json','performance_warnings.json','resource_release.json','latency_summary.csv','latency_two_sessions.csv','paired_statistics.csv','per_request.csv','runtime_length_only.diff','runtime_ast_audit.json','latency_vs_tokens.pdf','latency_vs_tokens.png','latency_vs_tokens.plot_receipt.json','GPU_telemetry_summary.json','input_gold_separation.json','CPU_entry_preflight.json','noop_dependency_receipt.json','input_manifest.jsonl.gz','input_path_mapping.json'):keep.append(p)
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(set(keep)):z.write(p,str(p.relative_to(O)))
 print(json.dumps(dict(status=v['status'],manifest_files=len(entries),manifest_bytes=sum(x['bytes'] for x in entries),archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=streamsha(archive),GPU_released=True),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
