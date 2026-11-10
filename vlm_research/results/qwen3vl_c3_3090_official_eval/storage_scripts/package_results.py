"""Archive completed/stopped research artifacts only. No model/data/push/shutdown."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,tarfile
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval');W=O.parents[2];C=Path('/home/featurize/work/ate-qwen3090-control-29535')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(8*2**20),b''):h.update(part)
 return h.hexdigest()
def main():
 state=read(C/'gpu_job.json');cmd=Path('/proc')/str(state['pid'])/'cmdline'
 assert not cmd.exists() or state['script'].encode() not in cmd.read_bytes(),'Authorized GPU worker still active; do not package a moving result set'
 status=read(O/'progress.json');assert status['status'] in ['GPU_STAGES_FINISHED_PENDING_CPU_AUDIT','BLOCKED'],status
 for n,h in read(O/'source_manifest.json')['files'].items():assert sha(W/n)==h,('Frozen inference source changed',n)
 for n,h in read(O/'analysis_manifest.json')['files'].items():assert sha(O/n)==h,('Analysis source changed',n)
 if (O/'partial_diagnostics_manifest.json').exists():
  for n,h in read(O/'partial_diagnostics_manifest.json')['files'].items():assert sha(O/n)==h,('Partial diagnostic source changed',n)
 if (O/'resume_scripts/local_resume_source_manifest.json').exists():
  for n,h in read(O/'resume_scripts/local_resume_source_manifest.json')['files'].items():assert sha(O/'resume_scripts'/n)==h,('Restart orchestration changed',n)
 if (O/'storage_report_manifest.json').exists():
  for n,h in read(O/'storage_report_manifest.json')['files'].items():assert sha(O/n)==h,('Storage report source changed',n)
 if (O/'protocol_description_manifest.json').exists():
  for n,h in read(O/'protocol_description_manifest.json')['files'].items():assert sha(O/n)==h,('Protocol description evidence changed',n)
 if (O/'storage_delivery_manifest.json').exists():
  for n,h in read(O/'storage_delivery_manifest.json')['files'].items():assert sha(O/n)==h,('Storage delivery source changed',n)
 folders={'historical_predictions','request_predictions','benchmark_predictions','efficiency_raw','smoke'};archives=O/'archives';archives.mkdir(exist_ok=True)
 raw=[];light=[]
 for p in O.rglob('*'):
  if not p.is_file() or p.name.endswith('.tmp') or p.name=='GPU_run.lock' or 'archives' in p.relative_to(O).parts or '__pycache__' in p.parts:continue
  (raw if p.relative_to(O).parts[0] in folders else light).append(p)
 receipts=[]
 for name,files in [('raw_predictions.tar.gz',raw),('light_evidence.tar.gz',light)]:
  dest=archives/name;tmp=dest.with_suffix('.gz.tmp')
  assert not dest.exists(),'Archive already exists; verify/reuse explicitly instead of overwriting'
  with tarfile.open(tmp,'w:gz',compresslevel=3) as tar:
   for p in sorted(files):tar.add(p,arcname=str(p.relative_to(O)),recursive=False)
  tmp.replace(dest);r=dict(path=str(dest),file=name,bytes=dest.stat().st_size,SHA256=sha(dest),member_files=len(files));receipts.append(r);print('ARCHIVE',json.dumps(r),flush=True)
 record=dict(status='SERVER_ARCHIVES_SHA256_VERIFIED',at_UTC=datetime.now(timezone.utc).isoformat(),GPU_worker_exited=True,progress_status=status['status'],archives=receipts,contains_model_weights_or_dataset_images=False,Windows_transfer_status='NOT_YET_PERFORMED')
 (O/'archive_manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 lines=[r['SHA256']+'  archives/'+r['file'] for r in receipts]
 for p in sorted(O.iterdir()):
  if p.is_file() and p.name not in ['SHA256SUMS','GPU_run.lock','execution.log'] and not p.name.endswith('.tmp'):lines.append(sha(p)+'  '+p.name)
 (O/'SHA256SUMS').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('LOCAL_ONLY_ARCHIVES_READY',flush=True)
if __name__=='__main__':main()
