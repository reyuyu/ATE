"""After GPU/CPU completion, copy verified result archives to preserved NFS.

Never touches models, dataset files, another worktree, Git remotes or shutdown.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess

O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 m=read(O/'local_disk_migration.json');job=read(Path('/home/featurize/work/ate-qwen3090-control-29535/gpu_job.json'))
 cmd=Path('/proc')/str(job['pid'])/'cmdline'
 assert not cmd.exists() or job['script'].encode() not in cmd.read_bytes(),'GPU worker still active'
 assert read(O/'progress.json')['status'] in ['GPU_STAGES_FINISHED_PENDING_CPU_AUDIT','BLOCKED']
 archive=read(O/'archive_manifest.json');assert archive['status']=='SERVER_ARCHIVES_SHA256_VERIFIED'
 mapping=next(x for x in m['storage_mappings'] if x['label']=='worktree')
 parent=Path(mapping['persistent_backup_path'])/'vlm_research/results/qwen3vl_c3_3090_official_eval'
 assert subprocess.check_output(['findmnt','-T',str(parent),'-n','-o','FSTYPE'],text=True).strip()=='nfs','Backup destination is not the original persistent NFS'
 dest=parent/'local_disk_final_archives';dest.mkdir(exist_ok=True)
 needed=sum(x['bytes'] for x in archive['archives'] if not (dest/x['file']).exists())
 assert shutil.disk_usage(dest).free>needed+128*2**20,'Insufficient persistent disk space; keep local results and perform Windows backup'
 out=[]
 for rec in archive['archives']:
  src=O/'archives'/rec['file'];target=dest/rec['file']
  assert src.stat().st_size==rec['bytes'] and sha(src)==rec['SHA256'],('Source archive changed',str(src))
  if target.exists():assert target.stat().st_size==rec['bytes'] and sha(target)==rec['SHA256'],('Preserve an existing different archive',str(target))
  else:
   tmp=target.with_suffix(target.suffix+'.copying');shutil.copy2(src,tmp)
   assert tmp.stat().st_size==rec['bytes'] and sha(tmp)==rec['SHA256'],('Persistent copy failed integrity',str(tmp))
   tmp.replace(target)
  out.append(dict(file=rec['file'],path=str(target),bytes=rec['bytes'],SHA256=rec['SHA256']))
  print('PERSISTENT_RESULT_ARCHIVE_VERIFIED',json.dumps(out[-1]),flush=True)
 # The raw and light archives are immutable. A readable delivery copy is separate.
 delivery=dest/'readable_delivery';delivery.mkdir(exist_ok=True)
 names=['report_zh.md','accuracy_summary.csv','paired_comparisons.csv','efficiency_summary.csv','historical_accuracy_summary.csv','historical_paired_comparisons.csv','partial_view_paired_diagnostics.csv','local_disk_migration.json','archive_manifest.json']
 for name in names:
  src=O/name
  if not src.exists():continue
  target=delivery/name
  if target.exists():assert sha(target)==sha(src),('Existing readable delivery differs; preserve it',name)
  else:shutil.copy2(src,target)
  assert sha(target)==sha(src)
 receipt=dict(status='PERSISTENT_RESULT_ARCHIVES_SHA256_VERIFIED',at_UTC=datetime.now(timezone.utc).isoformat(),destination_fstype='nfs',destination=str(dest),archives=out,model_and_dataset_originals_preserved=True,full_raw_and_light_archives_copied=True,Windows_backup_status='See separate Windows transfer receipt; not implied by this NFS copy')
 text=json.dumps(receipt,ensure_ascii=False,indent=2)+'\n'
 (O/'persistent_final_backup_receipt.json').write_text(text,encoding='utf-8')
 (dest/'persistent_final_backup_receipt.json').write_text(text,encoding='utf-8')
 print('PERSISTENT_FINAL_RESULT_BACKUP_COMPLETE',flush=True)
if __name__=='__main__':main()
