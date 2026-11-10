set -eu
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python3 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
L=Path('/home/featurize/ate-local-runs/qwen3090-29535-20261110');B=L/'persistent_work_disk';receipt=json.loads((L/'integrity_receipt.json').read_text())
assert receipt['status']=='LOCAL_COPY_INTEGRITY_PASS_PENDING_BIND_MOUNTS'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU must remain idle before switching storage'
assert shutil.disk_usage(L).free>50*2**30,'Unexpected local disk usage'
roots=[('worktree',Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval')),('data',Path('/home/featurize/work/ate-qwen3090-data')),('model',Path('/home/featurize/work/ate-models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')),('control',Path('/home/featurize/work/ate-qwen3090-control-29535'))]
for label,target in roots:
 assert target.is_dir() and (L/label).is_dir()
 assert subprocess.run(['mountpoint','-q',str(target)]).returncode!=0,('Target unexpectedly already mounted',str(target))
B.mkdir(exist_ok=True)
assert subprocess.run(['mountpoint','-q',str(B)]).returncode!=0,'Backup alias unexpectedly exists; inspect instead of stacking mounts'
subprocess.run(['sudo','-n','mount','--bind','/home/featurize/work',str(B)],check=True)
subprocess.run(['sudo','-n','mount','--make-private',str(B)],check=True)
created=[];mappings=[]
try:
 for label,target in roots:
  local=L/label
  subprocess.run(['sudo','-n','mount','--bind',str(local),str(target)],check=True);created.append(target)
  subprocess.run(['sudo','-n','mount','--make-private',str(target)],check=True)
  assert os.path.samefile(local,target)
  info=subprocess.check_output(['findmnt','-T',str(target),'-n','-o','SOURCE,FSTYPE,TARGET'],text=True).strip()
  fstype=subprocess.check_output(['findmnt','-T',str(target),'-n','-o','FSTYPE'],text=True).strip();assert fstype=='ext4',(str(target),info)
  backup=B/target.relative_to('/home/featurize/work')
  assert subprocess.check_output(['findmnt','-T',str(backup),'-n','-o','FSTYPE'],text=True).strip()=='nfs','Persistent checkpoint must remain on the original NFS mount'
  mappings.append(dict(label=label,original_absolute_path=str(target),active_local_path=str(local),persistent_backup_path=str(backup),active_mount=info))
  print('LOCAL_DISK_BIND_VERIFIED',json.dumps(mappings[-1]),flush=True)
 W=roots[0][1];O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval';R=O/'resume_scripts'
 for n,h in json.loads((O/'source_manifest.json').read_text())['files'].items():assert hashlib.sha256((W/n).read_bytes()).hexdigest()==h,n
 for n,h in json.loads((R/'local_resume_source_manifest.json').read_text())['files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
 receipt.update(status='LOCAL_EXT4_ACTIVE_PERSISTENT_CHECKPOINT_PRESERVED',activated_at_UTC=datetime.now(timezone.utc).isoformat(),storage_mappings=mappings,persistent_backup_alias=str(B),original_absolute_paths_and_manifest_bytes_unchanged=True,periodic_bulk_NFS_copy_not_started=True,persistent_backup_current_through='User-authorized pause; new progress will remain local until an explicit archive/checkpoint sync',restart_orchestration_manifest_SHA256=hashlib.sha256((R/'local_resume_source_manifest.json').read_bytes()).hexdigest())
 text=json.dumps(receipt,ensure_ascii=False,indent=2)+'\n';(O/'local_disk_migration.json').write_text(text,encoding='utf-8');(L/'integrity_receipt.json').write_text(text,encoding='utf-8')
 backupO=B/W.relative_to('/home/featurize/work')/'vlm_research/results/qwen3vl_c3_3090_official_eval'
 assert not (backupO/'local_disk_migration.json').exists(),'Preserve any prior migration receipt'
 (backupO/'local_disk_migration.json').write_text(text,encoding='utf-8')
 backupR=backupO/'resume_scripts';backupR.mkdir(exist_ok=True)
 for p in R.iterdir():
  if p.is_file():assert not (backupR/p.name).exists();shutil.copy2(p,backupR/p.name)
 print('LOCAL_DISK_SWITCH_COMPLETE_PERSISTENT_ORIGINALS_UNCHANGED',flush=True)
except BaseException:
 for target in reversed(created):subprocess.run(['sudo','-n','umount',str(target)],check=False)
 print('BIND_FAILURE_ROLLED_BACK_NO_GPU_STARTED',flush=True)
 raise
PY
