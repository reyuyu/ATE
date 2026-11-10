set -eu
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python3 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,time,shutil
L=Path('/home/featurize/ate-local-runs/qwen3090-29535-20261110')
assert json.loads((L/'migration_checkpoint/pause.json').read_text())['status']=='PAUSED_GPU_RELEASED_FOR_MIGRATION'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU must remain idle during migration'
roots=[('worktree',Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval')),('data',Path('/home/featurize/work/ate-qwen3090-data')),('model',Path('/home/featurize/work/ate-models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')),('control',Path('/home/featurize/work/ate-qwen3090-control-29535'))]
records=[];state=L/'copy_progress.json'
for label,source in roots:
 assert source.is_dir() and not source.is_symlink(),str(source)
 target=L/label;target.mkdir(exist_ok=True)
 record=dict(label=label,source=str(source),target=str(target),status='COPYING',started_at_UTC=datetime.now(timezone.utc).isoformat())
 records.append(record);state.write_text(json.dumps(dict(status='COPYING',roots=records),indent=2)+'\n')
 print('LOCAL_MIGRATION_START',json.dumps(record),flush=True)
 args=['rsync','-a','--no-owner','--no-group','--partial','--partial-dir=.rsync-partial','--info=progress2,stats2','--outbuf=L',str(source)+'/',str(target)+'/']
 p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 last=time.monotonic();tail=[]
 for line in p.stdout:
  line=line.strip()
  if not line:continue
  tail.append(line);tail=tail[-18:]
  if time.monotonic()-last>=25:
   print('LOCAL_MIGRATION_COPY',label,line,flush=True);last=time.monotonic()
 code=p.wait();record.update(exit_code=code,rsync_stats=tail,completed_at_UTC=datetime.now(timezone.utc).isoformat(),status='COPIED' if code==0 else 'BLOCKED')
 state.write_text(json.dumps(dict(status='COPYING' if code==0 else 'BLOCKED',roots=records),indent=2)+'\n')
 print('LOCAL_MIGRATION_ROOT_FINISHED',json.dumps(record),flush=True)
 assert code==0,('Copy failed; do not bind or resume',label,code)
state.write_text(json.dumps(dict(status='COPIED_PENDING_INTEGRITY_AND_BIND_MOUNTS',roots=records,free_local_GiB=round(shutil.disk_usage(L).free/2**30,3)),indent=2)+'\n')
print('LOCAL_MIGRATION_ALL_COPIES_READY',flush=True)
PY
