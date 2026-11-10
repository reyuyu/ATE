set -eu
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python3 - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,os,signal,subprocess,time,shutil
W=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval');O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval';C=Path('/home/featurize/work/ate-qwen3090-control-29535')
L=Path('/home/featurize/ate-local-runs/qwen3090-29535-20261110')
assert not L.exists(),'Migration root already exists; inspect and resume explicitly'
job=json.loads((C/'gpu_job.json').read_text());pid=int(job['pid']);cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
assert b'qwen3vl_c3_3090_official_eval/scripts/runner.py' in cmd,('Unexpected worker command',pid)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
assert [int(x.strip()) for x in apps if x.strip()]==[pid],('Unexpected GPU processes',apps)
L.mkdir(parents=True)
snap=L/'migration_checkpoint';snap.mkdir()
for base,names in [(O,['progress.json','seven_progress.json','historical_completion.json','validation.json','protocol.json','source_manifest.json','model_manifest.json']),(C,['gpu_job.json'])]:
 for name in names:
  src=base/name
  if src.exists():shutil.copy2(src,snap/name)
record=dict(status='USER_AUTHORIZED_PAUSE_FOR_LOCAL_DISK_MIGRATION',at_UTC=datetime.now(timezone.utc).isoformat(),PID=pid,signal='SIGTERM',local_root=str(L),persistent_checkpoint_root=str(O),reason='Avoid user concurrent persistent NFS I/O; preserve all completed predictions and frozen protocol',inflight_unsaved_generation_may_need_recompute=True,checkpoint_files_atomic=True)
(snap/'pause.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+25
while time.monotonic()<deadline:
 try:alive=b'qwen3vl_c3_3090_official_eval/scripts/runner.py' in (Path('/proc')/str(pid)/'cmdline').read_bytes()
 except OSError:alive=False
 if not alive:break
 time.sleep(.25)
assert not alive,('Worker did not exit; do not start migration/replacement',pid)
gpu_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
assert not gpu_apps,('GPU process remains; do not load second model',gpu_apps)
record.update(status='PAUSED_GPU_RELEASED_FOR_MIGRATION',paused_at_UTC=datetime.now(timezone.utc).isoformat(),GPU_compute_apps=gpu_apps)
(snap/'pause.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
for args in [['id'],['sudo','-n','true'],['unshare','-Ur','-m','true'],['which','rsync']]:
 r=subprocess.run(args,capture_output=True,text=True,check=False)
 print(json.dumps(dict(capability_probe=args,exit_code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip()),ensure_ascii=False))
PY
