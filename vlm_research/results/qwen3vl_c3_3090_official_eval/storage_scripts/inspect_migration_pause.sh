set -eu
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python3 - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,time
L=Path('/home/featurize/ate-local-runs/qwen3090-29535-20261110');p=L/'migration_checkpoint/pause.json';r=json.loads(p.read_text())
for i in range(30):
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
 if not apps:break
 time.sleep(.5)
assert not apps,('GPU process still clearing; do not load a replacement',apps)
r.update(status='PAUSED_GPU_RELEASED_FOR_MIGRATION',paused_at_UTC=datetime.now(timezone.utc).isoformat(),GPU_compute_apps=apps)
p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(r,ensure_ascii=False))
for args in [['id'],['sudo','-n','true'],['unshare','-Ur','-m','true'],['which','rsync']]:
 x=subprocess.run(args,capture_output=True,text=True,check=False)
 print(json.dumps(dict(capability_probe=args,exit_code=x.returncode,stdout=x.stdout.strip(),stderr=x.stderr.strip()),ensure_ascii=False))
PY
