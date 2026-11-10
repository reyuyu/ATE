set -eu
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python3 - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
W=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval');O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval';C=Path('/home/featurize/work/ate-qwen3090-control-29535')
assert json.loads((O/'local_disk_migration.json').read_text())['status']=='LOCAL_EXT4_ACTIVE_PERSISTENT_CHECKPOINT_PRESERVED'
for p in [W,C,Path('/home/featurize/work/ate-qwen3090-data')]:assert subprocess.check_output(['findmnt','-T',str(p),'-n','-o','FSTYPE'],text=True).strip()=='ext4'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Do not start a second GPU worker'
script=O/'resume_scripts/runner_local_resume.py';manifest=O/'resume_scripts/local_resume_source_manifest.json'
for n,h in json.loads(manifest.read_text())['files'].items():assert hashlib.sha256((script.parent/n).read_bytes()).hexdigest()==h,n
protocol=json.loads((O/'protocol.json').read_text())
env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',CUDA_VISIBLE_DEVICES=protocol['gpu_UUID'],PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
log=(O/'execution.log').open('ab',buffering=0)
log.write(('\nUSER_AUTHORIZED_LOCAL_DISK_STORAGE_RESTART '+datetime.now(timezone.utc).isoformat()+'\n').encode())
process=subprocess.Popen(['/home/featurize/ate-local-envs/qwen3vl-torch260-tf4576/bin/python','-u',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=W,start_new_session=True)
record=dict(pid=process.pid,script=str(script),GPU_UUID=protocol['gpu_UUID'],scope='Resume only pending authorized Qwen Original evaluation; no completed or blocked view retry',storage='LOCAL_EXT4_BIND_MOUNTS',restart_at_UTC=datetime.now(timezone.utc).isoformat(),restart_orchestration_manifest_SHA256=hashlib.sha256(manifest.read_bytes()).hexdigest())
state=C/'gpu_job.json';tmp=state.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(state)
(O/'local_disk_resume_job.json').write_text(json.dumps(record,indent=2)+'\n')
print('AUTHORIZED_LOCAL_DISK_SINGLE_GPU_JOB_STARTED',json.dumps(record))
PY
