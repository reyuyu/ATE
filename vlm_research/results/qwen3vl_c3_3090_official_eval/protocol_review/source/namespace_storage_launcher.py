"""Storage-only private mount namespace; frozen absolute paths and bytes remain intact."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,pwd,subprocess,sys

L=Path('/home/featurize/ate-local-runs/qwen3090-29535-20261110')
C=L/'control';W=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval')
O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval'
E='/home/featurize/ate-local-envs/qwen3vl-torch260-tf4576/bin/python'
mode=sys.argv[1];assert mode in ['check','run'] and os.geteuid()==0
config=json.loads((C/'namespace_storage_config.json').read_text())
assert os.readlink('/proc/self/ns/mnt')!=config['host_mount_namespace'],'Must isolate mounts from every other process'
base=L/'private_work_parent'
subprocess.run(['mount','--bind',str(base),'/home/featurize/work'],check=True)
for source,target in config['mounts']:
    subprocess.run(['mount','--bind',source,target],check=True)
    assert os.path.samefile(source,target),(source,target)
for p in ['/home/featurize/work',*[t for _,t in config['mounts']]]:
    assert subprocess.check_output(['findmnt','-T',p,'-n','-o','FSTYPE'],text=True).strip()=='ext4',p
account=pwd.getpwnam('featurize');os.initgroups(account.pw_name,account.pw_gid);os.setgid(account.pw_gid);os.setuid(account.pw_uid)
env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='' if mode=='check' else config['gpu_UUID'])
os.chdir(W)
if mode=='check':
    os.execve(E,[E,'-u',str(C/'namespace_storage_check.py')],env)
assert (L/'namespace_check_receipt.json').exists(),'CPU input/path verification required'
check=json.loads((L/'namespace_check_receipt.json').read_text());assert check['status']=='LOCAL_PARENT_NAMESPACE_CPU_CHECK_PASS'
for name,h in config['controller_files_SHA256'].items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==h,name
script=O/'resume_scripts/runner_local_resume.py'
record=dict(pid=os.getpid(),script=str(script),GPU_UUID=config['gpu_UUID'],scope='Resume only pending authorized evaluation; storage namespace removes residual NFS ancestor traversal',storage='PRIVATE_NAMESPACE_ALL_RUNTIME_PATH_ANCESTORS_LOCAL_EXT4',restart_at_UTC=datetime.now(timezone.utc).isoformat(),parent_mount_namespace=os.readlink('/proc/self/ns/mnt'),host_mount_namespace=config['host_mount_namespace'],frozen_runner_SHA256=hashlib.sha256(script.read_bytes()).hexdigest(),CPU_check_receipt_SHA256=hashlib.sha256((L/'namespace_check_receipt.json').read_bytes()).hexdigest())
state=C/'gpu_job.json';tmp=state.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(state)
(O/'local_parent_namespace_job.json').write_text(json.dumps(record,indent=2)+'\n')
print('STORAGE_ONLY_PRIVATE_NAMESPACE_SINGLE_GPU_RESUME',json.dumps(record),flush=True)
os.execve(E,[E,'-u',str(script)],env)
