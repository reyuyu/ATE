set -eu
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,urllib.request,shutil
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/featurize_29535_environment_restore';P=R/'vlm_research/results/featurize_20520_environment_restore'
O.mkdir(exist_ok=True)
owner={'endpoint':'workspace.featurize.cn:29535','task':'CPU-only environment and priority dataset recovery'}
f=O/'recovery_owner.json'
if f.exists():assert json.loads(f.read_text())==owner
else:f.write_text(json.dumps(owner,indent=2)+'\n')
for name in ('hf_revision_manifest.json','windows_archive_checksums.json','requirements.lock.txt'):
 src=P/name;dst=O/('previous_requirements.lock.txt' if name=='requirements.lock.txt' else name)
 if dst.exists():assert dst.read_bytes()==src.read_bytes()
 else:dst.write_bytes(src.read_bytes())
manifest=json.loads((O/'hf_revision_manifest.json').read_text());assert manifest['revision']=='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
assert subprocess.check_output(['findmnt','-T','/home/featurize','-n','-o','FSTYPE'],text=True).strip()=='ext4'
assert shutil.disk_usage('/home/featurize').free>30*2**30
record={'status':'PREPARING','started_at_UTC':datetime.now(timezone.utc).isoformat(),'endpoint':owner['endpoint'],
 'Git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'tracked_changes_before':subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=R,text=True).strip(),
 'existing_model_and_venv':'NOT_FOUND_IN_USER_ACCESSIBLE_LOCAL_AND_PERSISTENT_CANDIDATE_PATHS','private_root_cache':'NOT_ACCESSIBLE_TO_FEATURIZE; not a usable source',
 'model_manifest_sha256':hashlib.sha256((O/'hf_revision_manifest.json').read_bytes()).hexdigest(),'no_GPU_inference':True,'model_data_storage':'local ext4, not /home/featurize/work NFS'}
(O/'recovery_progress.json').write_text(json.dumps(record,indent=2)+'\n')
for name in ('/home/featurize/ate-local-models/llava-1.5-7b-hf','/home/featurize/ate-local-envs'):
 Path(name).mkdir(parents=True,exist_ok=True)
print('CPU_RECOVERY_DIRECTORY_READY',str(O),flush=True)
PY
