set -eu
python3 - <<'PY'
from pathlib import Path
import json,subprocess,shutil
L=Path('/home/featurize/ate-local-data-20520/ate-recovery');D=Path('/home/featurize/data/ate-recovery')
O=Path('/home/featurize/work/ATE/vlm_research/results/featurize_20520_dataset_restore')
assert subprocess.check_output(['findmnt','-T','/home/featurize','-n','-o','FSTYPE'],text=True).strip()=='ext4'
assert shutil.disk_usage('/home/featurize').free>8*2**30
L.mkdir(parents=True,exist_ok=True);D.parent.mkdir(parents=True,exist_ok=True)
if D.exists() or D.is_symlink():assert D.is_symlink() and D.resolve()==L.resolve(),'Refuse replacing another dataset root'
else:D.symlink_to(L,target_is_directory=True)
O.mkdir(parents=True,exist_ok=True)
for name in ('backup_packages','restored_bundles','normalized','images','.restore_state'):(L/name).mkdir(exist_ok=True)
owner=L/'.restore_owner.json'
record={'endpoint':'workspace.featurize.cn:20520','output':str(O),'local_root':str(L)}
if owner.exists():assert json.loads(owner.read_text())==record
else:owner.write_text(json.dumps(record,indent=2)+'\n')
print('LOCAL_DATA_ROOT_READY',str(D),'->',str(D.resolve()),flush=True)
print('DATA_MOUNT',subprocess.check_output(['findmnt','-T',str(L),'-n','-o','TARGET,FSTYPE'],text=True).strip(),flush=True)
PY
