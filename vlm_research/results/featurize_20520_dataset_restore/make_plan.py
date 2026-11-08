"""Freeze the seven dataset packages from the user's Windows backup."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile
H=Path(__file__).resolve().parent;R=Path(r'D:\ATE-dataset-backups-3090')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
index=json.loads((R/'archive_index.json').read_text(encoding='utf-8'));entries=[]
for e in index['datasets']:
 p=R/e['filename'];assert p.stat().st_size==e['archive_bytes']
 entries.append({'name':p.name,'local':str(p),'bytes':p.stat().st_size,'sha256':e['sha256'],'kind':'portable',
  'benchmark':e['benchmark'],'questions':e['questions'],'images':e['images'],'files':e['files'],'bundle_manifest_sha256':e['bundle_manifest_sha256']})
for name,source,kind in [('ocrbench_windows_restore.zip',R/'OCRBench-v1','OCRBench'),('ate-extra-images-and-metadata.zip',R/'extra_images','auxiliary')]:
 target=H/name
 if not target.exists():
  with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as z:
   for p in sorted(source.rglob('*')):
    if p.is_file():z.write(p,p.relative_to(R).as_posix())
   if kind=='auxiliary':
    for n in ['archive_index.json','SHA256SUMS.txt','download_receipt.json','上传说明_zh.md']:z.write(R/n,'backup_metadata/'+n)
 with zipfile.ZipFile(target) as z:
  if kind=='OCRBench':
   for p in source.iterdir():
    if p.is_file():assert z.read(p.relative_to(R).as_posix())==p.read_bytes()
   files=[p for p in source.rglob('*') if p.is_file()]
   assert {p.relative_to(R).as_posix() for p in files}==set(z.namelist())
   assert all(z.getinfo(p.relative_to(R).as_posix()).file_size==p.stat().st_size for p in files)
 entry={'name':name,'local':str(target),'bytes':target.stat().st_size,'sha256':sha(target),'kind':kind}
 if kind=='OCRBench':entry.update(benchmark='OCRBench-v1',questions=1000,images=930,official_manifest_sha256=sha(source/'file_manifest.json'))
 entries.append(entry)
plan={'task':'Windows dataset restore to Featurize port20520','created_at_UTC':datetime.now(timezone.utc).isoformat(),
 'endpoint':'workspace.featurize.cn:20520','source':str(R),'local_data_root':'/home/featurize/ate-local-data-20520/ate-recovery',
 'canonical_root':'/home/featurize/data/ate-recovery','output':'/home/featurize/work/ATE/vlm_research/results/featurize_20520_dataset_restore',
 'packages':entries,'concurrent_transfers':2,'no_public_dataset_downloads':True,'no_model_downloads_or_GPU_experiments':True}
(H/'datasets_restore_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('PLAN_READY',len(entries),'packages',sum(e['bytes'] for e in entries),'bytes',flush=True)
