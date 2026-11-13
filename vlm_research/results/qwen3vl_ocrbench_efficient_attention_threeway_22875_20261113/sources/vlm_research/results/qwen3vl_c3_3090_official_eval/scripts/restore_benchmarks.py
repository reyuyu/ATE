"""Restore original packages into this task's isolated persistent data directory."""
from pathlib import Path,PurePosixPath
import hashlib,json,os,zipfile
C=Path('/home/featurize/work/ate-qwen3090-control-29535');D=Path('/home/featurize/work/ate-qwen3090-data/benchmarks')
R=Path('/home/featurize/work/ATE');O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
plan=read(C/'datasets_restore_plan.json');transfer=read(C/'transfer_progress.json')
assert transfer['status']=='ALL_ARCHIVES_SHA256_VERIFIED'
receipt={r['name']:r for r in transfer['completed']}
oldplan=read(R/'vlm_research/results/featurize_29535_environment_restore/datasets_restore_plan.json')
oldreceipt=read(R/'vlm_research/results/featurize_29535_environment_restore/transfer_progress.json')
for e in oldreceipt['completed']:receipt[e['name']]=e
views={};images={};archives=[]
for entry in oldplan['packages']+plan['packages']:
 archive=D/'backup_packages'/entry['name']
 if entry in oldplan['packages']:archive=Path('/home/featurize/data/ate-recovery/backup_packages')/entry['name']
 assert archive.stat().st_size==entry['bytes']
 rec=receipt[entry['name']];assert rec['status']=='PASS' and rec['sha256']==entry['sha256']
 dest=D/'normalized' if entry['kind']=='OCRBench' else D/'restored_bundles'
 marker=D/'.restore_state'/(entry['name']+'.json')
 with zipfile.ZipFile(archive) as z:
  assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in z.namelist())
  assert all((i.external_attr>>16)&0o170000!=0o120000 for i in z.infolist())
  if marker.exists():assert read(marker)['SHA256']==entry['sha256']
  else:z.extractall(dest);write(marker,dict(SHA256=entry['sha256'],ZIP_CRC='PASS',files=len(z.namelist())))
  root=dest/z.namelist()[0].split('/')[0]
 archives.append(dict(name=entry['name'],path=str(archive),SHA256=entry['sha256'],reused_archive=entry in oldplan['packages']))
 if entry['kind']=='OCRBench':
  assert sha(root/'file_manifest.json')==entry['official_manifest_sha256']
  anns=read(root/'OCRBench.json');rawrows=[json.loads(l) for l in (root/'inputs.jsonl').read_text().splitlines()]
  assert len(anns)==len(rawrows)==1000
  files=read(root/'file_manifest.json');print('OCR_MANIFEST_SCHEMA',type(files).__name__,flush=True)
  views['OCRBench-v1']=dict(path=str(root/'inputs.jsonl'),annotations=str(root/'OCRBench.json'),root=str(root),inputs_SHA256=sha(root/'inputs.jsonl'),Gold_SHA256=sha(root/'OCRBench.json'),questions=1000)
  continue
 assert sha(root/'bundle_manifest.json')==entry['bundle_manifest_sha256']
 bundle=read(root/'bundle_manifest.json');hashes=read(root/'image_sha256.json')
 assert isinstance(hashes,dict)
 for path,h in hashes.items():
  p=root/path
  if isinstance(h,dict):h=h['sha256']
  assert p.is_file(),str(p)
  images[str(p)]=h
 for source in (root/'views').iterdir():
  spec=bundle['views'][source.name];assert sha(source/'inputs.jsonl')==spec['inputs_sha256'] and sha(source/'answers.json')==spec['answers_sha256']
  rows=[json.loads(l) for l in (source/'inputs.jsonl').read_text().splitlines()];gold=read(source/'answers.json')
  assert len(rows)==len({r['question_id'] for r in rows})==spec['questions'] and {r['question_id'] for r in rows}==set(gold)
  assert all((root/r['image_path']).is_file() for r in rows)
  views[source.name]=dict(root=str(root),path=str(source/'inputs.jsonl'),answers=str(source/'answers.json'),inputs_SHA256=spec['inputs_sha256'],Gold_SHA256=spec['answers_sha256'],questions=len(rows))
 print('PERSISTENT_BENCHMARK_RESTORE_PASS',entry['benchmark'],entry['questions'],flush=True)
assert len(views)==15
write(O/'data_restore.json',dict(status='PASS',views=views,image_SHA256_manifest=images,archives=archives,no_frozen_input_modified=True,all_images_present=True,MME_known_SHA_difference='Preserve original landmark/03d5e3bfc958be38.jpg',no_global_image_rehash=True))
print('SEVEN_ORIGINAL_DATASETS_READY',len(views),'views',flush=True)
