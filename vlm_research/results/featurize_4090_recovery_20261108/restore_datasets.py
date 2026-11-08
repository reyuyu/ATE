"""Restore portable Windows bundles; preserve every input field except paths."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil
D=Path('/home/featurize/data/ate-recovery');O=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
(D/'images').mkdir(exist_ok=True);(D/'normalized').mkdir(exist_ok=True)
plan=json.loads((O/'archive_transfer_manifest.json').read_text());details={}
accept=json.loads((O/'dataset_acceptance.json').read_text())
import time
for entry in sorted(plan['archives'],key=lambda e:('gqa' not in e['name'],e['bytes'])):
 p=D/'backup_packages'/entry['name']
 while not p.exists() or p.stat().st_size<entry['bytes']:time.sleep(10)
 assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
 with zipfile.ZipFile(p) as z:
  names=z.namelist()
  assert all((D/'restored_bundles'/n).resolve().is_relative_to((D/'restored_bundles').resolve()) for n in names)
  if entry['name']=='ocrbench_windows_restore.zip':
   dst=D/'normalized';z.extractall(dst)
   raw=dst/'OCRBench-v1';m=json.loads((raw/'file_manifest.json').read_text())
   assert sha(raw/'OCRBench.json')=='df75957e75ea052e9f38f2c67a4f6676103998517345cf49ef7d03b62c835864'
   assert len([x for x in (raw/'OCRBench_Images').rglob('*') if x.is_file()])==930
   assert len(json.loads((raw/'OCRBench.json').read_text()))==1000
   # Paths in the accepted OCR inputs already target this canonical directory.
   rows=[json.loads(l) for l in (raw/'inputs.jsonl').read_text().splitlines()]
   assert len(rows)==1000 and len({r['question_id'] for r in rows})==1000
   assert all(Path(r['image_path']).is_file() for r in rows)
   details['OCRBench-v1']={'questions':1000,'images':930,'files':len(names),'official_manifest_SHA256':sha(raw/'file_manifest.json'),'all_images_present':True,'archive_SHA256_verified':True}
   print('DATASET_RESTORED','OCRBench-v1',1000,flush=True);continue
  z.extractall(D/'restored_bundles')
  bundle=D/'restored_bundles'/names[0].split('/')[0]
 for image_dir in (bundle/'images').iterdir():
  dest=D/'images'/image_dir.name
  if dest.is_symlink():assert dest.resolve()==image_dir.resolve()
  elif dest.exists():raise RuntimeError('Existing image target requires review: '+str(dest))
  else:dest.symlink_to(image_dir,target_is_directory=True)
 for view in (bundle/'views').iterdir():
  dest=D/'normalized'/view.name;dest.mkdir(exist_ok=True)
  original=[json.loads(l) for l in (view/'inputs.jsonl').read_text().splitlines()]
  gold=json.loads((view/'answers.json').read_text());rows=[]
  for old in original:
   row=dict(old);row['image_path']=str(D/old['image_path']);assert Path(row['image_path']).is_file()
   assert {k:v for k,v in row.items() if k!='image_path'}=={k:v for k,v in old.items() if k!='image_path'}
   rows.append(row)
  assert len(rows)==len({r['question_id'] for r in rows}) and {r['question_id'] for r in rows}==set(gold)
  expected=accept[view.name]['inputs_sha256'];matching=None
  for ascii_flag,sort,separators in [(False,False,None),(False,True,None),(False,False,(',',':')),(False,True,(',',':')),(True,False,None)]:
   text=''.join(json.dumps(r,ensure_ascii=ascii_flag,sort_keys=sort,separators=separators)+'\n' for r in rows)
   if hashlib.sha256(text.encode()).hexdigest()==expected:matching=text;break
  if matching is None:raise RuntimeError('Restored input bytes do not match frozen acceptance: '+view.name)
  (dest/'inputs.jsonl').write_text(matching)
  for f in view.iterdir():
   if f.name!='inputs.jsonl' and f.is_file():shutil.copy2(f,dest/f.name)
  assert sha(dest/'answers.json')==accept[view.name]['answers_sha256']
  details[view.name]={'questions':len(rows),'inputs_sha256':sha(dest/'inputs.jsonl'),'answers_sha256':sha(dest/'answers.json'),
   'exact_3090_input_and_gold_files':True,'all_images_present':True,'portable_other_fields_unchanged':True}
  if view.name=='POPE-paper':assert Counter(r['strategy'] for r in rows)=={'random':2910,'popular':3000,'adversarial':3000}
  (O/'dataset_restore.partial.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
  print('DATASET_RESTORED',view.name,len(rows),'EXACT_FROZEN_SHA',flush=True)
r={'status':'PASS','at':datetime.now(timezone.utc).isoformat(),'local_data_root':str(D.resolve()),'datasets':details,
 'seven_benchmarks':7,'all_source_archives_SHA256_verified':True,'source_archives':plan['archives'],'image_checks':'Existence for all mappings; decode/pixel/hash tests on selected GPU cases. Prior complete image audit retained in Windows backup.'}
(O/'dataset_restore.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print('SEVEN_DATASETS_READY',len(details),flush=True)
