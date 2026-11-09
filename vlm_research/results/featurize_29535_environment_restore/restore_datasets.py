"""CPU-only restore of Windows packages on the new instance's ext4 disk."""
from pathlib import Path,PurePosixPath
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil,subprocess,os
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/featurize_29535_environment_restore'
D=Path('/home/featurize/data/ate-recovery');L=Path('/home/featurize/ate-local-data-29535/ate-recovery')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,r):
 p=Path(p);temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def exact_write(p,raw):
 if p.exists():
  if p.read_bytes()==raw:return
  raise RuntimeError('Preserve pre-existing differing restored metadata: '+str(p))
 temp=p.with_name(p.name+'.tmp');temp.write_bytes(raw);temp.replace(p)
plan=read(O/'datasets_restore_plan.json');transfer=read(O/'transfer_progress.json')
assert transfer['status']=='ALL_ARCHIVES_SHA256_VERIFIED' and len(transfer['completed'])==len(plan['packages'])
verified={e['name']:e for e in transfer['completed']}
assert D.is_symlink() and D.resolve()==L.resolve()
assert subprocess.check_output(['findmnt','-T',str(L),'-n','-o','FSTYPE'],text=True).strip()=='ext4'
owner=read(L/'.restore_owner.json');assert owner['endpoint']==plan['endpoint'] and owner['output']==str(O)
accept=read(R/'vlm_research/results/featurize_new3090_recovery/dataset_acceptance.json')
details={};base=[];representatives={};started=datetime.now(timezone.utc).isoformat()
for entry in plan['packages']:
 archive=D/'backup_packages'/entry['name'];assert archive.stat().st_size==entry['bytes']
 assert verified[entry['name']]['sha256']==entry['sha256'] and verified[entry['name']]['status']=='PASS'
 marker=D/'.restore_state'/(entry['name']+'.json')
 destination=D/'restored_bundles' if entry['kind']=='portable' else D/'normalized' if entry['kind']=='OCRBench' else D
 with zipfile.ZipFile(archive) as z:
  names=z.namelist();root=destination.resolve()
  assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and (destination/n).resolve().is_relative_to(root) for n in names)
  assert all((i.external_attr>>16)&0o170000!=0o120000 for i in z.infolist()),'ZIP symlink not allowed'
  if marker.exists():assert read(marker)['sha256']==entry['sha256']
  else:
   z.extractall(destination)
   save(marker,{'sha256':entry['sha256'],'files':len(names),'extraction_CRC_checks':'PASS'})
  if entry['kind']=='auxiliary':
   source=read(D/'extra_images/2346600.source.json');assert sha(D/'extra_images/2346600.jpg')==source['sha256']
   details['extra_images']={'files':3,'image_id':'2346600','image_sha256':source['sha256'],'stored_separately_from_frozen_testdev':True}
   print('AUXILIARY_FILES_RESTORED',flush=True);continue
  if entry['kind']=='OCRBench':
   raw=D/'normalized/OCRBench-v1';assert sha(raw/'file_manifest.json')==entry['official_manifest_sha256']
   annotations=read(raw/'OCRBench.json');rows=[json.loads(l) for l in (raw/'inputs.jsonl').read_text().splitlines()]
   images=[p for p in (raw/'OCRBench_Images').rglob('*') if p.is_file()]
   assert len(images)==930 and len(annotations)==len(rows)==1000 and len({r['question_id'] for r in rows})==1000
   for i,row in enumerate(rows):
    assert row['official_annotation_index']==i and row['official_id']==annotations[i]['id']
    assert Path(row['image_path']).is_file()
   details['OCRBench-v1']={'questions':1000,'images':930,'official_manifest_SHA256':sha(raw/'file_manifest.json'),
      'inputs_sha256':sha(raw/'inputs.jsonl'),'official_annotations_sha256':sha(raw/'OCRBench.json'),'all_images_present':True,'official_order_ID_mapping':'PASS','input_bytes_unchanged':True}
   details['OCRBench-v1']['matches_previous_verified_Windows_ZIP_SHA256']=entry['sha256']=='1280d885b8d544016d770fdea29aa41553e200d60e5284475630c76eb46e8f07'
   base.append({'benchmark':'OCRBench-v1','questions':1000,'images':930});representatives['OCRBench-v1']=rows[0]['image_path']
   print('DATASET_RESTORED','OCRBench-v1',1000,930,flush=True);continue
  assert len(names)==entry['files'];bundle=D/'restored_bundles'/names[0].split('/')[0]
 assert sha(bundle/'bundle_manifest.json')==entry['bundle_manifest_sha256']
 image_files=[p for p in (bundle/'images').rglob('*') if p.is_file()];assert len(image_files)==entry['images']
 base.append({'benchmark':entry['benchmark'],'questions':entry['questions'],'images':len(image_files)})
 for image_dir in (bundle/'images').iterdir():
  dest=D/'images'/image_dir.name
  if dest.is_symlink():assert dest.resolve()==image_dir.resolve()
  elif dest.exists():raise RuntimeError('Preserve another image target: '+str(dest))
  else:dest.symlink_to(image_dir,target_is_directory=True)
 for view in sorted((bundle/'views').iterdir()):
  dest=D/'normalized'/view.name;dest.mkdir(exist_ok=True)
  original=[json.loads(l) for l in (view/'inputs.jsonl').read_text().splitlines()];gold=read(view/'answers.json');rows=[]
  for old in original:
   row=dict(old);row['image_path']=str(D/old['image_path']);assert Path(row['image_path']).is_file()
   assert {k:v for k,v in row.items() if k!='image_path'}=={k:v for k,v in old.items() if k!='image_path'}
   rows.append(row)
  assert len(rows)==len({r['question_id'] for r in rows}) and {r['question_id'] for r in rows}==set(gold)
  expected=accept[view.name]['inputs_sha256'];matching=None
  for ascii_flag,sort,separators in [(False,False,None),(False,True,None),(False,False,(',',':')),(False,True,(',',':')),(True,False,None)]:
   raw=''.join(json.dumps(r,ensure_ascii=ascii_flag,sort_keys=sort,separators=separators)+'\n' for r in rows).encode()
   if hashlib.sha256(raw).hexdigest()==expected:matching=raw;break
  assert matching is not None,'Frozen input bytes differ: '+view.name
  exact_write(dest/'inputs.jsonl',matching)
  for f in view.iterdir():
   if f.name!='inputs.jsonl' and f.is_file():exact_write(dest/f.name,f.read_bytes())
  assert sha(dest/'answers.json')==accept[view.name]['answers_sha256']
  details[view.name]={'questions':len(rows),'unique_images':len({r['image_path'] for r in rows}),'inputs_sha256':sha(dest/'inputs.jsonl'),
   'answers_sha256':sha(dest/'answers.json'),'exact_previous_frozen_input_and_Gold_files':True,'all_images_present':True,'other_input_fields_unchanged':True}
  if view.name=='POPE-paper':
   counts=Counter(r['strategy'] for r in rows);assert counts=={'random':2910,'popular':3000,'adversarial':3000};details[view.name]['strategies']=dict(counts)
  representatives.setdefault(entry['benchmark'],rows[0]['image_path'])
  print('DATASET_RESTORED',view.name,len(rows),'EXACT_FROZEN_SHA',flush=True)
decode={}
try:
 from PIL import Image
 for name,path in representatives.items():
  with Image.open(path) as im:im.load();decode[name]={'status':'PASS','size':list(im.size),'mode':im.mode}
except ImportError:decode={'status':'NOT_RUN','reason':'Pillow not installed; archives CRC, manifest hashes and all image mappings verified'}
assert len(base)==2 and set(details)=={'GQA-testdev-balanced','TextVQA','TextVQA-OCR'}
for b in base:assert details[b['benchmark']]['questions']==b['questions']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
result={'status':'PASS','endpoint':plan['endpoint'],'started_at_UTC':started,'finished_at_UTC':datetime.now(timezone.utc).isoformat(),
 'source':plan['source'],'canonical_data_root':str(D),'physical_local_data_root':str(D.resolve()),'storage_FSTYPE':'ext4',
 'packages_SHA256_verified':len(plan['packages']),'total_transfer_bytes':sum(e['bytes'] for e in plan['packages']),'base_datasets':base,'datasets':details,
 'frozen_3_view_input_and_Gold_SHA256':'PASS','all_question_image_paths_present':True,'selected_image_decode':decode,
 'MME_known_SHA_difference':'Preserved original backup landmark/03d5e3bfc958be38.jpg; no replacement',
 'no_public_dataset_downloads':True,'no_model_downloads':True,'no_GPU_experiment_started':True,'GPU_compute_processes_after':gpu,
 'disk_free_GiB':round(shutil.disk_usage(L).free/2**30,2),'Git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()}
save(O/'dataset_restore.json',result)
print('PRIORITY_DATASETS_RESTORED_PASS',len(details),'views',flush=True)
