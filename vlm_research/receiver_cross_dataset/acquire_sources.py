import pathlib,requests,hashlib,json,datetime,concurrent.futures,zipfile,shutil
root=pathlib.Path('/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation');out=root/'sources';out.mkdir(exist_ok=True)
urls={
'vqa_questions.zip':'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
'vqa_annotations.zip':'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
'official_vqa_eval.py':'https://raw.githubusercontent.com/GT-Vision-Lab/VQA/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py',
'textvqa_loader.py':'https://huggingface.co/datasets/facebook/textvqa/raw/main/textvqa.py',
'textvqa_annotations.json':'https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json'}
def fetch(item):
 name,url=item;p=out/name;record=dict(name=name,url=url,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  if p.exists():record.update(status='EXISTING',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size);return record
  assert shutil.disk_usage('/data').free>25*1024**3
  session=requests.Session();session.trust_env=False
  with session.get(url,stream=True,timeout=(20,120)) as r:
   r.raise_for_status();record.update(ETag=r.headers.get('ETag'),last_modified=r.headers.get('Last-Modified'),final_url=r.url)
   tmp=p.with_suffix(p.suffix+'.partial');size=0
   with tmp.open('xb') as f:
    for b in r.iter_content(1024**2):
     size+=len(b);assert size<180*1024**2;f.write(b)
   tmp.rename(p);record.update(status='DOWNLOADED',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=size)
 except Exception as e:record.update(status='FAILED',error=repr(e))
 return record
records=list(concurrent.futures.ThreadPoolExecutor(4).map(fetch,urls.items()))
(root/'source_acquisition.json').write_text(json.dumps(records,ensure_ascii=False,sort_keys=True)+'\n')
for r in records:print(json.dumps(r),flush=True)
for name in ['vqa_questions.zip','vqa_annotations.zip']:
 p=out/name
 if p.exists():
  with zipfile.ZipFile(p) as z:
   assert z.testzip() is None
   for n in z.namelist():
    assert pathlib.PurePosixPath(n).name==n and n.endswith('.json');dest=out/n
    if not dest.exists():dest.write_bytes(z.read(n))
