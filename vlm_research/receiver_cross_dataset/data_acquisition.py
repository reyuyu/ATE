import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json, hashlib, pathlib, zipfile, shutil, datetime, concurrent.futures, requests, time
from PIL import Image
W=pathlib.Path('/data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation')
P=pathlib.Path('/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation')
O=W/'vlm_research/results/receiver_cross_dataset_validation'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
def key(dataset,r):return hashlib.sha256(f"20261104|{dataset}|{r['question_id']}|{r['image_id']}".encode()).hexdigest()
ids=json.loads((W/'vlm_research/results/semantic_single_gpu_search3h/split_manifest.json').read_text())['metadata']
gqa=[json.loads((pathlib.Path('/data1/likai/ATE-resources/experiments/gqa5h_receiver_search/samples')/(q+'.json')).read_text()) for q in sorted(ids)]
vgids={str(r['image_id']) for r in gqa};gsha={r['image_sha256'] for r in gqa}
with zipfile.ZipFile('/data/lk/ATE-resources/paper_rq1/annotations/vg_image_data.zip') as z:vg=json.loads(z.read('image_data.json'))
known=[r for r in vg if str(r['image_id']) in vgids]
coco={str(r['coco_id']) for r in known if r.get('coco_id')};flickr={str(r['flickr_id']) for r in known if r.get('flickr_id')}
save(O/'overlap_reference.json',dict(GQA_questions=len(gqa),GQA_images=len(gsha),GQA_VG_ids=sorted(vgids),COCO_ids=sorted(coco),Flickr_ids=sorted(flickr),GQA_image_sha256=sorted(gsha),VG_metadata_sha256=sha('/data/lk/ATE-resources/paper_rq1/annotations/vg_image_data.zip'),VG_history_ids_mapped=len(known),scope='Historical GQA3000; unknown IDs and re-encoded image identity remain limitations.'))
S=P/'sources'
questions=json.loads((S/'v2_OpenEnded_mscoco_val2014_questions.json').read_text())['questions']
anns={r['question_id']:r for r in json.loads((S/'v2_mscoco_val2014_annotations.json').read_text())['annotations']}
vqa=[]
for q in questions:
 a=anns[q['question_id']];assert a['image_id']==q['image_id'] and len(a['answers'])==10
 vqa.append(dict(question_id=str(q['question_id']),image_id=str(q['image_id']),question=q['question'],answers=[x['answer'] for x in a['answers']],answer_type=a['answer_type'],question_type=a['question_type'],metadata_overlap=str(q['image_id']) in coco))
tv=json.loads((S/'textvqa_annotations.json').read_text())['data'];text=[]
for r in tv:
 assert len(r['answers'])==10 and r['set_name']=='val'
 fid=pathlib.PurePosixPath(r['flickr_original_url']).name.split('_')[0]
 text.append(dict(question_id=str(r['question_id']),image_id=r['image_id'],question=r['question'],answers=r['answers'],answer_type='textvqa',question_type='OCR-oriented VQA',metadata_overlap=fid in flickr,flickr_id=fid))
save(O/'dataset_sources.json',dict(files={p.name:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in S.iterdir() if p.is_file() and not p.name.endswith('.partial')},full_validation_counts=dict(vqa_v2=len(vqa),textvqa=len(text))))
def fetch(dataset,r):
 iid=r['image_id'];dest=P/'images'/dataset/(iid+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True)
 local=pathlib.Path('/data/lk/ATE-resources/paper_rq1/images')/('coco2014' if dataset=='vqa_v2' else 'openimages')/(iid+'.jpg')
 receipt=dict(image_id=iid)
 try:
  if dest.exists():receipt['source']='existing_new_cache'
  elif local.exists():receipt['source']=str(local);shutil.copyfile(local,dest)
  elif dataset=='vqa_v2':
   assert shutil.disk_usage('/data').free>20*1024**3
   url=f'https://s3.amazonaws.com/images.cocodataset.org/val2014/COCO_val2014_{int(iid):012d}.jpg'
   s=requests.Session();s.trust_env=False
   response=s.get(url,timeout=(15,45));response.raise_for_status();assert len(response.content)<20*1024**2
   dest.write_bytes(response.content);receipt.update(source=url,ETag=response.headers.get('ETag'))
  else:return dict(status='MISSING_LOCAL_TEXTVQA',image_id=iid,path=str(dest))
  with Image.open(dest) as im:im.load();assert im.width>0 and im.height>0;size=list(im.size)
  h=sha(dest);r.update(image_path=str(dest),image_sha256=h,image_size=size,SHA_overlap=h in gsha)
  receipt.update(status='OK',sha256=h,bytes=dest.stat().st_size);return receipt
 except Exception as e:return dict(receipt,status='FAILED',error=repr(e))
for dataset,rows,n in [('vqa_v2',vqa,500),('textvqa',text,300)]:
 d=O/dataset;d.mkdir(parents=True,exist_ok=True)
 import collections,re
 save(d/'full_validation_cpu_firstwords.json',dict(n=len(rows),counts=dict(collections.Counter((re.findall('[a-z]+',r['question'].lower()) or [''])[0] for r in rows)),scope='Full validation text only; no GPU results or rule selection'))
 ordered=sorted(rows,key=lambda r:(r['metadata_overlap'],key(dataset,r)));selected=[];seen=set()
 for r in ordered:
  if not r['question'].strip() or r['image_id'] in seen:continue
  seen.add(r['image_id']);selected.append(r)
  if len(selected)==n:break
 assert len(selected)==n
 save(d/'selection_before_images.json',dict(seed=20261104,selection='Known non-overlap first; SHA256(seed|dataset|qid|iid), one question per image, no output-based selection',rows=selected))
 receipts=list(concurrent.futures.ThreadPoolExecutor(6).map(lambda r:fetch(dataset,r),selected))
 missing=[r for r,x in zip(selected,receipts) if x['status']=='MISSING_LOCAL_TEXTVQA']
 if missing:
  import sys;sys.path.insert(0,str(W/'vlm_research'))
  from paper_rq1.data import fetch_openimages
  try:
   rec=fetch_openimages([dict(image_identity='openimages:'+r['image_id'],image_path=str(P/'images'/dataset/(r['image_id']+'.jpg'))) for r in missing]);save(d/'remote_zip_ranges.json',rec)
  except Exception as e:save(d/'image_fetch_failure.json',dict(error=repr(e),missing=len(missing)))
  receipts=[fetch(dataset,r) for r in selected]
 save(d/'image_receipts.json',receipts)
 good=all(r['status']=='OK' for r in receipts)
 save(d/'sample_manifest.json',dict(status='FROZEN' if good else 'DATA_UNAVAILABLE',dataset=dataset,n=n,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=sha(d/'selection_before_images.json'),rows=selected))
 save(d/'dataset_overlap_audit.json',dict(n=n,metadata_overlap=sum(r['metadata_overlap'] for r in selected),SHA_overlap=sum(r.get('SHA_overlap',False) for r in selected),GQA_reference_images=len(gsha),method='VG coco_id/flickr_id plus exact image byte SHA256',unknown='Absence of a metadata or byte match does not exclude re-encoded/uncatalogued identity; model pretraining overlap not audited'))
 print(dataset,'FROZEN' if good else 'DATA_UNAVAILABLE',len(selected),collections.Counter(r['status'] for r in receipts),flush=True)
