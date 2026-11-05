import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
import json,hashlib,collections,datetime,subprocess,shutil
W=Path('/data/lk/ATE-worktrees/textvqa-decode-l0-l1-factorial');R=W/'vlm_research';O=R/'results/textvqa_decode_l0_l1_factorial';P=Path('/data/lk/ATE-resources/experiments/textvqa_decode_l0_l1_factorial')
OLD=Path('/data/lk/ATE-worktrees/textvqa-single-layer-visual-ablation/vlm_research');HO=OLD/'results/textvqa_single_layer_visual_ablation'
ANN=Path('/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/sources/textvqa_annotations.json')
BASE='4408b862e4760b6aadcb5135d2640f422778938b'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def key(iid):return hashlib.sha256(('20261105|textvqa_decode_l0_l1_factorial|'+iid).encode()).hexdigest()
def main():
 raw=read(ANN)['data'];assert len(raw)==5000 and all(x['set_name']=='val' and len(x['answers'])==10 for x in raw)
 byid={str(x['question_id']):x for x in raw};official_ids={x['image_id'] for x in raw}
 # Read frozen input selections only. Never load prediction CSVs or checkpoint directories here.
 roots=[Path('/data/lk/ATE-worktrees'),Path('/data1/likai/ATE-worktrees')]
 resultroots=[]
 for root in roots:
  if root.exists():resultroots.extend(x/'vlm_research/results' for x in root.iterdir() if x!=W and (x/'vlm_research/results').is_dir())
 resultroots.append(Path('/data/lk/深度学习/ATE/vlm_research/results'))
 blocked_ids=set();blocked_sha=set();sources=[];missing=[];path_hash={}
 prune={'checkpoints','input_specs','token_manifests','figures','images','raw_logits','raw_first_logits','__pycache__','.git','sample_logits','first_logits','logits','artifacts'}
 def visit(v,ids,hashes):
  if isinstance(v,list):
   for x in v:visit(x,ids,hashes)
  elif isinstance(v,dict):
   iid=str(v.get('image_id',''));identity=str(v.get('image_identity',''))
   if identity.startswith('openimages:'):iid=identity.split(':',1)[1]
   if iid in official_ids:ids.add(iid)
   q=str(v.get('question_id',''))
   if q in byid and v.get('question')==byid[q]['question']:ids.add(byid[q]['image_id'])
   for name in ['image_sha256','image_sha','image_bytes_sha256']:
    h=v.get(name)
    if isinstance(h,str) and len(h)==64:hashes.add(h)
   ip=v.get('image_path')
   if isinstance(ip,str) and (iid in official_ids or 'textvqa' in ip.lower() or '/openimages/' in ip):
    p=Path(ip)
    if p.is_file():
     if ip not in path_hash:path_hash[ip]=sha(p)
     hashes.add(path_hash[ip])
    else:missing.append(ip)
   for x in v.values():
    if isinstance(x,(list,dict)):visit(x,ids,hashes)
 for rr in resultroots:
  if not rr.exists():continue
  for dp,dirs,files in os.walk(rr):
   dirs[:]=[x for x in dirs if x not in prune and not any(z in x.lower() for z in ['checkpoint','logits','attention_arrays'])]
   for name in files:
    lower=name.lower()
    if not lower.endswith('.json') or not any(x in lower for x in ['manifest','selection_plan','selection_before','selected_samples','sample_freeze','selection_lock']):continue
    p=Path(dp)/name
    if p.stat().st_size>80*1024**2:continue
    try:v=read(p)
    except (ValueError,OSError):continue
    ids=set();hashes=set();visit(v,ids,hashes)
    if ids or hashes:
     blocked_ids.update(ids);blocked_sha.update(hashes);sources.append(dict(path=str(p),sha256=sha(p),TextVQA_image_ids=sorted(ids),image_sha256=sorted(hashes)))
  print('HISTORY_AUDIT',rr.parent.parent.name,'images',len(blocked_ids),flush=True)
 # Explicit phase4 input selections are relevant even when their source worktree was moved.
 extra=Path('/data/lk/ATE-resources/benchmarks/phase4_0/selection_lock.json')
 if extra.exists():
  ids=set();hashes=set();visit(read(extra),ids,hashes);blocked_ids.update(ids);blocked_sha.update(hashes);sources.append(dict(path=str(extra),sha256=sha(extra),TextVQA_image_ids=sorted(ids),image_sha256=sorted(hashes)))
 old=read(HO/'sample_manifest.json')['rows'];assert len(old)==300
 assert {x['image_id'] for x in old}<=blocked_ids and {x['image_sha256'] for x in old}<=blocked_sha
 write(O/'historical_image_inventory.json',dict(sources=sources,excluded_image_ids=sorted(blocked_ids),excluded_image_sha256=sorted(blocked_sha),missing_historical_image_paths=sorted(set(missing)),scope='Conservative union of frozen input manifests/selections across all available ATE worktrees plus phase4 input lock; includes input-only historical selections, not just successful predictions. No prediction files read.'))
 grouped=collections.defaultdict(list)
 for x in raw:grouped[x['image_id']].append(x)
 candidates=sorted(set(grouped)-blocked_ids,key=key)
 localdirs=[Path('/data/lk/ATE-resources/paper_rq1/images/openimages'),Path('/data/lk/ATE-resources/benchmarks/phase4_0/images/textvqa'),Path('/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/images/textvqa')]
 selected=[];attempts=[];seen=set();fetched=[]
 for iid in candidates:
  path=next((d/(iid+'.jpg') for d in localdirs if (d/(iid+'.jpg')).is_file()),P/'images'/(iid+'.jpg'))
  if not path.is_file():
   import sys;sys.path.insert(0,str(R))
   from paper_rq1.data import fetch_openimages
   try:
    assert shutil.disk_usage('/data').free>20*1024**3
    receipt=fetch_openimages([dict(image_identity='openimages:'+iid,image_path=str(path))]);fetched.append(receipt)
   except Exception as e:
    attempts.append(dict(image_id=iid,status='IMAGE_UNAVAILABLE',error=repr(e)));continue
  h=sha(path)
  if h in blocked_sha or h in seen:attempts.append(dict(image_id=iid,status='EXCLUDED_BYTE_SHA_OVERLAP',image_sha256=h));continue
  seen.add(h)
  from PIL import Image
  with Image.open(path) as im:im.load();size=list(im.size)
  for x in sorted(grouped[iid],key=lambda x:str(x['question_id'])):
   selected.append(dict(question_id=str(x['question_id']),image_id=iid,question=x['question'],answers=x['answers'],image_path=str(path),image_sha256=h,image_size=size,answer_type='textvqa',question_type='OCR-oriented VQA',cohort='new_images',set_name='val',flickr_original_url=x['flickr_original_url']))
  attempts.append(dict(image_id=iid,status='SELECTED_WHOLE_IMAGE_GROUP',questions=len(grouped[iid]),image_sha256=h))
  if len(selected)>=800:break
 assert selected and len({x['question_id'] for x in selected})==len(selected)
 assert not {x['image_id'] for x in selected}&blocked_ids and not {x['image_sha256'] for x in selected}&blocked_sha
 write(P/'data_selection.json',dict(new_rows=selected,old_rows=old,candidates=len(candidates),attempts=attempts,download_receipts=fetched,annotation_path=str(ANN),annotation_sha256=sha(ANN)))
 write(O/'image_overlap_audit.json',dict(passed=True,official_validation_questions=5000,official_validation_images=len(grouped),historical_TextVQA_image_IDs=len(blocked_ids),historical_image_SHA_union=len(blocked_sha),historical_input_sources=len(sources),new_questions=len(selected),new_images=len(seen),image_ID_overlap=0,image_SHA_overlap=0,old300_questions=300,all_questions_of_selected_images_included=True,selection_rule='Sort unseen official image IDs by SHA256(20261105|textvqa_decode_l0_l1_factorial|image_id), fetch/cache input images, exclude historical image-ID or exact byte-SHA matches, retain whole question groups until at least800; no predictions read',target=800,shortfall_reason=None if len(selected)>=800 else 'All verifiably available eligible images exhausted under resource constraints; never filled with historical rows',annotation_sha256=sha(ANN),limitations='Exact image IDs/byte SHA exclude known historical images; no guarantee against re-encoded near duplicates or undiscoverable experiments outside attached repositories; not an external dataset.'))
 print('DATA_SELECTION_COMPLETE',len(selected),len(seen),'history_ids',len(blocked_ids),flush=True)
if __name__=='__main__':main()
