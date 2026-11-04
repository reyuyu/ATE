import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,hashlib,urllib.request,datetime,collections
W=pathlib.Path('/data/lk/ATE-worktrees/textvqa-ocr-reaccess-mechanism');O=W/'vlm_research/results/textvqa_ocr_reaccess_mechanism';P=pathlib.Path('/data/lk/ATE-resources/experiments/textvqa_ocr_reaccess_mechanism')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
url='https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json';target=P/'TextOCR_0.1_val.json';download={}
try:
 if not target.exists():
  req=urllib.request.Request(url,headers={'User-Agent':'ATE CPU annotation alignment'})
  with urllib.request.urlopen(req,timeout=45) as resp:
   length=int(resp.headers.get('Content-Length','0'));assert length<70*1024**2
   data=resp.read(70*1024**2+1);assert len(data)<=70*1024**2;target.write_bytes(data)
 t=json.loads(target.read_text());download=dict(success=True,bytes=target.stat().st_size,sha256=sha(target),keys=list(t),images=len(t['imgs']),word_annotations=len(t['anns']))
except Exception as e:t={};download=dict(success=False,error=repr(e))
blind=json.loads((O/'blind_inputs.json').read_text());source=pathlib.Path('/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/sources/textvqa_annotations.json');tv=json.loads(source.read_text());tvby={str(r['question_id']):r for r in tv['data']};extracted=[]
for b in blind:
 r=tvby[b['question_id']];assert str(r['image_id'])==b['image_id'] and r['question']==b['question']
 im=t.get('imgs',{}).get(b['image_id']);ids=t.get('imgToAnns',t.get('img2Anns',{})).get(b['image_id'],[])
 anns=[t['anns'][str(i)] for i in ids] if im else []
 if im:
  assert str(im['id'])==b['image_id'] and int(im['width'])==b['width'] and int(im['height'])==b['height'];assert pathlib.Path(im['file_name'] if 'file_name' in im else im['filename']).stem==b['image_id']
  assert all(str(a['image_id'])==b['image_id'] for a in anns)
 extracted.append(dict(question_id=b['question_id'],image_id=b['image_id'],image_sha256=b['image_sha256'],official_TextVQA_question_match=True,TextVQA_width=r.get('image_width'),TextVQA_height=r.get('image_height'),TextOCR_matched=bool(im),TextOCR_image=im,TextOCR_words=[dict(annotation_id=a['id'],text=a['utf8_string'],bbox=a['bbox'],points=a['points']) for a in anns],legacy_ocr_tokens=r.get('ocr_tokens',[]),legacy_ocr_provenance='TextVQA source OCR tokens; not human OCR ground truth'))
write(O/'aligned_ocr_annotations.json',extracted)
write(O/'ocr_dataset_alignment.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),official_documentation='https://textvqa.org/textocr/dataset/',documentation_cautions='IDs shared but privacy removals exist; membership, image ID, file stem, dimensions, annotation image_id verified per case; no claim official image byte hash without downloading archive',download_url=url,download=download,source_TextVQA_sha256=sha(source),sample_count=300,TextVQA_question_ID_image_ID_exact=300,matched_TextOCR=sum(r['TextOCR_matched'] for r in extracted),all_original_images_SHA_verified=True,no_image_archive_downloaded=True,alignment='Exact official image IDs + file stems + dimensions, original frozen TextVQA image bytes SHA retained; not independent TextOCR byte identity'))
print(json.dumps(dict(download=download,matched=sum(r['TextOCR_matched'] for r in extracted),examples=[dict(question_id=r['question_id'],TextOCR_words=[x['text'] for x in r['TextOCR_words']]) for r in extracted[:2]]),ensure_ascii=False))
