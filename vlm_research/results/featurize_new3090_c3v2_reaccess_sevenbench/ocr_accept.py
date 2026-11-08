"""CPU-only recovery of the pinned official OCRBench v1 annotation and images."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path,PurePosixPath
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib,importlib.util,json,shutil,zipfile
from PIL import Image
base=Path('/home/featurize/data/ate-recovery')
src=base/'sources/ocrbench-v1';out=base/'normalized/OCRBench-v1';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
archive=src/'OCRBench_Images.official.zip'
assert archive.stat().st_size==109105033 and sha(archive)=='c82acb33a2f00cdb499ea13a3cee7f3d1584269af339aed3fcb4e381d5a9f975'
official=src/'official/OCRBench/OCRBench/OCRBench.json'
assert sha(official)=='df75957e75ea052e9f38f2c67a4f6676103998517345cf49ef7d03b62c835864'
shutil.copyfile(official,out/'OCRBench.json');data=json.loads(official.read_text());assert len(data)==1000
with zipfile.ZipFile(archive) as z:
 members=[p for p in z.infolist() if not p.is_dir()]
 assert len(members)==930 and len({p.filename for p in members})==930
 for entry in members:
  rel=PurePosixPath(entry.filename)
  assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='OCRBench_Images'
  dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  blob=z.read(entry)
  if not dest.is_file() or dest.read_bytes()!=blob:dest.write_bytes(blob)
images={};dimensions={}
for p in sorted((out/'OCRBench_Images').rglob('*')):
 if not p.is_file():continue
 with Image.open(p) as im:im.verify()
 with Image.open(p) as im:im.convert('RGB').load();dimensions[str(p.relative_to(out))]=list(im.size)
 images[str(p.relative_to(out))]={'bytes':p.stat().st_size,'sha256':sha(p),'dimensions':dimensions[str(p.relative_to(out))]}
assert len(images)==930
categories={'Regular Text Recognition':50,'Irregular Text Recognition':50,'Artistic Text Recognition':50,
 'Handwriting Recognition':50,'Digit String Recognition':50,'Non-Semantic Text Recognition':50,
 'Scene Text-centric VQA':200,'Doc-oriented VQA':200,'Key Information Extraction':200,
 'Handwritten Mathematical Expression Recognition':100}
assert Counter(r['type'] for r in data)==categories
spec=importlib.util.spec_from_file_location('ocr_reference_conversation',base/'sources/Information-Horizon/llava/conversation.py')
conversation=importlib.util.module_from_spec(spec);spec.loader.exec_module(conversation)
system=("A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.")
rows=[];used=set();identities=[]
for i,r in enumerate(data):
 assert set(['id','dataset_name','image_path','question','answers','type'])<=set(r)
 assert isinstance(r['question'],str) and r['question'].strip()
 answers=r['answers'] if isinstance(r['answers'],list) else [r['answers']]
 assert answers and all(isinstance(a,str) and a.strip() for a in answers)
 rel='OCRBench_Images/'+r['image_path'];assert rel in images;used.add(rel)
 user_text=r['question']+'\nAnswer the question using a single word or phrase.'
 conv=conversation.conv_templates['vicuna_v1'].copy();conv.append_message(conv.roles[0],'<image>\n'+user_text);conv.append_message(conv.roles[1],None)
 prompt=system+' USER: <image>\n'+user_text+' ASSISTANT:'
 assert conv.get_prompt()==prompt
 qid=f'{i:04d}';identities.append((r['dataset_name'],str(r['id'])))
 rows.append(dict(key='OCRBench-v1:'+qid,question_id=qid,official_id=r['id'],official_annotation_index=i,
  benchmark='OCRBench-v1',dataset_name=r['dataset_name'],category=r['type'],question=r['question'],answer=r['answers'],
  image_id=r['image_path'],image_path=str(out/rel),image_sha256=images[rel]['sha256'],
  input_view='OCRBench-v1-official',user_text=user_text,prompt=prompt,max_new_tokens=128,
  external_ocr=False,official_record_sha256=hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True).encode()).hexdigest()))
assert len(rows)==len({r['key'] for r in rows})==1000 and len(used)==930
(out/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))
discovery=json.loads((src/'discovery.json').read_text())
source={'official_repository':'https://github.com/Yuliang-Liu/MultimodalOCR','official_commit':discovery['official_commit'],
 'annotation':discovery['official_files']['OCRBench/OCRBench/OCRBench.json'],
 'image_download_url':'https://drive.google.com/file/d/1a3VRJx3V3SdOmPr7499Ky0Ug8AwqGUHO/view',
 'image_zip_sha256':sha(archive),'image_zip_bytes':archive.stat().st_size,'mirror_used':False,
 'transfer_route':'Official Google Drive downloaded on Windows, same ZIP uploaded via verified SSH/SFTP to Featurize; no re-encoding.',
 'raw_annotation_preserved':True,'full_test_or_v2_used':False,'restored_at':datetime.now(timezone(timedelta(hours=8))).isoformat()}
save('SOURCE.json',source)
save('protocol.json',dict(id='OCRBench-v1-official-LLaVA1.5-HF-physical-20261108',model='llava-hf/llava-1.5-7b-hf',
 revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',template='vicuna_v1',prompt_suffix='\nAnswer the question using a single word or phrase.',
 generation=dict(greedy=True,temperature=0,top_p=None,num_beams=1,max_new_tokens=128),
 preprocessing='Original RGB and official LLaVA pad-to-square mean background, CLIP processor 336; no external OCR.',
 scorer=dict(source=discovery['official_files']['OCRBench/scripts/LLaVA1_5.py'],
  rules='Official answer substring match: HME100k strips spaces and newlines case-sensitively; other datasets lowercase, strip outer whitespace and replace newline with space. Any accepted answer gives one point.',
  scale='Raw summed score 0–1000, not a percentage or VQA soft score.',categories=categories)))
save('statistics.json',dict(status='PASS',questions=1000,unique_images=930,decodable_images=930,categories=categories,
 original_raw_id_duplicates=1000-len({str(r['id']) for r in data}),
 dataset_scoped_id_duplicates=1000-len(set(identities)),
 identity_policy='ATE stable zero-based official annotation index; retain dataset_name and original id without changing Gold.',
 unique_ATE_question_keys=1000,missing_images=[],unused_images=[],invalid_gold=[],original_annotation_sha256=sha(official)))
(out/'README_zh.md').write_text('OCRBench v1 官方 1,000 题／930 张原始图片。原始 OCRBench.json 未改写；来源和固定哈希见 SOURCE.json。\n'
 'inputs.jsonl 记录官方行号、原始 ID、类别、图像和冻结 Prompt；image_path 是验收时服务器路径，迁移时按 image_id 定位到本目录 OCRBench_Images。\n'
 'Official raw id 不保证全局唯一；ATE question_id 使用官方行号，原始 ID 和答案完整保留。protocol.json 为官方 LLaVA1.5 Prompt、128 Token Greedy 和十子类 0–1000 评分。无额外 OCR 文本。\n')
payload={}
for p in sorted(out.rglob('*')):
 if p.is_file() and p.name not in ['file_manifest.json','SHA256SUMS.txt']:
  payload[str(p.relative_to(out))]={'bytes':p.stat().st_size,'sha256':sha(p)}
save('file_manifest.json',dict(dataset='OCRBench-v1',questions=1000,images=930,files=payload))
(out/'SHA256SUMS.txt').write_text(''.join(f"{v['sha256']}  {k}\n" for k,v in payload.items())+f"{sha(out/'file_manifest.json')}  file_manifest.json\n")
print('OCR_DATA_ACCEPTED',json.dumps(dict(path=str(out),questions=1000,images=930,files=len(payload)+2,
 bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),manifest_sha256=sha(out/'file_manifest.json'),
 statistics=json.loads((out/'statistics.json').read_text())),ensure_ascii=False),flush=True)
