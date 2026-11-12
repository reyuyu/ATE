import os,sys,json,hashlib,pathlib,subprocess,collections,datetime
W=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer')
O=W/'vlm_research/results/qwen3vl_c3v2_physical_transfer';O.mkdir(parents=True,exist_ok=True)
P=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_physical_transfer')
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
SEED=20261105
# Conservative: any input manifest entry counts as prior exposure, never use outputs.
historical=set();hist_ids=collections.defaultdict(set);seen=set();sources=[]
def walk(x):
 if isinstance(x,dict):
  h=x.get('image_sha256',x.get('image_sha'))
  if isinstance(h,str) and len(h)==64:historical.add(h)
  for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
for wt in pathlib.Path('/data/lk/ATE-worktrees').iterdir():
 if wt==W:continue
 r=wt/'vlm_research/results'
 if not r.exists():continue
 for root,ds,fs in os.walk(r):
  ds[:]=[d for d in ds if d not in ['checkpoints','input_specs','token_manifests','shard0','shard1','images','figures','raw']]
  for f in fs:
   if not f.endswith('.json') or not any(t in f.lower() for t in ['sample_manifest','data_manifest','selected_cases']):continue
   p=pathlib.Path(root)/f
   if p.stat().st_size>80_000_000:continue
   h=sha(p)
   if h in seen:continue
   seen.add(h)
   try:walk(json.loads(p.read_text()));sources.append(dict(path=str(p),sha256=h))
   except Exception:pass
write(O/'historical_manifest_sources.json',sources)
print('Historical distinct image SHA',len(historical),flush=True)
sys.path.insert(0,str(P/'cpu_tools'))
import pyarrow.parquet as pq
gsource=pathlib.Path('/data/lk/ATE-resources/paper_rq1/annotations/gqa_val_balanced.parquet')
grows=pq.read_table(gsource).to_pylist();byimage=collections.defaultdict(list)
for r in grows:byimage[r['imageId']].append(r)
groot=pathlib.Path('/data/lk/ATE-resources/data/image-tasks/MMEB/GQA')
gpool=[]
for im,rs in byimage.items():
 p=groot/f'image_{im}.jpg'
 if not p.exists():continue
 h=sha(p)
 if h in historical:continue
 gpool.append((h,im,p,rs))
gpool.sort(key=lambda x:digest([SEED,'GQA',x[0]]))
samples=[];count=0
for h,im,p,rs in gpool:
 if count>=1000:break
 for r in sorted(rs,key=lambda r:r['id']):
  samples.append(dict(dataset='GQA',task=r['types']['detailed'],question_id=r['id'],image_id=im,image_path=str(p),image_sha256=h,question=r['question'],answer=r['answer'],historical_overlap=False))
 count+=len(rs)
print('GQA new whole image groups',count,len({s['image_sha256'] for s in samples}),flush=True)
vroot=pathlib.Path('/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/sources')
anns=json.loads((vroot/'v2_mscoco_val2014_annotations.json').read_text())['annotations']
ques={r['question_id']:r['question'] for r in json.loads((vroot/'v2_OpenEnded_mscoco_val2014_questions.json').read_text())['questions']}
vby=collections.defaultdict(list)
for r in anns:vby[str(r['image_id'])].append(r)
# Search already cached COCO images; do not download images or datasets.
vfiles={}
for root,ds,fs in os.walk('/data/lk/ATE-resources'):
 ds[:]=[d for d in ds if d not in ['models','data','checkpoints','cpu_tools','textvqa','textcaps','token_manifests','.git','figures']]
 for f in fs:
  if not f.lower().endswith(('.jpg','.jpeg','.png')):continue
  stem=pathlib.Path(f).stem
  candidate=stem.split('_')[-1]
  if candidate.isdigit() and str(int(candidate)) in vby:vfiles.setdefault(str(int(candidate)),pathlib.Path(root)/f)
vpool=[]
for im,p in vfiles.items():
 h=sha(p)
 if h not in historical:vpool.append((h,im,p))
vpool.sort(key=lambda x:digest([SEED,'VQAv2',x[0]]))
vn=0
for h,im,p in vpool:
 if vn>=500:break
 for r in sorted(vby[im],key=lambda r:r['question_id']):
  samples.append(dict(dataset='VQAv2',task=r['answer_type'],question_id=str(r['question_id']),image_id=im,image_path=str(p),image_sha256=h,question=ques[r['question_id']],answers=[a['answer'] for a in r['answers']],historical_overlap=False))
 vn+=len(vby[im])
print('VQA new whole image groups',vn,len(vpool),flush=True)
# If no new local VQA image exists, retain zero formal VQA samples and use old
# frozen valid images only for clearly labelled smoke/pilot diagnostics.
oldv=json.loads(pathlib.Path('/data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation/vlm_research/results/receiver_cross_dataset_validation/vqa_v2/sample_manifest.json').read_text())['rows']
diagnostic=[dict(s,dataset='VQAv2',task=s['answer_type'],historical_overlap=True) for s in oldv[:12]] if vn==0 else []
if not samples:raise RuntimeError('No valid new samples')
for s in samples+diagnostic:s['id']=s['dataset']+'__'+s['question_id']
groups=collections.defaultdict(list)
for s in samples:groups[s['image_sha256']].append(s)
loads=[0,0];shards={}
for h,ss in sorted(groups.items(),key=lambda kv:(-len(kv[1]),digest([SEED,'shard',kv[0]]))):
 rank=min(range(2),key=lambda i:(loads[i],i));shards[h]=rank;loads[rank]+=len(ss)
for s in samples:s['shard']=shards[s['image_sha256']]
for i,s in enumerate(diagnostic):s['shard']=i%2
write(O/'sample_manifest.json',dict(samples=samples,diagnostic_only=diagnostic,selection='Complete question groups of local official images; exclude union of historical manifest SHAs; seeded SHA rank; stop after target reached, retain final group',seed=SEED,shard_counts=loads))
write(O/'image_overlap_audit.json',dict(historical_distinct_sha=len(historical),new_images_by_dataset=dict(collections.Counter(s['dataset'] for h,ss in groups.items() for s in ss[:1])),question_counts=dict(collections.Counter(s['dataset'] for s in samples)),old_main_images=0,gqa_local_new_pool_images=len(gpool),vqav2_local_new_pool_images=len(vpool),vqav2_zero_new_fallback='No historical images mixed into main; old images diagnostic-only' if vn==0 else None,source_hashes={str(gsource):sha(gsource),str(vroot/'v2_mscoco_val2014_annotations.json'):sha(vroot/'v2_mscoco_val2014_annotations.json'),str(vroot/'v2_OpenEnded_mscoco_val2014_questions.json'):sha(vroot/'v2_OpenEnded_mscoco_val2014_questions.json')}))
print('Preparing processor inputs',flush=True)
from transformers import AutoProcessor
from vlm_research.qwen3vl_early_windows.experiment import prepare,MODEL
processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
specs={}
for s in samples+diagnostic:
 _,layout=prepare(processor,s);specs[s['id']]=layout
write(O/'input_bindings.json',specs)
ordered=sorted(samples+diagnostic,key=lambda s:specs[s['id']]['input_length'])
smoke=[ordered[0]['id'],ordered[len(ordered)//2]['id'],ordered[-1]['id']]
assert len({specs[k]['image_tokens'] for k in smoke})>1
pilot=[];bench=[]
for dataset in ['GQA','VQAv2']:
 rows=[s for s in samples+diagnostic if s['dataset']==dataset]
 ranked=sorted(rows,key=lambda s:digest([SEED,'pilot',s['image_sha256'],s['id']]))
 pilot.extend(s['id'] for s in ranked[:10])
 used=set()
 for s in sorted(rows,key=lambda s:digest([SEED,'benchmark',s['image_sha256']])):
  if s['image_sha256'] not in used:bench.append(s['id']);used.add(s['image_sha256'])
  if len(used)>=20:break
write(O/'execution_selection.json',dict(smoke=smoke,pilot=pilot,benchmark=bench,rule='Input-only seeded SHA selection; smoke sorted prompt length min/median/max; no outputs read'))
model_files={str(p.relative_to(MODEL)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in MODEL.iterdir() if p.is_file()}
write(O/'model_manifest.json',dict(model=str(MODEL),revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',files=model_files))
write(O/'protocol.json',dict(seed=SEED,max_new_tokens=32,dtype='bfloat16',attention='eager',transformers='4.57.6',conditions=['N','F','P','B','C'],diagnostics=['W22-functional','A16-functional'],writer_cutoff=22,skip_layers=list(range(1,7)),answer_image_mask_all_non_native=True,smoke_tolerance=dict(native_noop='torch.equal all effective logits',functional_physical_max_abs=.5,functional_physical_mean_abs=.02,top1_agreement=1.,complete_token_agreement=1.,action='Any failure stops both workers before pilot/main/performance; do not relax'),comparisons=[['C','N'],['F','N'],['P','F'],['C','F'],['C','P'],['C','B']],bootstrap=20000,cluster='image_sha256, question equally weighted',multiplicity='Bonferroni within each dataset six comparisons; exploratory -1pp one-sided lower CI, no confirmatory claim',timing='Each image, each condition: three warmups + five repeats, interleaved; coldKV; original pixels; file IO/processor excluded; CUDA events and synchronized wall E2E',smoke_assertions='All36 MLP/norm and residual; native mRoPE and 3 deepstack; zero blocked attention; early skip affects P-1 only; actual compact KV and hidden rows; cleanup normal/exception; no fallback',data_manifest_sha256=sha(O/'sample_manifest.json'),input_bindings_sha256=sha(O/'input_bindings.json'),interaction_limitation='Native has direct answer image access; F/P/B/C do not. C-P and C-B conditional contributions, not pure four-cell factorial with Native.',source_reference='dfe0cbacd09de9dfeacff980b911e6aec22a7665'))
print('FROZEN',flush=True)
