"""CPU-only deterministic 100+100 input freeze. No weight/image downloads."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import sys,json,gzip,hashlib,subprocess
import torch,transformers
from transformers import AutoProcessor,AutoConfig
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/v2drop_quick_reproduction';D=Path('/home/featurize/data/ate-recovery');M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
sys.path[:0]=[str(R),str(O),str(R/'vlm_research/results/featurize_new3090_recovery/scripts')]
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from hf_adapter import official_selector
import ate_data
SEED='ATE-V2Drop-quickcheck-20261108-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rank(s):return hashlib.sha256((SEED+'|'+s['benchmark']+'|'+s['question_id']).encode()).hexdigest()
def gz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
assert not (O/'protocol.json').exists(),'Do not overwrite an existing frozen quickcheck'
G=R/'vlm_research/results/featurize_4090_gqa_official_aligned_threeway';gp=read(G/'protocol.json')
assert read(G/'run_metadata.json')['status']=='COMPLETE' and sha(G/'input_manifest.jsonl.gz')==gp['manifest_sha256']
gqa=[dict(s,benchmark='GQA',key='GQA:'+s['question_id'],max_new_tokens=32,input_view='GQA-official-aligned') for s in gz(G/'input_manifest.jsonl.gz')]
H=R/'vlm_research/results/featurize_new3090_recovery/history/cvpr2026-information-horizon-alignment/vlm_research/results/cvpr2026_information_horizon_alignment'
paper=read(H/'protocol_ate_paper.json');assert sha(H/'sample_manifest.jsonl.gz')==paper['artifacts']['sample_manifest.jsonl.gz']
text=[s for s in gz(H/'sample_manifest.jsonl.gz') if s['benchmark']=='TextVQA'];assert len(text)==5000
current={s['question_id']:s for s in ate_data.iter_inputs('TextVQA-OCR')};gold=ate_data.load_answers('TextVQA-OCR')
proc=AutoProcessor.from_pretrained(M,local_files_only=True);cfg=AutoConfig.from_pretrained(M,local_files_only=True)
proc.patch_size=cfg.vision_config.patch_size;proc.num_additional_image_tokens=1;proc.vision_feature_select_strategy=cfg.vision_feature_select_strategy
rows=[]
for bench,cohort in [('GQA',gqa),('TextVQA',text)]:
 for position,s in enumerate(sorted(cohort,key=rank)[:100]):
  if bench=='TextVQA':
   c=current[s['question_id']];assert s['prompt']==c['prompt'] and s['answer']==gold[s['question_id']] and s['paper_ocr'] and 'Reference OCR token:' in s['prompt']
   s=dict(s,image_path=c['image_path'],input_view='TextVQA-Paper-OCR')
  raw=frozen.image_ids(s['prompt'],proc.tokenizer,-200);ids=[]
  for v in raw:ids.extend([32000]*576 if v==-200 else [v])
  assert ids.count(32000)==576
  pixels=frozen.paper_pixels(s['image_path'],proc.image_processor).half()
  hashes={'input_ids':tensor_hash(torch.tensor([ids],dtype=torch.long)),'attention_mask':tensor_hash(torch.ones((1,len(ids)),dtype=torch.long)),'pixel_values':tensor_hash(pixels)}
  if bench=='GQA':assert ids==s['input_ids'] and hashes==s['input_hashes']
  row={k:s[k] for k in ['key','benchmark','question_id','image_id','image_path','image_sha256','question','prompt','answer','max_new_tokens','input_view']}
  row.update(selection_rank=position,selection_hash=rank(s),input_ids=ids,input_hashes=hashes,prompt_sha256=hashlib.sha256(s['prompt'].encode()).hexdigest(),image_token_start=ids.index(32000))
  row['sample_sha256']=digest(row);rows.append(row)
smoke=[]
for bench in ('GQA','TextVQA'):
 seen=set()
 for s in [s for s in rows if s['benchmark']==bench]:
  if s['image_sha256'] not in seen:smoke.append(s['key']);seen.add(s['image_sha256'])
  if len(seen)==2:break
profile=[s['key'] for bench in ('GQA','TextVQA') for s in rows if s['benchmark']==bench and s['selection_rank']<15]
assert len(smoke)==4 and len(profile)==30
select,reference,provenance=official_selector(O/'official_sources/V2Drop.py')
synthetic=[]
for start in (5,35):
 h=torch.arange((start+576+7)*32,dtype=torch.float32).sin().reshape(1,start+576+7,32);prev=h*0.9
 for layer in (3,17,22):
  k,selected,scores=select(h,prev,start,576,layer)
  p=h.new_zeros((1,35,32));a=torch.cat([p,h[:,start:]],dim=1);b=torch.cat([p,prev[:,start:]],dim=1)
  rk,rs,rv=reference(a,b,35,576,layer)
  assert torch.equal(rs-35,selected-start) and torch.equal(rv,scores)
  assert set(k.tolist())-set(range(start,start+576))==set(range(start))|set(range(start+576,start+576+7))
  synthetic.append({'image_start':start,'layer':layer,'official_selection_exact':True,'text_preserved':True})
save('selector_cpu_checks.json',synthetic)
with gzip.GzipFile(filename=str(O/'input_manifest.jsonl.gz'),mode='wb',mtime=0) as f:
 for s in rows:f.write((json.dumps(s,ensure_ascii=False,separators=(',',':'))+'\n').encode())
protocol={'status':'FROZEN','task':'V2Drop fixed quick reproduction','official_commit':'2a9431150f8ace3a1d9abcdc96a7d46acca597df','official_source':provenance,
 'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'model':'llava-hf/llava-1.5-7b-hf','revision':gp['revision'],'model_path':str(M),
 'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'float16','attention':'eager','batch_size':1,'greedy':True},
 'selection_rule':'Sort SHA256(seed|benchmark|question_id) ascending within the complete frozen manifest; take first100; no answer-based selection','seed':SEED,
 'samples':{'GQA':100,'TextVQA':100},'manifest_sha256':sha(O/'input_manifest.jsonl.gz'),'source_manifests':{'GQA':gp['manifest_sha256'],'TextVQA':sha(H/'sample_manifest.jsonl.gz')},
 'smoke_keys':smoke,'efficiency_keys':profile,'efficiency_selection':'First15 of each selected100 in frozen hash rank','efficiency_lengths':[32,128],
 'accuracy_methods':['Native','V2Drop'],'efficiency_methods':['Native','V2Drop','C3-V2','C3-V2+L11-L24'],'warmup_requests_per_method':2,'latency_repetitions_per_input':1,
 'preprocessing':'Unchanged frozen paper_pixels: RGB, CLIP-mean square-pad, CLIP336; same tensors for all methods','image_token_start_counts':{str(i):sum(s['image_token_start']==i for s in rows) for i in sorted({s['image_token_start'] for s in rows})},
 'scoring':{'GQA':'Frozen official rstrip periods, lowercase, exact match','TextVQA':'Frozen Paper-OCR ten-answer VQA soft scoring'},
 'frozen_ATE_sources':gp['frozen_sources'],'original_GQA_result_untouched':True,'no_weights_or_data_downloaded':True,
 'Flops_convention':'Actual aten matrix products and convolutions, MAC=2 FLOPs, includes vision+projector, language prefill and real autoregressive decode. Non-matrix elementwise/norm/softmax/index/sort not counted; selection overhead reported separately.',
 'fixed_length_definition':'32/128 output tokens = one prefill prediction +31/127 real decode calls; ignore EOS only during efficiency runs',
 'CUDA_initialized':torch.cuda.is_initialized()}
assert not protocol['CUDA_initialized'];save('protocol.json',protocol);save('progress.json',{'status':'PREPARED','phase':'CPU_FREEZE','accuracy_completed':0,'accuracy_expected':400,'efficiency_completed':0,'efficiency_expected':240})
print('PREPARED',len(rows),provenance,protocol['image_token_start_counts'],flush=True)
