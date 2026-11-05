import hashlib,json,os,datetime,subprocess
from pathlib import Path
from ..physical_visual_handoff.common import atomic,digest,read,sha
from ..benchmarks.scoring import textvqa_score

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/textvqa500_all_writer_c3_answer'
HISTORY=ROOT/'vlm_research/results/c3v2_middle_visual_reaccess'
PRIVATE=Path('/data/lk/ATE-resources/experiments/textvqa500_all_writer_c3_answer')
MODEL=Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
BASE='5627bc822a7827ae11edfaec84c31f6b73398e25'
UUIDS=['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
SEED=20261106

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a]).decode().strip()
def tensor_hash(x):return hashlib.sha256(x.detach().contiguous().cpu().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
def verify_sources():
 m=read(OUT/'source_manifest.json')
 for path,h in m['files'].items():assert sha(path if path.startswith('/') else ROOT/path)==h,path
 return digest(m)

def processor_cpu():
 from transformers import AutoProcessor,AutoConfig
 p=AutoProcessor.from_pretrained(MODEL,local_files_only=True);c=AutoConfig.from_pretrained(MODEL,local_files_only=True)
 p.patch_size=c.vision_config.patch_size;p.num_additional_image_tokens=1;p.vision_feature_select_strategy=c.vision_feature_select_strategy;p.tokenizer.padding_side='left'
 return p

def prepare_input(processor,s):
 import torch
 from PIL import Image
 with Image.open(s['image_path']) as source:image=source.convert('RGB')
 conversation=[{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}]
 prompt=processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
 x=processor(images=image,text=prompt,return_tensors='pt').to(dtype=torch.float16)
 m=s['input_mapping'];ids=x['input_ids'][0].tolist();assert ids==m['prompt_ids'] and [i for i,v in enumerate(ids) if v==32000]==m['image']
 assert len(m['image'])==576 and m['Pminus1']==len(ids)-1 and m['R']==list(range(m['image'][-1]+1,len(ids)-1))
 binding=dict(input_hashes={k:tensor_hash(v) for k,v in x.items()},prompt_ids=ids,Q=m['parts']['Q'],ASSISTANT=m['parts']['ASSISTANT'],Hint=m['parts']['Hint'],Options=m['parts']['Options'],writers=m['R'],R=m['R'],Pminus1=m['Pminus1'])
 return x,binding
