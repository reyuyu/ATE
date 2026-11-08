"""Local public dataset bridge. Gold is available only through load_answers()."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path('/home/featurize/data/ate-recovery')
REPO=Path('/home/featurize/work/ATE')
OUT=REPO/'vlm_research/results/featurize_new3090_recovery'
MODEL=ROOT/'models/llava-1.5-7b-hf'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dataset_dir(name):
 path=ROOT/'normalized'/name
 if not path.resolve().is_relative_to((ROOT/'normalized').resolve()):raise ValueError(name)
 return path
def iter_inputs(name,verify=True):
 d=dataset_dir(name)
 if verify:
  expected=json.loads((OUT/'dataset_acceptance.json').read_text())[name]['inputs_sha256']
  if sha(d/'inputs.jsonl')!=expected:raise ValueError('Input SHA256 changed after CPU acceptance')
 forbidden={'answer','answers','gold','label','lecture','solution','correct_answer'}
 for l in (d/'inputs.jsonl').open():
  row=json.loads(l)
  if forbidden.intersection(row):raise ValueError('Gold field in inputs')
  yield row
def load_answers(name):
 d=dataset_dir(name);expected=json.loads((OUT/'dataset_acceptance.json').read_text())[name]['answers_sha256']
 if sha(d/'answers.json')!=expected:raise ValueError('Gold SHA256 changed after CPU acceptance')
 return json.loads((d/'answers.json').read_text())
def processor_cpu():
 from transformers import AutoProcessor,AutoConfig
 p=AutoProcessor.from_pretrained(MODEL,local_files_only=True);c=AutoConfig.from_pretrained(MODEL,local_files_only=True)
 p.patch_size=c.vision_config.patch_size;p.num_additional_image_tokens=1;p.vision_feature_select_strategy=c.vision_feature_select_strategy;p.tokenizer.padding_side='left'
 return p
def prepare_input(processor,row,benchmark):
 from PIL import Image
 sys.path.insert(0,str(REPO))
 from vlm_research.seven_benchmark_quality.common import prompt
 b='POPE' if benchmark.startswith('POPE') else 'TextVQA' if benchmark.startswith('TextVQA') else benchmark.removesuffix('-paper')
 text=row.get('user_text') or prompt(row,b)
 rendered=row.get('prompt') or processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=text)])],add_generation_prompt=True)
 with Image.open(row['image_path']) as im:batch=processor(images=im.convert('RGB'),text=rendered,return_tensors='pt')
 return batch,rendered
if __name__=='__main__':
 import os
 os.environ['CUDA_VISIBLE_DEVICES']=''
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch
 p=processor_cpu();evidence={}
 for name in json.loads((OUT/'dataset_acceptance.json').read_text()):
  rows=list(iter_inputs(name));gold=load_answers(name);assert {r['question_id'] for r in rows}==set(gold)
  x,rendered=prepare_input(p,rows[0],name);assert x['pixel_values'].shape==(1,3,336,336)
  assert int((x['input_ids']==32000).sum())==576
  evidence[name]=dict(rows=len(rows),gold_bijection=True,pixel_shape=list(x['pixel_values'].shape))
 assert not torch.cuda.is_initialized()
 (OUT/'loader_acceptance.json').write_text(json.dumps(dict(success=True,cuda_initialized=False,datasets=evidence),indent=2)+'\n')
 print(json.dumps(evidence,indent=2))
