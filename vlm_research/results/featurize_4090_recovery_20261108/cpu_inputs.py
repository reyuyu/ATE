import os
os.environ.update(USE_TF='0',USE_FLAX='0',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
from pathlib import Path
import sys,json,torch,hashlib
O=Path(__file__).resolve().parent;sys.path[:0]=[str(O.parents[2]),str(O)]
from transformers import AutoProcessor,LlamaTokenizer
from vlm_research.information_horizon_alignment.engine import paper_pixels,image_ids
from vlm_research.information_horizon_alignment.common import tensor_sha,digest as paper_digest
from protocol_helpers import official_functions,make_inputs
from smoke_helpers import hashes,digest
M='/home/featurize/ate-local-models/llava-1.5-7b-hf';p=AutoProcessor.from_pretrained(M,local_files_only=True);t=LlamaTokenizer.from_pretrained(M,local_files_only=True,legacy=True);c,f=official_functions();cases=json.loads((O/'smoke_cases.json').read_text());checks=[]
s=cases['gqa_official'];x,prompt=make_inputs(s,'C',p,t,c,f);assert prompt==s['prompt'] and x['input_ids'][0].tolist()==s['input_ids'] and hashes(x)==s['input_hashes'];checks.append({'protocol':'GQA-official-aligned-vicuna-pad','qid':s['question_id'],'exact_frozen_input_tensors':True})
for c in cases['paper']:
 s=c['sample'];r=c['reference_3090'];pixels=paper_pixels(s['image_path'],p.image_processor);raw=image_ids(s['prompt'],p.tokenizer,-200);ids=[v for z in raw for v in ([32000]*576 if z==-200 else [z])]
 assert tensor_sha(pixels)==r['pixel_values_sha256'],(s['benchmark'],'pixels',tensor_sha(pixels),r['pixel_values_sha256'])
 assert paper_digest(ids)==r['input_ids_sha256'] and paper_digest(s['prompt'])==r['prompt_sha256']
 checks.append({'protocol':s['benchmark']+'-paper','qid':s['question_id'],'exact_frozen_prompt_ids_pixels':True})
(O/'input_cpu_preflight.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2)+'\n');print('CPU_INPUT_PREFLIGHT_PASS',checks)
