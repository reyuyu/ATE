"""Native Qwen chat input binding; original dataset and gold files stay read-only."""
from pathlib import Path
import hashlib,json
import torch
from PIL import Image
from vlm_research.qwen3vl_early_windows.experiment import tensor_digest
from vlm_research.qwen3vl_smoke.common import question_positions

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
verified={}
def prepare(processor,s):
 p=Path(s['image_path']);signature=(p.stat().st_size,p.stat().st_mtime_ns)
 if verified.get(str(p))!=signature:
  assert hashlib.sha256(p.read_bytes()).hexdigest()==s['image_sha256'],('Image hash mismatch',s['key']);verified[str(p)]=signature
 with Image.open(p) as handle:image=handle.convert('RGB')
 body=s['body']
 messages=[dict(role='user',content=[dict(type='image',image=str(p)),dict(type='text',text=body)])]
 prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 inputs=processor(text=[prompt],images=[image],return_tensors='pt',return_offsets_mapping=True)
 offsets=inputs.pop('offset_mapping')[0].tolist();ids=inputs['input_ids'][0].tolist()
 visual=[i for i,v in enumerate(ids) if v==processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
 grid=inputs['image_grid_thw'][0].tolist();count=grid[0]*grid[1]*grid[2]//processor.image_processor.merge_size**2
 assert len(visual)==count and visual==list(range(visual[0],visual[-1]+1)) and visual[-1]<len(ids)-1
 expanded=prompt.replace(processor.image_token,processor.image_token*count)
 assert processor.tokenizer(expanded)['input_ids']==ids
 layout=dict(prompt=prompt,input_ids=ids,input_length=len(ids),p_minus_1=len(ids)-1,image_positions=visual,image_tokens=count,question_positions=question_positions(offsets,expanded,body),image_grid_thw=grid,input_tensor_hashes={k:tensor_digest(v) for k,v in inputs.items()})
 assert inputs['attention_mask'].bool().all() and len(inputs['input_ids'])==1
 return inputs,layout
