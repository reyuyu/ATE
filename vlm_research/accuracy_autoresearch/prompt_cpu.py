"""Input-only checks for the two predeclared receiver structures."""
import torch
from transformers import AutoProcessor
from .common import *


def main():
    assert not torch.cuda.is_initialized()
    processor=AutoProcessor.from_pretrained('/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9',local_files_only=True)
    pool=read(OUT/'sample_manifest.json'); selected=read(OUT/'split_manifest.json')['coarse']; rows=[]
    for k in selected:
        s=pool[k]
        for variant in ('repeat-after-instruction','repeat-before-instruction'):
            text=prompt_text(s,variant)
            chat=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=text)])],add_generation_prompt=True)
            expanded=chat.replace('<image>','<image>'*576)
            enc=processor.tokenizer(expanded,return_offsets_mapping=True); ids=enc['input_ids']
            image=[i for i,x in enumerate(ids) if x==32000]
            assert len(image)==576 and ids[-1]==s['input_mapping']['prompt_ids'][-1]==29901
            assert text.count(s['question'])==2
            begin=expanded.rindex('ASSISTANT')
            assistant=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>begin and a<begin+9]
            assert [ids[i] for i in assistant]==[319,1799,9047,13566] and max(assistant)<len(ids)-1
            assert all(image[-1]<i<len(ids)-1 for i in assistant)
            rows.append(dict(sample=k,variant=variant,prompt_tokens=len(ids),
                added_tokens=len(ids)-len(s['input_mapping']['prompt_ids']),image_tokens=576,
                last_original_prompt_query=len(ids)-1,assistant_positions=assistant,passed=True))
    atomic(OUT/'prompt_structure_cpu.json',dict(status='PASS',at=now(),checks=rows,
        no_cuda_initialized=not torch.cuda.is_initialized(),selection_addendum_sha256=sha(OUT/'selection_addendum_before_prompt_search.json')))
    print('PASS',len(rows),'dynamic token mappings; no model loading')


if __name__=='__main__': main()
