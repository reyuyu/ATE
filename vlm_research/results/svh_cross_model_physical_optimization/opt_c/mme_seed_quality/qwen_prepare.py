"""Actual Processor audit of every source image, original versus 2048 cap."""
import csv
import gc
import gzip
import inspect
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from .common import *

MAX_IMAGE_TOKENS=2048

def audit_images():
    torch.set_num_threads(1)
    original=AutoProcessor.from_pretrained(MODELS['qwen'],local_files_only=True)
    capped=AutoProcessor.from_pretrained(MODELS['qwen'],local_files_only=True)
    assert original.image_processor.patch_size==16 and original.image_processor.merge_size==2
    capped.image_processor.size={**dict(capped.image_processor.size),'longest_edge':MAX_IMAGE_TOKENS*32*32}
    source=Path(inspect.getfile(type(original.image_processor)))
    atomic(OUT/'qwen_processor_2048.json',dict(max_image_tokens=MAX_IMAGE_TOKENS,
        original=original.image_processor.to_dict(),capped=capped.image_processor.to_dict(),
        implementation_path=str(source),implementation_sha256=sha(source),
        patch_size=16,merge_size=2,temporal_patch_size=original.image_processor.temporal_patch_size,
        prompt='unchanged original Qwen3-VL HF chat template; question only; no OCR',
        output_gate='Actual merged image positions <=2048 for every image; no token truncation'))
    samples=read(OUT/'sample_manifest.json');images=read(OUT/'image_manifest.json')
    cache=PRIVATE/'qwen_cpu_image_audit';cache.mkdir(exist_ok=True)
    message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':'Image preprocessing audit.'}]}]
    prompt=original.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
    assert capped.apply_chat_template(message,tokenize=False,add_generation_prompt=True)==prompt
    image_id=original.tokenizer.convert_tokens_to_ids(original.image_token)
    data=[]
    for i,(path,meta) in enumerate(sorted(images.items())):
        dst=cache/(digest(path)+'.json')
        if dst.exists():
            row=read(dst);assert row['image_sha256']==meta['sha256']
        else:
            row=dict(image_path=path,image_sha256=meta['sha256'],original_width=meta['width'],original_height=meta['height'])
            with Image.open(path) as im:image=im.convert('RGB')
            for name,processor in [('original',original),('cap2048',capped)]:
                inputs=processor(images=image,text=prompt,return_tensors='pt')
                grid=inputs['image_grid_thw'][0].tolist()
                count=grid[0]*grid[1]*grid[2]//processor.image_processor.merge_size**2
                pos=(inputs['input_ids'][0]==image_id).nonzero().flatten().tolist()
                assert len(pos)==count and pos==list(range(pos[0],pos[-1]+1))
                assert torch.isfinite(inputs['pixel_values']).all() and inputs['pixel_values'].numel()>0
                if name=='cap2048':assert count<=MAX_IMAGE_TOKENS,(path,count)
                row[name+'_tokens']=count;row[name+'_grid_thw']=grid
                row[name+'_image_positions']=[pos[0],pos[-1]]
                row[name+'_pixel_values_sha256']=tensor_sha(inputs['pixel_values'])
                row[name+'_pixel_shape']=list(inputs['pixel_values'].shape)
                h,w=grid[1]*16,grid[2]*16
                row[name+'_resized_height']=h;row[name+'_resized_width']=w
                row[name+'_aspect_relative_error']=abs((w/h)/(meta['width']/meta['height'])-1)
                del inputs
            del image
            atomic(dst,row)
        data.append(row)
        if (i+1)%100==0:print('QWEN_CPU_IMAGES',i+1,'/',len(images),flush=True)
    assert len(data)==len(images) and max(r['cap2048_tokens'] for r in data)<=2048
    atomic(OUT/'qwen_image_processor_audit.json',data)
    with (OUT/'qwen_visual_token_statistics.csv').open('w',newline='') as f:
        columns=list(data[0]);w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in data)
    imap={r['image_path']:r for r in data};bindings={};summary={}
    for bench in ('MME','SEED'):
        paths={s['image_path'] for s in samples if s['benchmark']==bench}
        summary[bench]={}
        for name in ('original','cap2048'):
            values=np.array([imap[p][name+'_tokens'] for p in paths])
            summary[bench][name]=dict(images=len(values),min=int(values.min()),
                p50=float(np.percentile(values,50)),p90=float(np.percentile(values,90)),
                p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)),
                max=int(values.max()),above_2048=int((values>2048).sum()))
    for s in samples:
        message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':s['question']}]}]
        prompt=capped.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
        ir=imap[s['image_path']];count=ir['cap2048_tokens']
        expanded=prompt.replace(capped.image_token,capped.image_token*count)
        ids=capped.tokenizer(expanded)['input_ids'];assert ids.count(image_id)==count
        b=dict(prompt=prompt,input_ids=ids,image_tokens=count,image_grid_thw=ir['cap2048_grid_thw'],
            pixel_values_sha256=ir['cap2048_pixel_values_sha256'])
        if s['benchmark']=='SEED':
            cands=[]
            for answer in s['choices']:
                full=capped.tokenizer(expanded+answer)['input_ids']
                assert full[:len(ids)]==ids,(s['key'],'candidate boundary')
                tail=full[len(ids):];assert tail;cands.append(tail)
            b['candidate_token_ids']=cands
        bindings[s['key']]=b
    (OUT/'qwen_input_bindings.json.gz').write_bytes(gzip.compress(json.dumps(bindings,sort_keys=True).encode(),mtime=0))
    atomic(OUT/'qwen_processor_validation.json',dict(status='PASSED',at=now(),statistics=summary,
        actual_processor_every_image=True,no_images_removed=True,actual_cap_exceedances=0,
        image_audit_sha256=sha(OUT/'qwen_image_processor_audit.json'),
        bindings_sha256=sha(OUT/'qwen_input_bindings.json.gz'),processor_sha256=sha(OUT/'qwen_processor_2048.json')))
    print('QWEN_PROCESSOR_AUDIT_PASSED',summary,flush=True)

if __name__=='__main__':audit_images()
