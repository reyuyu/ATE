import math
import json
import os
from pathlib import Path
import subprocess

from query_role_attention.common import ROOT, MODEL, REVISION, TASKS, SEED, read, write, sha256, digest, csv_write
from visual_write_window.common import sources as inherited_sources
from relay_source_causality.common import hardware

BASE = '6b9467d6fb4e0f8535040809bee60397ca798ea8'
OUT = ROOT/'results/gqa3000_visual_writing_necessity'
POOL = ROOT/'results/rq2_visual_sufficiency_full3000/sample_manifest.json'
HISTORY = ROOT/'results/visual_write_window_240'
CONDITIONS = ['All-Writer', 'No-Writer']+[f'Drop-L{k}' for k in range(32)]


def progress_write(path,value):
    path=Path(path);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temp.open('x') as stream:
            json.dump(value,stream,allow_nan=False,indent=2)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def writer_layers(condition):
    if condition=='All-Writer':return list(range(32))
    if condition=='No-Writer':return []
    if condition not in CONDITIONS:raise ValueError('Unfrozen condition '+condition)
    return [i for i in range(32) if i!=int(condition[6:])]


def source_hashes():
    return {**inherited_sources(),**{str(p.relative_to(ROOT)):sha256(p) for p in
        sorted(Path(__file__).parent.glob('*.py')) if p.name in ['__init__.py','common.py','engine.py','run.py','tests.py']}}


def shards(samples):
    groups={}
    for s in samples:groups.setdefault(s['image_sha256'],[]).append(s['question_id'])
    result=[[],[]];owners={}
    for image,ids in sorted(groups.items(),key=lambda p:(-len(p[1]),p[0])):
        index=min(range(2),key=lambda i:(len(result[i]),i))
        result[index].extend(sorted(ids));owners[image]=index
    assert sorted(q for r in result for q in r)==sorted(s['question_id'] for s in samples)
    assert all(s['question_id'] in result[owners[s['image_sha256']]] for s in samples)
    return result,owners


def gold_tokens(processor,sample):
    chat=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),
        dict(type='text',text=sample['prompt_text'])])],tokenize=False,add_generation_prompt=True)
    assert digest(chat)==sample['input_mapping']['chat_prompt_sha256']
    prefix=processor.tokenizer.encode(chat,add_special_tokens=True)
    joined=processor.tokenizer.encode(chat+' '+sample['answer'],add_special_tokens=True)
    assert joined[:len(prefix)]==prefix,'Gold continuation retokenizes the original prompt boundary'
    gold=joined[len(prefix):]
    assert gold and not any(t in processor.tokenizer.all_special_ids for t in gold)
    expanded=[]
    for t in prefix:expanded.extend([t]*576 if t==32000 else [t])
    assert expanded==sample['input_mapping']['prompt_ids']
    return dict(token_ids=gold,answer=sample['answer'],separator=' ',joint_tokenization_prefix_exact=True,
        no_extra_BOS=True,EOS_excluded=True,chat_sha256=digest(chat))


def preflight(gpu):
    state=hardware();uuid=gpu['uuid']
    v=next(r for r in state['table'].splitlines() if r.split(',')[1].strip()==uuid).split(',')
    busy=[r for r in state['compute_processes'].splitlines() if r.split(',')[0].strip()==uuid]
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==uuid
    assert int(v[3])<200 and int(v[4])>22000 and int(v[5])==0 and not busy,'GPU no longer idle; stop'
    return dict(passed=True,selected=gpu,**state)


def verify():
    p=read(OUT/'protocol.json')
    assert p['sources']==source_hashes(),'Frozen GPU source changed'
    for name,h in p['bindings'].items():assert sha256(OUT/name)==h,name
    for name,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(name)==h,name
    for name,info in p['model_files'].items():
        st=Path(name).stat();assert st.st_size==info['bytes'] and st.st_mtime_ns==info['mtime_ns'],name
    assert p['conditions']=={c:writer_layers(c) for c in CONDITIONS}
    return p
