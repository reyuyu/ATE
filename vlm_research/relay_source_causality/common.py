import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from query_role_attention.common import ROOT, MODEL, REVISION, TASKS, SEED, LAYERS, read, write, sha256, digest, csv_write
from query_role_formal.common import sources as historical_sources
from first_answer_attention.engine import first_metrics
from paper_rq1.scorers import gqa_prediction

BASE='8019c6009bb1d8c9f2dc3ffd8c18bf4dfb8c1fd1'
OLD=ROOT/'results/query_role_attention_boundary_240_formal'
OUT=ROOT/'results/relay_source_causality_240'
RAW=Path('/data/lk/ATE-resources/experiments/relay_source_causality_240')
BASELINES=['Full','R-Bypass']
NEW=['R-Image-OFF','R-Text-OFF']
CONDITIONS=BASELINES+NEW
SWAP=ROOT/'results/rq2_visual_text_handoff_240/image_swap_manifest.json'
MANIFEST_SHA='258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c'

def sources():
    return {**historical_sources(), **{str(p.relative_to(ROOT)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}

def hardware():
    fields=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    return dict(table=fields,compute_processes=apps)

def select_gpu():
    state=hardware();busy={line.split(',')[0].strip() for line in state['compute_processes'].splitlines()}
    candidates=[]
    for line in state['table'].splitlines():
        index,uuid,name,used,free,util=[x.strip() for x in line.split(',')]
        if uuid not in busy and int(used)<200 and int(free)>22000 and int(util)==0 and '3090' in name:
            candidates.append(dict(physical_index=int(index),uuid=uuid,name=name,logical='cuda:0'))
    state['selected']=candidates[0] if candidates else None
    write(OUT/'gpu_selection.json',state)
    if not candidates: raise RuntimeError('No truly idle GPU; stop without polling')
    return state['selected']

def preflight(gpu):
    state=hardware();uuid=gpu['uuid']
    row=next(x for x in state['table'].splitlines() if x.split(',')[1].strip()==uuid)
    fields=[x.strip() for x in row.split(',')]
    apps=[x for x in state['compute_processes'].splitlines() if x.split(',')[0].strip()==uuid]
    state.update(selected=gpu,passed=int(fields[3])<200 and int(fields[4])>22000 and int(fields[5])==0 and not apps)
    write(OUT/'gpu_preflight.json',state)
    assert state['passed'],'GPU no longer idle; stop without polling'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==uuid
    return state

def history_hashes():
    complete=read(OLD/'completion.json')
    expected={**complete['artifacts'],'completion.json':sha256(OLD/'completion.json')}
    for rel,h in expected.items(): assert sha256(OLD/rel)==h,rel
    return expected

def checkpoint(sample,condition):
    return OLD/f"checkpoints/{sample['task']}__{sample['question_id']}/{condition}.json"

def audit_history():
    p=read(OLD/'protocol.json');v=read(OLD/'validation.json');m=read(OLD/'sample_manifest.json')
    assert p['code_commit']=='cc0d8526cea480bc8d01148a7bf50452675bbf12'
    assert v['passed'] and read(OLD/'completion.json')['formal_generations']==960
    assert p['generation']['logits_to_keep']==1 and p['model_revision']==REVISION
    assert p['sources']==historical_sources()
    assert sha256(OLD/'sample_manifest.json')==MANIFEST_SHA
    assert len(m['samples'])==240 and len({s['question_id'] for s in m['samples']})==240
    assert len({s['image_sha256'] for s in m['samples']})==240
    assert all(sum(s['task']==t for s in m['samples'])==40 for t in TASKS)
    mapping=read(OLD/'input_mappings.json');binding=digest(p)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    raw_hashes={};cp_hashes={};correct={c:0 for c in BASELINES}
    for i,s in enumerate(m['samples']):
        assert sha256(s['image_path'])==s['image_sha256']
        for c in BASELINES:
            path=checkpoint(s,c);row=read(path);g=row['generation']
            assert sha256(path)==v['checkpoint_hashes'][str(path.relative_to(OLD))]
            assert row['binding']==binding and row['sample_sha256']==digest(s)
            assert row['condition']==c and row['gold']==s['answer'] and row['question_id']==s['question_id']
            assert row['input_hashes']==mapping[i]['input_hashes'] and g['query_layout']==mapping[i]['role_layout']
            assert row['result_sha256']==digest(g)
            assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text']
            assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'])
            assert sha256(row['raw']['path'])==row['raw']['sha256']==v['raw_hashes'][row['raw']['path']]
            with np.load(row['raw']['path']) as a:logits=torch.from_numpy(a['first_logits'].copy())
            assert torch.isfinite(logits).all() and int(logits.argmax())==g['token_ids'][0]
            if c=='Full':full=logits
            assert all(g[k]==value for k,value in first_metrics(full,logits).items())
            assert digest(logits.tolist())==g['first_logits_sha256']
            cp_hashes[str(path.relative_to(OLD))]=sha256(path);raw_hashes[row['raw']['path']]=row['raw']['sha256']
            correct[c]+=g['correct']
    assert correct==dict(zip(BASELINES,[206,138]))
    return dict(passed=True,questions=240,baseline_records=480,correct=correct,checkpoint_hashes=cp_hashes,raw_hashes=raw_hashes,history_artifacts=history_hashes(),new_generations=0)

def verify():
    p=read(OUT/'protocol.json');old=read(OLD/'protocol.json')
    assert p['sources']==sources()
    assert p['history_artifacts']==history_hashes()
    for n,h in p['bindings'].items():assert sha256(OUT/n)==h,n
    for path,h in {**old['transformers_sources'],**old['scorer_sources']}.items():assert sha256(path)==h,path
    for path,info in p['model_files'].items():
        st=Path(path).stat();assert st.st_size==info['bytes'] and st.st_mtime_ns==info['mtime_ns'],path
    return p
