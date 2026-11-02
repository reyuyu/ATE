import os,subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from query_role_attention.common import ROOT,MODEL,REVISION,TASKS,SEED,read,write,sha256,digest,csv_write
from visual_write_window.common import sources as previous_sources,MANIFEST_SHA
from relay_source_causality.common import hardware,audit_history as audit_native
from visual_write_window.analyze import validate as validate_historical
from first_answer_attention.engine import first_metrics

BASE='6b9467d6fb4e0f8535040809bee60397ca798ea8'
OLD=ROOT/'results/visual_write_window_240'
NATIVE=ROOT/'results/query_role_attention_boundary_240_formal'
OUT=ROOT/'results/sparse_visual_writing_frontier_240'
RAW=Path('/data/lk/ATE-resources/experiments/sparse_visual_writing_frontier_240')
NEW=['Keep-0-15','Keep-0-11','Keep-4-15','Keep-4-11','Keep-8-15','Keep-0-3+8-11']
BASELINES=['All-Writer','Drop-W3','Only-W2','No-Writer']
KEEP={'All-Writer':list(range(32)),'Drop-W3':list(range(12))+list(range(16,32)),'Only-W2':list(range(8,12)),'No-Writer':[],
      'Keep-0-15':list(range(16)),'Keep-0-11':list(range(12)),'Keep-4-15':list(range(4,16)),
      'Keep-4-11':list(range(4,12)),'Keep-8-15':list(range(8,16)),'Keep-0-3+8-11':list(range(4))+list(range(8,12))}
CONDITIONS=BASELINES+NEW
SAME_COUNT=[('Keep-0-11','Keep-4-15'),('Keep-4-11','Keep-8-15'),('Keep-4-11','Keep-0-3+8-11'),('Keep-8-15','Keep-0-3+8-11')]

def writer_layers(c):return KEEP[c]
def layer_mode(c,layer):return 'Native' if c=='Native' else ('All-Writer' if layer in writer_layers(c) else 'No-Writer')
def sources():
    return {**previous_sources(),**{str(p.relative_to(ROOT)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}

def checkpoint(s,c):return OLD/f"checkpoints/{s['task']}__{s['question_id']}/{c}.json"
def history_artifacts():
    c=read(OLD/'completion.json');assert c['status']=='COMPLETE'
    files={**c['artifacts'],'completion.json':sha256(OLD/'completion.json')}
    for name,h in files.items():assert sha256(OLD/name)==h,name
    return files

def audit_history():
    p=read(OLD/'protocol.json');assert p['code_commit']=='3b1950f9780676bac874d0a97b53341ab8c57ec7' and p['sources']==previous_sources()
    assert p['model_revision']==REVISION and p['generation']['logits_to_keep']==1
    assert read(OLD/'validation.json')['passed'] and read(OLD/'delivery_read_only_validation.json')['passed']
    assert sha256(OLD/'sample_manifest.json')==MANIFEST_SHA==sha256(NATIVE/'sample_manifest.json')
    assert sha256(OLD/'input_mappings.json')==sha256(NATIVE/'input_mappings.json')
    samples=read(OLD/'sample_manifest.json')['samples'];mapping=read(OLD/'input_mappings.json')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);hashes={};raws={};scores={c:0 for c in BASELINES}
    for i,s in enumerate(samples):
        assert sha256(s['image_path'])==s['image_sha256'];reference=None
        for c in BASELINES:
            f=checkpoint(s,c);r=read(f);g,v=validate_historical(r,s,tok,digest(p),mapping[i]['input_hashes'])
            assert g['query_layout']==mapping[i]['role_layout'] and g['writer_layers']==KEEP[c]
            if c=='All-Writer':reference=v
            else:assert all(g[k]==value for k,value in first_metrics(reference,v).items())
            hashes[str(f.relative_to(OLD))]=sha256(f);raws[r['raw']['path']]=r['raw']['sha256'];scores[c]+=g['correct']
    assert scores==dict(zip(BASELINES,[202,205,188,153]))
    native=audit_native();assert native['correct']['Full']==206
    # This verifies both native Full and its old paired R control, without pooling either into the frontier.
    return dict(passed=True,questions=240,reused_main_records=960,correct=scores,history_artifacts=history_artifacts(),checkpoint_hashes=hashes,raw_hashes=raws,
        native_full=dict(passed=True,records=240,correct=206,accuracy_pct=100*206/240,path=str(NATIVE),audit=native,
            distinction='Native Full allows P-1/decode direct image access. It is an external protocol reference, never the All-Writer baseline.'),new_generations=0)

def select_gpu():
    state=hardware();busy={s.split(',')[0].strip() for s in state['compute_processes'].splitlines()};idle=[]
    for line in state['table'].splitlines():
        index,uuid,name,used,free,util=[v.strip() for v in line.split(',')]
        if uuid not in busy and int(used)<200 and int(free)>22000 and int(util)==0:idle.append(dict(physical_index=int(index),uuid=uuid,name=name,logical='cuda:0'))
    state['selected']=idle[0] if idle else None;write(OUT/'gpu_selection.json',state)
    if not idle:raise RuntimeError('No idle GPU. Stop without a wait job.')
    return idle[0]

def preflight(gpu):
    state=hardware();line=next(s for s in state['table'].splitlines() if s.split(',')[1].strip()==gpu['uuid']);v=[s.strip() for s in line.split(',')]
    busy=[s for s in state['compute_processes'].splitlines() if s.split(',')[0].strip()==gpu['uuid']]
    state.update(selected=gpu,passed=int(v[3])<200 and int(v[4])>22000 and int(v[5])==0 and not busy);write(OUT/'gpu_preflight.json',state)
    assert state['passed'] and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid'],'GPU no longer idle; stop'

def verify():
    p=read(OUT/'protocol.json');assert p['sources']==sources() and p['conditions']==KEEP
    for name,h in p['bindings'].items():assert sha256(OUT/name)==h,name
    for name,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(name)==h,name
    for name,v in p['model_files'].items():
        st=Path(name).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],name
    return p
