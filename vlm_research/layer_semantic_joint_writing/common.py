import os,subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from query_role_attention.common import ROOT,MODEL,REVISION,TASKS,read,write,sha256,digest,csv_write
from sparse_visual_writing_frontier.common import sources as layer_sources,select_gpu as unused_select
from semantic_visual_writers.common import sources as semantic_sources
from semantic_visual_writers.analyze import validate as semantic_validate
from sparse_visual_writing_frontier.analyze import validate_new as layer_validate
from visual_write_window.analyze import validate as all_validate,raw
from visual_write_window.common import MANIFEST_SHA
from relay_source_causality.common import hardware

BASE='2a45dcdfb6f203037472801bd99f9731e9a8fffb'
SEM_COMMIT='9687b298efcdca651cc5446b5cb07da4a2b4e37c'
SEMROOT=Path('/data/lk/ATE-worktrees/semantic-visual-writers-240')
SEM=SEMROOT/'vlm_research/results/semantic_visual_writers_240'
LAYER=ROOT/'results/sparse_visual_writing_frontier_240'
ALL=ROOT/'results/visual_write_window_240'
OUT=ROOT/'results/layer_semantic_joint_writing_240'
RAW=Path('/data/lk/ATE-resources/experiments/layer_semantic_joint_writing_240')
NEW=['Joint-16','Joint-12','Joint-8']
MATCH={'Joint-16':'Keep-0-15','Joint-12':'Keep-0-11','Joint-8':'Keep-4-11'}
BASELINES=['All-Writer','Semantic-k','Keep-0-15','Keep-0-11','Keep-4-11']
CONDITIONS=BASELINES+NEW
KEEP={'All-Writer':list(range(32)),'Semantic-k':list(range(32)),
      'Keep-0-15':list(range(16)),'Keep-0-11':list(range(12)),'Keep-4-11':list(range(4,12)),
      'Joint-16':list(range(16)),'Joint-12':list(range(12)),'Joint-8':list(range(4,12))}
SMOKE_IDS=['00150746','11697551','11127723']
SEED=20261103

def key(s):return s['task']+'__'+s['question_id']
def sources():
    return {**layer_sources(),**semantic_sources(),**{str(p.relative_to(ROOT)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
def selected(c,layout,spec):
    assert c in CONDITIONS+['Native']
    return list(spec['selection']['writers']['Semantic-k']) if c in NEW+['Semantic-k'] else list(layout['R'])
def writers(c,layer,layout,spec):
    return selected(c,layout,spec) if c=='Native' or layer in KEEP[c] else []
def historical_path(s,c):
    parent=ALL if c=='All-Writer' else SEM if c=='Semantic-k' else LAYER
    return parent/'checkpoints'/key(s)/(c+'.json')
def reused_path(s,c):return OUT/'reused_checkpoints'/key(s)/(c+'.json')
def validate_reused(row,s,c,tok,mapping,spec,protocols):
    if c=='Semantic-k':g,v=semantic_validate(row,s,spec,mapping['input_hashes'],protocols['semantic'],tok)
    elif c=='All-Writer':g,v=all_validate(row,s,tok,digest(protocols['all']),mapping['input_hashes'])
    else:g,v=layer_validate(row,s,tok,digest(protocols['layer']),mapping['input_hashes'])
    assert g['query_layout']==mapping['role_layout']
    return g,v

def audit_history():
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    paths={'layer':LAYER,'semantic':SEM,'all':ALL};protocols={k:read(p/'protocol.json') for k,p in paths.items()}
    artifacts={};records={};raws={};correct={c:0 for c in BASELINES}
    for label,p in paths.items():
        co=read(p/'completion.json');assert co['status']=='COMPLETE' and read(p/'validation.json')['passed']
        artifacts[label]={**co['artifacts'],'completion.json':sha256(p/'completion.json')}
        for n,h in artifacts[label].items():assert sha256(p/n)==h,(label,n)
        pr=protocols[label]
        for n,h in pr['sources'].items():assert sha256(ROOT/n)==h,(label,n)
        assert pr['model_revision']==REVISION and pr['dtype']=='float16' and pr['attention']=='eager' and pr['transformers']=='4.49.0'
        assert pr['generation']['logits_to_keep']==1 and pr['generation']['max_new_tokens']==32
        assert sha256(p/'sample_manifest.json')==MANIFEST_SHA and sha256(p/'input_mappings.json')==sha256(ALL/'input_mappings.json')
        for field in ['model_files','transformers_sources','scorer_sources','generation']:assert pr[field]==protocols['all'][field],(label,field)
    specs=read(SEM/'token_group_manifest.json');samples=read(ALL/'sample_manifest.json')['samples'];mapping=read(ALL/'input_mappings.json')
    assert len(samples)==240 and len({s['image_sha256'] for s in samples})==240
    for i,s in enumerate(samples):
        spec=specs['main'][key(s)];layout=mapping[i]['role_layout'];sel=spec['selection']
        assert spec['selection_binding']==digest(sel) and spec['question']==s['question']
        assert spec['relay_positions']==layout['R'] and spec['prompt_ids_sha256']==digest(s['input_mapping']['prompt_ids'])
        assert 0<len(sel['writers']['Semantic-k'])==sel['k'] and set(sel['writers']['Semantic-k'])<=set(layout['R'])
        assert sha256(s['image_path'])==s['image_sha256']
        for c in BASELINES:
            p=historical_path(s,c);row=read(p);g,v=validate_reused(row,s,c,tok,mapping[i],spec,protocols)
            records[key(s)+'/'+c+'.json']=dict(path=str(p),sha256=sha256(p),commit=SEM_COMMIT if c=='Semantic-k' else BASE,
                raw=row['raw'],correct=g['correct'])
            correct[c]+=g['correct'];raws[row['raw']['path']]=row['raw']['sha256']
        # Semantic experiment's reused All must be the same result, not just the same accuracy.
        sa=read(SEM/'checkpoints'/key(s)/'All-Writer.json');a=read(historical_path(s,'All-Writer'))
        assert sa['generation']['token_ids']==a['generation']['token_ids'] and sa['raw']==a['raw']
    assert correct==dict(zip(BASELINES,[202,202,200,201,198]))
    budget=[len(specs['main'][key(s)]['selection']['writers']['Semantic-k'])/len(mapping[i]['role_layout']['R']) for i,s in enumerate(samples)]
    return dict(passed=True,records=records,raw_hashes=raws,artifact_hashes=artifacts,protocols=protocols,correct=correct,
        semantic_selection_sha256=sha256(SEM/'token_group_manifest.json'),mean_token_ratio=float(np.mean(budget)),
        pooled_token_ratio=sum(specs['main'][key(s)]['selection']['k'] for s in samples)/sum(len(m['role_layout']['R']) for m in mapping),
        new_generations=0,reused_records=1200,selection_recomputed=False)

def select_gpu():
    state=hardware();busy={s.split(',')[0].strip() for s in state['compute_processes'].splitlines()};idle=[]
    for line in state['table'].splitlines():
        index,uuid,name,used,free,util=[v.strip() for v in line.split(',')]
        if uuid not in busy and int(used)<200 and int(free)>22000 and int(util)==0:idle.append(dict(physical_index=int(index),uuid=uuid,name=name,logical='cuda:0'))
    state['selected']=idle[0] if idle else None;write(OUT/'gpu_selection.json',state)
    assert idle,'No idle GPU: stop without waiting task';return idle[0]
def preflight(gpu):
    state=hardware();v=[s.strip() for s in next(s for s in state['table'].splitlines() if s.split(',')[1].strip()==gpu['uuid']).split(',')]
    busy=[s for s in state['compute_processes'].splitlines() if s.split(',')[0].strip()==gpu['uuid']]
    state.update(selected=gpu,passed=int(v[3])<200 and int(v[4])>22000 and int(v[5])==0 and not busy);write(OUT/'gpu_preflight.json',state)
    assert state['passed'] and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid'],'GPU occupied; stop'
def verify():
    p=read(OUT/'protocol.json');assert p['sources']==sources() and p['conditions']==KEEP
    for n,h in p['bindings'].items():assert sha256(OUT/n)==h,n
    for n,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(n)==h,n
    for n,v in p['model_files'].items():
        st=Path(n).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],n
    return p
