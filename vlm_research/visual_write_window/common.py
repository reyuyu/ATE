import os,subprocess
from pathlib import Path
from query_role_attention.common import ROOT,MODEL,REVISION,TASKS,SEED,read,write,sha256,digest,csv_write
from relay_source_causality.common import sources as prior_sources,hardware,audit_history

BASE='e139c76e32edd3eacdf44d7172069bc37b230b60'
OLD=ROOT/'results/query_role_attention_boundary_240_formal'
OUT=ROOT/'results/visual_write_window_240'
RAW=Path('/data/lk/ATE-resources/experiments/visual_write_window_240')
CONDITIONS=['All-Writer','No-Writer']+[f'Only-W{k}' for k in range(8)]+[f'Drop-W{k}' for k in range(8)]
WINDOWS={f'W{k}':list(range(4*k,4*k+4)) for k in range(8)}
SWAP=ROOT/'results/rq2_visual_text_handoff_240/image_swap_manifest.json'
MANIFEST_SHA='258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c'

def writer_layers(c):
    if c=='All-Writer':return list(range(32))
    if c=='No-Writer':return []
    if c.startswith('Only-W') and c in CONDITIONS:return WINDOWS[c[5:]]
    if c.startswith('Drop-W') and c in CONDITIONS:return [k for k in range(32) if k not in WINDOWS[c[5:]]]
    raise ValueError('Condition not frozen: '+c)

def sources():
    return {**prior_sources(),**{str(p.relative_to(ROOT)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}

def select_gpu():
    state=hardware();busy={x.split(',')[0].strip() for x in state['compute_processes'].splitlines()};candidates=[]
    for row in state['table'].splitlines():
        index,uuid,name,used,free,util=[x.strip() for x in row.split(',')]
        if uuid not in busy and int(used)<200 and int(free)>22000 and int(util)==0:
            candidates.append(dict(physical_index=int(index),uuid=uuid,name=name,logical='cuda:0'))
    state['selected']=candidates[0] if candidates else None;write(OUT/'gpu_selection.json',state)
    if not candidates:raise RuntimeError('No idle GPU; stop without waiting')
    return candidates[0]

def preflight(gpu):
    state=hardware();row=next(x for x in state['table'].splitlines() if x.split(',')[1].strip()==gpu['uuid']);v=[x.strip() for x in row.split(',')]
    busy=[x for x in state['compute_processes'].splitlines() if x.split(',')[0].strip()==gpu['uuid']]
    state.update(selected=gpu,passed=int(v[3])<200 and int(v[4])>22000 and int(v[5])==0 and not busy)
    write(OUT/'gpu_preflight.json',state)
    assert state['passed'] and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid'],'GPU preflight failed; stop'

def verify():
    p=read(OUT/'protocol.json');assert p['sources']==sources()
    for name,h in p['bindings'].items():assert sha256(OUT/name)==h,name
    for name,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(name)==h,name
    for path,info in p['model_files'].items():
        st=Path(path).stat();assert st.st_size==info['bytes'] and st.st_mtime_ns==info['mtime_ns'],path
    assert p['conditions']=={c:writer_layers(c) for c in CONDITIONS}
    return p
