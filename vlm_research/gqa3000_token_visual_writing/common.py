import hashlib,json,os,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch
from query_role_attention.common import ROOT,MODEL,REVISION,TASKS,read,sha256,digest
from semantic_visual_writers.common import sources as inherited_sources
from relay_source_causality.common import hardware

BASE='a125cc0f72f9fab3cabc24bf29ee458e23a38b69'
SEM_COMMIT='9687b298efcdca651cc5446b5cb07da4a2b4e37c'
HISTORY=ROOT/'results/rq2_visual_sufficiency_full3000'
SEM=Path('/data/lk/ATE-worktrees/semantic-visual-writers-240/vlm_research/results/semantic_visual_writers_240')
OUT=ROOT/'results/gqa3000_token_visual_writing'
RAW=Path('/data/lk/ATE-resources/experiments/gqa3000_token_visual_writing')
MANIFEST_SHA='b5f6dec34cc5c96c6cce2d080b141bbbc8379d3d4b72e67f322ac1e63aa8e9f2'
SEEDS=[20261103,20261107,20261111]
BOOTSTRAP_SEED=20261104
BUDGETS=[20,40,60,80,100]
NATURAL=['All-Writer','No-Writer','Semantic-k','Question-Last-k','R-Last-k']+[
    f'{source}-Random-k-{seed}' for source in ['Question','R'] for seed in SEEDS]+['Drop-Semantic']+[
    f'Question-Stratified-k-{seed}' for seed in SEEDS[:2]]
CURVE=[name for b in BUDGETS for name in [f'Semantic-first-Q{b}',f'Question-Last-Q{b}']+[
    f'{source}-Random-{seed}-Q{b}' for source in ['Question','R'] for seed in SEEDS[:2]]]
CONDITIONS=NATURAL+CURVE
EXECUTION_FILES=['__init__.py','common.py','selection.py','prepare.py','engine.py','worker.py','driver.py','tests.py']

def key(s):return s['task']+'__'+s['question_id']
def sources():
    d={**inherited_sources()}
    for name in EXECUTION_FILES:
        p=Path(__file__).parent/name;d[str(p.relative_to(ROOT))]=sha256(p)
    return d
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert read(path)==obj,'Immutable record differs: '+str(path);return
    data=(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name+'.')
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)
def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists(),path
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name+'.')
    try:
        with os.fdopen(fd,'wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)
    return dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size)
def layout(s):
    m=s['input_mapping'];n=len(m['prompt_ids']);I=m['visual_positions'];R=list(range(I[-1]+1,n-1))
    assert len(I)==576 and m['p_minus_one']==n-1
    return dict(prompt_length=n,I=I,R=R,A=[n-1],other=list(range(I[0])),counts=dict(I=576,R=len(R),A=1,other=I[0]))
def gold_losses(logits,ids):
    a=torch.from_numpy(np.asarray(logits).copy()).double()
    assert a.ndim==2 and a.shape[0]==len(ids) and len(ids)>0 and torch.isfinite(a).all()
    values=-a.log_softmax(-1)[torch.arange(len(ids)),torch.tensor(ids)]
    return values.tolist(),float(values.mean())
def choose_gpus():
    state=hardware();busy={v.split(',')[0].strip() for v in state['compute_processes'].splitlines()};idle=[]
    for row in state['table'].splitlines():
        index,uuid,name,used,free,util=[s.strip() for s in row.split(',')]
        if uuid not in busy and int(used)<200 and int(free)>22000 and int(util)==0:idle.append(dict(index=int(index),uuid=uuid,name=name,logical='cuda:0'))
    state['selected']=idle[:2];atomic_json(OUT/'gpu_selection.json',state)
    assert len(idle)>=2,'Fewer than two idle GPUs. Stop without polling or preempting.';return idle[:2]
def preflight(gpu,rank,attempt='initial'):
    state=hardware();fields=[s.strip() for s in next(v for v in state['table'].splitlines() if v.split(',')[1].strip()==gpu['uuid']).split(',')]
    busy=[v for v in state['compute_processes'].splitlines() if v.split(',')[0].strip()==gpu['uuid']]
    state.update(selected=gpu,passed=not busy and int(fields[3])<200 and int(fields[4])>22000 and int(fields[5])==0)
    atomic_json(OUT/f'shard{rank}/attempts/{attempt}/gpu_preflight.json',state)
    assert state['passed'] and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid'],'Selected GPU became occupied'
def verify():
    p=read(OUT/'protocol.json');assert p['sources']==sources()
    for n,h in p['bindings'].items():assert sha256(OUT/n)==h,n
    for n,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(n)==h,n
    for n,info in p['model_files'].items():
        s=Path(n).stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],n
    return p
def assert_no_stop():
    assert not (OUT/'STOP.json').exists(),'A worker failed. Stop all experiment work.'
def confirmatory_decisions(nll_replicates,accuracy_replicates):
    upper=float(np.percentile(nll_replicates,97.5));lower=float(np.percentile(accuracy_replicates,2.5))
    return dict(family_alpha=.05,tests=2,one_sided_alpha_each=.025,nll_superiority_upper_bound=upper,
        nll_superiority_rejected=upper<0,accuracy_NI_lower_bound_pp=lower,accuracy_NI_minus2_rejected=lower>-2.,
        accuracy_minus1_sensitivity=lower>-1.,ci_containing_zero_not_an_NI_rule=True)
