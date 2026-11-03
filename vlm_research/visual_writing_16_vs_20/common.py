import json
from pathlib import Path

from visual_writing_necessity.common import (ROOT,MODEL,REVISION,TASKS,read,write,sha256,digest,
    preflight,progress_write,shards,hardware,source_hashes as parent_sources)

BASE='afddd2a9972e6decddbb535f4b524225e52eba0e'
HISTORY=ROOT/'results/gqa3000_visual_writing_necessity'
OUT=ROOT/'results/gqa3000_visual_writing_16_vs_20'
CONDITIONS=['Keep-0-15','Keep-0-19']
SEED=20261103


def writers(condition):
    if condition=='All-Writer' or condition=='Native':return list(range(32))
    if condition not in CONDITIONS:raise ValueError(condition)
    return list(range(16 if condition=='Keep-0-15' else 20))


def sources():
    return {**parent_sources(),**{str(p.relative_to(ROOT)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))
        if p.name in ['__init__.py','common.py','engine.py','run.py','tests.py']}}


def verify():
    p=read(OUT/'protocol.json');assert p['sources']==sources(),'Frozen execution source changed'
    for name,h in p['bindings'].items():assert sha256(OUT/name)==h,name
    for name,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(name)==h,name
    for name,info in p['model_files'].items():
        st=Path(name).stat();assert st.st_size==info['bytes'] and st.st_mtime_ns==info['mtime_ns'],name
    assert p['conditions']=={c:writers(c) for c in CONDITIONS}
    return p


def noninferiority(lower,margin):
    return float(lower)>-float(margin)
