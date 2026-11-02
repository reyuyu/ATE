from pathlib import Path

from query_role_attention.common import (ROOT, OUT as OLD, MODEL, UUID, REVISION,
    TASKS, CONDITIONS, COMPARISONS, SEED, LAYERS, read, write, sha256, digest, csv_write, preflight)
from query_role_smoke_fix.run import sources as repair_sources

BASE = '2654e80a4b9448dd864ae6c726998e4931bf772a'
OUT = ROOT/'results/query_role_attention_boundary_240_formal'
REPAIR = ROOT/'results/query_role_smoke_logits_fix'
RAW = Path('/data/lk/ATE-resources/experiments/query_role_attention_boundary_240_formal')
MANIFEST_SHA = '258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c'
TASK_TEXT_SHA = '258c32d470270829e126ae9abf3a030dbc1485b2f329dff826f0a3e686c'


def sources():
    return {**repair_sources(), **{str(p.relative_to(ROOT)):sha256(p)
        for p in sorted(Path(__file__).parent.glob('*.py'))}}


def historical():
    return {str(p.relative_to(ROOT)):sha256(p) for directory in (OLD,REPAIR)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def verify():
    p=read(OUT/'protocol.json')
    assert p['sources']==sources(), 'Frozen source changed'
    assert p['historical_files']==historical(), 'Historical artifact changed'
    for name,value in p['bindings'].items():
        assert sha256(OUT/name)==value,name
    for path,h in {**p['transformers_sources'],**p['scorer_sources']}.items():
        assert sha256(path)==h,path
    for path,info in p['model_files'].items():
        stat=Path(path).stat()
        assert stat.st_size==info['bytes'] and stat.st_mtime_ns==info['mtime_ns'],path
    return p


def divergence(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
                min(len(a),len(b)) if len(a)!=len(b) else None)


def repeated_trigram(tokens):
    content=[t for t in tokens if t!=2]
    triples=[tuple(content[i:i+3]) for i in range(len(content)-2)]
    return len(set(triples))!=len(triples)
