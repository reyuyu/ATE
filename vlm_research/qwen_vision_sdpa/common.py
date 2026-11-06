"""Isolated output paths and immutable references to the preceding experiment."""
from ..mme_seed_quality.common import (
    Path,os,json,sha,digest,atomic,read,tensor_sha,hook_counts,now,git_head,
    MODELS,UUIDS,TASKS,mme_parse,
)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/qwen3vl_mme_seed_2048_vision_sdpa'
PREVIOUS=ROOT/'vlm_research/results/mme_seed_physical_c3_quality'
PRIVATE=Path('/data/lk/ATE-resources/experiments/qwen3vl_mme_seed_2048_vision_sdpa')
PREVIOUS_PRIVATE=Path('/data/lk/ATE-resources/experiments/mme_seed_physical_c3_quality')
PARENT='213d4e3be766444e1ecd9a90ac479836f7756e9b'
BACKENDS={'vision_config':'sdpa','text_config':'eager'}


def source_hashes():
    from ..mme_seed_quality.common import source_hashes as previous_hashes
    result=previous_hashes()
    for p in (ROOT/'vlm_research/qwen_vision_sdpa').glob('*.py'):
        if p.name not in ('analyze.py','report.py','delivery.py'):
            result[str(p.relative_to(ROOT))]=sha(p)
    return result


def write_checkpoint(path,payload):
    """Commit a condition, then atomically commit its byte-level SHA index."""
    payload=dict(payload)
    payload['payload_sha256']=digest(payload)
    atomic(path,payload)
    index_path=path.parent/'checkpoint_index.json'
    index=read(index_path) if index_path.exists() else {}
    index[path.name]=sha(path)
    atomic(index_path,index)


def checkpoint_complete(path,tag,sample):
    index_path=path.parent/'checkpoint_index.json'
    if not path.exists() or not index_path.exists():return False
    index=read(index_path)
    if path.name not in index:return False
    assert sha(path)==index[path.name],f'Checkpoint byte SHA mismatch: {path}'
    record=read(path);claimed=record.pop('payload_sha256')
    assert digest(record)==claimed,f'Checkpoint payload SHA mismatch: {path}'
    for k in ('source_manifest_sha256','protocol_sha256','sample_manifest_sha256','model_manifest_sha256'):
        assert record[k]==tag[k],(path,k)
    assert record['sample_sha256']==digest(sample)
    assert record['uuid']==UUIDS[sample['shard']]
    return True
