"""Local immutable receipts; never mutate the running FastV worktree's source."""
from pathlib import Path

from ..fastv_svh.common import (
    CAPS, COUNTS, GATE, MODEL, TOKENIZER, atomic, digest, hardware, idle_uuids,
    read, rows, sha, sources as inherited_sources,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path('/data/lk/ATE-worktrees/fastv-k3-svh-sixbench-3090')
PRIOR = PARENT / 'vlm_research/results/fastv_k3_svh_sixbench_3090'
OUT = ROOT / 'vlm_research/results/sparsevlm192_svh_sixbench_3090'
MME_OUT = PRIOR / 'mme_authorized'
MANIFEST = PRIOR / 'sample_manifest.jsonl.gz'
PARTITION = PRIOR / 'partition.json'
SERVICE = 'ate-fastv-svh-sixbench-3090.service'
OFFICIAL_COMMIT = '734d9003879689713a132b83297dfad060a5c9af'
ARCHIVE_COMMIT = '5ee2a0dbe78df336229b55fed14a4a6d0d880bed'
OFFICIAL = ROOT / 'vlm_research/external_baselines/sparsevlm_icml2025/official_source'
BASE_GPUS = ('GPU-a63c6362-dcfd-4a26-c44b-014b40127c89',
             'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6')


def sources():
    result = inherited_sources()
    paths = list((ROOT / 'vlm_research/sparsevlm_svh').glob('*.py'))
    paths += list(OFFICIAL.rglob('*.py'))
    paths += [ROOT / 'vlm_research/external_baselines/e2e32_compatibility/e2e32_official_compat.py']
    paths += list((ROOT / '.runtime-deps/einops').rglob('*.py'))
    result.update({str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)})
    return result


def verify(stage='sparse'):
    path = OUT / ('protocol.json' if stage == 'sparse' else 'mme_protocol.json')
    p = read(path)
    assert p['sources'] == sources(), 'Immutable executable snapshot changed'
    assert p['manifest_sha256'] == sha(MANIFEST)
    assert p['partition_sha256'] == sha(PARTITION)
    assert p['parent_protocol_sha256'] == digest(read(PRIOR / 'protocol.json'))
    for name, expected in p['model_file_stats'].items():
        stat = (MODEL / name).stat()
        assert [stat.st_size, stat.st_mtime_ns, stat.st_ino] == expected, f'Model file changed: {name}'
    return p


def destination(stage):
    assert stage in ('sparse', 'mme')
    return OUT if stage == 'sparse' else MME_OUT


def selected_samples(stage):
    samples = rows(MANIFEST)
    assert len(samples) == sum(COUNTS.values()) == 35256
    if stage == 'mme':
        samples = [s for s in samples if s['benchmark'] == 'MME']
        assert len(samples) == 2374
    return samples


def check_idle(uuids):
    h = hardware()
    available = idle_uuids(h)
    assert len(set(uuids)) == len(uuids)
    assert all(u in available for u in uuids), f'No preemption or idle-GPU polling: {h}'
    return h
