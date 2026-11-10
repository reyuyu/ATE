"""Bounded, local-only artifacts and pinned six-benchmark protocol."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/fastv_k3_svh_sixbench_3090'
MODEL = Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
TOKENIZER = Path('/data/lk/ATE-resources/information-horizon-alignment/runtime-staging/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234')
OLD = Path('/data/lk/ATE-worktrees/llava-svh-r-middle16-3090-sevenbench/vlm_research/results/svh_r_middle16_sevenbench_3090')
BASE = 'e281754ef17b3accc09d6938b5b9f020048b2aec'
FASTV_ARCHIVE = '5ee2a0dbe78df336229b55fed14a4a6d0d880bed'
FASTV_OFFICIAL = 'd1659729b5bf1be225e99ee15783deeea80f63b1'
GQA_REF = '71a5762cd8ccb8cc99efe9b6b0267d8abef70c46'
PAPER_REF = '0414e553ba6fd1e21626ecf741489d225b44279c'
COUNTS = {'GQA': 12578, 'ScienceQA-IMG': 2017, 'POPE': 8910,
          'MME': 2374, 'MMBench': 4377, 'TextVQA': 5000}
CAPS = {'GQA': 32, 'ScienceQA-IMG': 1024, 'POPE': 128,
        'MME': 128, 'MMBench': 1024, 'TextVQA': 128}
GATE = dict(logits_atol=.10, logits_rtol=.003, hidden_relative_rmse=.01,
            hidden_cosine_distance=1e-4, native_noop='torch.equal',
            pruning_disabled_svh='torch.equal', selector_top192='torch.equal')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def hardware():
    result = {}
    for label, query, flag in [('gpus', 'index,uuid,name,memory.used,memory.free,utilization.gpu', '--query-gpu='),
                               ('processes', 'gpu_uuid,pid,process_name,used_memory', '--query-compute-apps=')]:
        result[label] = subprocess.check_output(['nvidia-smi', flag+query,
            '--format=csv,noheader,nounits'], text=True).strip()
    pids = [line.split(',')[1].strip() for line in result['processes'].splitlines() if line.strip()]
    if pids:
        result['process_owners'] = subprocess.check_output(['ps', '-o', 'user,pid,etime,args', '-p', ','.join(pids)], text=True)
    return result


def idle_uuids(h):
    result = []
    for fields in csv.reader(h['gpus'].splitlines()):
        _, uuid, name, used, free, util = [v.strip() for v in fields]
        if ('RTX 3090' in name and uuid not in h['processes'] and
                int(used) < 200 and int(free) > 22000 and int(util) == 0):
            result.append(uuid)
    return result


def sources():
    paths = list((ROOT / 'vlm_research/fastv_svh').glob('*.py'))
    paths += list((ROOT / 'vlm_research/external_baselines/fastv').rglob('*.py'))
    paths += [ROOT / 'vlm_research' / p for p in (
        'physical_handoff_fastpath/engine.py', 'ate_fulljoint_v2/engine.py',
        'c3v2_middle_reaccess/engine.py', 'c3v2_middle_reaccess/common.py',
        'information_horizon_alignment/engine.py', 'information_horizon_alignment/common.py',
        'information_horizon_alignment/score.py', 'information_horizon_alignment/analyze.py',
        'mme_seed_quality/engine.py', 'mme_seed_quality/common.py',
        'physical_visual_handoff/engine.py', 'physical_visual_handoff/common.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def verify():
    protocol = read(OUT / 'protocol.json')
    assert protocol['sources'] == sources(), 'Source changed after freeze'
    assert protocol['manifest_sha256'] == sha(OUT / 'sample_manifest.jsonl.gz')
    assert protocol['partition_sha256'] == sha(OUT / 'partition.json')
    return protocol
