"""Experiment-local bindings; historical implementations remain untouched."""
import json
import os
from pathlib import Path
import subprocess

from ..qwen3vl_early_windows.experiment import MODEL, REVISION, digest, sha256
from ..qwen3vl_query_role.common import read, seal, key

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/qwen3vl_visual_writing_necessity_3000'
HISTORY = ROOT / 'vlm_research/results/three_vlm_layer_sensitivity_3000'
WRITER240 = ROOT / 'vlm_research/results/qwen3vl_semantic_visual_writers_240'
CONDITIONS = ['All-Writer', 'No-Writer'] + [f'Drop-L{i}' for i in range(36)]
# Run the structural negative control immediately after its paired baseline.
ORDER = ['All-Writer', 'Drop-L35', 'No-Writer'] + [f'Drop-L{i}' for i in range(35)]
SEED = 20261103
LLAVA_COMMIT = 'afddd2a9972e6decddbb535f4b524225e52eba0e'
LLAVA = Path('/data/lk/ATE-worktrees/gqa3000-visual-writing-necessity/vlm_research/results/gqa3000_visual_writing_necessity')


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def atomic(path, value):
    """Replace mutable progress only; scientific records use immutable seal()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def source_hashes():
    paths = [Path(__file__).parent / name for name in ('__init__.py', 'common.py', 'engine.py', 'run.py', 'tests.py')]
    paths += [ROOT / 'vlm_research' / name for name in (
        'qwen3vl_semantic_visual_writers/engine.py', 'qwen3vl_query_role/engine.py',
        'qwen3vl_query_role/common.py', 'qwen3vl_query_role/run.py',
        'qwen3vl_early_windows/experiment.py', 'qwen3vl_smoke/common.py')]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def writer_layers(condition):
    assert condition in CONDITIONS
    if condition == 'All-Writer':
        return list(range(36))
    if condition == 'No-Writer':
        return []
    drop = int(condition.removeprefix('Drop-L'))
    return [i for i in range(36) if i != drop]


def hardware():
    rows = command('nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
                   '--format=csv,noheader,nounits')
    processes = command('nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits')
    return dict(gpus=[dict(zip(('index', 'uuid', 'name', 'used_mb', 'free_mb', 'utilization'),
                              [v.strip() for v in line.split(',')])) for line in rows.splitlines()],
                compute=processes)


def assert_idle(uuid, snapshot=None):
    snapshot = snapshot or hardware()
    gpu = next(g for g in snapshot['gpus'] if g['uuid'] == uuid)
    assert uuid not in snapshot['compute'], f'External compute process on {uuid}'
    assert int(gpu['used_mb']) < 200 and int(gpu['free_mb']) > 22000 and int(gpu['utilization']) == 0, gpu
    return gpu


def verify():
    p = read(OUT / 'protocol.json')
    assert p['sources'] == source_hashes(), 'Frozen execution source changed'
    for name, expected in p['artifacts'].items():
        assert sha256(OUT / name) == expected, name
    for path, expected in p['official_sources'].items():
        assert sha256(path) == expected, path
    return p


def checkpoint(shard, identifier, condition):
    return OUT / f'shard{shard}' / 'checkpoints' / identifier / f'{condition}.json'
