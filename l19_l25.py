"""Configure the single requested seven-layer window on the frozen runtime."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD_ROOT = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-l24-l34-discovery')
sys.path.insert(0, str(OLD_ROOT))
import l24_l34 as runtime
from l24_l34 import MODEL, SEED, TASKS, UUIDS, read_store, sha, table, write

ORIGINAL_OUT = runtime.OUT
OUT = ROOT / 'vlm_research/results/qwen3vl_svhr_l19_l25_discovery'
LAYERS = list(range(19, 26))
NAME = 'L19_L25'


def freeze():
    assert not (OUT / 'protocol.json').exists(), 'Never replace a frozen protocol'
    old = json.loads((ORIGINAL_OUT / 'protocol.json').read_text())
    for path, expected in old['source_sha256'].items():
        assert sha(path) == expected, ('ORIGINAL_SOURCE_CHANGED', path)
    for name, binding in [('sample_manifest.json', 'manifest_sha256'), ('paired_baselines.sqlite', 'baseline_sha256')]:
        assert sha(ORIGINAL_OUT / name) == old[binding]
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ORIGINAL_OUT / name, OUT / name)
    config_path = Path(MODEL) / 'config.json'
    count = json.loads(config_path.read_text())['text_config']['num_hidden_layers']
    assert count == 36 and LAYERS == list(range(19, 26)) and max(LAYERS) < count
    rows = json.loads((OUT / 'sample_manifest.json').read_text())
    assert len(rows) == 585 and all(r['split'] == 'Discovery' for r in rows)
    p = dict(old)
    p.update(experiment='QWEN3VL_SVHR_L19_L25_DISCOVERY', condition=NAME, layers=LAYERS, k=len(LAYERS),
        user_request_label='L19-L25', indexing='ZERO_BASED_INCLUSIVE', model_layer_count=count,
        planned_records=585, source_sha256=dict(old['source_sha256']),
        manifest_sha256=sha(OUT / 'sample_manifest.json'), baseline_sha256=sha(OUT / 'paired_baselines.sqlite'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        previous_result_commit='32eb6cfc38843ad90a2cd8fe8630a4344bcf0478',
        storage='FULL_FUNCTIONAL_REFERENCE_NOT_A_PHYSICAL_SPEED_TEST', performance='NOT_PLANNED', time=time.time())
    for path in [ROOT/'l19_l25.py', ROOT/'l19_l25_tests.py', ROOT/'l19_l25_cpu.py',
                 OLD_ROOT/'cpu_cache_checks.py', config_path]:
        p['source_sha256'][str(path)] = sha(path)
    write(OUT / 'protocol.json', p)
    print('FROZEN', NAME, LAYERS, len(rows), flush=True)


def frozen():
    p = json.loads((OUT / 'protocol.json').read_text())
    assert p['condition'] == NAME and p['layers'] == LAYERS and p['k'] == 7
    assert sha(OUT / 'sample_manifest.json') == p['manifest_sha256']
    assert sha(OUT / 'paired_baselines.sqlite') == p['baseline_sha256']
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, ('FROZEN_SOURCE_CHANGED', path)
    return p


runtime.ROOT = ROOT
runtime.OUT = OUT
runtime.LAYERS = LAYERS
runtime.NAME = NAME
runtime.__file__ = __file__
runtime.frozen = frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', type=int)
    opt = parser.parse_args()
    if opt.freeze:
        freeze()
    elif opt.launch:
        runtime.launch()
    elif opt.worker is not None:
        runtime.worker(opt.worker)
    else:
        parser.error('Choose --freeze, --launch or --worker')
