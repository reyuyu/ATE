"""One requested final-layer window on the unchanged no-Smoke worker."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_ROOT = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-l19-trimmed-discovery')
sys.path.insert(0, str(BASE_ROOT))
import l19_trimmed as runtime
from l19_trimmed import MODEL, SEED, TASKS, UUIDS, read_store, sha, table, write

BASE_OUT = runtime.OUT
OUT = ROOT/'vlm_research/results/qwen3vl_svhr_l19_l35_discovery'
NAME = 'L19_L35'
LAYERS = list(range(19, 36))
CONDITIONS = {NAME: LAYERS}
EXPECTED = 585


def freeze():
    assert not (OUT/'protocol.json').exists(), 'Never replace frozen protocol'
    old = json.loads((BASE_OUT/'protocol.json').read_text())
    for path, expected in old['source_sha256'].items():
        assert sha(path) == expected, ('HISTORICAL_SOURCE_CHANGED', path)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, binding in [('sample_manifest.json', 'manifest_sha256'), ('paired_baselines.sqlite', 'baseline_sha256')]:
        assert sha(BASE_OUT/name) == old[binding]
        shutil.copyfile(BASE_OUT/name, OUT/name)
    rows = json.loads((OUT/'sample_manifest.json').read_text())
    assert len(rows) == len({r['key'] for r in rows}) == EXPECTED
    assert all(r['split'] == 'Discovery' for r in rows)
    assert sum(r['analysis_task'] == 'TextVQA' for r in rows) == 250
    config = Path(MODEL)/'config.json'
    assert json.loads(config.read_text())['text_config']['num_hidden_layers'] == 36
    assert LAYERS == list(range(19, 36)) and len(LAYERS) == 17
    p = dict(old)
    p.update(experiment='QWEN3VL_SVHR_L19_L35_DISCOVERY', conditions=CONDITIONS,
        expected_per_condition=585, expected_total=EXPECTED, planned_records=EXPECTED,
        source_sha256=dict(old['source_sha256']), manifest_sha256=sha(OUT/'sample_manifest.json'),
        baseline_sha256=sha(OUT/'paired_baselines.sqlite'), previous_protocol_sha256=sha(BASE_OUT/'protocol.json'),
        previous_result_commit='947da3efe4a32461b7b7d2039bc4e7a30c5d5cb9',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        smoke='NOT_RUN_USER_REQUEST', new_gpu_numerical_validation='NOT_RUN', time=time.time())
    p['source_sha256'][str(ROOT/'l19_l35.py')] = sha(ROOT/'l19_l35.py')
    write(OUT/'protocol.json', p)
    print('FROZEN', CONDITIONS, 'EXPECTED', EXPECTED, flush=True)


def frozen():
    p = json.loads((OUT/'protocol.json').read_text())
    assert p['conditions'] == CONDITIONS and p['expected_total'] == EXPECTED
    assert p['smoke'] == 'NOT_RUN_USER_REQUEST'
    assert sha(OUT/'sample_manifest.json') == p['manifest_sha256']
    assert sha(OUT/'paired_baselines.sqlite') == p['baseline_sha256']
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, ('FROZEN_SOURCE_CHANGED', path)
    return p


runtime.ROOT = ROOT
runtime.OUT = OUT
runtime.CONDITIONS = CONDITIONS
runtime.EXPECTED = EXPECTED
runtime.frozen = frozen


def launch():
    frozen()
    assert not (OUT/'launch.json').exists(), 'Reject duplicate launch'
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(uuid in processes for uuid in UUIDS), processes
    assert shutil.disk_usage('/data').free >= 2*2**30
    write(OUT/'gpu_preflight.json', dict(time=time.time(), processes=processes, authorized=UUIDS,
        sample=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)))
    children, streams = [], []
    started = time.time()
    try:
        for shard, uuid in enumerate(UUIDS):
            stream = (OUT/f'worker{shard}.log').open('a')
            streams.append(stream)
            children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(shard)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, TOKENIZERS_PARALLELISM='false'),
                stdout=stream, stderr=subprocess.STDOUT))
        write(OUT/'launch.json', dict(pid=os.getpid(), worker_pids=[p.pid for p in children], time=started,
                                    smoke='NOT_RUN_USER_REQUEST'))
        while not all((OUT/f'done{i}.json').exists() for i in [0, 1]):
            assert not any((OUT/f'failure{i}.json').exists() for i in [0, 1]), 'WORKER_FAILURE'
            assert all(c.poll() in [None, 0] for c in children), 'Worker exited with failure'
            assert time.time()-started < 3600, 'Bounded execution timeout'
            time.sleep(2)
        for child in children:
            assert child.wait(timeout=40) == 0
        counts = [json.loads((OUT/f'done{i}.json').read_text())['rows'] for i in [0, 1]]
        assert sum(counts) == EXPECTED
        write(OUT/'generation_complete.json', dict(status='COMPLETE_585_GENERATIONS', expected=EXPECTED,
            actual=sum(counts), time=time.time(), wall_seconds=time.time()-started))
    except BaseException as exc:
        write(OUT/'coordinator_failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        write(OUT/'STOP.json', dict(time=time.time(), scope='ONLY_THIS_EXPERIMENT_OWN_WORKERS'))
        for child in children:
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait()
        for stream in streams:
            stream.close()
        write(OUT/'gpu_release_audit.json', dict(time=time.time(), exit_codes=[c.returncode for c in children],
            processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv'], text=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', type=int)
    opt = parser.parse_args()
    if opt.freeze:
        freeze()
    elif opt.launch:
        launch()
    elif opt.worker is not None:
        runtime.worker(opt.worker)
    else:
        parser.error('Choose --freeze, --launch or --worker')
