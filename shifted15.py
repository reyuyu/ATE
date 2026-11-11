"""Four user-frozen 15-layer windows; reuse the immutable multi-set worker."""
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
BASE_ROOT = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-late-windows-discovery')
sys.path.insert(0, str(BASE_ROOT))
import late_windows as runtime
from late_windows import MODEL, SEED, TASKS, UUIDS, read_store, sha, table, write

BASE_OUT = runtime.OUT
OUT = ROOT/'vlm_research/results/qwen3vl_svhr_shifted15_discovery'
CONDITIONS = {f'L{s}_L{s+14}': list(range(s, s+15)) for s in [19, 18, 17, 16]}
EXPECTED = 585*len(CONDITIONS)


def freeze():
    assert not (OUT/'protocol.json').exists(), 'Never replace a frozen protocol'
    old = json.loads((BASE_OUT/'protocol.json').read_text())
    for path, expected in old['source_sha256'].items():
        assert sha(path) == expected, ('HISTORICAL_SOURCE_CHANGED', path)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, binding in [('sample_manifest.json', 'manifest_sha256'), ('paired_baselines.sqlite', 'baseline_sha256')]:
        assert sha(BASE_OUT/name) == old[binding]
        shutil.copyfile(BASE_OUT/name, OUT/name)
    rows = json.loads((OUT/'sample_manifest.json').read_text())
    assert len(rows) == 585 and all(r['split'] == 'Discovery' for r in rows)
    config = Path(MODEL)/'config.json'
    assert json.loads(config.read_text())['text_config']['num_hidden_layers'] == 36
    p = dict(old)
    p.update(experiment='QWEN3VL_SVHR_FOUR_SHIFTED15_DISCOVERY', conditions=CONDITIONS,
        expected_per_condition=585, expected_total=EXPECTED, source_sha256=dict(old['source_sha256']),
        manifest_sha256=sha(OUT/'sample_manifest.json'), baseline_sha256=sha(OUT/'paired_baselines.sqlite'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        previous_result_commit='b77a1403ce63cb278e0f040e569cf960295886bd',
        previous_protocol_sha256=sha(BASE_OUT/'protocol.json'), time=time.time())
    for path in [ROOT/'shifted15.py', ROOT/'shifted15_tests.py', ROOT/'shifted15_cpu.py', config]:
        p['source_sha256'][str(path)] = sha(path)
    write(OUT/'protocol.json', p)
    print('FROZEN', CONDITIONS, 'EXPECTED', EXPECTED, flush=True)


def frozen():
    p = json.loads((OUT/'protocol.json').read_text())
    assert p['conditions'] == CONDITIONS and p['expected_total'] == EXPECTED
    assert sha(OUT/'sample_manifest.json') == p['manifest_sha256']
    assert sha(OUT/'paired_baselines.sqlite') == p['baseline_sha256']
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, ('FROZEN_SOURCE_CHANGED', path)
    return p


# Only run configuration is rebound; historical on-disk code stays unchanged.
runtime.ROOT = ROOT
runtime.OUT = OUT
runtime.CONDITIONS = CONDITIONS
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
        write(OUT/'launch.json', dict(pid=os.getpid(), worker_pids=[p.pid for p in children], time=started))
        while not all((OUT/f'smoke{i}.json').exists() for i in [0, 1]):
            assert all(c.poll() is None for c in children), 'Worker exited before Smoke'
            assert time.time()-started < 900, 'Smoke timeout'
            time.sleep(1)
        write(OUT/'START_QUALITY.json', dict(time=time.time(), both_gpu_smoke='PASS', conditions=CONDITIONS))
        while not all((OUT/f'done{i}.json').exists() for i in [0, 1]):
            assert not any((OUT/f'failure{i}.json').exists() for i in [0, 1]), 'WORKER_FAILURE'
            assert all(c.poll() in [None, 0] for c in children), 'Worker exited with failure'
            assert time.time()-started < 3600, 'Bounded execution timeout'
            time.sleep(2)
        for child in children:
            assert child.wait(timeout=40) == 0
        counts = [json.loads((OUT/f'done{i}.json').read_text())['rows'] for i in [0, 1]]
        assert sum(counts) == EXPECTED
        write(OUT/'generation_complete.json', dict(status='COMPLETE_2340_GENERATIONS', expected=EXPECTED,
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
