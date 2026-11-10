"""Explicit GPU launch only; no automatic idle-GPU polling or remote publication."""
import argparse
from datetime import datetime
import gc
import importlib.metadata
import json
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

from .common import *


def assert_idle(uuid):
    h = hardware()
    assert uuid in idle_uuids(h), f'GPU not idle; do not preempt: {h}'
    return h


def accepted_samples(protocol, samples):
    acceptance = read(OUT / 'data_acceptance.json')
    assert sha(OUT / 'data_acceptance.json') == protocol['data_acceptance_sha256']
    if protocol['all_six_ready']:
        assert acceptance['status'] == 'READY'
        return samples
    schedule = protocol.get('execution_schedule')
    assert schedule and schedule['deferred_benchmarks'] == ['MME'], 'No implicit dataset exclusion'
    assert schedule['ready_benchmarks'] == [b for b in COUNTS if b != 'MME']
    assert acceptance['status'] == 'BLOCKED_MME_REFERENCE_BYTES'
    assert acceptance['blockers'] and all(b['key'].startswith('MME:') for b in acceptance['blockers'])
    ready = [s for s in samples if s['benchmark'] in schedule['ready_benchmarks']]
    assert len(samples) == 35256 and len(ready) == schedule['ready_questions'] == 32882
    assert schedule['requested_questions'] == 35256 and schedule['deferred_questions'] == 2374
    return ready


def child(mode, uuid, rank=0):
    p = verify()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == uuid
    assert_idle(uuid)
    assert importlib.metadata.version('transformers') == p['transformers']
    import torch
    assert torch.__version__ == p['torch']
    from transformers import AutoTokenizer
    from ..information_horizon_alignment import common as shared
    from ..information_horizon_alignment.engine import load
    from .smoke import run as smoke
    from .engine import generate
    from .inputs import prepare, score
    # Set only this process's scorer root; installed/shared files stay unchanged.
    shared.AUTHOR = Path('/data/lk/ATE-resources/Information-Horizon')
    from ..information_horizon_alignment import score as scorer
    scorer.AUTHOR = shared.AUTHOR
    dest = OUT / ('gpu_smoke' if mode == 'smoke-child' else f'worker{rank}')
    dest.mkdir(parents=True, exist_ok=True)
    adapter, db = None, None
    start = time.time()
    try:
        samples = rows(OUT / 'sample_manifest.jsonl.gz')
        adapter = load('B')
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=False, legacy=True,
                                                  local_files_only=True)
        versions = [(v._version, v.data_ptr()) for v in adapter.model.parameters()]
        atomic(dest / 'resources.json', dict(uuid=uuid, pid=os.getpid(), hardware=hardware(),
            code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            protocol_sha256=digest(p), python=sys.executable, torch=torch.__version__,
            transformers=importlib.metadata.version('transformers')))
        if mode == 'smoke-child':
            gate = smoke(adapter, tokenizer, samples)
            gate.update(protocol_sha256=digest(p), uuid=uuid, elapsed_seconds=time.time()-start)
            atomic(OUT / 'gpu_smoke.json', gate)
            print('GPU_SMOKE_PASS', uuid, flush=True)
            return
        gate = read(OUT / 'gpu_smoke.json')
        assert gate['passed'] and gate['protocol_sha256'] == digest(p)
        runnable = accepted_samples(p, samples)
        assignment = read(OUT / 'partition.json')['assignment']
        mine = [s for s in runnable if assignment[s['key']] == rank]
        db = sqlite3.connect(dest / 'results.sqlite')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
        index, done = {s['key']:s for s in mine}, set()
        for key, raw, checksum in db.execute('SELECT key,payload,sha256 FROM results'):
            row = json.loads(raw)
            assert key in index and digest(row) == checksum
            assert row['sample_sha256'] == digest(index[key]) and row['protocol_sha256'] == digest(p)
            assert row['gpu_uuid'] == uuid, 'Resume must keep the original physical device'
            done.add(key)
        for sample in mine:
            assert not (OUT / 'STOP_REQUESTED').exists()
            assert not any(OUT.glob('worker*/STOP.json')), 'Previous failure requires investigation'
            if sample['key'] in done:
                continue
            if len(done) % 100 == 0:
                verify()
                assert shutil.disk_usage(OUT).free > 2*1024**3
                assert versions == [(v._version, v.data_ptr()) for v in adapter.model.parameters()]
            assert sha(sample['image_path']) == sample['image_sha256']
            layout, binding = prepare(adapter, sample, tokenizer)
            assert layout.length+sample['max_new_tokens'] <= adapter.model.config.text_config.max_position_embeddings
            generated, _ = generate(adapter, layout, sample['max_new_tokens'])
            row = dict(key=sample['key'], benchmark=sample['benchmark'], question_id=sample['question_id'],
                image_id=sample['image_id'], image_sha256=sample['image_sha256'], gold=sample['answer'],
                question=sample['question'], input_binding=binding, generation=generated,
                scoring=score(sample, generated['text']), rank=rank, gpu_uuid=uuid,
                sample_sha256=digest(sample), protocol_sha256=digest(p))
            with db:
                db.execute('INSERT INTO results VALUES(?,?,?)', (sample['key'], json.dumps(row, ensure_ascii=False), digest(row)))
            done.add(sample['key'])
            adapter.inputs = {}
            if len(done) % 25 == 0 or len(done) == len(mine):
                atomic(dest / 'progress.json', dict(status='RUNNING', completed=len(done), expected=len(mine),
                    elapsed_seconds=time.time()-start, last_key=sample['key']))
                print('PROGRESS', rank, len(done), '/', len(mine), flush=True)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        atomic(dest / 'completion.json', dict(status='COMPLETE_SHARD', completed=len(done),
                                               expected=len(mine), elapsed_seconds=time.time()-start))
    except BaseException:
        atomic(dest / 'STOP.json', dict(traceback=traceback.format_exc(), uuid=uuid, rank=rank))
        raise
    finally:
        if db is not None:
            db.close()
        adapter = None
        gc.collect()
        torch.cuda.empty_cache()
        atomic(dest / 'release.json', dict(uuid=uuid, pid=os.getpid(), allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(), context_released_on_exit=True,
            finished_at=datetime.now().astimezone().isoformat()))


def spawn(mode, uuid, rank, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONUNBUFFERED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return subprocess.Popen([sys.executable, '-m', 'vlm_research.fastv_svh.run', mode,
                             '--gpus', uuid, '--rank', str(rank)], cwd=ROOT, env=env,
                             stdout=log, stderr=subprocess.STDOUT)


def launch(mode, uuids):
    p = verify()
    assert len(set(uuids)) == len(uuids)
    assert len(uuids) == (1 if mode == 'smoke' else 2)
    for uuid in uuids:
        assert_idle(uuid)
    if mode == 'driver':
        gate = read(OUT / 'gpu_smoke.json')
        assert gate['passed'] and gate['protocol_sha256'] == digest(p)
        accepted_samples(p, rows(OUT / 'sample_manifest.jsonl.gz'))
    # Refuse mutable, uncommitted executable sources even if hashes were captured.
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--',
        'vlm_research/fastv_svh', 'vlm_research/external_baselines/fastv'], cwd=ROOT, text=True)
    assert not dirty.strip(), 'Commit frozen executable code before GPU launch'
    logs, processes = [], []
    try:
        for rank, uuid in enumerate(uuids):
            log = (OUT / f'{mode}{rank}.log').open('a', buffering=1)
            logs.append(log)
            processes.append(spawn('smoke-child' if mode=='smoke' else 'worker', uuid, rank, log))
        while any(c.poll() is None for c in processes):
            if any(c.poll() not in (None, 0) for c in processes):
                raise RuntimeError('Task-owned worker failed; preserve diagnostics and stop its peer')
            time.sleep(5)
        assert all(c.returncode == 0 for c in processes)
        if mode == 'driver':
            from .analyze import main as analyze
            analyze()
    finally:
        for c in processes:
            if c.poll() is None:
                c.terminate()
        for c in processes:
            try:
                c.wait(timeout=60)
            except subprocess.TimeoutExpired:
                c.kill()
                c.wait()
        for log in logs:
            log.close()
        atomic(OUT / 'gpu_release_audit.json', hardware())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'driver', 'smoke-child', 'worker'])
    parser.add_argument('--gpus', nargs='+', required=True, help='Explicit physical GPU UUID(s), not ordinal IDs')
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    if args.mode in ('smoke', 'driver'):
        launch(args.mode, args.gpus)
    else:
        assert len(args.gpus) == 1
        child(args.mode, args.gpus[0], args.rank)
