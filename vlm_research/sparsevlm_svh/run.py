"""Fixed two-device launch, atomic individual checkpoints, no GPU polling."""
import argparse
from datetime import datetime
import gc
import importlib.metadata
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

from .common import *


def child(stage, mode, uuid, rank):
    p = verify(stage)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == uuid
    check_idle([uuid])
    import torch
    from transformers import AutoTokenizer
    from ..information_horizon_alignment import common as shared
    from ..information_horizon_alignment import score as scorer
    from ..information_horizon_alignment.engine import load
    from ..fastv_svh.inputs import prepare
    from .checkpoint_audit import score_immutable
    shared.AUTHOR = scorer.AUTHOR = Path('/data/lk/ATE-resources/Information-Horizon')
    assert torch.__version__ == p['torch']
    assert importlib.metadata.version('transformers') == p['transformers']
    output = destination(stage)
    target = output / ('gpu_smoke' if mode == 'smoke-child' else f'worker{rank}')
    target.mkdir(parents=True, exist_ok=True)
    adapter = db = None
    started = time.time()
    try:
        adapter = load('B')
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=False, legacy=True, local_files_only=True)
        versions = [(v._version, v.data_ptr()) for v in adapter.model.parameters()]
        atomic(target / 'resources.json', dict(uuid=uuid, pid=os.getpid(), hardware=hardware(),
            protocol_sha256=digest(p), python=sys.executable,
            code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
        if mode == 'smoke-child':
            assert stage == 'sparse'
            from .smoke import run as smoke
            result = smoke(adapter, tokenizer, selected_samples(stage))
            result.update(protocol_sha256=digest(p), uuid=uuid, elapsed_seconds=time.time() - started)
            atomic(output / 'gpu_smoke.json', result)
            print('SIX_IMAGE_GPU_SMOKE_PASS', flush=True)
            return
        if stage == 'sparse':
            from .engine import generate
            gate = read(output / 'gpu_smoke.json')
            assert gate['passed'] and gate['actual_gpu_images'] == 6 and gate['protocol_sha256'] == digest(p)
        else:
            from ..fastv_svh.engine import generate
            parent_gate = read(PRIOR / 'gpu_smoke.json')
            assert parent_gate['passed'] and parent_gate['protocol_sha256'] == p['parent_protocol_sha256']
        samples = selected_samples(stage)
        assignment = read(PARTITION)['assignment']
        mine = [s for s in samples if assignment[s['key']] == rank]
        index = {s['key']: s for s in mine}
        db = sqlite3.connect(target / 'results.sqlite')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
        done = set()
        for key, raw, checksum in db.execute('SELECT key,payload,sha256 FROM results'):
            record = json.loads(raw)
            assert key in index and digest(record) == checksum
            assert record['protocol_sha256'] == digest(p) and record['sample_sha256'] == digest(index[key])
            assert record['gpu_uuid'] == uuid
            done.add(key)
        for sample in mine:
            assert not (output / 'STOP_REQUESTED').exists()
            assert not any(output.glob('worker*/STOP.json')), 'Failure requires investigation, not silent retry'
            if sample['key'] in done:
                continue
            if len(done) % 100 == 0:
                verify(stage)
                assert shutil.disk_usage(output).free > 2 * 1024**3
                assert versions == [(v._version, v.data_ptr()) for v in adapter.model.parameters()]
            assert sha(sample['image_path']) == sample['image_sha256']
            layout, binding = prepare(adapter, sample, tokenizer)
            assert layout.length + sample['max_new_tokens'] <= adapter.model.config.text_config.max_position_embeddings
            generated, _ = generate(adapter, layout, sample['max_new_tokens'])
            record = dict(key=sample['key'], benchmark=sample['benchmark'], question_id=sample['question_id'],
                image_id=sample['image_id'], image_sha256=sample['image_sha256'], gold=sample['answer'],
                question=sample['question'], input_binding=binding, generation=generated,
                scoring=score_immutable(sample, generated['text']), rank=rank, gpu_uuid=uuid,
                sample_sha256=digest(sample), protocol_sha256=digest(p))
            with db:
                db.execute('INSERT INTO results VALUES(?,?,?)', (sample['key'], json.dumps(record, ensure_ascii=False), digest(record)))
            done.add(sample['key'])
            adapter.inputs = {}
            if len(done) % 25 == 0 or len(done) == len(mine):
                atomic(target / 'progress.json', dict(status='RUNNING', completed=len(done), expected=len(mine),
                    elapsed_seconds=time.time() - started, last_key=sample['key']))
                print('PROGRESS', stage, rank, len(done), '/', len(mine), flush=True)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        atomic(target / 'completion.json', dict(status='COMPLETE_SHARD', completed=len(done), expected=len(mine),
            elapsed_seconds=time.time() - started))
    except BaseException:
        atomic(target / 'STOP.json', dict(traceback=traceback.format_exc(), stage=stage, uuid=uuid, rank=rank))
        raise
    finally:
        if db is not None:
            db.close()
        adapter = None
        gc.collect()
        torch.cuda.empty_cache()
        atomic(target / 'release.json', dict(uuid=uuid, pid=os.getpid(), allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(), context_released_on_exit=True,
            finished_at=datetime.now().astimezone().isoformat()))


def driver(stage, mode, uuids):
    p = verify(stage)
    assert len(uuids) == (1 if mode == 'smoke' else 2)
    resources = check_idle(uuids)
    assert shutil.disk_usage(OUT).free > 2 * 1024**3
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--',
        'vlm_research/sparsevlm_svh', 'vlm_research/fastv_svh',
        'vlm_research/external_baselines'], cwd=ROOT, text=True).strip(), 'Commit execution snapshot first'
    if mode == 'driver' and stage == 'sparse':
        gate = read(OUT / 'gpu_smoke.json')
        assert gate['passed'] and gate['protocol_sha256'] == digest(p)
    output = destination(stage)
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / 'launch.json', dict(gpus=uuids, resources=resources, protocol_sha256=digest(p), stage=stage, mode=mode))
    logs, processes = [], []
    try:
        for rank, uuid in enumerate(uuids):
            log = (output / f'{mode}{rank}.log').open('a', buffering=1)
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONUNBUFFERED='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            processes.append(subprocess.Popen([sys.executable, '-m', 'vlm_research.sparsevlm_svh.run',
                'smoke-child' if mode == 'smoke' else 'worker', '--stage', stage, '--gpus', uuid,
                '--rank', str(rank)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Owned worker failed; preserve diagnostics and stop only its owned peer')
            time.sleep(5)
        assert all(p.returncode == 0 for p in processes)
        if mode == 'driver':
            from .analyze import analyze
            analyze(stage)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        atomic(output / 'gpu_release_audit.json', hardware())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'smoke-child', 'driver', 'worker'])
    parser.add_argument('--stage', choices=['mme', 'sparse'], required=True)
    parser.add_argument('--gpus', nargs='+', required=True)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    if args.mode in ('smoke', 'driver'):
        driver(args.stage, args.mode, args.gpus)
    else:
        assert len(args.gpus) == 1
        child(args.stage, args.mode, args.gpus[0], args.rank)
