"""Quality-only resident workers with immutable input and source bindings."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import gc
import importlib.metadata
import json
import os
import sqlite3
import subprocess
import time
import traceback

from .common import *


def hardware():
    result = {}
    for name, query in [('gpus', 'index,uuid,memory.used,memory.free,utilization.gpu'),
                        ('processes', 'gpu_uuid,pid,process_name,used_memory')]:
        flag = '--query-gpu=' if name == 'gpus' else '--query-compute-apps='
        result[name] = subprocess.check_output(['nvidia-smi', flag + query, '--format=csv,noheader,nounits'], text=True)
    return result


def assert_idle(uuid):
    h = hardware()
    gpu = next(line.split(',') for line in h['gpus'].splitlines() if uuid in line)
    assert uuid not in h['processes'], h
    assert int(gpu[2]) < 200 and int(gpu[3]) > 22000 and int(gpu[4]) == 0, h
    return h


@contextmanager
def execution_trace(adapter, layout, algorithm):
    import torch
    events, handles, incoming = [], [], {}
    counts = {name: [0] * 32 for name in ['norm', 'post_norm', 'mlp']}
    checks = [0]
    try:
        for i, layer in enumerate(adapter.layers):
            def before(module, args, kw, i=i):
                t = counts['mlp'][i]
                pos = (list(range(layout.length)) if t == 0 and i < algorithm.exit_before
                       else list(layout.text) if t == 0 else [layout.length+t-1])
                assert args[0].shape[1] == len(pos)
                assert kw['cache_position'].tolist() == pos == kw['position_ids'][0].tolist()
                if i in algorithm.answer_skip:
                    incoming[i] = args[0][:, -1].detach().clone()
                events.append(dict(layer=i, step=t, rows=len(pos), original_positions_sha256=digest(pos)))
            def norm(module, args, output, i=i):
                counts['norm'][i] += 1
            def post(module, args, i=i):
                counts['post_norm'][i] += 1
                if i in algorithm.answer_skip:
                    assert torch.equal(args[0][:, -1], incoming.pop(i))
                    checks[0] += 1
            def mlp(module, args, output, i=i):
                counts['mlp'][i] += 1
            handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(layer.input_layernorm.register_forward_hook(norm))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(post))
            handles.append(layer.mlp.register_forward_hook(mlp))
        yield dict(events=events, counts=counts, residual_checks=checks)
    finally:
        for handle in handles:
            handle.remove()


def smoke(adapter, tokenizer, samples, rank):
    import torch
    from .engine import generate, prepare_input
    from ..information_horizon_alignment import engine as frozen
    from ..physical_visual_handoff.engine import equivalence
    from ..fastp2_bypass_writer.run import operators
    from ..information_horizon_alignment.common import hook_counts, tensor_sha
    report = []
    selected = [next(s for s in samples if s['benchmark'] == b)
                for b in ['GQA', 'ScienceQA-IMG', 'TextVQA', 'POPE', 'MMBench']]
    for s in selected:
        layout, binding = prepare_input(adapter, s, tokenizer)
        before = hook_counts(adapter.model)
        hashes = {k: tensor_sha(v) for k, v in adapter.inputs.items()}
        native, nl = generate(adapter, layout, cap=3, native=True, capture=True)
        noop, pl = generate(adapter, layout, cap=3, native=True, noop=True, capture=True)
        assert native['token_ids'] == noop['token_ids'] and all(torch.equal(x, y) for x, y in zip(nl, pl))
        hf = adapter.model.generate(**adapter.inputs, max_new_tokens=3, do_sample=False,
                                    num_beams=1, use_cache=True, logits_to_keep=1)
        assert hf[0, layout.length:].tolist() == native['token_ids']
        old, ol = generate(adapter, layout, OLD_REACCESS, cap=3, capture=True)
        reference, rl = frozen.hf_generate(adapter, layout, 'Physical-L11-L24', 3, capture=True)
        assert old['token_ids'] == reference['token_ids']
        assert len(ol) == len(rl) and all(torch.equal(x, y) for x, y in zip(ol, rl))
        with operators(adapter) as (calls, mlp, shapes), execution_trace(adapter, layout, MIDDLE16) as trace:
            actual, av = generate(adapter, layout, cap=3, capture=True, audit=True)
        dense, dv = generate(adapter, layout, cap=3, capture=True, audit=True, dense=True)
        steps = len(actual['token_ids'])
        assert steps >= 2, 'Smoke must include real incremental Decode'
        gate = equivalence('llava', dv, av, dense.pop('_hidden'), actual.pop('_hidden'))
        assert gate['passed'] and actual['token_ids'] == dense['token_ids'], gate
        ar, dr = actual.pop('_rotary'), dense.pop('_rotary')
        assert len(ar) == len(dr) == steps
        assert all(x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x) for x, y in zip(ar, dr))
        assert all(trace['counts'][name] == [steps]*32 for name in trace['counts'])
        assert trace['residual_checks'][0] == 4*steps
        for layer in MIDDLE16.answer_skip:
            assert all(calls[layer, op] == 1 for op in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
            observed = {r['op']: r['shape'][1] for r in shapes if r['layer'] == layer}
            assert observed == dict(q_proj=layout.length-1, k_proj=layout.length,
                                    v_proj=layout.length, o_proj=layout.length-1)
        assert all(r['blocked_probability_max'] == 0 for r in actual['attention_audit'])
        assert actual['kv_audit']['retained_image_layers'] == list(range(8, 24))
        for r in actual['visual_audit']:
            assert r['image_keys_permitted'] == (8 <= r['layer'] < 24)
            assert r['answer_image_mass'] > 0 if r['image_keys_permitted'] else r['answer_image_mass'] == 0
            if r['prefill'] and r['layer'] >= 20:
                assert r['relay_image_mass'] == 0
            if r['prefill'] and r['layer'] >= 24 or not r['prefill'] and r['layer'] not in range(8, 24):
                assert not r['image_keys_present']
        assert hashes == {k: tensor_sha(v) for k, v in adapter.inputs.items()}
        assert before == hook_counts(adapter.model)
        vectors = OUT / f'worker{rank}/smoke_vectors'
        vectors.mkdir(parents=True, exist_ok=True)
        vector_path = vectors / (s['benchmark'] + '.pt')
        torch.save(dict(native=nl, noop=pl, historical=ol, physical=av, dense=dv), vector_path)
        report.append(dict(key=s['key'], input_binding=binding, numerical_gate=gate,
                           strict_native_noop=True, strict_historical_old_reaccess=True,
                           native_tokens=native['token_ids'], old_tokens=old['token_ids'],
                           physical=actual, trace=trace, projection_shapes=shapes,
                           vector_path=str(vector_path), vector_sha256=sha(vector_path)))
        atomic(OUT / f'worker{rank}/smoke_progress.json', dict(status='RUNNING', cases=report))
        print('SMOKE_PASS', rank, s['benchmark'], gate, flush=True)
        del ar, dr, nl, pl, ol, rl, av, dv, dense, actual, hf
        adapter.inputs = {}
        gc.collect()
    atomic(OUT / f'worker{rank}/smoke.json', dict(status='PASS', sources=sources(), cases=report))


def score(s, text):
    if s['benchmark'] == 'GQA':
        p = text.rstrip('.').lower()
        return dict(score=float(p == s['answer']), parsed=p)
    from ..information_horizon_alignment import score as inherited
    return inherited.score(s, text)


def database(rank, protocol):
    path = OUT / f'worker{rank}/results.sqlite'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)')
    old = db.execute('SELECT value FROM metadata WHERE key="protocol"').fetchone()
    assert old is None or old[0] == digest(protocol)
    with db:
        db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)', ('protocol', digest(protocol)))
    return db


def worker(rank):
    p = verify()
    resources = read(OUT / 'resources.json')
    uuid = resources['workers'][rank]['uuid']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == uuid
    atomic(OUT / f'worker{rank}/preflight.json', dict(uuid=uuid, snapshot=assert_idle(uuid), pid=os.getpid()))
    import torch
    import transformers
    from transformers import LlamaTokenizer
    from ..mme_seed_quality.engine import load
    from ..information_horizon_alignment import score as inherited
    from .engine import generate, prepare_input
    assert torch.__version__ == p['torch'] and transformers.__version__ == p['transformers']
    inherited.AUTHOR = Path('/data/lk/ATE-resources/Information-Horizon')
    inherited.text_evaluator()
    inherited.matcher()
    a = db = None
    start = time.time()
    try:
        samples = rows(OUT / 'sample_manifest.jsonl.gz')
        a = load('llava')
        a.tokenizer = a.processor.tokenizer
        a.image_processor = a.processor.image_processor
        assert {v.dtype for v in a.model.parameters()} == {torch.float16}
        tokenizer = LlamaTokenizer.from_pretrained(ORIGINAL_TOKENIZER, local_files_only=True, legacy=True)
        versions = [(v._version, v.data_ptr()) for v in a.model.parameters()]
        smoke(a, tokenizer, samples, rank)
        # Neither shard may start formal inference while a peer has a failed gate.
        wait_start = time.time()
        while not all((OUT / f'worker{i}/smoke.json').exists() for i in range(len(resources['workers']))):
            assert not list(OUT.glob('worker*/STOP.json')), 'Peer gate failed'
            assert time.time() - wait_start < 900, 'Peer smoke timeout'
            time.sleep(1)
        verify()
        partition = read(OUT / 'partition.json')['assignment']
        mine = [s for s in samples if partition.get(s['key']) == rank]
        db = database(rank, p)
        lookup = {s['key']: s for s in mine}
        done = set()
        for key, raw, h in db.execute('SELECT key,payload,sha256 FROM results'):
            r = json.loads(raw)
            assert digest(r) == h and r['sample_sha256'] == digest(lookup[key])
            assert r['protocol_sha256'] == digest(p)
            done.add(key)
        for s in mine:
            assert not (OUT / 'STOP_REQUESTED').exists() and not list(OUT.glob('worker*/STOP.json'))
            if s['key'] in done:
                continue
            assert Path(s['image_path']).is_file()
            if len(done) % 250 == 0:
                verify()
                assert __import__('shutil').disk_usage(OUT).free > 2 * 1024**3, 'Disk safety reserve reached'
                assert versions == [(v._version, v.data_ptr()) for v in a.model.parameters()]
            layout, binding = prepare_input(a, s, tokenizer)
            assert layout.length + s['max_new_tokens'] <= a.model.config.text_config.max_position_embeddings
            g, _ = generate(a, layout, cap=s['max_new_tokens'])
            r = dict(key=s['key'], benchmark=s['benchmark'], question_id=s['question_id'],
                     image_sha256=s['image_sha256'], image_id=s['image_id'], question=s['question'],
                     gold=s['answer'], sample_sha256=digest(s), protocol_sha256=digest(p),
                     generation=g, scoring=score(s, g['text']), input_binding=binding,
                     rank=rank, gpu_uuid=uuid, code_commit=resources['code_commit'])
            with db:
                db.execute('INSERT INTO results VALUES(?,?,?)', (s['key'], json.dumps(r, ensure_ascii=False), digest(r)))
            done.add(s['key'])
            a.inputs = {}
            if len(done) % 25 == 0 or len(done) == len(mine):
                state = dict(status='RUNNING', completed=len(done), expected=len(mine),
                             elapsed_seconds=time.time()-start, last_key=s['key'], gpu_uuid=uuid)
                atomic(OUT / f'worker{rank}/progress.json', state)
                print('PROGRESS', rank, len(done), '/', len(mine), s['key'], flush=True)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        atomic(OUT / f'worker{rank}/completion.json', dict(status='COMPLETE_RUNNABLE_SHARD',
               completed=len(done), expected=len(mine), elapsed_seconds=time.time()-start))
    except BaseException:
        atomic(OUT / f'worker{rank}/STOP.json', dict(traceback=traceback.format_exc(), rank=rank, uuid=uuid))
        raise
    finally:
        if db is not None:
            db.close()
        a = None
        gc.collect()
        torch.cuda.empty_cache()
        atomic(OUT / f'worker{rank}/release.json', dict(pid=os.getpid(), uuid=uuid,
               allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
               remaining_context_released_on_process_exit=True))


def driver():
    p = verify()
    resources = read(OUT / 'resources.json')
    for gpu in resources['workers']:
        assert_idle(gpu['uuid'])
    streams, processes = [], []
    try:
        for rank, gpu in enumerate(resources['workers']):
            log = (OUT / f'worker{rank}.log').open('a', buffering=1)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONUNBUFFERED='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            child = subprocess.Popen([resources['python'], '-m', 'vlm_research.svh_middle16.run',
                                      'worker', '--rank', str(rank)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(child)
            streams.append(log)
            # First card must finish its gate before another model is loaded.
            if rank == 0:
                while not (OUT / 'worker0/smoke.json').exists():
                    if child.poll() is not None:
                        raise RuntimeError('First-GPU smoke stopped; see worker0 log/STOP')
                    time.sleep(2)
        while any(c.poll() is None for c in processes):
            if any(c.poll() not in (None, 0) for c in processes):
                raise RuntimeError('Worker failed; stop only task-owned peers')
            time.sleep(5)
        assert all(c.returncode == 0 for c in processes)
        from .analyze import main as analyze
        analyze()
    finally:
        for child in processes:
            if child.poll() is None:
                child.terminate()
        for child in processes:
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in streams:
            log.close()
        atomic(OUT / 'gpu_release_audit.json', hardware())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['worker', 'driver'])
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    driver() if args.mode == 'driver' else worker(args.rank)
