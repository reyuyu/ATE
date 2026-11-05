"""Authorized two-card quality-only execution. No benchmark or profiler entry point."""
import argparse
import gc
import os
import subprocess
import sys
import time
import traceback

import torch

from .common import *
from .engine import generate, prepare
from ..c3v2_middle_reaccess.run import compact
from ..fastp2_bypass_writer.common import hardware
from ..fastp2_bypass_writer.run import operators
from ..p1_layerwise_accuracy.engine import Adapter, tensor_hash
from ..physical_visual_handoff.engine import equivalence


def freeze():
    assert not (OUT / 'execution_freeze.json').exists(), 'Already frozen'
    devices = [dict(index=0, uuid='GPU-a63c6362-dcfd-4a26-c44b-014b40127c89'),
               dict(index=1, uuid='GPU-b08489af-0a8a-eb48-55ba-8346733af417')]
    resource = hardware()
    assert all(d['uuid'] not in resource['processes'] for d in devices), 'Authorized GPU occupied'
    result = subprocess.run([sys.executable, '-m', 'vlm_research.middle10_attribution.tests'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / 'cpu_tests.log').write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    atomic(OUT / 'gpu_authorization.json', dict(at=now(), status='USER_AUTHORIZED',
        devices=devices, user_reply='确认', request='Authorize physical GPU0 and GPU1 for fixed A-E experiment.',
        scope='Current task only; never use GPU2 or GPU3'))
    p = read(OUT / 'protocol.json')
    p.update(GPU_authorization='USER_AUTHORIZED', devices=devices,
        inference_sources=sources(),
        candidate_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    atomic(OUT / 'protocol.json', p)
    atomic(OUT / 'execution_freeze.json', dict(at=now(), sources=sources(), devices=devices,
        weights=read(OUT / 'source_manifest.json')['weights'], code_commit=p['candidate_source_commit'],
        **{n+'_sha256': sha(OUT / (n+'.json')) for n in
           ('sample_manifest', 'partition', 'protocol', 'gpu_authorization')}))
    atomic(OUT / 'authorized_preflight.json', dict(at=now(), **resource))
    print('FROZEN: quality only, physical GPU0/1, no timing.', flush=True)


def verify_freeze():
    f = read(OUT / 'execution_freeze.json')
    assert f['sources'] == sources(), 'Frozen inference source changed'
    for n in ('sample_manifest', 'partition', 'protocol', 'gpu_authorization'):
        assert f[n+'_sha256'] == sha(OUT / (n+'.json')), n
    return f


def check_stop():
    assert not list(OUT.glob('worker*/STOP.json')), 'Task-owned worker failed'


def smoke(a, rank, samples, p):
    checks = []
    for key in p['smoke']:
        check_stop()
        s = samples[key]
        layout, _, binding = prepare(a, s)
        native = generate(a, layout, 'Native', 3, fixed=True, capture='all')
        noop = generate(a, layout, 'Native-Noop', 3, fixed=True, capture='all')
        assert len(native[2]) == len(noop[2]) == 3
        assert all(torch.equal(x, y) for x, y in zip(native[2], noop[2]))
        with torch.inference_mode():
            hf = a.model.generate(**a.inputs, max_new_tokens=32, do_sample=False,
                                  num_beams=1, use_cache=True, logits_to_keep=1)
        natural = generate(a, layout, 'Native', capture='all')
        old = historical(key, 'Native')
        assert binding == old['input_binding']
        assert hf[0, layout.length:].tolist() == natural[0]['token_ids'] == old['generation']['token_ids']
        assert natural[0]['first_logits_sha256'] == old['generation']['first_logits_sha256']
        frozen = generate(a, layout, 'E3-Physical', capture='all')
        assert frozen[0]['token_ids'] == historical(key, 'E3')['generation']['token_ids']
        assert frozen[0]['first_logits_sha256'] == historical(key, 'E3')['generation']['first_logits_sha256']
        del hf, frozen, noop
        details = []
        first_A = None
        for label in LABELS:
            with operators(a) as (calls, mlp, shapes):
                actual = generate(a, layout, label, 3, fixed=True, capture='all', audit=True)
            reference = generate(a, layout, label, 3, fixed=True, capture='all', reference=True)
            e = equivalence('llava', reference[1], actual[1], [], [])
            assert e['passed'] and actual[0]['token_ids'] == reference[0]['token_ids'], (key, label, e)
            assert all(mlp[i] == 3 for i in range(32))
            assert all(calls[i, op] == 3 for i in range(32) for op in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
            assert all(r['shape'][1] in (layout.length, 1) for r in shapes)
            assert len(actual[0]['audit']) == 96
            assert all(r['blocked_probability_max'] == 0 for r in actual[0]['audit'])
            if label == 'A':
                frozen_dense = generate(a, layout, 'E3-Dense', 3, fixed=True, capture='all')
                assert all(torch.equal(x, y) for x, y in zip(actual[2], frozen_dense[2]))
                first_A = actual[1][0].clone()
                del frozen_dense
            if label in ('D', 'E'):
                assert torch.equal(actual[1][0], native[1][0]), 'Native Prefill not restored'
            g = actual[0]
            detail = dict(condition=label, numerical=e, MLP_complete=True,
                native_prefill_first_logits_exact=label in ('D', 'E'),
                first_logits_sha256=g['first_logits_sha256'],
                visual_audit=g['visual_audit'], mask_audit=g['audit'], projection_shapes=shapes)
            del actual, reference
            actual = generate(a, layout, label, capture='all')
            reference = generate(a, layout, label, capture='all', reference=True)
            e = equivalence('llava', reference[1], actual[1], [], [])
            assert e['passed'] and actual[0]['token_ids'] == reference[0]['token_ids'], (key, label, e)
            detail.update(natural_numerical=e, natural_tokens=actual[0]['token_ids'])
            details.append(detail)
            del actual, reference
            a.verify()
        replay = generate(a, layout, 'A', 3, fixed=True, capture='all')
        assert torch.equal(replay[1][0], first_A), 'Cache/hook contamination across conditions'
        del replay, native, natural
        checks.append(dict(key=key, native_noop_exact=True, native_HF_tokens_exact=True,
            historical_native_E3_exact=True, cache_replay_exact=True, conditions=details))
        print('SMOKE', rank, key, 'PASS', flush=True)
        atomic(OUT / f'worker{rank}/smoke_partial.json', dict(sources=sources(), checks=checks))
    atomic(OUT / f'worker{rank}/smoke.json', dict(at=now(), status='PASS', sources=sources(), checks=checks))


def worker(rank):
    f = verify_freeze()
    d = f['devices'][rank]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == d['uuid']
    resource = hardware()
    assert d['uuid'] not in resource['processes'], 'Device is no longer idle'
    atomic(OUT / f'worker{rank}/preflight.json', dict(at=now(), pid=os.getpid(), **resource))
    a = None
    try:
        samples, p = read(OUT / 'sample_manifest.json'), read(OUT / 'partition.json')
        a = Adapter('llava')
        atomic(OUT / f'worker{rank}/environment.json', a.environment)
        if not (OUT / f'worker{rank}/smoke.json').exists():
            smoke(a, rank, samples, p)
        start = time.monotonic()
        while not all((OUT / f'worker{r}/smoke.json').exists() for r in range(2)):
            check_stop()
            assert time.monotonic()-start < 900
            time.sleep(1)
        assert all(read(OUT / f'worker{r}/smoke.json')['sources'] == sources() for r in range(2))
        keys = [k for k in sorted(samples) if p['image_shard'][samples[k]['image_sha256']] == rank]
        assert len(keys) == 250
        start = time.monotonic()
        for ix, key in enumerate(keys):
            check_stop()
            s = samples[key]
            layout, _, binding = prepare(a, s)
            assert all(historical(key, label)['input_binding'] == binding for label in BASELINES)
            for label in LABELS:
                path = checkpoint(key, label)
                if path.exists():
                    old = load(path)
                    assert old['sources'] == sources() and old['input_binding'] == binding
                    assert old['sample_sha256'] == digest(s)
                    continue
                g, logits, *_ = generate(a, layout, label)
                if label in ('D', 'E'):
                    assert g['first_logits_sha256'] == historical(key, 'Native')['generation']['first_logits_sha256']
                    assert g['first_top1_token'] == historical(key, 'Native')['generation']['first_top1_token']
                full_token = historical(key, 'Native')['generation']['first_top1_token']
                g['native_first_token_logp'] = float(logits[0].log_softmax(-1)[0, full_token])
                save(path, dict(at=now(), key=key, condition=label, rank=rank,
                    sample_sha256=digest(s), input_binding=binding, sources=sources(),
                    score=score(s, g['text']), generation=compact(g)))
                del g, logits
            a.verify()
            if (ix+1) % 10 == 0 or ix+1 == len(keys):
                progress = dict(at=now(), stage='quality_only', questions=ix+1, total=250,
                    conditions=5, elapsed_seconds=time.monotonic()-start, performance_requests=0)
                atomic(OUT / f'worker{rank}/progress.json', progress)
                print('QUALITY', rank, progress, flush=True)
        atomic(OUT / f'worker{rank}/completion.json', dict(at=now(), status='COMPLETE',
            quality_generations=1250, performance_requests=0))
    except BaseException:
        atomic(OUT / f'worker{rank}/STOP.json', dict(at=now(), traceback=traceback.format_exc()))
        raise
    finally:
        del a
        gc.collect()
        torch.cuda.empty_cache()
        atomic(OUT / f'worker{rank}/release.json', dict(at=now(), pid=os.getpid(),
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            process_exit_releases_remaining=True))


def driver():
    f = verify_freeze()
    resource = hardware()
    assert all(d['uuid'] not in resource['processes'] for d in f['devices'])
    atomic(OUT / 'driver_preflight.json', dict(at=now(), **resource))
    jobs, streams = [], []
    try:
        for rank, d in enumerate(f['devices']):
            path = OUT / f'worker{rank}/execution.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('a')
            streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=d['uuid'], OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
            jobs.append(subprocess.Popen([sys.executable, '-u', '-m', 'vlm_research.middle10_attribution.run',
                '--rank', str(rank)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT))
        atomic(OUT / 'driver_pids.json', dict(at=now(), driver=os.getpid(), workers=[p.pid for p in jobs]))
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None, 0) for p in jobs):
                raise RuntimeError('Quality worker failed; stop only task-owned peer')
            time.sleep(1)
        codes = [p.returncode for p in jobs]
        atomic(OUT / 'driver_exit.json', dict(at=now(), returncodes=codes))
        assert codes == [0, 0]
    finally:
        for p in jobs:
            if p.poll() is None:
                p.terminate()
        for p in jobs:
            if p.poll() is None:
                p.wait(timeout=60)
        for s in streams:
            s.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--driver', action='store_true')
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.driver:
        driver()
    else:
        assert args.rank in (0, 1)
        worker(args.rank)
