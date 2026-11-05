"""Two authorized resident workers; correctness and quality, never timing."""
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
from ..p1_layerwise_accuracy.engine import Adapter
from ..physical_visual_handoff.engine import equivalence


def freeze(devices, authorization_text):
    assert not (OUT / 'execution_freeze.json').exists(), 'Already frozen'
    assert authorization_text.strip(), 'Fresh explicit GPU authorization is required'
    assert len(devices) == 2 and len(set(devices)) == 2
    resource = hardware()
    rows = {int(r.split(',')[0]): [v.strip() for v in r.split(',')]
            for r in resource['gpus'].strip().splitlines()}
    selected = [dict(index=i, uuid=rows[i][1]) for i in devices]
    assert all(d['uuid'] not in resource['processes'] for d in selected), 'GPU occupied'
    assert all(int(rows[i][4]) >= 22000 and int(rows[i][5]) == 0 for i in devices)
    result = subprocess.run([sys.executable, '-m', 'vlm_research.middle10_decode_regions.tests'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / 'cpu_tests.log').write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    atomic(OUT / 'gpu_authorization.json', dict(at=now(), status='USER_AUTHORIZED',
        devices=selected, user_reply=authorization_text, scope='This fixed Early/Late quality task only'))
    p = read(OUT / 'protocol.json')
    p.update(GPU_authorization='USER_AUTHORIZED', devices=selected, inference_sources=sources(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    atomic(OUT / 'protocol.json', p)
    atomic(OUT / 'execution_freeze.json', dict(at=now(), sources=sources(), devices=selected,
        weights=read(OUT / 'source_manifest.json')['weights'], code_commit=p['code_commit'],
        **{n+'_sha256': sha(OUT / (n+'.json')) for n in
           ('sample_manifest', 'partition', 'protocol', 'gpu_authorization')}))
    atomic(OUT / 'authorized_preflight.json', dict(at=now(), **resource))
    print('FROZEN: only 1000 new quality generations; zero performance requests.', flush=True)


def verify_freeze():
    f = read(OUT / 'execution_freeze.json')
    assert f['sources'] == sources(), 'Frozen inference source changed'
    for n in ('sample_manifest', 'partition', 'protocol', 'gpu_authorization'):
        assert f[n+'_sha256'] == sha(OUT / (n+'.json')), n
    return f


def check_stop():
    assert not list(OUT.glob('worker*/STOP.json')), 'A task-owned worker failed'


def smoke(a, rank, samples, partition):
    checks = []
    for key in partition['smoke']:
        check_stop()
        layout, _, binding = prepare(a, samples[key])
        native = generate(a, layout, 'Native', max_tokens=3, fixed=True, capture='all')
        noop = generate(a, layout, 'Native-Noop', max_tokens=3, fixed=True, capture='all')
        assert len(native[2]) == len(noop[2]) == 3
        assert all(torch.equal(x, y) for x, y in zip(native[2], noop[2]))
        with torch.inference_mode():
            hf = a.model.generate(**a.inputs, max_new_tokens=32, do_sample=False,
                                  num_beams=1, use_cache=True, logits_to_keep=1)
        old_native, old_e = historical(key, 'Native'), historical(key, 'E')
        assert binding == old_native['input_binding'] == old_e['input_binding']
        assert hf[0, layout.length:].tolist() == old_native['generation']['token_ids']
        assert native[0]['first_logits_sha256'] == old_native['generation']['first_logits_sha256']
        baseline = generate(a, layout, 'E', capture='all')
        assert baseline[0]['token_ids'] == old_e['generation']['token_ids']
        assert baseline[0]['first_logits_sha256'] == old_e['generation']['first_logits_sha256']
        del hf, baseline, noop
        details = []
        for label in LABELS:
            with operators(a) as (calls, mlp, shapes):
                actual = generate(a, layout, label, max_tokens=3, fixed=True, capture='all', audit=True)
            reference = generate(a, layout, label, max_tokens=3, fixed=True, capture='all', reference=True)
            e = equivalence('llava', reference[1], actual[1], [], [])
            assert e['passed'] and actual[0]['token_ids'] == reference[0]['token_ids'], (key, label, e)
            assert torch.equal(native[1][0], actual[1][0]), 'Prefill must remain strictly native'
            assert all(mlp[i] == 3 for i in range(32))
            assert all(calls[i, op] == 3 for i in range(32) for op in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
            assert all(r['shape'][1] in (layout.length, 1) for r in shapes)
            assert len(actual[0]['audit']) == 96
            assert all(r['blocked_probability_max'] == 0 for r in actual[0]['audit'])
            assert len(actual[0]['rotary_audit']) == 3
            assert all(r['position_ids'] ==
                       (list(range(layout.length)) if i == 0 else [layout.length+i-1])
                       for i, r in enumerate(actual[0]['rotary_audit']))
            detail = dict(condition=label, numerical=e, native_prefill_exact=True,
                MLP_complete=True, projection_shapes=shapes,
                visual_audit=actual[0]['visual_audit'], mask_audit=actual[0]['audit'],
                rotary_audit=actual[0]['rotary_audit'])
            del actual, reference
            actual = generate(a, layout, label, capture='all')
            reference = generate(a, layout, label, capture='all', reference=True)
            e = equivalence('llava', reference[1], actual[1], [], [])
            assert e['passed'] and actual[0]['token_ids'] == reference[0]['token_ids'], (key, label, e)
            detail.update(natural_numerical=e, natural_tokens=actual[0]['token_ids'])
            details.append(detail)
            del actual, reference
            a.verify()
        replay = generate(a, layout, 'E', capture='all')
        assert replay[0]['token_ids'] == old_e['generation']['token_ids']
        assert replay[0]['first_logits_sha256'] == old_e['generation']['first_logits_sha256']
        del replay, native
        checks.append(dict(key=key, native_noop_exact=True, native_HF_and_history_exact=True,
            historical_E_exact=True, cache_replay_exact=True, conditions=details))
        atomic(OUT / f'worker{rank}/smoke_partial.json', dict(sources=sources(), checks=checks))
        print('SMOKE', rank, key, 'PASS', flush=True)
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
            sample = samples[key]
            layout, _, binding = prepare(a, sample)
            assert binding == historical(key, 'Native')['input_binding'] == historical(key, 'E')['input_binding']
            for label in LABELS:
                path = checkpoint(key, label)
                if path.exists():
                    old = load(path)
                    assert old['sources'] == sources() and old['input_binding'] == binding
                    assert old['sample_sha256'] == digest(sample) and old['rank'] == rank
                    continue
                g, logits, *_ = generate(a, layout, label)
                assert g['first_logits_sha256'] == historical(key, 'Native')['generation']['first_logits_sha256']
                save(path, dict(at=now(), key=key, condition=label, rank=rank,
                    sample_sha256=digest(sample), input_binding=binding, sources=sources(),
                    score=score(sample, g['text']), generation=compact(g)))
                del g, logits
            a.verify()
            if (ix+1) % 10 == 0 or ix+1 == len(keys):
                progress = dict(at=now(), stage='quality_only', questions=ix+1, total=250,
                    conditions=2, elapsed_seconds=time.monotonic()-start, performance_requests=0)
                atomic(OUT / f'worker{rank}/progress.json', progress)
                print('QUALITY', rank, progress, flush=True)
        atomic(OUT / f'worker{rank}/completion.json', dict(at=now(), status='COMPLETE',
            quality_generations=500, performance_requests=0))
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
            jobs.append(subprocess.Popen([sys.executable, '-u', '-m', 'vlm_research.middle10_decode_regions.run',
                '--rank', str(rank)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT))
        atomic(OUT / 'driver_pids.json', dict(at=now(), driver=os.getpid(), workers=[p.pid for p in jobs]))
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None, 0) for p in jobs):
                raise RuntimeError('Worker failed; stop only task-owned peer')
            time.sleep(1)
        codes = [p.returncode for p in jobs]
        atomic(OUT / 'driver_exit.json', dict(at=now(), returncodes=codes))
        assert codes == [0, 0]
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
        for job in jobs:
            if job.poll() is None:
                job.wait(timeout=60)
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--authorization-text', default='')
    parser.add_argument('--driver', action='store_true')
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze([int(v) for v in args.devices.split(',')], args.authorization_text)
    elif args.driver:
        driver()
    else:
        assert args.rank in (0, 1)
        worker(args.rank)
