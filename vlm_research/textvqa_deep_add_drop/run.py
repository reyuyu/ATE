"""Two resident models; all22 conditions paired within each image shard."""
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


def freeze():
    assert not (OUT / 'execution_freeze.json').exists(), 'Already frozen'
    resource = hardware()
    gpu_rows = {int(r.split(',')[0]): [v.strip() for v in r.split(',')]
                for r in resource['gpus'].strip().splitlines()}
    devices = [dict(index=i, uuid=gpu_rows[i][1]) for i in (0, 1)]
    assert all(d['uuid'] not in resource['processes'] for d in devices), 'Selected GPU occupied'
    assert all(int(gpu_rows[i][4]) >= 22000 and int(gpu_rows[i][5]) == 0 for i in (0, 1))
    p = read(OUT / 'protocol.json')
    assert p['source_hashes'] == sources(), 'Source changed after CPU preparation'
    p.update(devices=devices, authorization='User standing authorization: 确认允许，以后都确认; current request permits at most two idle GPUs.',
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    atomic(OUT / 'protocol.json', p)
    atomic(OUT / 'execution_freeze.json', dict(at=now(), sources=sources(), devices=devices,
        code_commit=p['code_commit'], weights=read(OUT / 'source_manifest.json')['weights'],
        **{n+'_sha256': sha(OUT / (n+'.json')) for n in
           ('sample_manifest', 'selection', 'partition', 'source_manifest', 'protocol')}))
    atomic(OUT / 'authorized_preflight.json', dict(at=now(), **resource))
    print('FROZEN: physical GPU0/1, Decode Add/Drop only, no timing.', flush=True)


def verify_freeze():
    f = read(OUT / 'execution_freeze.json')
    assert f['sources'] == sources(), 'Frozen inference source changed'
    for n in ('sample_manifest', 'selection', 'partition', 'source_manifest', 'protocol'):
        assert f[n+'_sha256'] == sha(OUT / (n+'.json')), n
    return f


def check_stop():
    assert not list(OUT.glob('worker*/STOP.json')), 'Task-owned worker failed'


def smoke(a, rank, samples, partition):
    checks = []
    for key in partition['smoke']:
        check_stop()
        layout, _, binding = prepare(a, samples[key])
        native = generate(a, layout, 'Native', max_tokens=3, fixed=True, capture='all')
        noop = generate(a, layout, 'Native-Noop', max_tokens=3, fixed=True, capture='all')
        assert len(native[2]) == len(noop[2]) == 3
        assert all(torch.equal(x, y) for x, y in zip(native[2], noop[2]))
        for label in BASELINES:
            baseline = generate(a, layout, label, capture='all')
            old = historical(key, label)
            assert binding == old['input_binding']
            assert baseline[0]['token_ids'] == old['generation']['token_ids']
            assert baseline[0]['first_logits_sha256'] == old['generation']['first_logits_sha256']
            del baseline
        details = []
        for label in LABELS:
            with operators(a) as (calls, mlp, shapes):
                actual = generate(a, layout, label, max_tokens=3, fixed=True, capture='all', audit=True)
            reference = generate(a, layout, label, max_tokens=3, fixed=True, capture='all', reference=True)
            e = equivalence('llava', reference[1], actual[1], [], [])
            assert e['passed'] and actual[0]['token_ids'] == reference[0]['token_ids'], (key, label, e)
            assert torch.equal(actual[1][0], native[1][0]), 'Native Prefill changed'
            assert all(mlp[i] == 3 for i in range(32))
            assert all(calls[i, op] == 3 for i in range(32) for op in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
            assert all(r['shape'][1] in (layout.length, 1) for r in shapes)
            assert len(actual[0]['audit']) == 96
            assert all(r['blocked_probability_max'] == 0 for r in actual[0]['audit'])
            assert [r['position_ids'] for r in actual[0]['rotary_audit']] == [list(range(layout.length)), [layout.length], [layout.length+1]]
            details.append(dict(condition=label, numerical=e, native_prefill_exact=True,
                mask_audit=actual[0]['audit'], visual_audit=actual[0]['visual_audit'],
                rotary_audit=actual[0]['rotary_audit'], MLP_complete=True,
                three_step_tokens=actual[0]['token_ids']))
            del actual, reference
            a.verify()
        replay = generate(a, layout, 'Late', capture='all')
        assert replay[0]['token_ids'] == historical(key, 'Late')['generation']['token_ids']
        assert replay[0]['first_logits_sha256'] == historical(key, 'Native')['generation']['first_logits_sha256']
        del native, noop, replay
        checks.append(dict(key=key, native_noop_exact=True, all3_historical_baselines_exact=True,
            cache_replay_exact=True, conditions=details))
        atomic(OUT / f'worker{rank}/smoke_partial.json', dict(sources=sources(), checks=checks))
        print('SMOKE', rank, key, 'PASS', flush=True)
    atomic(OUT / f'worker{rank}/smoke.json', dict(at=now(), status='PASS', sources=sources(), checks=checks))


def worker(rank):
    f = verify_freeze()
    d = f['devices'][rank]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == d['uuid']
    resource = hardware()
    assert d['uuid'] not in resource['processes'], 'GPU no longer idle'
    atomic(OUT / f'worker{rank}/preflight.json', dict(at=now(), pid=os.getpid(), **resource))
    a = None
    try:
        samples, partition = read(OUT / 'sample_manifest.json'), read(OUT / 'partition.json')
        a = Adapter('llava')
        atomic(OUT / f'worker{rank}/environment.json', a.environment)
        if not (OUT / f'worker{rank}/smoke.json').exists():
            smoke(a, rank, samples, partition)
        start = time.monotonic()
        while not all((OUT / f'worker{r}/smoke.json').exists() for r in range(2)):
            check_stop()
            assert time.monotonic()-start < 900
            time.sleep(1)
        assert all(read(OUT / f'worker{r}/smoke.json')['sources'] == sources() for r in range(2))
        keys = [k for k in sorted(samples) if partition['image_shard'][samples[k]['image_sha256']] == rank]
        assert len(keys) == partition['question_counts'][rank]
        start = time.monotonic()
        for ix, key in enumerate(keys):
            check_stop()
            sample = samples[key]
            layout, _, binding = prepare(a, sample)
            assert all(binding == historical(key, label)['input_binding'] for label in BASELINES)
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
            if (ix+1) % 5 == 0 or ix+1 == len(keys):
                progress = dict(at=now(), stage='quality_only', questions=ix+1, total=len(keys),
                    conditions=22, generations=(ix+1)*22, elapsed_seconds=time.monotonic()-start,
                    performance_requests=0)
                atomic(OUT / f'worker{rank}/progress.json', progress)
                print('QUALITY', rank, progress, flush=True)
        atomic(OUT / f'worker{rank}/completion.json', dict(at=now(), status='COMPLETE',
            quality_generations=len(keys)*22, performance_requests=0))
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
            jobs.append(subprocess.Popen([sys.executable, '-u', '-m', 'vlm_research.textvqa_deep_add_drop.run',
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
