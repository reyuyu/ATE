"""Two resident, paired image-shard workers with a global Smoke barrier."""
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


def verify_freeze():
    f = read(OUT / 'execution_freeze.json')
    assert f['sources'] == sources(), 'Execution source changed after freeze'
    for name in ('sample_manifest', 'partition', 'protocol', 'gpu_authorization'):
        assert sha(OUT / (name + '.json')) == f[name + '_sha256'], name
    return f


def check_stop():
    assert not list(OUT.glob('worker*/STOP.json')), 'Task-owned worker failed'


def barrier(filename):
    start = time.monotonic()
    while not all((OUT / f'worker{r}' / filename).exists() for r in range(2)):
        check_stop()
        assert time.monotonic() - start < 14400, 'Peer stage exceeded four hours'
        time.sleep(1)


def historical(key, label):
    return load(HISTORY / 'checkpoints' / key / (label + '.json'))


def smoke(a, rank, samples, partition):
    checks = []
    for key in partition['smoke']:
        s = samples[key]
        layout, _, binding = prepare(a, s)
        native = generate(a, layout, 'Native', 3, fixed=True, capture='all')
        noop = generate(a, layout, 'Native', 3, fixed=True, capture='all', noop=True)
        assert all(torch.equal(x, y) for x, y in zip(native[2], noop[2]))
        assert len(native[2]) == len(noop[2]) == 3
        with torch.inference_mode():
            hf = a.model.generate(**a.inputs, max_new_tokens=32, do_sample=False,
                                  num_beams=1, use_cache=True, logits_to_keep=1)
        native_natural = generate(a, layout, 'Native', 32, capture='first')
        assert hf[0, layout.length:].tolist() == native_natural[0]['token_ids']
        del hf, native, noop, native_natural
        baseline_checks = []
        for label in BASELINES:
            g, logits, *_ = generate(a, layout, label, 32, capture='first')
            old = historical(key, label)
            assert old['input_binding'] == binding and old['sample_sha256'] == digest(s)
            assert g['token_ids'] == old['generation']['token_ids'], (key, label, 'historical tokens')
            assert tensor_hash(logits[0]) == old['generation']['first_logits_sha256'], (key, label, 'H1 logits')
            baseline_checks.append(dict(condition=label, history_tokens_exact=True,
                                        history_first_logits_exact=True))
            del g, logits
        details = []
        for label in NEW:
            with operators(a) as (calls, mlp, shapes):
                actual = generate(a, layout, label, 3, fixed=True, capture='all', audit=True)
            ref = generate(a, layout, label, 3, fixed=True, capture='all', audit=True, dense=True)
            e = equivalence('llava', ref[1], actual[1], ref[3], actual[3])
            assert e['passed'] and actual[0]['token_ids'] == ref[0]['token_ids'], (key, label, e)
            assert all(mlp[i] == 3 for i in range(32))
            for row in shapes:
                i, op = row['layer'], row['op']
                n = layout.length if i < CUTOFF[label] else len(layout.text)
                if i in (2, 3, 4, 5) and op in ('q_proj', 'o_proj') and label != 'E4':
                    n -= 1
                assert row['shape'][1] in (1, n), (label, row, n)
            assert all(calls[i, op] == 1 for i in (2, 3, 4, 5)
                       for op in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
            assert actual[0]['kv']['retained_image_layers'] == list(KEEP[label])
            assert actual[0]['kv']['historical_cache_copies'] == 0
            assert actual[0]['kv']['append_concatenations'] == 0
            assert actual[0]['audit'] and all(r['blocked_probability_max'] == 0 for r in actual[0]['audit'])
            for x, y in zip(actual[4], ref[4]):
                assert x.keys() == y.keys() and all(torch.equal(x[f], y[f]) for f in x)
            detail = dict(condition=label, numerical=e, projection_shapes=shapes,
                          kv=actual[0]['kv'], actual_visual_probabilities=actual[0]['visual_audit'])
            del actual, ref
            actual = generate(a, layout, label, 32, capture='all')
            ref = generate(a, layout, label, 32, capture='all', dense=True)
            e = equivalence('llava', ref[1], actual[1], [], [])
            assert e['passed'] and actual[0]['token_ids'] == ref[0]['token_ids'], (key, label, e)
            detail.update(natural_numerical=e, natural_tokens=actual[0]['token_ids'],
                          first_logits_sha256=tensor_hash(actual[1][0]))
            details.append(detail)
            del actual, ref
            a.verify()
        checks.append(dict(key=key, native_noop_strict=True, native_official_generate_exact=True,
                           baselines=baseline_checks, conditions=details))
        atomic(OUT / f'worker{rank}/smoke_partial.json', dict(checks=checks))
        print('SMOKE PASS', rank, key, flush=True)
    atomic(OUT / f'worker{rank}/smoke.json', dict(status='PASS', at=now(), sources=sources(), checks=checks))


def quality(a, rank, samples, partition):
    keys = sorted((k for k, s in samples.items() if partition['image_shard'][s['image_sha256']] == rank),
                  key=lambda k: (samples[k]['image_sha256'], k))
    started = time.monotonic()
    for ix, key in enumerate(keys):
        check_stop()
        s = samples[key]
        layout, _, binding = prepare(a, s)
        for label in BASELINES:
            old = historical(key, label)
            assert old['input_binding'] == binding and old['sample_sha256'] == digest(s)
        for label in NEW:
            path = quality_path(key, label)
            if path.exists():
                old = load(path)
                assert old['sources'] == sources() and old['sample_sha256'] == digest(s)
                assert old['input_binding'] == binding
                continue
            g, logits, *_ = generate(a, layout, label, 32, capture='first')
            lp = logits[0].log_softmax(-1)
            g.update(first_logits_sha256=tensor_hash(logits[0]), first_top1_token=int(logits[0].argmax()),
                     first_top1_logp=float(lp.max()))
            g['native_first_token_logp'] = float(lp[0, historical(key, 'Native')['generation']['first_top1_token']])
            g['middle10_first_token_logp'] = float(lp[0, historical(key, 'C3+Middle10')['generation']['first_top1_token']])
            save(path, dict(key=key, condition=label, rank=rank, at=now(), sources=sources(),
                sample_sha256=digest(s), input_binding=binding, generation=compact(g), score=score(s, g['text'])))
            del g, logits, lp
            a.verify()
        if ix % 10 == 0 or ix + 1 == len(keys):
            progress = dict(at=now(), stage='quality', questions=ix+1, total=len(keys),
                            elapsed_seconds=time.monotonic()-started)
            atomic(OUT / f'worker{rank}/progress.json', progress)
            print('QUALITY', rank, progress, flush=True)
    atomic(OUT / f'worker{rank}/quality_complete.json', dict(status='COMPLETE', questions=len(keys), at=now()))


def performance_selection(samples):
    means = {label: sum(load(quality_path(k, label))['score'] for k in samples)/len(samples) for label in NEW}
    m10 = sum(historical(k, 'C3+Middle10')['score'] for k in samples)/len(samples)
    eligible = [label for label in NEW if means[label] > m10 + 1e-12]
    value = dict(at=now(), criterion='Positive complete500 point-estimate quality gain versus M10; not a significance claim',
                 middle10_accuracy=m10, new_accuracy=means,
                 conditions=['Native', 'C3-V2', 'C3+Middle10']+eligible if eligible else [])
    atomic(OUT / 'performance_selection.json', value)
    return value['conditions']


def performance(a, rank, samples, partition, labels):
    keys = [k for k in partition['performance48'] if partition['image_shard'][samples[k]['image_sha256']] == rank]
    started = time.monotonic()
    for ix, key in enumerate(keys):
        check_stop()
        s = samples[key]
        layout, _, binding = prepare(a, s)
        for mode in ('natural_eos', 'fixed16'):
            for repeat in range(-3, 5):
                shift = (ix+repeat+3) % len(labels)
                for label in labels[shift:]+labels[:shift]:
                    path = performance_path(key, mode, repeat, label)
                    if path.exists():
                        old = load(path)
                        assert old['sources'] == sources() and old['sample_sha256'] == digest(s)
                        assert old['input_binding'] == binding
                        continue
                    g, *_ = generate(a, layout, label, 16 if mode == 'fixed16' else 32, fixed=mode == 'fixed16')
                    if mode == 'natural_eos':
                        expected = (historical(key, label) if label in BASELINES else load(quality_path(key, label)))['generation']['token_ids']
                        assert g['token_ids'] == expected, 'Performance versus quality token mismatch'
                    else:
                        assert len(g['token_ids']) == 16
                    save(path, dict(key=key, condition=label, rank=rank, at=now(), mode=mode,
                        repeat=repeat, warmup=repeat<0, sources=sources(), sample_sha256=digest(s),
                        input_binding=binding, generation=compact(g)))
                    del g
            a.verify()
        progress = dict(at=now(), stage='performance', questions=ix+1, total=len(keys),
                        elapsed_seconds=time.monotonic()-started)
        atomic(OUT / f'worker{rank}/progress.json', progress)
        print('PERFORMANCE', rank, progress, flush=True)
    atomic(OUT / f'worker{rank}/performance_complete.json', dict(status='COMPLETE', at=now(), labels=labels, questions=len(keys)))


def worker(rank):
    freeze = verify_freeze()
    device = freeze['devices'][rank]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == device['uuid']
    resources = hardware()
    assert device['uuid'] not in resources['processes'], 'GPU no longer idle'
    atomic(OUT / f'worker{rank}/preflight.json', dict(at=now(), pid=os.getpid(), **resources))
    a = None
    try:
        samples, partition = read(OUT / 'sample_manifest.json'), read(OUT / 'partition.json')
        a = Adapter('llava')
        atomic(OUT / f'worker{rank}/environment.json', a.environment)
        if not (OUT / f'worker{rank}/smoke.json').exists():
            smoke(a, rank, samples, partition)
        else:
            assert read(OUT / f'worker{rank}/smoke.json')['sources'] == sources()
        barrier('smoke.json')
        quality(a, rank, samples, partition)
        barrier('quality_complete.json')
        if rank == 0:
            performance_selection(samples)
        start = time.monotonic()
        while not (OUT / 'performance_selection.json').exists():
            check_stop()
            assert time.monotonic()-start < 120
            time.sleep(1)
        labels = read(OUT / 'performance_selection.json')['conditions']
        if labels:
            performance(a, rank, samples, partition, labels)
        atomic(OUT / f'worker{rank}/completion.json', dict(status='COMPLETE', at=now(), performance_conditions=labels))
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
    freeze = verify_freeze()
    resource = hardware()
    assert all(d['uuid'] not in resource['processes'] for d in freeze['devices'])
    atomic(OUT / 'driver_preflight.json', dict(at=now(), **resource))
    processes, streams = [], []
    try:
        for rank, device in enumerate(freeze['devices']):
            path = OUT / f'worker{rank}/execution.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('a')
            streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device['uuid'], PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            processes.append(subprocess.Popen([sys.executable, '-u', '-m', 'vlm_research.middle10_recovery.run',
                '--rank', str(rank)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT))
        atomic(OUT / 'driver_pids.json', dict(at=now(), driver=os.getpid(), workers=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (0, None) for p in processes):
                raise RuntimeError('Task worker failed; terminate only its task-owned peer')
            time.sleep(1)
        codes = [p.returncode for p in processes]
        atomic(OUT / 'driver_exit.json', dict(at=now(), returncodes=codes))
        assert codes == [0, 0]
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            if p.poll() is None:
                p.wait(timeout=60)
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', action='store_true')
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.driver:
        driver()
    else:
        assert args.rank in (0, 1)
        worker(args.rank)
