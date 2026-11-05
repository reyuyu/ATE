"""One new condition on the original500, without any performance measurements."""
import argparse
import gc
import os
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import torch

from .common import *
from .engine import generate, prepare
from ..c3v2_middle_reaccess.run import compact
from ..fastp2_bypass_writer.common import hardware
from ..p1_layerwise_accuracy.engine import Adapter
from ..textvqa_deep_add_drop import run as audited_runner


def prepare_cpu():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'execution_freeze.json').exists()
    test = subprocess.run([sys.executable, '-m', 'vlm_research.textvqa_decode_l11_l24.tests'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / 'cpu_tests.log').write_text(test.stdout)
    assert test.returncode == 0, test.stdout
    old = read(PREVIOUS / 'execution_freeze.json')
    assert sha(PREVIOUS / 'sample_manifest.json') == old['sample_manifest_sha256']
    assert old['sources'] == previous.sources()
    samples = read(PREVIOUS / 'sample_manifest.json')
    subset = read(SUBSET / 'sample_manifest.json')
    assert len(samples) == 500 and len(subset) == 199
    assert all(samples[k] == s for k, s in subset.items())
    atomic(OUT / 'sample_manifest.json', samples)
    groups = {}
    for key, sample in samples.items():
        assert sample['dataset'] == 'textvqa' and len(sample['answer']) == 10
        groups.setdefault(sample['image_sha256'], []).append(key)
    loads, assignment = [0, 0], {}
    for image in sorted(groups, key=lambda h: (-len(groups[h]), stable(h))):
        rank = min(range(2), key=lambda r: (loads[r], r))
        assignment[image] = rank
        loads[rank] += len(groups[image])
    atomic(OUT / 'partition.json', dict(image_shard=assignment, question_counts=loads,
        smoke=read(SUBSET / 'partition.json')['smoke'], subset199_keys=sorted(subset),
        subset199_manifest_sha256=sha(SUBSET / 'sample_manifest.json')))
    for path, expected in {s['image_path']: s['image_sha256'] for s in samples.values()}.items():
        assert sha(path) == expected, path
    for path, expected in old['weights'].items():
        assert sha(path) == expected, path
    roots = {'Native': previous.HISTORY, 'E': previous.PREVIOUS, 'Late': PREVIOUS}
    freezes = {c: read(p / 'execution_freeze.json') for c, p in roots.items()}
    records = []
    for key, sample in samples.items():
        native = historical(key, 'Native')
        for label in BASELINES:
            value = historical(key, label)
            assert value['key'] == key and value['condition'] == label
            assert value['sample_sha256'] == digest(sample)
            assert value['sources'] == freezes[label]['sources']
            assert value['input_binding'] == native['input_binding']
            assert value['generation']['first_logits_sha256'] == native['generation']['first_logits_sha256']
            assert value['score'] == score(sample, value['generation']['text'])
            records.append(dict(key=key, condition=label, path=str(historical_path(key, label)),
                sha256=sha(historical_path(key, label))))
    atomic(OUT / 'source_manifest.json', dict(weights=old['weights'], reused_records=records))
    atomic(OUT / 'protocol.json', dict(created_at=now(), base_commit='84367c439741b00287c5cc58bac86f769a186a7e',
        model='LLaVA-1.5-7B', revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
        dtype='FP16', attention='eager', transformers='4.49.0', logits_to_keep=1,
        condition=CONDITION, first_layer=FIRST_LAYER, decode_image_layers=KEEP, greedy=True, max_new_tokens=32,
        questions=500, image_clusters=len(groups), subset_questions=199,
        entire_prefill_and_p1='Native in all32 layers', decode_text_attention='Native in all32 layers',
        independent_KV=True, no_performance_tests=True, no_physical_optimization=True,
        bootstrap=dict(cluster='Image SHA256', replicates=20000, seed=SEED, question_equal=True),
        source_hashes=sources(), tolerances=TOLERANCES['llava']))
    print(dict(status='CPU_READY', questions=500, shard_questions=loads), flush=True)


def freeze():
    assert not (OUT / 'execution_freeze.json').exists()
    resource = hardware()
    rows = {int(r.split(',')[0]): [v.strip() for v in r.split(',')]
            for r in resource['gpus'].strip().splitlines()}
    devices = [dict(index=i, uuid=rows[i][1]) for i in (0, 1)]
    assert all(d['uuid'] not in resource['processes'] for d in devices)
    assert all(int(rows[i][4]) >= 22000 and int(rows[i][5]) == 0 for i in (0, 1))
    p = read(OUT / 'protocol.json')
    assert p['source_hashes'] == sources()
    p.update(devices=devices, authorization='Standing user authorization: 确认允许，以后都确认',
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    atomic(OUT / 'protocol.json', p)
    atomic(OUT / 'execution_freeze.json', dict(at=now(), sources=sources(), devices=devices,
        code_commit=p['code_commit'], **{n+'_sha256': sha(OUT / (n+'.json')) for n in
        ('sample_manifest', 'partition', 'source_manifest', 'protocol')}))
    atomic(OUT / 'authorized_preflight.json', dict(at=now(), **resource))


def verify_freeze():
    f = read(OUT / 'execution_freeze.json')
    assert f['sources'] == sources()
    for n in ('sample_manifest', 'partition', 'source_manifest', 'protocol'):
        assert f[n+'_sha256'] == sha(OUT / (n+'.json'))
    return f


def check_stop():
    assert not list(OUT.glob('worker*/STOP.json')), 'A task-owned worker failed'


def worker(rank):
    f = verify_freeze()
    device = f['devices'][rank]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == device['uuid']
    resource = hardware()
    assert device['uuid'] not in resource['processes']
    atomic(OUT / f'worker{rank}/preflight.json', dict(at=now(), pid=os.getpid(), **resource))
    a = None
    try:
        samples, p = read(OUT / 'sample_manifest.json'), read(OUT / 'partition.json')
        a = Adapter('llava')
        atomic(OUT / f'worker{rank}/environment.json', a.environment)
        if not (OUT / f'worker{rank}/smoke.json').exists():
            # Same audited smoke; only its output directory and one candidate differ.
            with patch.multiple(audited_runner, OUT=OUT, LABELS=(CONDITION,),
                    generate=generate, sources=sources, check_stop=check_stop):
                audited_runner.smoke(a, rank, samples, p)
        waiting = time.monotonic()
        while not all((OUT / f'worker{r}/smoke.json').exists() for r in range(2)):
            check_stop()
            assert time.monotonic()-waiting < 900
            time.sleep(1)
        assert all(read(OUT / f'worker{r}/smoke.json')['sources'] == sources() for r in range(2))
        keys = [k for k in sorted(samples) if p['image_shard'][samples[k]['image_sha256']] == rank]
        assert len(keys) == p['question_counts'][rank]
        started = time.monotonic()
        for ix, key in enumerate(keys):
            check_stop()
            sample = samples[key]
            layout, _, binding = prepare(a, sample)
            assert binding == historical(key, 'Native')['input_binding']
            path = checkpoint(key)
            if path.exists():
                value = load(path)
                assert value['sources'] == sources() and value['input_binding'] == binding
                assert value['sample_sha256'] == digest(sample) and value['rank'] == rank
            else:
                g, logits, *_ = generate(a, layout, CONDITION)
                assert g['first_logits_sha256'] == historical(key, 'Native')['generation']['first_logits_sha256']
                save(path, dict(at=now(), key=key, condition=CONDITION, rank=rank,
                    sample_sha256=digest(sample), input_binding=binding, sources=sources(),
                    score=score(sample, g['text']), generation=compact(g)))
                del g, logits
            a.verify()
            if (ix+1) % 10 == 0 or ix+1 == len(keys):
                progress = dict(at=now(), questions=ix+1, total=len(keys),
                    elapsed_seconds=time.monotonic()-started, performance_requests=0)
                atomic(OUT / f'worker{rank}/progress.json', progress)
                print('QUALITY', rank, progress, flush=True)
        atomic(OUT / f'worker{rank}/completion.json', dict(at=now(), status='COMPLETE',
            quality_generations=len(keys), performance_requests=0))
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
        for rank, device in enumerate(f['devices']):
            path = OUT / f'worker{rank}/execution.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('a')
            streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device['uuid'], OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
            jobs.append(subprocess.Popen([sys.executable, '-u', '-m',
                'vlm_research.textvqa_decode_l11_l24.run', '--rank', str(rank)], cwd=ROOT,
                env=env, stdout=stream, stderr=subprocess.STDOUT))
        atomic(OUT / 'driver_pids.json', dict(at=now(), driver=os.getpid(), workers=[p.pid for p in jobs]))
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None, 0) for p in jobs):
                raise RuntimeError('Task-owned worker failed')
            time.sleep(1)
        atomic(OUT / 'driver_exit.json', dict(at=now(), returncodes=[p.returncode for p in jobs]))
        assert all(p.returncode == 0 for p in jobs)
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
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--driver', action='store_true')
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare_cpu()
    elif args.freeze:
        freeze()
    elif args.driver:
        driver()
    else:
        assert args.rank in (0, 1)
        worker(args.rank)
