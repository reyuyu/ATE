"""Resource-only adapter around the immutable original worker."""
import argparse
from contextlib import contextmanager
import io
import json
import subprocess
import time
import traceback
import unittest
from benchmarks.dataset_registry import digest, freeze_json, sha256
from . import run


def sources():
    return {n: sha256(run.ROOT / n) for n in
        ['four_layer_attention/idle_gpu.py', 'test_four_layer_idle_gpu.py']}


@contextmanager
def schedule(rank, physical_gpu, metadata):
    if rank not in range(3) or physical_gpu not in (1, 2, 3):
        raise ValueError('Only original shards and physical GPU1-3 are allowed')
    original_gpus, original_keys = run.GPUS, run.ENV_KEYS
    preflight, atomic = run.preflight, run.atomic_json
    selected_uuid = original_gpus[physical_gpu - 1]
    mapping = list(original_gpus)
    mapping[rank] = selected_uuid

    def check(logical_rank):
        assert logical_rank == rank
        return preflight(physical_gpu - 1)

    def save(path, payload):
        if path == run.OUT / f'shard{rank}/smoke.json':
            payload['environment'].update(physical_gpu=physical_gpu, logical_worker=rank,
                scheduling_amendment=metadata, actual_gpu_uuid=selected_uuid)
        return atomic(path, payload)

    run.GPUS = mapping
    run.ENV_KEYS = [key for key in original_keys if key != 'cuda_visible_devices']
    run.preflight, run.atomic_json = check, save
    try:
        yield
    finally:
        run.GPUS, run.ENV_KEYS = original_gpus, original_keys
        run.preflight, run.atomic_json = preflight, atomic


def cpu():
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_four_layer_idle_gpu'))
    freeze_json(run.OUT / 'scheduling_cpu.json', dict(passed=result.wasSuccessful(),
        tests=result.testsRun, source=sources(), output=output.getvalue()))
    print(output.getvalue(), flush=True)
    assert result.wasSuccessful()


def freeze(rank, physical_gpu):
    config = run.verify()
    tested = run.read(run.OUT / 'scheduling_cpu.json')
    assert tested['passed'] and tested['source'] == sources()
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *sources()], cwd=run.ROOT, text=True).strip()
    assert not (run.OUT / f'shard{rank}/smoke.json').exists()
    freeze_json(run.OUT / 'scheduling_amendment.json', dict(protocol_binding=digest(config),
        source=sources(), cpu_sha256=sha256(run.OUT / 'scheduling_cpu.json'),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=run.ROOT, text=True).strip(),
        logical_worker=rank, original_physical_gpu=rank + 1, actual_physical_gpu=physical_gpu,
        actual_gpu_uuid=run.GPUS[physical_gpu - 1],
        reason='Original GPU remained occupied by an external job; use a completed and idle allowed GPU.',
        invariants='Original worker function, cohort, generators, hooks, conditions, scores and all tolerances unchanged.',
        compatibility='Only CUDA UUID may differ. All other original environment keys match; exact Full/E replay is mandatory.',
        report_note='This resource-only amendment supersedes the original physical assignment for this shard only.'))


def worker():
    metadata = run.read(run.OUT / 'scheduling_amendment.json')
    assert metadata['protocol_binding'] == digest(run.verify()) and metadata['source'] == sources()
    assert metadata['cpu_sha256'] == sha256(run.OUT / 'scheduling_cpu.json')
    with schedule(metadata['logical_worker'], metadata['actual_physical_gpu'], metadata):
        run.worker(metadata['logical_worker'])


def analyze():
    from . import analyze as original
    if (run.OUT / 'scheduling_amendment.json').exists():
        metadata = run.read(run.OUT / 'scheduling_amendment.json')
        assert metadata['protocol_binding'] == digest(run.verify()) and metadata['source'] == sources()
        smoke = run.read(run.OUT / f"shard{metadata['logical_worker']}/smoke.json")
        assert smoke['environment']['scheduling_amendment'] == metadata
        assert smoke['environment']['physical_gpu'] == metadata['actual_physical_gpu']
        report = original.report

        def annotated_report(*args):
            report(*args)
            path = run.OUT / 'report.md'
            text = path.read_text()
            text = text.replace('复用原三卡图片分配，不干扰GPU0或其他GPU进程；GPU忙时先等待。',
                '保留原图片分片；不干扰GPU0或其他进程。原GPU1持续被占用，分片0转至已完成并空闲的GPU'
                + str(metadata['actual_physical_gpu']) + '，通过历史Full/E精确复现后才开始；资源变更见`scheduling_amendment.json`。')
            path.write_text(text)

        original.report = annotated_report
    original.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cpu', 'freeze', 'worker', 'analyze'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--physical-gpu', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    try:
        if args.stage == 'freeze': freeze(args.rank, args.physical_gpu)
        else: globals()[args.stage]()
    except BaseException:
        if args.stage == 'worker':
            freeze_json(run.OUT / f'failures/scheduled_{time.time_ns()}.json',
                dict(traceback=traceback.format_exc()))
        raise
