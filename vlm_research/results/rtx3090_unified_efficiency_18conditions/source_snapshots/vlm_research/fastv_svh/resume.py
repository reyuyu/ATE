"""Preserve the original blocked freeze; explicitly schedule five ready benchmarks."""
from collections import Counter
from datetime import datetime
import shutil
import subprocess
import sys

from .common import *

PARENT = 'cb2f794dc448306dbf21b72b98a802f2a35a6111'


def main():
    p = read(OUT / 'protocol.json')
    assert not p.get('execution_schedule'), 'Already scheduled; do not silently refreeze'
    assert not list(OUT.glob('worker*/results.sqlite')), 'No refreeze over formal predictions'
    for relative, expected in p['sources'].items():
        original = subprocess.check_output(['git', 'show', f'{PARENT}:{relative}'], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == expected, relative
    assert sha(OUT / 'sample_manifest.jsonl.gz') == p['manifest_sha256']
    assert sha(OUT / 'partition.json') == p['partition_sha256']
    assert sha(OUT / 'data_acceptance.json') == p['data_acceptance_sha256']
    previous_gate = read(OUT / 'gpu_smoke.json')
    assert previous_gate['passed'] and previous_gate['protocol_sha256'] == digest(p)
    stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
    archive = OUT / 'revisions' / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('protocol.json', 'validation.json', 'completion.json', 'artifact_sha256.json',
                 'gpu_smoke.json', 'gpu_release_audit.json', 'cpu_tests.json', 'cpu_tests.log', 'smoke0.log'):
        if (OUT / name).exists():
            shutil.copy2(OUT / name, archive / name)
    if (OUT / 'gpu_smoke').is_dir():
        shutil.copytree(OUT / 'gpu_smoke', archive / 'gpu_smoke')
    test = subprocess.run([sys.executable, '-m', 'vlm_research.fastv_svh.tests'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
    (OUT / 'cpu_tests.log').write_text(test.stdout+test.stderr)
    atomic(OUT / 'cpu_tests.json', dict(passed=test.returncode==0, returncode=test.returncode,
                                       log_sha256=sha(OUT / 'cpu_tests.log')))
    assert test.returncode == 0, test.stderr
    schedule = dict(ready_benchmarks=[b for b in COUNTS if b!='MME'], deferred_benchmarks=['MME'],
        ready_questions=32882, requested_questions=35256, deferred_questions=2374,
        reason='MME frozen reference image bytes unavailable; whole benchmark explicitly deferred',
        user_instruction='Use idle GPUs to start; algorithm, samples and numerical gates unchanged',
        created_at=datetime.now().astimezone().isoformat())
    previous_hash = digest(p)
    p.update(sources=sources(), status='FROZEN_FIVE_READY_ONE_DEFERRED',
             execution_schedule=schedule, previous_protocol_sha256=previous_hash,
             previous_freeze_archive=str(archive), gpu_smoke='PENDING_FRESH_SMOKE')
    atomic(OUT / 'protocol.json', p)
    samples = rows(OUT / 'sample_manifest.jsonl.gz')
    assignment = read(OUT / 'partition.json')['assignment']
    counts = Counter(assignment[s['key']] for s in samples if s['benchmark']!='MME')
    atomic(OUT / 'resume_schedule.json', dict(**schedule, rank_question_counts=dict(counts),
        protocol_sha256=digest(p), previous_protocol_sha256=previous_hash,
        previous_gpu_gate_passed=True, fresh_gpu_gate_required=True))
    atomic(OUT / 'completion.json', dict(status='GPU_SMOKE_PENDING', completed=0, requested=35256,
        scheduled=32882, deferred=2374, deferred_benchmarks=['MME']))
    atomic(OUT / 'validation.json', dict(status='CPU_PASS_FRESH_GPU_SMOKE_PENDING',
        cpu_tests_passed=True, previous_gpu_gate_preserved=str(archive / 'gpu_smoke.json'),
        protocol_sha256=digest(p), algorithm_engine_sha256=sha(ROOT/'vlm_research/fastv_svh/engine.py')))
    print('RESUMED_READY_BENCHMARK_SCHEDULE', dict(counts), 'TOTAL', sum(counts.values()), flush=True)


if __name__ == '__main__':
    main()
