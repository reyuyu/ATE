"""Archive the failed gate and prove no quality generation was unlocked."""
import json
import os
import subprocess
from collections import Counter

from .qwen3vl_svh_fasttrack_runtime.run import OUT, ROOT, BASE, GPUS, RUNTIME, check
from .qwen3vl_svh_preflight48.freeze import sha, table, write
from .qwen3vl_svh_preflight48.run import resource


def main():
    protocol = check()
    gate = json.loads((OUT / 'numerical_completion.json').read_text())
    assert gate['status'] == 'FAIL_OR_PARTIAL' and gate['passed'] == 14 and gate['completed'] == 15
    assert not gate['pretest_allowed'] and not gate['full_evaluation_allowed']
    assert (OUT / 'STOP_NUMERICAL.json').exists()
    assert not (ROOT / 'vlm_research/results/qwen3vl_svh_fasttrack_pretest8_runtime').exists()
    cases = json.loads((OUT / 'numerical_cases.json').read_text())
    validated = {r['key']: r for r in gate['rows']}
    table(OUT / 'all24_status.csv', [dict(key=c['sample']['key'], task=c['sample']['benchmark'],
        quantile=c['quantile'], status=validated.get(c['sample']['key'], {}).get('status', 'NOT_RUN_AFTER_FAILURE')) for c in cases])
    a = json.loads((ROOT / 'vlm_research/results/qwen3vl_svh_sqa7908_repair/verify_completion.json').read_text())
    b = json.loads((ROOT / 'vlm_research/results/qwen3vl_svh_vision_memory/memory_accept_completion.json').read_text())
    assert a['status'] == b['status'] == 'PASS' and a['completed'] == 14
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    command = ['/data/lk/深度学习/.envs/ate-qwen3vl/bin/python', '-m', 'unittest',
        'vlm_research.qwen3vl_svh_sqa7908_repair.tests', 'vlm_research.qwen3vl_svh_vision_memory.tests', '-v']
    cpu = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    write(OUT / 'cpu_regression.json', dict(command=command, exit_code=cpu.returncode,
        stdout=cpu.stdout, stderr=cpu.stderr, status='PASS' if cpu.returncode == 0 else 'FAIL'))
    assert cpu.returncode == 0
    released = resource()
    assert all(uuid not in released['processes'] for uuid in GPUS)
    released.update(task_gpu_uuids=GPUS, own_gpu_processes_remaining=False, status='RELEASED', other_gpus_untouched=True)
    write(OUT / 'gpu_release.json', released)
    write(OUT / 'completion.json', dict(status='BLOCKED_NUMERICAL', plan='B: ATE unified HF; not author official',
        plan_A='BLOCKED', base_commit='d5ca88d7de604728eb1f0bcea3185f01d266e125',
        runtime=RUNTIME, original_eager_status='BLOCKED_MEMORY', sqa7908_repair='PASS',
        original_eager_full_generation_regressions=14, memory_repair='PASS_NEW_RUNTIME',
        numerical24=dict(passed=14, failed=1, not_run=9), pretest48='NOT_RUN', full_evaluation='NOT_RUN',
        fast_track_validated=False, full_59_numerical_validated=False, gpu_resources='RELEASED',
        unresolved='MME P99 language Prefill attention probability mismatch; no numerical gate relaxation'))
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json']
    write(OUT / 'artifact_sha256.json', dict(files={str(p.relative_to(ROOT)): sha(p) for p in files},
        input_protocol_sha256=sha(OUT / 'protocol.json'), tested_sources=protocol['source_files'],
        historical_base=str(BASE), finalizer_sha256=sha(__file__)))
    print('ARCHIVED: SQA fixed, new-runtime memory PASS, numerical24 14/1/9; no evaluation; GPUs released', flush=True)


if __name__ == '__main__':
    main()
