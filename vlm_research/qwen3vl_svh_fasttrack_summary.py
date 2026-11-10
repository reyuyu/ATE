"""CPU-only archive of an unsuccessful fast-track gate; no evaluation unlock."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'vlm_research/results/qwen3vl_svh_fasttrack_sixbench'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = read(OUT / 'protocol.json')
    assert protocol['gate'] == dict(mean_abs=.02, max_abs=.5, top1_equal=True, complete_tokens_equal=True)
    for name, expected in {**protocol['source_files'], **protocol['scorer_sources']}.items():
        assert sha(ROOT / name) == expected
    for name, expected in protocol['frozen_files'].items():
        assert sha(OUT / name) == expected
    assert sha(protocol['inventory_path']) == protocol['inventory_sha256']
    tests = []
    for module in ['qwen3vl_svh_fasttrack.tests', 'qwen3vl_svh_l28_repair.tests']:
        test = subprocess.run([sys.executable, '-m', 'vlm_research.' + module], cwd=ROOT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
        (OUT / (module.split('.')[0] + '_cpu_tests.txt')).write_text(test.stdout + test.stderr)
        assert test.returncode == 0
        tests.append(dict(module=module, status='PASS'))
    cases, comparisons, summary = read(OUT / 'numerical_cases.json'), [], []
    failures = []
    for case in cases:
        r = case['sample']
        folder = OUT / 'numerical' / r['key'].replace(':', '_').replace('/', '_')
        if not (folder / 'validation.json').exists():
            summary.append(dict(key=r['key'], task=r['benchmark'], quantile=case['quantile'], status='NOT_RUN_AFTER_STOP'))
            continue
        validation = read(folder / 'validation.json')
        with np.load(folder / 'Native_logits.npz') as a, np.load(folder / 'Noop_logits.npz') as b:
            assert np.array_equal(a['logits'], b['logits']) and np.array_equal(a['token_ids'], b['token_ids'])
        with np.load(folder / 'Dense_logits.npz') as a, np.load(folder / 'Physical_logits.npz') as b:
            assert a['logits'].shape == b['logits'].shape
            same_tokens = np.array_equal(a['token_ids'], b['token_ids'])
            checks = []
            for step, (x, y) in enumerate(zip(a['logits'], b['logits'])):
                assert np.isfinite(x).all() and np.isfinite(y).all()
                d = np.abs(x.astype(np.float64) - y.astype(np.float64))
                passed = float(d.mean()) <= .02 and float(d.max()) <= .5 and x.argmax() == y.argmax() and same_tokens
                checks.append(passed)
                comparisons.append(dict(key=r['key'], step=step, equal=np.array_equal(x, y),
                    mean_abs_cpu_float64=float(d.mean()), max_abs=float(d.max()), tokens_equal=same_tokens,
                    native_noop_equal=True, pass_gate=passed))
            assert all(checks) == (validation['status'] == 'PASS')
        first = read(folder / 'first_operator_difference.json')['first'] if validation['status'] == 'FAIL' else None
        if first:
            failures.append(dict(key=r['key'], first=first, validation=validation))
        summary.append(dict(key=r['key'], task=r['benchmark'], quantile=case['quantile'],
            status=validation['status'], worst_mean_abs=validation['mean_abs'], worst_max_abs=validation['max_abs'],
            all_tokens_equal=validation['all_tokens_equal'], native_noop_equal=validation['native_noop_equal'],
            first_operator=first['operator'] if first else None, first_operator_layer=first['layer'] if first else None,
            peak_allocated_MiB=validation['peak_allocated']/2**20, AV_extra_GFLOPs=validation['AV_extra_flops']/1e9))
    assert len(failures) == 2
    assert sum(r['status'] == 'PASS' for r in summary) == 2
    assert not (OUT / 'pretest').exists() and not (OUT / 'full').exists()
    table(OUT / 'numerical_summary.csv', summary)
    table(OUT / 'independent_cpu_logits_recheck.csv', comparisons)
    write(OUT / 'failure_operators.json', failures)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], text=True)
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
    assert not any(gpu in processes for gpu in protocol['gpu_uuids'])
    write(OUT / 'gpu_release_audit.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        used_uuids=protocol['gpu_uuids'], no_task_compute_processes=True, processes=processes, gpus=gpus,
        gpu0_and_gpu2_not_used=True, no_foreign_process_signalled=True))
    write(OUT / 'validation.json', dict(status='FAIL', FAST_TRACK_VALIDATED=False,
        FULL_59_NUMERICAL_VALIDATED=False, stage1_attempted=4, stage1_passed=2, stage1_failed=2,
        stage1_not_run=20, cpu_tests=tests, frozen_source_and_inputs_unchanged=True,
        independent_cpu_recheck=True, native_noop_all_four_strict=True,
        phase2_generations=0, phase3_generations=0, full_evaluation_unlocked=False,
        no_oom_observed_in_attempted_cases=True, longest_input_memory_acceptance='NOT_RUN', gate=protocol['gate']))
    write(OUT / 'completion.json', dict(status='BLOCKED_NUMERICAL', plan='B_ATE_UNIFIED_HF_NON_AUTHOR_OFFICIAL',
        plan_A='BLOCKED', FAST_TRACK_VALIDATED=False, FULL_59_NUMERICAL_VALIDATED=False,
        numerical_attempted=4, numerical_required=24, pretest_completed=0, pretest_expected=96,
        full_questions_completed=0, full_questions_expected=35346,
        full_generations_completed=0, full_generations_expected=70692,
        gpu_released=True, pushed=False, no_quality_scores_available=True))
    for shard in range(2):
        (OUT / f'worker{shard}_execution.txt').write_text((OUT / f'worker{shard}.log').read_text())
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}
    hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    write(OUT / 'artifact_hashes.json', hashes)
    print(json.dumps(read(OUT / 'completion.json'), indent=2))


if __name__ == '__main__':
    main()
