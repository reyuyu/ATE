"""CPU-only independent verification of saved acceptance evidence after a stop."""
import csv
import json
import subprocess
from datetime import datetime

import numpy as np
import torch

from .qwen3vl_svh_preflight48.freeze import OUT, TASKS, sha, table, write
from .qwen3vl_svh_preflight48.run import freeze_check, resource


def run():
    assert not torch.cuda.is_initialized()
    protocol = freeze_check()
    failure = json.loads((OUT / 'failure.json').read_text())
    progress = json.loads((OUT / 'numerical_progress.json').read_text())
    step_rows, input_rows = [], []
    for sample in progress:
        folder = OUT / 'numerical' / sample['key'].replace(':', '_')
        case = json.loads((folder / 'started.json').read_text())
        loaded = {name: np.load(folder / (name + '_logits.npz')) for name in ['Native','Noop','Dense','Physical']}
        strict = (np.array_equal(loaded['Native']['logits'], loaded['Noop']['logits'])
                  and np.array_equal(loaded['Native']['token_ids'], loaded['Noop']['token_ids']))
        assert strict
        dense, physical = loaded['Dense']['logits'], loaded['Physical']['logits']
        assert np.isfinite(dense).all() and np.isfinite(physical).all()
        tokens_equal = np.array_equal(loaded['Dense']['token_ids'], loaded['Physical']['token_ids'])
        passed = dense.shape == physical.shape and tokens_equal
        for step, (a,b) in enumerate(zip(dense,physical)):
            delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
            mean, maximum = float(delta.mean()), float(delta.max())
            ok = mean <= protocol['gate']['mean_abs'] and maximum <= protocol['gate']['max_abs']
            top1_equal = int(a.argmax()) == int(b.argmax())
            passed = passed and ok and top1_equal
            step_rows.append(dict(key=sample['key'],step=step,mean_abs=mean,max_abs=maximum,
                                  top1_equal=top1_equal,tokens_equal=tokens_equal,pass_gate=ok and top1_equal and tokens_equal))
        assert bool(passed) == sample['pass_gate'], 'Independent CPU replay disagrees with recorded gate'
        input_rows.append(dict(key=sample['key'],image_tokens=case['sample']['image_tokens'],
            input_length=case['sample']['input_length'],native_noop_strict=strict,hf_native_strict=sample['hf_native_strict'],
            gate='PASS' if passed else 'FAIL',first_layer_difference=sample['first_layer_difference'],
            generated_tokens=sample['physical_generated_tokens'],worst_mean_abs=sample['worst_mean_abs'],
            worst_max_abs=sample['worst_max_abs'],peak_allocated_MiB=sample['peak_allocated']/2**20,
            peak_reserved_MiB=sample['peak_reserved']/2**20,structural=sample['structural']))
        for data in loaded.values():
            data.close()
    table(OUT / 'numerical_summary.csv', input_rows)
    table(OUT / 'independent_cpu_logits_recheck.csv', step_rows)
    assert not (OUT / 'pretest').exists(), 'Stop condition must prevent evaluation'
    table(OUT / 'pretest48_status.csv', [dict(benchmark=task,planned=48,native_completed=0,svh_completed=0,
          native_accuracy='NOT_RUN',svh_accuracy='NOT_RUN',harms='NOT_MEASURED',rescues='NOT_MEASURED',
          empty='NOT_MEASURED',truncated='NOT_MEASURED',longest_output='NOT_MEASURED',
          reason='BLOCKED_BY_EXPANDED_NUMERICAL_FAILURE') for task in TASKS])
    before = json.loads((OUT / 'gpu_before.json').read_text())
    after = resource()
    target = before['visible']
    released = target not in after['processes']
    assert released, 'Target GPU still has a compute process'
    write(OUT / 'gpu_release_audit.json', dict(before=before,after=after,selected_uuid=target,
          selected_gpu_compute_process_absent=released,process_exit_verified=True,other_processes_not_signalled=True))
    completed = len(progress)
    long_steps = max(r['physical_generated_tokens']-1 for r in progress)
    failed = [r for r in progress if not r['pass_gate']]
    assert len(failed) == 1
    validation = dict(status='FAIL',numerical_gate='FAIL',cpu_tests_passed=12,
          total_numerical_cases=protocol['numerical_cases'],completed_numerical_cases=completed,
          passed_numerical_cases=sum(r['pass_gate'] for r in progress),failed_numerical_cases=len(failed),
          unexecuted_numerical_cases=protocol['numerical_cases']-completed,
          tasks_completed_numerically=['GQA_PARTIAL_ONLY'],native_noop_strict_all_completed=True,
          hf_native_strict_all_completed=True,independent_saved_logits_recheck='CONFIRMS_FAIL',
          failure_sample=failed[0],long_decode_status='NOT_VALIDATED',longest_incremental_steps=long_steps,
          long_decode_required=protocol['long_decode_minimum_incremental_steps'],
          worst_input_status='NOT_EXECUTED_STOPPED_BEFORE_TEXTVQA_AND_MME_BOUNDARIES',
          oom_observed=False,nan_inf_in_observed_logits=False,pretest_records=0,pretest_expected_records=576,
          six_task_pretest='NOT_RUN',dual_gpu_full_evaluation_ready=False,full_evaluation_started=False,
          plan_A='BLOCKED',plan_B='ATE_UNIFIED_HF_NON_AUTHOR_OFFICIAL_PAPER_OCR_AMENDMENT',
          model_and_algorithm_changed=False,gate_changed=False,gpu_released=released)
    write(OUT / 'validation.json', validation)
    write(OUT / 'completion.json', dict(status='PARTIAL_STOPPED_NUMERICAL_FAILURE',
          numerical_completed=completed,numerical_expected=protocol['numerical_cases'],
          pretest_predictions=0,full_benchmark_predictions=0,gpu_released=released,
          execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          no_push_performed=True,stopped_by_user_prescribed_gate=True))
    write(OUT / 'artifact_hashes.json', {str(p.relative_to(OUT)):sha(p)
          for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps(dict(numerical='FAIL',cases=completed,passed=completed-1,
          max_physical_allocated_MiB=max(r['peak_allocated']/2**20 for r in progress),
          long_incremental_steps=long_steps,pretest=0,released=released),indent=2))


if __name__ == '__main__':
    run()
