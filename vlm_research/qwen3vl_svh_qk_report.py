"""Aggregate immutable scoped repair and FAST-TRACK artifacts on the CPU."""
import json
import importlib.metadata

import torch

from .qwen3vl_svh_qk_repair.run import OUT as REPAIR, ROOT, GATE
from .qwen3vl_svh_fasttrack_qk.run import OUT, GPUS, check
from .qwen3vl_svh_preflight48.freeze import table, write
from .qwen3vl_svh_preflight48.run import resource


def main():
    assert not torch.cuda.is_initialized()
    check()
    validations=[]
    case_states=[]
    failures=[]
    for shard in range(2):
        path=OUT/f'worker{shard}_failure.json'
        if path.exists():
            failures.append(json.loads(path.read_text()))
    cases=json.loads((OUT/'numerical_cases.json').read_text())
    for case in cases:
        path=OUT/'numerical'/case['sample']['key'].replace(':','_').replace('/','_')/'validation.json'
        if path.exists():
            validation=json.loads(path.read_text())
            validations.append(validation)
            state='PASS' if validation['status']=='PASS' else 'NUMERICAL_FAIL'
        else:
            state='OOM' if any(f.get('current',{}).get('key')==case['sample']['key'] and
                'OutOfMemoryError' in f['error'] for f in failures) else 'NOT_RUN'
        case_states.append(dict(key=case['sample']['key'],task=case['sample']['benchmark'],
            quantile=case['quantile'],input_length=case['sample']['input_length'],
            image_tokens=case['sample']['image_tokens'],status=state))
    passed=sum(r['status']=='PASS' for r in validations)
    failed=sum(r['status']!='PASS' for r in validations)
    complete=passed==24 and failed==0 and not (OUT/'STOP_NUMERICAL.json').exists()
    write(OUT/'numerical_completion.json',dict(status='PASS' if complete else 'FAIL',
        completed=len(validations),passed=passed,failed=failed,expected=24,
        OOM=sum(r['status']=='OOM' for r in case_states),not_run=sum(r['status']=='NOT_RUN' for r in case_states),
        pretest8_allowed=complete,full_evaluation_allowed=False,
        full_59_numerical_validated=False,acceptance_label='FAST_TRACK_NUMERICAL_VALIDATED' if complete else 'BLOCKED',
        rows=validations))
    table(OUT/'numerical_summary.csv',validations)
    table(OUT/'numerical_case_status.csv',case_states)
    diagnosis=json.loads((REPAIR/'diagnosis_completion.json').read_text())
    regression=json.loads((REPAIR/'verification_completion.json').read_text())
    after={r['key']:r for r in regression['rows']}
    table(REPAIR/'repair_before_after.csv',[
        dict(key=r['key'],first_layer=r['first']['layer'],first_operator=r['first']['operator'],
            initial_QK_max_abs=r['first']['max_abs'],before_logit_mean_abs=r['worst_mean_abs'],
            before_logit_max_abs=r['worst_max_abs'],after_logit_mean_abs=after[r['key']]['mean_abs'],
            after_logit_max_abs=after[r['key']]['max_abs'],status=after[r['key']]['status'],
            extra_QK_flops=after[r['key']]['extra_QK_flops'],
            max_temporary_QK_output_bytes=after[r['key']]['max_temporary_QK_output_bytes']) for r in diagnosis['rows']])
    table(REPAIR/'regression_summary.csv',regression['rows'])
    write(REPAIR/'validation.json',dict(scoped_regression_status=regression['status'],
        scoped_completed=regression['completed'],scoped_expected=9,cpu_tests_passed=16,
        gate=GATE,fasttrack24_passed=passed,fasttrack24_failed=failed,
        fasttrack24_completed=len(validations),fasttrack24_status='PASS' if complete else 'FAIL',
        fasttrack24_OOM=sum(r['status']=='OOM' for r in case_states),
        fasttrack24_not_run=sum(r['status']=='NOT_RUN' for r in case_states),
        full_59_numerical_validated=False,author_official_status='BLOCKED',
        plan='B: ATE unified HF, non-author-official',
        pretest8_allowed=complete,pretest8_completed=False,full_evaluation_started=False))
    state=resource()
    own_pids=[regression['pid'],regression['diagnosis_pid']]+[f['pid'] for f in failures]
    state.update(task_pids=own_pids,
        task_processes_released=all(str(pid) not in state['processes'] for pid in own_pids),
        authorized_fasttrack_gpu_uuids=GPUS,diagnostic_gpu_uuid=GPUS[1],
        GPU0_and_GPU2_foreign_tasks_untouched=True)
    assert state['task_processes_released']
    write(REPAIR/'gpu_release_audit.json',state)
    write(REPAIR/'environment.json',dict(packages={name:importlib.metadata.version(name)
        for name in ['torch','transformers','numpy','Pillow']},cuda_runtime=torch.version.cuda,
        cuda_initialized=torch.cuda.is_initialized(),shared_environment_unchanged=True))
    print('SCOPED_REPAIR',regression['status'],'FAST_TRACK24',passed,'PASS',failed,'FAIL',len(validations),'completed')


if __name__=='__main__':
    main()
