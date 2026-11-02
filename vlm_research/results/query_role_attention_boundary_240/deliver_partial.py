"""CPU-only record of a stopped validation gate, never a substitute result."""
import csv
import json
import os
from pathlib import Path
import subprocess

from vlm_research.query_role_attention.common import (OUT, CONDITIONS, COMPARISONS, TASKS,
    read, write, verify, sha256, digest)


def table(name, fields, rows):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = verify()
    samples = read(OUT/'sample_manifest.json')
    checkpoints = list((OUT/'checkpoints').glob('*/*.json'))
    assert not checkpoints
    assert len(samples['samples']) == samples['unique_image_sha256'] == 240
    assert read(OUT/'cpu_validation.json')['tests'] == 11
    assert read(OUT/'cpu_validation.json')['passed']
    for path, info in protocol['model_files'].items():
        assert sha256(path) == info['sha256'], path
    pid_records = [read(path)['pid'] for path in OUT.rglob('environment_*.json')]
    assert all(not Path(f'/proc/{pid}').exists() for pid in pid_records)
    apps = subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'],text=True).strip()
    gpu = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                  '--format=csv,noheader'],text=True).strip()
    write(OUT/'gpu_release.json',dict(our_model_processes_exited=True, pids=pid_records,
        physical_gpu=1, current_compute_processes=apps, currently_idle=not bool(apps), device_table=gpu))
    validation = dict(passed=False, status='partial_validation_gate_failed',
        cpu_passed=True, cpu_tests=11, planned_smoke_questions=3, fully_passed_smoke_questions=0,
        formal_questions_completed=0, formal_generations_completed=0, formal_generations_planned=960,
        no_formal_scores_or_confidence_intervals=True,
        failure_stage='Native HF vs custom Full effective prediction logits exact equality',
        native_vs_custom_complete_tokens_equal=True, native_vs_custom_effective_logits_exact=False,
        full_noop_all_valid_logits_check_completed=False, role_gpu_smoke_completed=False,
        maximum_logit_difference=None, difference_vectors_saved=False,
        no_tolerance_relaxation=True, sources_unchanged_since_refreeze=True,
        model_full_sha256_unchanged=True, source_binding=digest(protocol),
        gpu_processes_exited=True, failures=[str(p.relative_to(OUT)) for p in sorted((OUT/'failures').glob('*.json'))],
        no_raw_formal_logits_created=True, no_accuracy_or_mechanism_claim=True)
    write(OUT/'validation.json',validation)
    write(OUT/'gpu_smoke.json',dict(passed=False, status='partial', binding=digest(protocol),
        attempted_question_id=samples['samples'][samples['smoke_indices'][0]]['question_id'],
        completed_questions=0, planned_questions=3, formal_started=False,
        reason=validation['failure_stage'], failure_records=validation['failures']))
    table('accuracy_summary.csv',['condition','planned_questions','completed_questions','accuracy_pct','delta_full_pp','status'],
        [dict(condition=c, planned_questions=240, completed_questions=0, accuracy_pct=None,
              delta_full_pp=None, status='not_run') for c in CONDITIONS])
    table('task_accuracy.csv',['task','condition','planned_questions','completed_questions','accuracy_pct','status'],
        [dict(task=t,condition=c,planned_questions=40,completed_questions=0,accuracy_pct=None,status='not_run')
         for t in TASKS for c in CONDITIONS])
    table('paired_comparisons.csv',['comparison','completed_pairs','delta_accuracy_pp','ci95_low_pp','ci95_high_pp','status'],
        [dict(comparison=f'{a} - {b}',completed_pairs=0,delta_accuracy_pp=None,
              ci95_low_pp=None,ci95_high_pp=None,status='not_run') for a,b in COMPARISONS])
    table('per_question.csv',['question_id','task','image_sha256','condition','text','token_ids','correct'],[])
    artifacts = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ['completion.json','worker.lock']]
    write(OUT/'completion.json',dict(complete=False,status='partial_validation_gate_failed',
        formal_generations=0, frozen_code_commit=protocol['code_commit'],
        artifacts={str(p.relative_to(OUT)):sha256(p) for p in sorted(artifacts)}))
    print(json.dumps(validation,indent=2),flush=True)


if __name__ == '__main__':
    main()
