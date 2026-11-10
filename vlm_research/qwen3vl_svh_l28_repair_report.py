"""CPU-only independent recheck of the frozen six-input repair artifacts."""
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'vlm_research/results/qwen3vl_svh_l28_20705853_repair'
OLD = ROOT / 'vlm_research/results/qwen3vl_svh_expanded_preflight48'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def compare(folder, reference, candidate, phase, question):
    with np.load(folder / (reference + '.npz')) as a, np.load(folder / (candidate + '.npz')) as b:
        x, y = a['logits'], b['logits']
        tokens_equal = np.array_equal(a['token_ids'], b['token_ids'])
        assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        rows = []
        for step, (xx, yy) in enumerate(zip(x, y)):
            error = np.abs(xx.astype(np.float64) - yy.astype(np.float64))
            mean, maximum = float(error.mean()), float(error.max())
            top1 = int(xx.argmax()) == int(yy.argmax())
            rows.append(dict(question_id=question, phase=phase, step=step,
                mean_abs_cpu_float64=mean, max_abs=maximum, equal=np.array_equal(xx, yy),
                top1_equal=top1, complete_tokens_equal=tokens_equal,
                pass_gate=mean <= .02 and maximum <= .5 and top1 and tokens_equal))
        return rows


def main():
    protocol = read(OUT / 'fixed_protocol.json')
    assert protocol['gate'] == dict(mean_abs=.02, max_abs=.5, top1_equal=True, complete_tokens_equal=True)
    for name, expected in protocol['source_files'].items():
        assert sha(ROOT / name) == expected, ('Frozen source changed', name)
    verified = read(OUT / 'independent_verification/completion.json')
    assert verified['status'] == 'PASS' and verified['inputs_completed'] == 6 and verified['new_process']
    history = {r['key'].split(':')[1]: r for r in read(OLD / 'numerical_progress.json')}
    rows = compare(OUT / 'diagnosis_original', 'Dense_logits', 'Physical_logits', 'before', '20705853')
    assert not any(r['pass_gate'] for r in rows)
    summary = []
    for entry in verified['rows']:
        qid = entry['question_id']
        folder = OUT / 'independent_verification' / qid
        noop = compare(folder, 'Native_logits', 'Noop_logits', 'native_noop', qid)
        assert all(r['equal'] for r in noop)
        fixed = compare(folder, 'Dense_logits', 'Fixed_logits', 'after', qid)
        assert all(r['pass_gate'] for r in fixed)
        rows.extend(noop + fixed)
        for condition in ('Native', 'Dense'):
            with np.load(OLD / 'numerical' / ('GQA_' + qid) / (condition + '_logits.npz')) as a:
                with np.load(folder / (condition + '_logits.npz')) as b:
                    assert all(np.array_equal(a[k], b[k]) for k in ('logits', 'token_ids'))
        result = read(folder / 'Fixed.json')
        audit = result['audit']
        assert result['eos'] and not result['truncated'] and not result['empty']
        assert audit['deepstack'] == [0, 1, 2]
        assert all(r['forbidden_probability'] == 0 for r in audit['probability'])
        assert len(audit['AV_query_row_alignment']) == 14
        assert all(r['image_hidden_rows'] == r['persistent_dummy_KV'] == 0
                   for r in audit['AV_query_row_alignment'])
        real_rows = entry['input_length'] - entry['image_tokens']
        post_exit = [r for r in audit['layers'] if r['step'] == 0 and r['layer'] >= 22]
        assert len(post_exit) == 14 and all(r['hidden_rows'] == real_rows for r in post_exit)
        assert all(r['real_rows'] == real_rows and not r['dense_projection_executed']
                   for r in audit['aligned_linears'])
        assert all(r['live_tokens'] <= real_rows + result['generated_tokens']
                   for r in audit['kv']['layers'])
        previous = history[qid]
        if qid != '20705853':
            assert previous['worst_mean_abs'] == entry['worst_mean_abs']
            assert previous['worst_max_abs'] == entry['worst_max_abs']
        summary.append(dict(question_id=qid, prior_pass=previous['pass_gate'],
            prior_mean_abs=previous['worst_mean_abs'], prior_max_abs=previous['worst_max_abs'],
            repaired_mean_abs=entry['worst_mean_abs'], repaired_max_abs=entry['worst_max_abs'],
            pass_gate=True, native_noop_equal=True, historical_native_dense_equal=True,
            eos=result['eos'], generated_tokens=result['generated_tokens'],
            extra_AV_flops=entry['extra_AV_alignment_flops']))
    target = OUT / 'independent_verification/20705853'
    diagnostic = compare(target, 'Dense_two_decode_logits', 'Fixed_two_decode_logits', 'fixed_prefix_diagnostic', '20705853')
    assert len(diagnostic) == 3 and all(r['equal'] for r in diagnostic)
    assert all(r['equal'] for r in rows if r['question_id'] == '20705853' and r['phase'] == 'after')
    rows.extend(diagnostic)
    for path in (target / 'semantics.json', target / 'two_decode_semantics.json'):
        semantics = read(path)
        assert semantics['deepstack_exact'] and semantics['mrope_original_positions_exact']
        assert all(r['k']['equal'] and r['v']['equal'] for r in semantics['cache_comparisons'])
        assert all(r['blocked_probability'] == 0 for r in semantics['masks'])
    raw = read(OUT / 'diagnosis_original/raw_trace.json')
    assert sha(Path(raw['path'])) == raw['sha256']
    protected = ['vlm_research/qwen3vl_svh_numerical_repair', 'vlm_research/qwen3vl_c3v2_physical',
                 'vlm_research/results/qwen3vl_svh_expanded_preflight48']
    assert not subprocess.check_output(['git', 'diff', protocol['base_commit'], '--', *protected], cwd=ROOT)
    table(OUT / 'numerical_audit.csv', rows)
    table(OUT / 'regression_summary.csv', summary)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], text=True)
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
    assert protocol['gpu_uuid'] not in processes
    write(OUT / 'gpu_release_audit.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        used_uuid=protocol['gpu_uuid'], no_task_compute_process=True, gpus=gpus, processes=processes,
        only_one_gpu_used=True, other_processes_not_signalled=True))
    write(OUT / 'validation.json', dict(status='PASS', scope='20705853 and five prior passing inputs only',
        cpu_recheck_float64=True, frozen_source_hashes_equal=True, historical_native_dense_logits_equal=True,
        gate=protocol['gate'], six_inputs_pass=True, target_natural_logits_exact=True,
        target_fixed_prefix_two_decode_logits_exact=True, second_decode_after_natural_eos=True,
        prior_five_metrics_unchanged=True, protected_history_unchanged=True, raw_trace_sha256_verified=True,
        full_59_case_acceptance_not_run=True, six_task_pretest_not_run=True, speed_not_measured=True))
    write(OUT / 'completion.json', dict(status='COMPLETE', scope='single-failure numerical repair',
        repair_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        primary_failure_inputs=1, prior_passing_regressions=5, independent_process_verification=True,
        gpu_released=True, pushed=False, author_protocol_A='BLOCKED', full_evaluation_ready='NOT_ESTABLISHED'))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(OUT.rglob('*'))
              if p.is_file() and p.name != 'artifact_hashes.json'}
    hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    write(OUT / 'artifact_hashes.json', hashes)
    print(json.dumps(dict(status='PASS', saved_logit_steps_checked=len(rows), inputs=6,
                          gpu_released=True, output=str(OUT)), indent=2))


if __name__ == '__main__':
    main()
