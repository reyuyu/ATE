"""Rebind presentation artifacts after a protected duplicate CPU analysis."""
import json
from pathlib import Path

from vlm_research.relay_source_causality.common import OUT, read, sha256, verify


def main():
    protocol = verify()
    completion = read(OUT / 'completion.json')
    validation = read(OUT / 'validation.json')
    assert completion['complete'] and completion['status'] == 'COMPLETE'
    assert validation['passed'] and validation['first_metric_replay_difference'] == 0
    assert completion['new_main_generations'] == 480
    assert completion['swap_generations'] == 240
    assert completion['text_diagnostic_generations'] == 24
    assert not Path('/proc/' + str(read(OUT / 'environment.json')['pid'])).exists()
    snapshot = OUT / 'completion_before_delivery.json'
    if not snapshot.exists():
        snapshot.write_text(json.dumps(completion, ensure_ascii=False, indent=2) + '\n')
    original = read(snapshot)
    changed = [n for n, expected in original['artifacts'].items()
               if sha256(OUT / n) != expected]
    allowed = {'report_zh.md', 'figure_image_swap.pdf', 'figure_sources.pdf', 'figure_tasks.pdf'}
    assert set(changed) <= allowed, changed
    for n, expected in validation['checkpoint_hashes'].items():
        assert sha256(OUT / n) == expected, n
    for n, expected in validation['raw_hashes'].items():
        assert sha256(n) == expected, n
    failures = sorted((OUT / 'failures').glob('*.json'))
    assert failures
    for path in failures:
        failure = read(path)
        assert 'Never overwrite a different completed checkpoint' in failure['traceback']
        assert "gpu_release.json" in failure['traceback']
    audit = dict(
        passed=True, frozen_execution_commit=protocol['code_commit'],
        changed_presentation_files=changed,
        unchanged_scientific_csv_and_png=True,
        immutable_checkpoint_hashes_verified=len(validation['checkpoint_hashes']),
        immutable_raw_hashes_verified=len(validation['raw_hashes']),
        failures_preserved={str(p.relative_to(OUT)): sha256(p) for p in failures},
        failure_scope='Duplicate CPU reporting rejected overwrite of existing GPU release audit; not a GPU numerical or generation failure',
        new_model_generations=0, tolerances_relaxed=False,
        sources_and_historical_artifacts_unchanged=True,
        original_completion_snapshot_sha256=sha256(snapshot))
    (OUT / 'postprocess_delivery_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    completion['delivery_audit'] = 'postprocess_delivery_audit.json'
    completion['artifacts'] = {
        str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name not in ('completion.json', 'worker.lock')
        and '__pycache__' not in p.parts}
    (OUT / 'completion.json').write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
