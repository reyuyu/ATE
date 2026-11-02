"""Index existing harm cases and validate delivery, without model execution."""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    completion = load(ROOT / 'completion.json')
    assert completion['status'] == 'COMPLETE'
    assert completion['formal_generations'] == 960
    validation = load(ROOT / 'validation.json')
    assert validation['passed'] and validation['raw_logits_verified'] == 960
    with (ROOT / 'per_question.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert len(rows) == 960
    harms = [r for r in rows if r['condition'] != 'Full'
             and int(r['correct_to_wrong']) == 1]
    counts = {c: sum(r['condition'] == c for r in harms)
              for c in ('I-Bypass', 'R-Bypass', 'A-Bypass')}
    assert counts == {'I-Bypass': 4, 'R-Bypass': 78, 'A-Bypass': 2}
    with (ROOT / 'accuracy_harm_cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(harms)
    joint = Path('/data/lk/ATE-worktrees/attention-relay-joint-240/vlm_research/results/attention_relay_joint_240/sample_manifest.json')
    formal = load(ROOT / 'sample_manifest.json')
    previous = load(joint)
    def key(s):
        return (s['question_id'], s['task'], s['image_sha256'],
                s['input_mapping']['prompt_ids'])
    assert [key(s) for s in formal['samples']] == [key(s) for s in previous['samples']]
    cohort = dict(same_240_questions_images_order_and_prompt_ids=True,
                  formal_manifest_sha256=sha(ROOT / 'sample_manifest.json'),
                  experiment_two_manifest_sha256=sha(joint),
                  experiment_two_path=str(joint),
                  scores_or_generation_conditions_pooled=False)
    (ROOT / 'experiment_two_cohort_audit.json').write_text(
        json.dumps(cohort, ensure_ascii=False, indent=2) + '\n')
    for name, expected in validation['checkpoint_hashes'].items():
        assert sha(ROOT / name) == expected, name
    release = load(ROOT / 'gpu_release.json')
    assert release['our_gpu_processes_exited']
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and p.name not in ('completion.json', 'worker.lock')
             and '__pycache__' not in p.parts]
    completion['artifacts'] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    completion['delivery_audit'] = dict(
        accuracy_harm_rows=len(harms), harm_counts=counts,
        experiment_two_cohort_identical=True,
        no_new_generations=True, frozen_execution_sources_unchanged=True)
    (ROOT / 'completion.json').write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(completion['delivery_audit'], indent=2))


if __name__ == '__main__':
    main()
