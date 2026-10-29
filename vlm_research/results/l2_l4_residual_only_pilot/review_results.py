"""CPU-only post-run audit of every saved candidate and generation prefix."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.dataset_registry import digest, freeze_json, sha256
from residual_only_pilot.run import CONDITIONS, LAYERS, OUTPUT, UNITS, verify


def main():
    config = verify(OUTPUT)
    aggregate = json.loads((OUTPUT/'aggregate.json').read_text())
    assert aggregate['status'] == 'complete'
    assert aggregate['independent_decoding_and_scoring_passed'] is True
    counts = {condition: 0 for condition in CONDITIONS}
    questions, seen = 0, set()
    for unit in UNITS:
        for sample in config['samples'][unit]:
            key = unit, sample['question_id']
            assert key not in seen
            seen.add(key)
            path = OUTPUT/unit/f'{sample["question_id"]}.json'
            row = json.loads(path.read_text())
            assert set(row['conditions']) == set(CONDITIONS)
            assert row['manifest_sha256'] == digest(config)
            assert row['answer'] == sample['answer']
            assert row['image_sha256'] == sample['image_sha256']
            assert row['p_minus_1'] == sample['metadata']['input_mapping']['p_minus_1']
            assert row['prompt_length'] == row['p_minus_1']+1
            assert sha256(path.with_suffix('.npz')) == row['npz_sha256']
            for condition, result in row['conditions'].items():
                assert result['use_cache'] is False
                assert 1 <= len(result['generated_token_ids']) <= 32
                assert len(result['generation_audits']) == len(result['generated_token_ids'])
                assert len(result['candidate_audits']) == len(result['candidate_scores'])
                assert len(result['candidate_scores']) == len(sample['candidates'] or [])
                for stage in ('candidate_audits', 'generation_audits'):
                    for step, audit in enumerate(result[stage]):
                        counts[condition] += 1
                        assert audit['condition'] == condition
                        assert audit['original_p_minus_1'] == row['p_minus_1']
                        assert audit['all_decoder_layers_visited_once'] is True
                        assert len(audit['target_sublayer_calls']) == 6
                        assert all(n == int(condition != 'Whole-Skip') for n in audit['target_sublayer_calls'].values())
                        if condition == 'Full':
                            assert audit['edits'] == audit['residual_identity'] == []
                            continue
                        identities = audit['residual_identity']
                        assert [r['layer'] for r in identities] == list(LAYERS)
                        assert all(r['target_residual_exact'] and r['p_minus_1'] == row['p_minus_1'] for r in identities)
                        assert all(r['all_tokens_identity'] == (condition == 'Whole-Skip') for r in identities)
                        if condition == 'Whole-Skip':
                            assert audit['edits'] == []
                            continue
                        edits = audit['edits']
                        assert [(r['layer'], r['kind']) for r in edits] == [(i, kind) for i in LAYERS for kind in ('attention', 'mlp')]
                        for edit in edits:
                            assert edit['query'] == row['p_minus_1']
                            assert edit['target_zero'] and edit['other_tokens_bitwise_exact'] and edit['original_tensor_unchanged']
                            if stage == 'generation_audits':
                                assert edit['shape'][1] == row['prompt_length']+step
            questions += 1
    weights = json.loads((OUTPUT/'weight_audit.json').read_text())
    assert weights['passed'] and weights['before'] == weights['after']
    assert questions == 64 and len(seen) == 64
    assert aggregate['questions'] == questions and aggregate['condition_rows'] == questions*3
    assert not list(OUTPUT.glob('failure_*.json'))
    report = {'passed': True, 'questions': questions, 'condition_rows': 192,
              'forwards_audited_by_condition': counts,
              'all_original_p_minus_1_targets_and_prefix_lengths_verified': True,
              'all_joint_non_target_outputs_unchanged': True,
              'all_whole_skip_layers_attention_and_mlp_call_counts_zero': True,
              'all_residual_identities_exact': True, 'all_frozen_sources_and_images_unchanged': True,
              'model_before_after_sha256': weights['before'],
              'manifest_sha256': digest(config), 'code_commit': aggregate['code_commit'],
              'review_source_sha256': sha256(__file__),
              'artifacts_sha256': {name: sha256(OUTPUT/name) for name in
                                  ('report.md', 'aggregate.json', 'per_question.csv', 'smoke.json', 'weight_audit.json')}}
    freeze_json(OUTPUT/'independent_review.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
