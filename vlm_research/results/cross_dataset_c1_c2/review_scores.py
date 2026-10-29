"""Post-run CPU-only review; does not modify frozen execution sources."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.dataset_registry import digest, freeze_json, load_manifest, sha256
from benchmarks.protocols import MODEL_PATH
from benchmarks.scoring import diagnostic_candidate_metrics, generated_score
from cross_dataset_c1_c2.run import OUTPUT, SUITE, history_row, load_config


def scalar_score(metrics):
    keys = ('official_generation_correct', 'official_generation_vqa_soft_accuracy',
            'mechanism_generation_literal_exact_match')
    return float(next(metrics[key] for key in keys if key in metrics))


def main():
    config = load_config(OUTPUT)
    assert json.loads((OUTPUT/'decision.json').read_text())['reuse_history'] is True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    samples = {(unit, row['question_id']): row
               for unit in config['manifest_sha256']
               for row in load_manifest(SUITE/f'manifests/{unit}.json')['samples']}
    with (OUTPUT/'per_sample.csv').open() as stream:
        flat = list(csv.DictReader(stream))
    lookup = {(r['unit'], r['question_id'], r['condition']): r for r in flat}
    assert len(flat) == len(lookup) == 7860
    aggregate = json.loads((OUTPUT/'aggregate.json').read_text())
    generated, candidate = defaultdict(list), []
    decoded = closed = 0
    for job in config['jobs']:
        unit, qid = job['unit'], job['question_id']
        sample = samples[unit, qid]
        path = OUTPUT/f'shard{job["shard"]}'/unit/f'{qid}.json'
        current = json.loads(path.read_text())
        historical = json.loads(history_row(job).read_text())
        assert current['image_sha256'] == sample['image_sha256'] == job['image_sha256']
        assert current['answer'] == sample['answer'] == historical['answer']
        assert current['config_sha256'] == digest(config)
        assert current['all_causal_prefixes_exact'] is True
        results = {'Full': historical['conditions']['Full'],
                   'C1': current['conditions']['C1'],
                   'C2': historical['conditions']['C2']}
        native = results['Full']
        for condition, result in results.items():
            text = tokenizer.decode(result['generated_token_ids'], skip_special_tokens=True)
            assert text == result['generated_text']
            decoded += 1
            metric = generated_score(sample, text)
            for key, value in metric.items():
                assert result['metrics'][key] == value
            value = scalar_score(metric)
            generated[unit, condition].append(value)
            row = lookup[unit, qid, condition]
            assert row['generated_text'] == text
            assert float(row['generated_score']) == value
            assert float(row['generated_score_gain']) == value-scalar_score(native['metrics'])
            if sample['answer_type'] == 'closed':
                assert [c['candidate'] for c in result['candidate_scores']] == sample['candidates']
                scores = [c['logp'] for c in result['candidate_scores']]
                expected = diagnostic_candidate_metrics(sample, scores)
                for key, diagnostic in expected.items():
                    assert result['metrics'][key] == diagnostic
                assert float(row['candidate_margin']) == expected['diagnostic_candidate_margin']
                assert float(row['gold_absolute_logp']) == expected['diagnostic_gold_sequence_logp']
                closed += 1
                for c in result['candidate_scores']:
                    candidate.append({'unit': unit, 'question_id': qid, 'condition': condition,
                                      'candidate': c['candidate'], 'is_gold': c['candidate'] == sample['answer'],
                                      'full_vocabulary_sequence_logp': c['logp'],
                                      'answer_token_count': len(c['records'])})
            else:
                assert result['candidate_scores'] == []
                assert row['candidate_margin'] == row['gold_absolute_logp'] == ''
    for (unit, condition), values in generated.items():
        assert float(np.mean(values)) == aggregate['units'][unit]['conditions'][condition]['generated_score']
    with (OUTPUT/'candidate_logp.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidate[0]))
        writer.writeheader()
        writer.writerows(candidate)
    review = {'passed': True, 'gpu_used_for_review': False,
              'questions': len(config['jobs']), 'paired_condition_rows': len(flat),
              'token_sequences_independently_decoded': decoded,
              'generated_scores_independently_recomputed': decoded,
              'closed_condition_metrics_independently_recomputed': closed,
              'candidate_sequence_rows': len(candidate),
              'all_manifest_image_and_answer_bindings_exact': True,
              'all_frozen_execution_source_hashes_unchanged': True,
              'aggregate_means_match_per_question_recomputation': True,
              'all_c1_raw_logits_independently_verified': aggregate['all_raw_c1_logits_independently_verified'],
              'raw_logp_max_error': aggregate['max_float64_logp_recompute_error'],
              'config_sha256': digest(config),
              'review_source_sha256': sha256(__file__),
              'artifacts_sha256': {name: sha256(OUTPUT/name) for name in
                                  ('aggregate.json', 'per_sample.csv', 'candidate_logp.csv', 'report.md')}}
    freeze_json(OUTPUT/'independent_review.json', review)
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
