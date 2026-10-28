"""CPU-only consistency audit of this completed run; prints JSON, changes no files."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
def read(name):
    return json.loads((RUN / name).read_text())
def check():
    config, validation = read('experiment_config.json'), read('validation.json')
    rows = read('per_condition_results.json')
    assert validation['passed'] and validation['status'] == 'completed'
    assert len(rows) == 6 and {(r['source_layer'], r['lambda']) for r in rows} == {(s,l) for s in (23,24) for l in (0,.5,1)}
    assert config['question_id'] == '15133706'
    for name, expected in config['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    with (RUN/'per_condition_results.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows)
    logits = np.load(RUN/'prediction_logits.npz')
    assert np.array_equal(logits['native_full'][0], logits['native_full'][1])
    mapping = read('input_mapping.json')['candidate_mapping']
    ids = {c['label']: c['answer_records'][0]['token_id'] for c in mapping}
    audits = []
    for row, csv_row in zip(rows, csv_rows):
        assert row['question_id'] == csv_row['question_id'] == '15133706'
        for field in ('margin','correct_logp','wrong_logp','gain_vs_full'):
            assert row[field] == float(csv_row[field])
        source, strength = row['source_layer'], row['lambda']
        values = logits[f'L{source}_lambda{strength:g}']
        assert np.isfinite(values).all() and np.array_equal(values[0],values[1])
        assert row['margin'] == float(values[0,ids['right']] - values[0,ids['left']])
        assert row['margin'] == row['correct_logp'] - row['wrong_logp']
        assert row['gain_vs_full'] == row['margin'] - row['full_margin']
        if strength == 0:
            assert np.array_equal(values,logits['native_full'])
        conditions = read(f'audit_L{source}_lambda{strength:g}.json')
        assert [c['candidate'] for c in conditions] == ['left','right']
        for condition in conditions:
            aa = condition['layer_audits']
            assert [a['layer'] for a in aa] == list(range(24,32))
            assert condition['question_hidden_bitwise_full']
            for a in aa:
                assert a['passed'] and a['finite_nonnegative']
                assert a['shape'] == [1,32,604,604] and a['query'] == 602 and a['visual_keys'] == 576
                assert a['other_queries_bitwise_exact'] and a['nonvisual_bitwise_exact']
                assert a['formula_rounded_target_bitwise_exact'] and a['conditional_rounding_bound_passed']
                assert np.allclose(a['visual_mass_after'],a['visual_mass_before'],atol=2e-5,rtol=.001)
            audits.extend(aa)
    assert len(audits) == validation['target_layer_audit_count'] == 96
    assert max(a['row_sum_max_abs_error'] for a in audits) == validation['max_row_sum_abs_error']
    assert max(a['visual_mass_max_abs_error'] for a in audits) == validation['max_visual_mass_abs_error']
    assert sum(a['conditional_nominal_exceeded'] for a in audits) == validation['nominal_conditional_threshold_exceeded_audits']
    return dict(passed=True,cpu_only=True,question_id='15133706',conditions=len(rows),layer_audits=len(audits),code_hashes_checked=len(config['code_sha256']),raw_logits_csv_json_consistent=True)
if __name__ == '__main__':
    print(json.dumps(check(),indent=2))
