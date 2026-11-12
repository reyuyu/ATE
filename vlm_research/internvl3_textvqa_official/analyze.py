"""Strict checkpoint audit, pinned official scoring and independent MMF scoring."""
import argparse
import csv
import sqlite3

from .common import *


def csv_output(path, rows, columns):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=True) if isinstance(row.get(k), (list, dict))
                             else row.get(k, '') for k in columns})
    os.replace(temporary, path)


def paired_cluster_ci(differences, image_hashes, replicates=20000, seed=20261113):
    import numpy as np
    groups = sorted(set(image_hashes))
    index = {key: i for i, key in enumerate(groups)}
    labels = np.array([index[key] for key in image_hashes])
    sums = np.bincount(labels, weights=differences, minlength=len(groups))
    counts = np.bincount(labels, minlength=len(groups))
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, replicates, 128):
        draws = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)),
                               size=min(128, replicates-start))
        values.extend(((draws@sums)/(draws@counts)).tolist())
    low, high = np.quantile(values, [.025, .975])
    return dict(lower_pp=float(low)*100, upper_pp=float(high)*100,
                replicates=replicates, seed=seed, image_clusters=len(groups))


def analyze(require_native=False):
    import numpy as np
    rows = manifest()
    frozen = {row['question_id']: row for row in rows}
    assert len(frozen) == len(rows) == 5000
    scope, reference = upstream(), independent_scorer()
    ph = sha(OUT/'protocol.json')
    records = []
    database = OUT/'server_only/predictions.sqlite'
    if database.exists():
        with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
            assert db.execute('SELECT value FROM meta WHERE key="protocol_sha256"').fetchone()[0] == ph
            for method, qid, expected, raw in db.execute('SELECT method,qid,sha256,payload FROM records ORDER BY method,qid'):
                assert hashlib.sha256(raw.encode()).hexdigest() == expected
                value = json.loads(raw)
                row = frozen[qid]
                assert value['method'] == method and value['question_id'] == qid and method in ['Native', 'Late9']
                for key in ['input_binding_sha256', 'prompt_sha256', 'input_ids_sha256', 'pixels_bf16_sha256',
                            'image_sha256', 'gold', 'image_tokens', 'tiles']:
                    assert value[key] == row[key], ('INPUT_BINDING_CHANGED', qid, key)
                assert value['protocol_sha256'] == ph
                actual, normalized = score(scope, value['text'], value['gold'])
                assert abs(actual-value['score']) < 1e-12 and normalized == value['normalized_prediction']
                value['independent_mmf_score'] = reference.eval_pred_list(
                    [dict(pred_answer=value['text'], gt_answers=value['gold'])], disable_tqdm=True)
                value['independent_score_difference'] = value['independent_mmf_score']-actual
                records.append(value)
    summary = []
    for method in ['Native', 'Late9']:
        subset = [r for r in records if r['method'] == method]
        assert len({r['question_id'] for r in subset}) == len(subset)
        complete = len(subset) == 5000
        score_percent = float(np.mean([r['score'] for r in subset])*100) if subset else None
        summary.append(dict(method=method, completed=len(subset), required=5000,
            status='COMPLETE' if complete else 'PARTIAL' if subset else 'NOT_RUN',
            accuracy_percent=score_percent if complete else None,
            partial_accuracy_percent=score_percent if subset and not complete else None,
            independent_mmf_accuracy_percent=float(np.mean([r['independent_mmf_score'] for r in subset])*100) if complete else None,
            independent_score_mismatch_questions=sum(abs(r['independent_score_difference']) > 1e-12 for r in subset),
            eos=sum(r['eos'] for r in subset), truncated=sum(r['truncated'] for r in subset),
            empty=sum(r['empty'] for r in subset),
            mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in subset])) if subset else None))
    native, late = summary
    gap = native['accuracy_percent']-80.2 if native['status'] == 'COMPLETE' else None
    gate = 'PASS' if gap is not None and abs(gap) <= 1 else 'BLOCKED_PROTOCOL_GAP' if gap is not None else 'NOT_RUN_OR_INCOMPLETE'
    result = dict(status='COMPLETE' if all(x['status'] == 'COMPLETE' for x in summary) else 'PARTIAL',
        protocol_sha256=ph, native_protocol_gate=gate, official_reference_percent=80.2,
        native_minus_reference_pp=gap, methods=summary,
        source_scope='Pinned release-time official HF implementation; authors exact leaderboard run commit unpublished',
        historical_single_tile_results_reused=False, independent_scorer_sha256=sha(INDEPENDENT_SCORER))
    paired = []
    if late['status'] == 'COMPLETE':
        assert native['status'] == 'COMPLETE' and gate == 'PASS'
        by_method = {method: {r['question_id']: r for r in records if r['method'] == method} for method in ['Native', 'Late9']}
        differences = [by_method['Late9'][qid]['score']-by_method['Native'][qid]['score'] for qid in frozen]
        hashes = [frozen[qid]['image_sha256'] for qid in frozen]
        paired = [dict(comparison='Late9-Native', delta_accuracy_pp=float(np.mean(differences))*100,
            soft_score_decreased=sum(d < 0 for d in differences), soft_score_increased=sum(d > 0 for d in differences),
            unchanged=sum(d == 0 for d in differences),
            **paired_cluster_ci(differences, hashes))]
        result['paired'] = paired
    columns = ['method', 'question_id', 'text', 'token_ids', 'gold', 'score', 'independent_mmf_score',
        'independent_score_difference', 'normalized_prediction', 'image_sha256', 'input_binding_sha256',
        'prompt_sha256', 'input_ids_sha256', 'pixels_bf16_sha256', 'tiles', 'image_tokens',
        'generated_tokens', 'eos', 'empty', 'truncated', 'gpu_uuid', 'protocol_sha256']
    csv_output(OUT/'per_question.csv', records, columns)
    csv_output(OUT/'accuracy_summary.csv', summary, list(summary[0]))
    csv_output(OUT/'paired_comparisons.csv', paired, ['comparison', 'delta_accuracy_pp', 'soft_score_decreased',
        'soft_score_increased', 'unchanged', 'lower_pp', 'upper_pp', 'replicates', 'seed', 'image_clusters'])
    if records:
        with gzip.open(OUT/'raw_predictions.jsonl.gz', 'wt') as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True)+'\n')
    write(OUT/'completion.json', result)
    if require_native:
        assert native['status'] == 'COMPLETE', 'NATIVE_5000_NOT_COMPLETE'
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    analyze()
