"""Independent checkpoint digest/score recomputation and paired cluster bootstrap."""
import argparse
import csv
from collections import defaultdict
import numpy as np
from .common import *
from ..benchmarks.scoring import pope_metrics


def interval(rows):
    clusters = defaultdict(lambda: [0., 0.])
    for h, delta in rows:
        clusters[h][0] += delta
        clusters[h][1] += 1
    values = np.array(list(clusters.values()), dtype=np.float64)
    rng = np.random.default_rng(20261106)
    draws = []
    for start in range(0, 20000, 50):
        indices = rng.integers(0, len(values), size=(min(50, 20000-start), len(values)))
        totals = values[indices].sum(axis=1)
        draws.extend((totals[:, 0] / totals[:, 1]).tolist())
    return (np.quantile(draws, [.025, .975]) * 100).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=ORDER, required=True)
    bench = parser.parse_args().benchmark
    protocol = verify()
    rows = samples(bench)
    records = {}
    for rank in range(2):
        path = OUT / f'worker{rank}/results.sqlite'
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        for key, condition, payload, checksum in db.execute(
                'SELECT key,condition,payload,sha256 FROM results WHERE benchmark=?', (bench,)):
            result = json.loads(payload)
            assert digest(result) == checksum
            assert (key, condition) not in records, 'Duplicate cross-shard checkpoint'
            records[key, condition] = result
        db.close()
    assert len(records) == len(rows) * 2, 'Incomplete split: do not publish an Accuracy'
    paired = []
    csv_path = OUT / f'{bench}_per_question.csv.gz'
    with gzip.open(csv_path, 'wt', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['key', 'qid', 'image_sha256', 'condition', 'text', 'token_ids', 'score',
                         'eos', 'truncated', 'empty', 'parse_failed'])
        for sample in rows:
            scores = []
            for condition in CONDITIONS:
                record = records[sample['key'], condition]
                assert record['sample_sha256'] == digest(sample)
                assert record['protocol_sha256'] == digest(protocol)
                assert record['image_sha256'] == sample['image_sha256']
                new_score = score(sample, record['generation']['text'])
                assert new_score == record['scoring'], 'Independent scorer mismatch'
                gen = record['generation']
                writer.writerow([sample['key'], sample['source_question_id'], sample['image_sha256'],
                    condition, gen['text'], json.dumps(gen['token_ids']), new_score['score'],
                    gen['eos'], gen['truncated'], gen['empty'], new_score.get('parse_failed')])
                scores.append(new_score['score'])
            paired.append(dict(sample=sample, native=scores[0], physical=scores[1]))
    output = dict(benchmark=bench, status='COMPLETE', questions=len(rows),
                  image_clusters=len({s['image_sha256'] for s in rows}), at=now(),
                  question_record_digest_verified=True, independent_scoring_recomputed=True)
    if bench == 'MM-Vet':
        output.update(status='GENERATED_AWAITING_OFFICIAL_SCORING', scores=None,
                      note='No paid external judge called; no fabricated score/CI')
    else:
        units = paired
        if bench == 'MMBench':
            cycles = defaultdict(list)
            for p in paired:
                cycles[int(p['sample']['source_question_id']) % 1000000].append(p)
            units = [dict(sample=g[0]['sample'], native=float(all(v['native'] for v in g)),
                          physical=float(all(v['physical'] for v in g))) for g in cycles.values()]
            output.update(scoring='Official exact_matching + circular all-rotation hit',
                          circular_groups=len(units), record_level_only_secondary=dict(
                          native=100*np.mean([v['native'] for v in paired]),
                          physical=100*np.mean([v['physical'] for v in paired])))
        output.update(native=100*np.mean([v['native'] for v in units]),
            physical=100*np.mean([v['physical'] for v in units]),
            delta_pp=100*np.mean([v['physical']-v['native'] for v in units]),
            paired_cluster_bootstrap_95CI_pp=interval([(v['sample']['image_sha256'],
                v['physical']-v['native']) for v in units]), bootstrap_replicates=20000,
            ci_scope='Exploratory, question-weighted within image SHA clusters',
            harmed=sum(v['physical'] < v['native'] for v in units),
            rescued=sum(v['physical'] > v['native'] for v in units))
        if bench == 'POPE':
            output['strategies'] = {}
            for strategy in ('random', 'popular', 'adversarial'):
                selected = [s for s in rows if s['raw']['strategy'] == strategy]
                output['strategies'][strategy] = {c: pope_metrics(
                    [records[s['key'], c]['generation']['text'] for s in selected],
                    [s['answer'] for s in selected]) for c in CONDITIONS}
        if bench == 'ScienceQA-IMG':
            output['subject'] = {}
            for group in sorted({s.get('groups', {}).get('subject', 'unknown') for s in rows}):
                selected = [p for p in paired if p['sample'].get('groups', {}).get('subject') == group]
                output['subject'][group] = dict(questions=len(selected),
                    native=100*np.mean([v['native'] for v in selected]),
                    physical=100*np.mean([v['physical'] for v in selected]))
    atomic(OUT / f'summaries/{bench}.json', output)
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
