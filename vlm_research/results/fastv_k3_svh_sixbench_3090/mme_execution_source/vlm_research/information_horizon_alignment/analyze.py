"""CPU-only exact checkpoint verification and explicitly partial-aware scoring."""
import argparse
from collections import Counter, defaultdict
import gzip
import sqlite3

import numpy as np

from .common import *
from .report import write_csv
from .score import score


def collect(rows):
    lookup = {s['key']: s for s in rows}
    records = {}
    inventory = []
    for stage in CONDITIONS:
        p = verify(stage)
        for path in sorted((OUT / 'workers' / stage).glob('rank*/results.sqlite')):
            rank = int(path.parent.name.removeprefix('rank'))
            db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
            assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', str(path)
            assert db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()[0] == digest(p)
            n = 0
            for key, condition, bench, payload, expected in db.execute('SELECT * FROM results'):
                value = json.loads(payload)
                assert digest(value) == expected, (key, condition)
                s = lookup[key]
                assert value['sample_sha256'] == digest(s)
                assert value['protocol_sha256'] == digest(p)
                assert condition in CONDITIONS[stage] and value['stage'] == stage
                assert value['rank'] == s['shard'] == rank and value['gpu_uuid'] == UUIDS[rank]
                assert bench == s['benchmark'] == value['benchmark']
                assert value['image_sha256'] == s['image_sha256']
                assert value['binding']['prompt'] == s['prompt']
                assert len(value['binding']['image_positions']) == 576
                assert value['binding']['raw_image_placeholder_ids'].count(-200) == 1
                g = value['result']
                assert g['all_effective_logits_finite'] and g['independent_cache']
                rebuilt = score(s, g['text'])
                for field, actual in rebuilt.items():
                    assert actual == g[field], (key, condition, field)
                assert (key, condition) not in records
                records[key, condition] = value
                n += 1
            db.close()
            inventory.append(dict(stage=stage, rank=rank, path=str(path), records=n, sha256=sha(path)))
    return records, inventory


def mme_score(rows, correct):
    from ..mme_seed_quality.common import TASKS
    task_rows = []
    for task in sum(TASKS.values(), []):
        indices = [i for i, s in enumerate(rows) if s['task'] == task]
        pairs = defaultdict(list)
        for i in indices:
            pairs[rows[i]['pair_id']].append(bool(correct[i]))
        assert pairs and all(len(p) == 2 for p in pairs.values())
        acc = 100 * np.mean([correct[i] for i in indices])
        plus = 100 * np.mean([all(p) for p in pairs.values()])
        task_rows.append(dict(task=task, accuracy=float(acc), accuracy_plus=float(plus),
                              score=float(acc + plus), questions=len(indices), pairs=len(pairs)))
    totals = {label: sum(r['score'] for r in task_rows if r['task'] in tasks)
              for label, tasks in dict(**TASKS, Total=sum(TASKS.values(), [])).items()}
    return totals, task_rows


def circular_rows(rows, values):
    groups = defaultdict(list)
    for s, v in zip(rows, values):
        groups[s['circular_group']].append((s, v))
    collapsed, correct = [], []
    for group, items in sorted(groups.items()):
        assert len({s['image_sha256'] for s, _ in items}) == 1
        collapsed.append(dict(image_sha256=items[0][0]['image_sha256'], circular_group=group))
        correct.append(float(all(v for _, v in items)))
    return collapsed, np.asarray(correct)


def paired_boot(rows, a, b, seed=20261106, draws=20000):
    assert len(rows)==len(a)==len(b) and rows
    clusters = sorted({s['image_sha256'] for s in rows})
    index = {h: i for i, h in enumerate(clusters)}
    numerator = np.zeros(len(clusters)); denominator = np.zeros(len(clusters))
    for s, left, right in zip(rows, a, b):
        j = index[s['image_sha256']]
        numerator[j] += right - left; denominator[j] += 1
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, draws, 100):
        w = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=min(100, draws - start))
        estimates.extend(100 * (w @ numerator) / (w @ denominator))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return dict(delta_pp=float(100 * (np.mean(b) - np.mean(a))), ci_low=float(low), ci_high=float(high),
                bootstrap=draws, unit='percentage points', cluster='Image SHA256', ci='exploratory 95%')


def summarize(rows, records):
    coverage, summary, partial, anomalies, comparisons = [], [], [], [], []
    for bench in ORDER:
        subset = [s for s in rows if s['benchmark'] == bench]
        for stage, conditions in CONDITIONS.items():
            for condition in conditions:
                present = [s for s in subset if (s['key'], condition) in records]
                values = [records[s['key'], condition]['result'] for s in present]
                state = 'COMPLETE' if len(present) == len(subset) else 'PARTIAL' if present else 'NOT_RUN'
                base = dict(benchmark=bench, stage=stage, condition=condition, expected=len(subset), completed=len(present), status=state)
                coverage.append(base)
                metric = 'MME Total /2800' if bench == 'MME' else 'Circular Accuracy' if bench == 'MMBench' else 'Soft Accuracy' if bench == 'TextVQA' else 'Accuracy'
                value = ''
                if state == 'COMPLETE':
                    scores = [v['score'] for v in values]
                    if bench == 'MME':
                        totals, task_rows = mme_score(present, scores)
                        value = totals['Total']
                        write_csv(OUT / f'mme_tasks_{condition}.csv', task_rows)
                    elif bench == 'MMBench':
                        _, groups = circular_rows(present, scores)
                        value = float(100 * np.mean(groups))
                        summary.append(dict(**base, metric='Row Accuracy', value=float(100 * np.mean(scores)), unit='%'))
                    else:
                        value = float(100 * np.mean(scores))
                summary.append(dict(**base, metric=metric, value=value, unit='points' if bench == 'MME' else '%'))
                if state == 'PARTIAL':
                    partial.append(dict(**base, completed_prefix_mean_score_percent=float(100 * np.mean([v['score'] for v in values])),
                                        warning='PARTIAL ONLY; not the full benchmark score, not a random evaluation subset'))
                anomalies.append(dict(**base, eos=sum(v['eos'] for v in values),
                                      truncated=sum(v['truncated'] for v in values), empty=sum(v['empty'] for v in values),
                                      repeated_3gram=sum(v['repeated_3gram'] for v in values),
                                      parse_failed=sum(v.get('parse_failed', False) for v in values)))
        for left, right in [('HF-Native', 'Physical-C3'), ('HF-Native', 'Physical-L11-L24'), ('Physical-C3', 'Physical-L11-L24')]:
            paired = [s for s in subset if (s['key'], left) in records and (s['key'], right) in records]
            base = dict(benchmark=bench, comparison=right + ' minus ' + left, expected=len(subset), paired=len(paired))
            if len(paired) != len(subset):
                comparisons.append(dict(**base, status='PARTIAL' if paired else 'NOT_RUN', delta='', ci_low='', ci_high=''))
                continue
            a = np.asarray([records[s['key'], left]['result']['score'] for s in paired])
            b = np.asarray([records[s['key'], right]['result']['score'] for s in paired])
            if bench == 'MME':
                from ..mme_seed_quality.analyze import mme_boot
                _, totals, _ = mme_boot(paired, a, b)
                comparisons.extend(dict(**base, status='COMPLETE', **t) for t in totals)
            else:
                if bench == 'MMBench':
                    paired, a = circular_rows(subset, a)
                    _, b = circular_rows(subset, b)
                comparisons.append(dict(**base, status='COMPLETE',
                                        harm=int(np.sum(b < a)), recovered=int(np.sum(b > a)),
                                        binary_flips=bench != 'TextVQA', **paired_boot(paired, a, b)))
    return coverage, summary, partial, anomalies, comparisons


def main():
    p = argparse.ArgumentParser()
    p.parse_args()
    rows = manifest()
    records, inventory = collect(rows)
    lookup = {s['key']:s for s in rows}
    tables = summarize(rows, records)
    for name, table in zip(('coverage', 'accuracy_summary', 'partial_scores', 'output_anomalies', 'paired_comparisons'), tables):
        write_csv(OUT / (name + '.csv'), table)
    export = []
    with gzip.GzipFile(filename=str(OUT / 'per_question.jsonl.gz'), mode='wb', mtime=0) as stream:
        for (key, condition), record in sorted(records.items()):
            stream.write((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
            s = lookup[key]
            export.append(dict(key=key, condition=condition, benchmark=record['benchmark'], image_sha256=record['image_sha256'],
                               question=s['question'], gold=s['answer'], **record['result']))
    write_csv(OUT / 'per_question.csv', export)
    completed = len(records)
    expected = len(rows) * sum(map(len, CONDITIONS.values()))
    completion = dict(at=now(), status='COMPLETE' if completed == expected else 'PARTIAL',
                      answers_completed=completed, answers_expected=expected,
                      stage_A_completed=sum(r['stage'] == 'A' for r in records.values()), stage_A_expected=len(rows),
                      stage_B_completed=sum(r['stage'] == 'B' for r in records.values()), stage_B_expected=len(rows) * 3,
                      checkpoint_inventory=inventory, no_duplicate_records=True,
                      all_checkpoint_hashes_valid=True, all_saved_scores_recomputed=True,
                      no_partial_score_reported_as_full=True)
    atomic(OUT / 'completion.json', completion)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == '__main__':
    main()
