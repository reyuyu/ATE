"""CPU-only final audit and portable exports; inference sources stay frozen."""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from vlm_research.seven_benchmark_quality.common import (
    CONDITIONS, COUNTS, OUT, atomic, digest, now, read, samples, score, sha, verify,
)


def csv_out(name, rows):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = verify()
    plan = read(OUT / 'execution_scope/plan.json')
    active = plan['run']
    assert read(OUT / 'generation_complete.json')['run'] == active
    exit_audit = read(OUT / 'gpu_supervisor_exit.json')
    assert exit_audit['exit_codes'] == [0, 0]
    audit = dict(status='PASSED', at=now(), protocol_sha256=digest(protocol),
                 inference_sources_unchanged=True, artifact_bindings_verified=True,
                 finalizer_sha256=sha(__file__), per_benchmark={}, workers=[])
    seen_images = {}
    total = 0
    changed = []
    quality_rows = []
    for rank in (0, 1):
        directory = OUT / f'worker{rank}'
        db = sqlite3.connect(f'file:{directory}/results.sqlite?mode=ro', uri=True)
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()[0] == digest(protocol)
        assert not (directory / 'failure.json').exists()
        assert read(directory / 'complete.json')['status'] == 'GENERATED_REQUESTED_FIVE_SPLITS'
        records_count = 0
        with gzip.open(OUT / f'worker{rank}_records.jsonl.gz', 'wt') as exported:
            for bench in active:
                expected = {s['key']: s for s in samples(bench) if s['shard'] == rank}
                found = set()
                paired = {}
                stats = {c: Counter() for c in CONDITIONS}
                for key, condition, payload, checksum in db.execute(
                        'SELECT key,condition,payload,sha256 FROM results WHERE benchmark=? ORDER BY key,condition', (bench,)):
                    result = json.loads(payload)
                    assert digest(result) == checksum
                    sample = expected[key]
                    assert condition in CONDITIONS
                    assert (key, condition) not in found
                    found.add((key, condition))
                    assert result['sample_sha256'] == digest(sample)
                    assert result['protocol_sha256'] == digest(protocol)
                    assert result['image_sha256'] == sample['image_sha256']
                    assert score(sample, result['generation']['text']) == result['scoring']
                    previous_rank = seen_images.setdefault(sample['image_sha256'], rank)
                    assert previous_rank == rank, 'Image cluster split across devices'
                    binding = result['input_binding']
                    assert binding['p_minus_1'] == binding['input_length'] - 1
                    assert binding['image_tokens'] == len(binding['image_positions']) == 576
                    assert binding['p_minus_1'] not in binding['R']
                    assert not set(binding['R']).intersection(binding['image_positions'])
                    gen = result['generation']
                    assert 0 < len(gen['token_ids']) <= 32
                    assert gen['first_token'] == gen['token_ids'][0]
                    for name in ('eos', 'empty', 'truncated', 'repeated_3gram'):
                        stats[condition][name] += int(gen[name])
                    stats[condition]['parse_failed'] += int(result['scoring'].get('parse_failed', False))
                    stats[condition]['generated_tokens'] += len(gen['token_ids'])
                    stats[condition]['questions'] += 1
                    paired.setdefault(key, {})[condition] = result
                    exported.write(json.dumps(dict(rank=rank, sha256=checksum, record=result),
                                              ensure_ascii=False, separators=(',', ':')) + '\n')
                    records_count += 1
                assert found == {(key, c) for key in expected for c in CONDITIONS}
                for key, conditions in paired.items():
                    native, physical = (conditions[c] for c in CONDITIONS)
                    assert native['input_binding'] == physical['input_binding']
                    n, p = native['scoring']['score'], physical['scoring']['score']
                    if n is not None and p != n:
                        sample = expected[key]
                        changed.append(dict(benchmark=bench, key=key,
                            question_id=sample['source_question_id'], image_sha256=sample['image_sha256'],
                            gold=json.dumps(sample['answer'], ensure_ascii=False),
                            native_answer=native['generation']['text'], physical_answer=physical['generation']['text'],
                            native_score=n, physical_score=p, delta_score=p-n))
                for condition, values in stats.items():
                    quality_rows.append(dict(benchmark=bench, rank=rank, condition=condition, **values))
            assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0] == records_count
            for bench in plan['deferred']:
                assert db.execute('SELECT COUNT(*) FROM results WHERE benchmark=?', (bench,)).fetchone()[0] == 0
        db.close()
        total += records_count
        audit['workers'].append(dict(rank=rank, records=records_count,
            sqlite_integrity='ok', sqlite_sha256=sha(directory / 'results.sqlite'),
            environment=read(directory / 'environment.json'), release=read(directory / 'release.json')))
    assert total == 41224
    summary_rows = []
    for bench in active:
        result = read(OUT / f'summaries/{bench}.json')
        assert result['questions'] == COUNTS[bench]
        assert result['question_record_digest_verified'] and result['independent_scoring_recomputed']
        audit['per_benchmark'][bench] = result
        ci = result.get('paired_cluster_bootstrap_95CI_pp', [None, None])
        summary_rows.append(dict(benchmark=bench, questions=result['questions'],
            scoring_units=result.get('circular_groups', result['questions']),
            image_clusters=result['image_clusters'], status=result['status'],
            native=result.get('native'), physical=result.get('physical'), delta_pp=result.get('delta_pp'),
            ci_low_pp=ci[0], ci_high_pp=ci[1], harmed=result.get('harmed'), rescued=result.get('rescued')))
    smokes = []
    selection = read(OUT / 'smoke_selection.json')
    for rank in (0, 1):
        for bench in active:
            for key in selection[bench][str(rank)]:
                result = read(OUT / f'smoke/worker{rank}/{key}.json')
                assert result['status'] == 'PASSED'
                flags = {k: v for k, v in result.items() if isinstance(v, bool)}
                assert flags and all(flags.values())
                smokes.append(dict(rank=rank, key=key, flags=flags,
                    source_sha256=sha(OUT / f'smoke/worker{rank}/{key}.json'),
                    dense_reference=result['dense_reference_same_prefix_diff'],
                    frozen_tolerances=result['frozen_tolerances']))
    assert len(smokes) == 30
    with gzip.open(OUT / 'execution_audit.json.gz', 'wt') as handle:
        json.dump(dict(smokes=[read(p) for p in sorted((OUT / 'smoke').rglob('*.json'))],
            generation_complete=read(OUT / 'generation_complete.json'),
            supervisor_exit=exit_audit,
            worker_logs={str(rank): (OUT / f'worker{rank}/run.log').read_text() for rank in (0, 1)}), handle)
    csv_out('accuracy_summary.csv', summary_rows)
    csv_out('outcome_audit.csv', quality_rows)
    csv_out('score_changed_cases.csv', changed)
    audit.update(records=total, paired_questions=total//2, gpu_smokes=smokes,
                 deferred_have_zero_records=True, independent_score_recomputation=True,
                 input_binding_paired_exact=True, duplicate_or_missing_records=0,
                 gpu_snapshot=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                     '--format=csv,noheader'], text=True), paid_judge_called=False)
    atomic(OUT / 'validation.json', audit)
    atomic(OUT / 'completion.json', dict(status='GENERATION_COMPLETE_FOUR_SCORED_ONE_AWAITING_JUDGE',
        at=now(), records=total, paired_questions=total//2, scored=active[:-1],
        awaiting_official_scoring=['MM-Vet'], deferred=plan['deferred'],
        qwen_status=protocol['qwen_status'], owned_gpu_processes_exited=True))
    artifacts = [p for p in OUT.glob('*') if p.is_file() and p.name not in
                 ('artifact_hashes.json', 'supervisor.lock', 'running.json')]
    artifacts += list((OUT / 'summaries').glob('*.json'))
    atomic(OUT / 'artifact_hashes.json', {str(p.relative_to(OUT)): sha(p) for p in artifacts})
    print(json.dumps(dict(status='PASSED', records=total, paired_questions=total//2,
                         gpu_smokes=len(smokes), missing_or_duplicate=0)))


if __name__ == '__main__':
    main()
