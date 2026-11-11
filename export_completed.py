"""Export completed stages without modifying live checkpoints or frozen sources."""
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import statistics
import subprocess
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'vlm_research/results/qwen3vl_svhr_l11_l24_budget_search'
DEST = OUT / 'completed_snapshot'
STAGES = {'Discovery': (585, 41), 'Validation': (1321, 6)}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_records(path, stages=None):
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True, timeout=20)
    try:
        db.execute('BEGIN')
        query = 'SELECT id,payload,sha256 FROM records'
        params = []
        if stages:
            query += ' WHERE ' + ' OR '.join('id LIKE ?' for _ in stages)
            params = [stage + '|Unified|%' for stage in stages]
        for key, blob, expected in db.execute(query + ' ORDER BY id', params):
            raw = zlib.decompress(blob)
            assert hashlib.sha256(raw).hexdigest() == expected, (path, key)
            yield key, json.loads(raw), expected
    finally:
        db.close()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def write_jsonl(path, records):
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def export():
    DEST.mkdir(exist_ok=True)
    manifest = json.loads((OUT / 'sample_manifest.json').read_text())
    selected = {r['key']: r for r in manifest if r['split'] in STAGES}
    seen = set()
    results = defaultdict(list)
    for shard in [0, 1]:
        for key, row, checksum in read_records(OUT / f'worker{shard}.sqlite', STAGES):
            assert key not in seen, ('DUPLICATE_CHECKPOINT', key)
            seen.add(key)
            assert row['stage'] in STAGES and row['view'] == 'Unified'
            spec = selected[row['key']]
            assert row['image_sha256'] == spec['image_sha256']
            if 'score' in row:
                assert row['prompt_sha256'] == spec['prompt_sha256']
            else:
                assert row['status'] == 'BLOCKED_HISTORICAL_OOM'
            assert row.get('prefill_layers', row.get('layers')) == row.get('decode_layers', row.get('layers'))
            if 'score' in row:
                assert math.isfinite(row['score'])
            results[row['stage']].append((key, row, checksum))
    stage_counts = {}
    for stage, (n, conditions) in STAGES.items():
        rows = sorted(results[stage], key=lambda x: x[0])
        counts = Counter(row['condition'] for _, row, _ in rows)
        assert len(counts) == conditions and set(counts.values()) == {n}, (stage, counts)
        assert len(rows) == n * conditions
        write_jsonl(DEST / f'{stage}_full_records.jsonl.gz',
                    (dict(checkpoint_id=key, checkpoint_payload_sha256=checksum, record=row) for key, row, checksum in rows))
        columns = ['key', 'benchmark', 'analysis_task', 'condition', 'image_sha256', 'question', 'gold',
                   'text', 'token_ids', 'score', 'status', 'state', 'eos', 'truncated', 'empty', 'generated_tokens',
                   'first_token_flip_vs_svh', 'layers', 'prefill_layers', 'decode_layers', 'physical_svhr',
                   'strict_numerical_status', 'input_binding_sha256', 'prompt_sha256', 'protocol_sha256', 'gpu_uuid']
        flat = [dict(stage=stage, checkpoint_id=key, checkpoint_payload_sha256=checksum,
                     **{column: row.get(column) for column in columns}) for key, row, checksum in rows]
        write_csv(DEST / f'{stage}_per_question.csv', flat)
        blocked = [dict(checkpoint_id=key, **row) for key, row, _ in rows if 'score' not in row]
        write_jsonl(DEST / f'{stage}_blocked_records.jsonl.gz', blocked)
        stage_counts[stage] = dict(records=len(rows), conditions=dict(counts),
                                   valid_records=len(rows)-len(blocked), blocked_records=len(blocked))
    baseline = []
    for key, row, checksum in read_records(OUT / 'historical.sqlite'):
        if row['key'] not in selected:
            continue
        spec = selected[row['key']]
        assert row['image_sha256'] == spec['image_sha256']
        assert row['prompt_sha256'] == spec['prompt_sha256']
        baseline.append(dict(checkpoint_id=key, checkpoint_payload_sha256=checksum,
                             split=spec['split'], record=row))
    assert len(baseline) == 2 * len(selected), (len(baseline), len(selected))
    assert len({r['checkpoint_id'] for r in baseline}) == len(baseline)
    write_jsonl(DEST / 'paired_historical_baselines.jsonl.gz', baseline)

    performance = []
    for shard in [0, 1]:
        assert (OUT / f'jobs/013_Performance.done{shard}').exists()
        for key, row, checksum in read_records(OUT / f'performance{shard}.sqlite'):
            performance.append(dict(checkpoint_id=key, checkpoint_payload_sha256=checksum, **row))
    assert len(performance) == 1200
    assert len({r['checkpoint_id'] for r in performance}) == len(performance)
    assert len({r['key'] for r in performance}) == 10
    write_csv(DEST / 'performance_all_repeats.csv', performance)
    write_jsonl(DEST / 'performance_full_records.jsonl.gz', performance)
    measured = [r for r in performance if not r['warmup']]
    assert len(measured) == 750
    grouped = defaultdict(list)
    for row in measured:
        grouped[row['mode'], row['condition'], row['key']].append(row)
    assert set(map(len, grouped.values())) == {5}
    per_input = []
    metrics = ['e2e_ms', 'prefill_including_vision_ms', 'vision_ms', 'decode_ms', 'tpot_ms',
               'generated_tokens', 'kv_live_bytes', 'kv_capacity_bytes', 'peak_allocated', 'peak_reserved']
    for (mode, condition, key), rows in sorted(grouped.items()):
        item = dict(mode=mode, condition=condition, key=key, image_sha256=rows[0]['image_sha256'],
                    gpu_uuid=rows[0]['gpu_uuid'], repeats=5)
        item.update({metric: statistics.mean(r[metric] for r in rows) for metric in metrics})
        sequences = {tuple(r['tokens']) for r in rows}
        assert len(sequences) == 1, ('NONDETERMINISTIC_TOKENS', mode, condition, key)
        item['tokens'] = list(next(iter(sequences)))
        if all(r['persistent_image_bytes'] is not None for r in rows):
            item['persistent_image_bytes'] = statistics.mean(r['persistent_image_bytes'] for r in rows)
        else:
            item['persistent_image_bytes'] = 'NOT_RECORDED_AS_SEPARATE_FIELD'
        per_input.append(item)
    write_csv(DEST / 'performance_per_input.csv', per_input)
    lookup = {(r['mode'], r['condition'], r['key']): r for r in per_input}
    aggregate = []
    for mode, condition in sorted({(r['mode'], r['condition']) for r in per_input}):
        rows = [r for r in per_input if (r['mode'], r['condition']) == (mode, condition)]
        native = [lookup[mode, 'Native', r['key']] for r in rows]
        svh = [lookup[mode, 'SVH', r['key']] for r in rows]
        item = dict(mode=mode, condition=condition, images=len(rows), repeats_per_image=5)
        item.update({metric: statistics.mean(r[metric] for r in rows) for metric in metrics})
        item['native_e2e_over_candidate_e2e'] = statistics.mean(r['e2e_ms'] for r in native) / item['e2e_ms']
        item['svh_e2e_over_candidate_e2e'] = statistics.mean(r['e2e_ms'] for r in svh) / item['e2e_ms']
        same = [r for r, n in zip(rows, native) if r['tokens'] == n['tokens']]
        item['same_native_sequence_images'] = len(same)
        item['same_sequence_native_speedup'] = (statistics.mean(lookup[mode, 'Native', r['key']]['e2e_ms'] for r in same)
                                               / statistics.mean(r['e2e_ms'] for r in same)) if same else None
        item['timing_boundary'] = performance[0]['timing_boundary']
        item['hardware_tflops'] = 'NOT_MEASURED_NO_HARDWARE_COUNTERS'
        aggregate.append(item)
    write_csv(DEST / 'performance_summary.csv', aggregate)
    partial = []
    assert (OUT / 'wider_pipeline_failure.json').exists(), 'Only archive stopped partial checkpoints'
    for shard in [0, 1]:
        for key, row, checksum in read_records(OUT / f'worker{shard}.sqlite', ['Held-out']):
            partial.append(dict(checkpoint_id=key, checkpoint_payload_sha256=checksum, record=row))
    assert len({r['checkpoint_id'] for r in partial}) == len(partial)
    write_jsonl(DEST / 'PARTIAL_Held-out_raw_records.jsonl.gz', partial)
    partial_counts = Counter(r['record']['condition'] for r in partial)
    write_csv(DEST / 'PARTIAL_Held-out_coverage.csv',
              [dict(condition=name, checkpoint_rows=n, planned_rows=1641, status='PARTIAL_STOPPED_OOM')
               for name, n in sorted(partial_counts.items())])
    gpu_processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv'], text=True)
    own_uuids = ['GPU-b08489af-0a8a-eb48-55ba-8346733af417', 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
    assert all(uuid not in gpu_processes for uuid in own_uuids), 'Owned GPU context still present'
    (DEST / 'gpu_release_snapshot.json').write_text(json.dumps(dict(
        timestamp=datetime.now(timezone.utc).isoformat(), owned_gpu_uuids=own_uuids,
        no_compute_process_on_owned_gpus=True, compute_processes_csv=gpu_processes,
        other_gpu_processes_untouched=True), indent=2) + '\n')
    audit = dict(status='COMPLETED_STAGES_ONLY_EXPORTED', overall_experiment='PARTIAL_HELD_OUT_BLOCKED_OOM',
                 created_at=datetime.now(timezone.utc).isoformat(), source_commit=subprocess.check_output(
                     ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 stages=stage_counts, paired_baseline_records=len(baseline), performance_records=len(performance),
                 performance_measured_records=len(measured), performance_images=10,
                 image_sha_and_prompt_binding='VERIFIED', payload_hashes='VERIFIED',
                 historical_strict_numerical_status='FAIL_RETAINED', held_out_outcomes_exported='RAW_PARTIAL_ONLY_NO_FINAL_SCORE',
                 held_out_checkpoint_rows=len(partial), held_out_planned_rows=6564,
                 live_sqlite_files_exported=False, model_weights_and_images_exported=False,
                 artifacts={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                            for p in sorted(DEST.iterdir()) if p.is_file() and p.name != 'snapshot_audit.json'})
    (DEST / 'snapshot_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ['status', 'stages', 'paired_baseline_records', 'performance_records']}, indent=2))
    for r in aggregate:
        if r['mode'] in ['natural', 'fixed32']:
            print(r['mode'], r['condition'], round(r['e2e_ms'], 2),
                  'Native speedup', round(r['native_e2e_over_candidate_e2e'], 3))


if __name__ == '__main__':
    export()
