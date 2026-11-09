"""CPU data acceptance, path restoration, frozen cohorts and image-cluster shards."""
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import subprocess

from .common import *


def git_file(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def snapshot(commit, relative):
    data = git_file(commit, relative)
    dest = OUT / 'references' / commit / Path(relative).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dict(source_commit=commit, git_path=relative, path=str(dest), sha256=sha(dest))


def partition(samples, workers=2):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    groups = {}
    for s in samples:
        im = s['image_sha256']
        find(im)
        if s['benchmark'] == 'MMBench':
            g = s['circular_group']
            if g in groups:
                parent[find(im)] = find(groups[g])
            groups[g] = im
    units = defaultdict(list)
    for s in samples:
        units[find(s['image_sha256'])].append(s['key'])
    loads = [0] * workers
    assignment = {}
    for key, values in sorted(units.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rank = min(range(workers), key=lambda r: (loads[r], r))
        for value in values:
            assignment[value] = rank
        loads[rank] += len(values)
    return dict(workers=workers, loads=loads, assignment=assignment)


def main():
    assert not (OUT / 'protocol.json').exists(), 'Protocol already frozen'
    OUT.mkdir(parents=True, exist_ok=True)
    inventories = {r['dataset']: r for r in read(AUDIT / 'data_inventory.json')}
    references = [snapshot(GQA_REF, 'vlm_research/results/featurize_4090_gqa_official_aligned_threeway/protocol.json')]
    assert references[0]['sha256'] == GQA_PROTOCOL_SHA
    for name in ['paper_protocol.json', 'ocr_protocol.json', 'ocr_SOURCE.json', 'seven_runner.py',
                 'report_zh.md', 'score_summary.json', 'model_acceptance.json']:
        references.append(snapshot(PAPER_REF, f'vlm_research/results/featurize_new3090_c3v2_reaccess_sevenbench/{name}'))
    registry = dict(status='MISSING_IN_BOTH_REFERENCE_COMMITS', requested_path='vlm_research/results/featurize_dataset_backup_3090/dataset_registry.json')
    atomic(OUT / 'dataset_registry_lookup.json', registry)
    acceptance = read(OUT / 'references' / PAPER_REF / 'model_acceptance.json')
    model_files = {}
    for name, expected in acceptance['files'].items():
        actual = dict(bytes=(MODEL / name).stat().st_size, sha256=sha(MODEL / name))
        assert actual['bytes'] == expected['bytes'] and actual['sha256'] == expected['sha256'], name
        model_files[name] = actual
    for path in sorted(MODEL.iterdir()):
        if path.is_file() and not path.name.endswith('.safetensors'):
            model_files[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    atomic(OUT / 'model_files_sha256.json', dict(revision=MODEL.name, files=model_files, all_weight_shards_match=True))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    test = subprocess.run([__import__('sys').executable, '-m', 'vlm_research.svh_middle16.tests'],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    (OUT / 'cpu_tests.log').write_text(test.stdout + test.stderr)
    atomic(OUT / 'cpu_tests.json', dict(returncode=test.returncode, passed=test.returncode==0,
                                      log_sha256=sha(OUT / 'cpu_tests.log')))
    assert test.returncode == 0, test.stderr
    paper = rows(PAPER / 'sample_manifest.jsonl.gz')
    assert dict(Counter(r['benchmark'] for r in paper)) == {k: COUNTS[k] for k in ['ScienceQA-IMG', 'TextVQA', 'POPE', 'MME', 'MMBench']}
    statuses = {b: {'status': 'READY', 'expected': n} for b, n in COUNTS.items()}
    mme = next(r for r in paper if r['benchmark'] == 'MME' and r['image_id'] == 'landmark/03d5e3bfc958be38.jpg')
    current = sha(mme['image_path'])
    expected_mme = 'ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a'
    if current != expected_mme:
        statuses['MME'].update(status='BLOCKED', reason='Reference 0414 recovered-image bytes absent; original local bytes must not be substituted',
                               path=mme['image_path'], actual_sha256=current, expected_sha256=expected_mme)
    ocr = read(AUDIT / 'official_provenance.json')
    if ocr['ocr_mirror_comparison']['official_zip_image_byte_parity'] != 'verified':
        statuses['OCRBench-v1'].update(status='BLOCKED', reason='Official fixed image ZIP unavailable; HF mirror is not verified byte-identical',
                                      annotation_verified=ocr['files'][0]['ok'], image_cache=str(Path('/data/lk/ATE-benchmarks-20261105/datasets/OCRBench-v1')))
    gold = read('/data/lk/ATE-resources/data/gqa-testdev-balanced-official/testdev_balanced_questions.json')
    gqa = [json.loads(line) for line in (AUDIT / 'data/GQA-testdev-balanced/inputs.jsonl').read_text().splitlines()]
    samples = [dict(r, key='GQA:' + r['question_id'], benchmark='GQA', answer=gold[r['question_id']]['answer'],
                    max_new_tokens=32, prompt=SYSTEM + ' USER: <image>\n' + r['user_text'] + ' ASSISTANT:') for r in gqa]
    samples += paper
    assert len(samples) == 35256 and len({r['key'] for r in samples}) == len(samples)
    for s in samples:
        assert inventories[s['benchmark']]['status'] == 'ready'
        assert Path(s['image_path']).is_file()
        assert s['prompt'] == SYSTEM + ' USER: <image>\n' + s['user_text'] + ' ASSISTANT:'
    # Keep every restored source row in the manifest, including blocked MME.
    dest = OUT / 'sample_manifest.jsonl.gz'
    with dest.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as f:
            for s in samples:
                f.write((json.dumps(s, ensure_ascii=False) + '\n').encode())
    runnable = [s for s in samples if statuses[s['benchmark']]['status'] == 'READY']
    atomic(OUT / 'partition.json', partition(runnable))
    atomic(OUT / 'data_acceptance.json', dict(benchmarks=statuses, requested_total=36256,
           restored_manifest_rows=len(samples), runnable_total=len(runnable),
           prior_audit=str(AUDIT / 'data_inventory.json'), prior_audit_sha256=sha(AUDIT / 'data_inventory.json'),
           official_ocr_missing_rows=1000, no_resampling=True))
    protocol = dict(status='FROZEN_WITH_DATA_BLOCKERS', algorithm=asdict(MIDDLE16), base_commit=BASE_REF,
                    model_revision=MODEL.name, model_path=str(MODEL), dtype='float16', attention='eager',
                    torch='2.6.0+cu124', transformers='4.49.0', logits_to_keep=1, batch_size=1,
                    greedy=True, condition_count=1, requested_counts=COUNTS,
                    sources=sources(), references=references,
                    manifest_sha256=sha(dest), partition_sha256=sha(OUT / 'partition.json'),
                    numerical_gate=dict(logits_atol=0.10, logits_rtol=0.003, hidden_relative_rmse=0.01, hidden_cosine_distance=1e-4),
                    data_blockers={k: v for k, v in statuses.items() if v['status'] != 'READY'},
                    controls='Only row-bound same-protocol records may be reused; unavailable controls remain missing',
                    inference_cap_policy='Frozen per-benchmark cap; never reuse old GQA outputs for Middle16')
    atomic(OUT / 'protocol.json', protocol)
    print('PREPARED', len(samples), 'RUNNABLE', len(runnable), 'BLOCKED', list(protocol['data_blockers']))


if __name__ == '__main__':
    main()
