"""Bind author evaluation inputs to existing, SHA-verified local images."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import io
import json
import math
import re
import subprocess
import zipfile

from .common import *

EVAL_SHA = '2df36a33e3d3947d3e351e627449ee7cc7725d69ecc8dcbb2103d90172fbe17a'
EVAL_REV = '50547ed80a7844484d2849431144225841b9e935'


def annotation_files():
    archive = RES / 'eval.zip'
    assert sha(archive) == EVAL_SHA
    wanted = ('llava_textvqa_val_v051_ocr.jsonl', 'llava_test_CQM-A.json',
              'llava_mme.jsonl', 'llava_pope_test.jsonl', 'convert_answer_to_mme.py')
    with zipfile.ZipFile(archive) as z:
        paths = {name: [v for v in z.namelist() if v.endswith('/' + name)] for name in wanted}
        assert all(len(v) == 1 for v in paths.values()), paths
        for name, values in paths.items():
            path = OUT / 'official' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(values[0])
            if path.exists():
                assert path.read_bytes() == data
            else:
                path.write_bytes(data)
        atomic(OUT / 'official/eval_archive_provenance.json', dict(
            official_source='https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view',
            mirror='Xiaodong/LLaVA-1.5-Eval', revision=EVAL_REV,
            archive_sha256=EVAL_SHA, paths=paths,
            limitation='Mirror provenance is attributed; original Google Drive bytes not independently available.'))


def build_samples():
    local = {}
    for bench in ORDER:
        entries = [json.loads(x) for x in (DATA / bench / 'inputs.jsonl').open()]
        answers = read(DATA / bench / 'answers.json')
        local[bench] = (entries, answers)
    result = []

    def add(bench, old, text, author_id, **extra):
        assert text and '<image>' not in text
        row = dict(key=bench + ':' + old['question_id'], benchmark=bench,
                   question_id=old['question_id'], image_id=old['image_id'],
                   image_path=old['image_path'], question=old['question'],
                   author_question_id=author_id, user_text=text, prompt=prompt(text),
                   answer=local[bench][1][old['question_id']],
                   max_new_tokens=1024 if bench in ('ScienceQA-IMG', 'MMBench') else 128,
                   raw=old, **extra)
        result.append(row)

    text_lookup = {(s['image_id'], s['question'].lower()): s for s in local['TextVQA'][0]}
    for row in (json.loads(x) for x in (OUT / 'official/llava_textvqa_val_v051_ocr.jsonl').open()):
        text = row['text']
        if text.startswith('OCR tokens: '):
            q = re.search(r'Question: (.*?) Short answer:', text, re.DOTALL).group(1)
        else:
            lines = text.split('\n')
            q = lines[1] if text.startswith('Reference OCR token:') else lines[0]
        old = text_lookup[(str(row['question_id']), q.lower())]
        add('TextVQA', old, text, row['question_id'], author_image=row['image'],
            ocr_text=text, paper_ocr=True)

    science_lookup = {s['question_id']: s for s in local['ScienceQA-IMG'][0]}
    problems = read(DATA / 'ScienceQA-IMG/official_problems_gold_only.json')
    for row in read(OUT / 'official/llava_test_CQM-A.json'):
        if 'image' not in row:
            continue
        old = science_lookup[str(row['id'])]
        q = row['conversations'][0]['value'].replace('<image>', '').strip()
        gold = problems[old['question_id']]
        assert gold['question'] in q
        for choice in gold['choices']:
            assert choice in q
        # Only human input is consumed. The file's assistant answer is never passed to the model.
        add('ScienceQA-IMG', old, q + '\n' + MC, row['id'],
            author_image=row['image'], options=dict(zip('ABCDE', gold['choices'])))

    def mme_core(text):
        return text.replace('Answer the question using a single word or phrase.', '').replace('Please answer yes or no.', '').strip()
    mme_lookup = defaultdict(list)
    for old in local['MME'][0]:
        mme_lookup[(old['image_id'], mme_core(old['question']))].append(old)
    for row in (json.loads(x) for x in (OUT / 'official/llava_mme.jsonl').open()):
        image = row['image']
        image = image.replace('/images/', '/')
        matches = mme_lookup[(image, mme_core(row['text']))]
        assert len(matches) == 1, (row, matches)
        old = matches[0]
        add('MME', old, row['text'], row['question_id'], author_image=row['image'],
            task=old['subtask'], pair_id=old['image_id'])

    pope_lookup = defaultdict(list)
    for s in local['POPE'][0]:
        pope_lookup[(s['strategy'], s['image_id'], s['question'])].append(s)
    for row in (json.loads(x) for x in (OUT / 'official/llava_pope_test.jsonl').open()):
        strategy = row['category']
        question = row['text'].replace('\nAnswer the question using a single word or phrase.', '')
        key = (strategy, row['image'], question.replace('imange', 'image'))
        matches = pope_lookup[key]
        assert matches, key
        assert len({local['POPE'][1][s['question_id']] for s in matches}) == 1
        old = matches[0]
        add('POPE', old, row['text'], row['question_id'], author_image=row['image'], strategy=strategy,
            author_question_typo_preserved=(question != old['question']))
        result[-1]['key'] = 'POPE:' + str(row['question_id'])

    csv.field_size_limit(100 * 1024**2)
    table = csv.DictReader((DATA / 'MMBench/mmbench_dev_20230712.tsv').open(), delimiter='\t')
    mm_lookup = {s['question_id']: s for s in local['MMBench'][0]}
    for row in table:
        old = mm_lookup[str(row['index'])]
        options = {k: row[k] for k in 'ABCD' if row.get(k) and row[k].lower() not in ('nan','none')}
        q = row['question']
        if row['hint'] and row['hint'].lower() not in ('nan','none'):
            q = row['hint'] + '\n' + q
        q += ''.join('\n' + k + '. ' + value for k, value in options.items())
        add('MMBench', old, q + '\n' + MC, int(row['index']), options=options,
            circular_group=int(row['index']) % 1000000)
    expected = dict(TextVQA=5000, MME=2374, **{'ScienceQA-IMG': 2017}, MMBench=4377, POPE=8910)
    assert dict(Counter(s['benchmark'] for s in result)) == expected
    assert len({s['key'] for s in result}) == len(result)
    return result


def shard(rows):
    # Union identical image bytes across benchmarks, plus every circular group.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(a)] = find(b)
    groups = {}
    for row in rows:
        im = row['image_sha256']
        if row['benchmark'] == 'MMBench':
            group = str(row['circular_group'])
            if group in groups:
                union(im, groups[group])
            groups[group] = im
        find(im)
    clusters = defaultdict(list)
    for row in rows:
        clusters[find(row['image_sha256'])].append(row)
    loads = [0] * 4
    for key, values in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rank = min(range(4), key=lambda r: (loads[r], r))
        for row in values:
            row['shard'] = rank
            row['execution_cluster'] = key
        loads[rank] += len(values)
    assert len({(s['image_sha256'], s['shard']) for s in rows}) == len({s['image_sha256'] for s in rows})
    return loads


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    annotation_files()
    rows = build_samples()
    paths = sorted({s['image_path'] for s in rows})
    with ThreadPoolExecutor(max_workers=12) as pool:
        hashes = dict(zip(paths, pool.map(sha, paths)))
    for row in rows:
        row['image_sha256'] = hashes[row['image_path']]
    loads = shard(rows)
    rows.sort(key=lambda s: (ORDER.index(s['benchmark']), s['key']))
    output = OUT / 'sample_manifest.jsonl.gz'
    with gzip.GzipFile(filename=str(output), mode='wb', mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n').encode())
    source = {str(p.relative_to(AUTHOR)): sha(p) for p in sorted(AUTHOR.rglob('*.py'))}
    atomic(OUT / 'author_repo_freeze.json', dict(commit=AUTHOR_SHA,
        remote='https://github.com/YahongWang1/Information-Horizon', sources=source,
        tracked_worktree_clean=True, native_revision=NATIVE_REV))
    audit = dict(at=now(), counts=dict(Counter(s['benchmark'] for s in rows)),
        total=len(rows), unique_image_bytes=len(set(hashes.values())), shards=loads,
        source_inputs={b: sha(DATA/b/'inputs.jsonl') for b in ORDER},
        source_answers={b: sha(DATA/b/'answers.json') for b in ORDER},
        paper_files={p.name: sha(p) for p in (OUT/'official').iterdir() if p.is_file()},
        native_max_new_tokens={b: (1024 if b in ('ScienceQA-IMG','MMBench') else 128) for b in ORDER},
        all_images_sha256_checked=True, no_sample_reselection=True,
        textvqa_historical_image_only_is_not_reused=True)
    atomic(OUT/'dataset_audit.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
