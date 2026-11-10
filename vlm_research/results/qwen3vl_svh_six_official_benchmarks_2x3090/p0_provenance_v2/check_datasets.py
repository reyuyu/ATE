"""CPU-only ID/gold provenance checks; does not decode images or import torch."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = Path('/data/lk/ATE-worktrees/fastv-k3-svh-sixbench-3090/vlm_research/results/fastv_k3_svh_sixbench_3090/sample_manifest.jsonl.gz')
PHASE = Path('/data/lk/ATE-resources/benchmarks/phase4_0/sources')
DATA = Path('/data/lk/ATE-benchmarks-20261105/datasets')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run():
    grouped = collections.defaultdict(list)
    with gzip.open(MANIFEST, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            grouped[row['benchmark']].append(row)
    report = {'scope': 'CPU_STATIC_IDS_GOLD_ONLY_NO_NEW_IMAGE_HASH_AUDIT',
              'manifest': str(MANIFEST), 'manifest_sha256': sha(MANIFEST),
              'inventory': {}, 'annotations': []}

    def annotate(path):
        report['annotations'].append({'path': str(path), 'sha256': sha(path)})

    for task, rows in grouped.items():
        report['inventory'][task] = dict(rows=len(rows), unique_question_ids=len({r['question_id'] for r in rows}),
                                        image_sha_clusters=len({r['image_sha256'] for r in rows}),
                                        missing_image_files=sum(not Path(r['image_path']).is_file() for r in rows))

    gqa_path = Path('/data/lk/ATE-resources/data/gqa-testdev-balanced-official/testdev_balanced_questions.json')
    annotate(gqa_path)
    gqa = json.loads(gqa_path.read_text())
    report['GQA'] = dict(source_rows=len(gqa), compared_rows=len(grouped['GQA']),
                        mismatches=[r['question_id'] for r in grouped['GQA']
                                    if r['question_id'] not in gqa or any([
                                        gqa[r['question_id']]['question'] != r['question'],
                                        gqa[r['question_id']]['answer'] != r['answer'],
                                        str(gqa[r['question_id']]['imageId']) != str(r['image_id'])])])

    problems_path = PHASE / 'scienceqa/data/scienceqa/problems.json'
    splits_path = PHASE / 'scienceqa/data/scienceqa/pid_splits.json'
    annotate(problems_path)
    annotate(splits_path)
    problems = json.loads(problems_path.read_text())
    test = json.loads(splits_path.read_text())['test']
    image_ids = {str(i) for i in test if problems[str(i)]['image']}
    sqa = grouped['ScienceQA-IMG']
    report['ScienceQA-IMG'] = dict(official_test_rows=len(test), official_image_test_rows=len(image_ids),
                                  identical_id_set=image_ids == {r['question_id'] for r in sqa},
                                  gold_choice_hint_mismatches=[])
    for r in sqa:
        src = problems[r['question_id']]
        if not (isinstance(r['answer'], int) and r['answer'] == src['answer']
                and r['question'] == src['question']
                and list(r['options'].values()) == src['choices']
                and r['raw']['hint'] == src['hint']):
            report['ScienceQA-IMG']['gold_choice_hint_mismatches'].append(r['question_id'])

    tv_path = PHASE / 'textvqa/TextVQA_0.5.1_val.json'
    annotate(tv_path)
    tv = {str(r['question_id']): r for r in json.loads(tv_path.read_text())['data']}
    report['TextVQA'] = dict(source_rows=len(tv), identical_id_set=set(tv) == {r['question_id'] for r in grouped['TextVQA']},
                            question_image_ten_answers_mismatches=[])
    for r in grouped['TextVQA']:
        s = tv[r['question_id']]
        if not (r['question'] == s['question'] and r['image_id'] == s['image_id']
                and r['answer'] == s['answers'] and len(s['answers']) == 10):
            report['TextVQA']['question_image_ten_answers_mismatches'].append(r['question_id'])

    pope = {}
    omitted_rows = []
    old = {(r['strategy'], int(r['question_id'].split(':')[-1])): r for r in grouped['POPE']}
    for strategy in ('random', 'popular', 'adversarial'):
        path = PHASE / f'pope/output/coco/coco_pope_{strategy}.json'
        annotate(path)
        src = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        omitted = [r for r in src if (strategy, r['question_id']) not in old]
        mismatches = []
        for r in src:
            previous = old.get((strategy, r['question_id']))
            if previous and (previous['question'] != r['text'] or previous['image_id'] != r['image']
                             or previous['answer'] != r['label']):
                mismatches.append(r['question_id'])
        pope[strategy] = dict(original_rows=len(src), old_rows=sum(s == strategy for s, _ in old),
                              omitted_ids=[r['question_id'] for r in omitted], mismatches=mismatches,
                              omitted_gold_counts=dict(collections.Counter(r['label'] for r in omitted)),
                              original_gold_counts=dict(collections.Counter(r['label'] for r in src)))
        for r in omitted:
            image = DATA / 'COCO2014/val2014' / r['image']
            omitted_rows.append(dict(strategy=strategy, question_id=r['question_id'], image=r['image'],
                                     question=r['text'], gold=r['label'], image_path=str(image), image_exists=image.is_file()))
    report['POPE'] = pope
    report['POPE']['exclusion_reason'] = 'UNRESOLVED: row difference verified; causal reason for legacy omission not inferred'
    report['POPE']['id_namespace'] = 'Join by (strategy, raw question_id suffix), not globally offset author_question_id'
    with (ROOT / 'pope_9000_vs_8910_missing.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(omitted_rows[0]))
        writer.writeheader()
        writer.writerows(omitted_rows)

    mme_bad = []
    mme_label_case_normalizations = 0
    mme_annotations = set()
    pairs = collections.defaultdict(list)
    for r in grouped['MME']:
        task, name = r['image_id'].split('/', 1)
        path = DATA / 'MME/MME_Benchmark_release_version' / task / Path(name).with_suffix('.txt')
        if not path.exists():
            path = path.parent / 'questions_answers_YN' / path.name
        if not path.exists():
            mme_bad.append({'id': r['question_id'], 'reason': 'raw annotation missing', 'path': str(path)})
            continue
        raw_lines = [line.strip().split('\t') for line in path.read_text().splitlines() if line.strip()]
        lines = [[q, a.lower()] for q, a in raw_lines]
        mme_label_case_normalizations += sum(q == r['question'] and a != a.lower() for q, a in raw_lines)
        mme_annotations.add(path)
        if [r['question'], r['answer']] not in lines:
            mme_bad.append({'id': r['question_id'], 'reason': 'question/gold mismatch'})
        pairs[r['pair_id']].append(r['answer'])
    report['MME'] = dict(tasks=dict(collections.Counter(r['task'] for r in grouped['MME'])),
                         pair_count=len(pairs), pair_size_histogram=dict(collections.Counter(map(len, pairs.values()))),
                         raw_question_gold_failures=mme_bad,
                         raw_gold_case_normalization_count=mme_label_case_normalizations,
                         raw_gold_case_rule='Official calculation.py lowercases Gold; question text is not normalized')
    for path in sorted(mme_annotations):
        annotate(path)

    tsv = DATA / 'MMBench/mmbench_dev_20230712.tsv'
    annotate(tsv)
    csv.field_size_limit(1024 * 1024 * 1024)
    source = {}
    with tsv.open() as stream:
        for r in csv.DictReader(stream, delimiter='\t'):
            source[str(r['index'])] = {k: r[k] for k in ['index', 'question', 'answer', 'hint', 'A', 'B', 'C', 'D'] if k in r}
    mismatches = []
    for r in grouped['MMBench']:
        src = source[r['question_id']]
        if not (src['answer'] == r['answer'] and src['question'] == r['question']
                and all(src[k] == v for k, v in r['options'].items())):
            mismatches.append(r['question_id'])
    report['MMBench'] = dict(source_path=str(tsv), resolved_path=str(tsv.resolve()), source_rows=len(source),
                             id_sets_equal=set(source) == {r['question_id'] for r in grouped['MMBench']},
                             question_choices_gold_mismatches=mismatches,
                             circular_groups=len({r['circular_group'] for r in grouped['MMBench']}),
                             version_evidence='legacy filename 20230712; NOT attested MMBench-V1.1',
                             author_report_version='MMBench-V1.1')
    (ROOT / 'dataset_id_gold_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'scope': report['scope'], 'inventory': report['inventory'],
                      'gqa_mismatches': len(report['GQA']['mismatches']),
                      'sqa_mismatches': len(report['ScienceQA-IMG']['gold_choice_hint_mismatches']),
                      'textvqa_mismatches': len(report['TextVQA']['question_image_ten_answers_mismatches']),
                      'mme_mismatches': len(mme_bad), 'mmbench_mismatches': len(mismatches),
                      'pope_omitted': len(omitted_rows)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
