"""Render paired TextVQA failures without loading a model or changing results."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vlm_research/results/c3v2_middle_visual_reaccess'
OUT = SOURCE / 'native_success_middle10_failure_cases'
CASES = (
    ('34822', 'Time reading', (565, 130, 775, 290)),
    ('39376', 'Jersey number', (380, 685, 595, 855)),
    ('39441', 'Bottle number', (250, 670, 475, 960)),
    ('39065', 'Author name', (640, 380, 1010, 550)),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel(fig, slot, case):
    grid = slot.subgridspec(2, 2, width_ratios=[1.0, 1.0],
                           height_ratios=[1.0, 0.92], hspace=0.13, wspace=0.13)
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(case['image'])
    ax.set_title('Original image', fontsize=12, loc='left', pad=8)
    ax.axis('off')
    detail = fig.add_subplot(grid[0, 1])
    detail.imshow(case['image'].crop(case['detail_crop_xyxy']))
    detail.set_title('Reading detail (crop, no enhancement)', fontsize=11,
                     loc='left', pad=8)
    detail.axis('off')
    text = fig.add_subplot(grid[1, 1])
    text.axis('off')
    text.text(0, 1.0, f"ID {case['question_id']} | {case['category']}",
              fontsize=13, fontweight='bold', va='top', transform=text.transAxes)
    text.text(0, 0.83, 'Q: ' + case['question'], fontsize=12, va='top',
              wrap=True, transform=text.transAxes)
    text.text(0, 0.58, 'Human consensus: ' + case['majority_answer'],
              fontsize=12, va='top', transform=text.transAxes)
    text.text(0, 0.39, 'Native: ' + case['native_answer'] + '  [score 1.0]',
              color='#116644', fontsize=13, fontweight='bold', va='top',
              transform=text.transAxes)
    text.text(0, 0.18, 'Middle10: ' + case['middle10_answer'] + '  [score 0.0]',
              color='#b02c38', fontsize=13, fontweight='bold', va='top',
              transform=text.transAxes)


def render(cases, name):
    fig = plt.figure(figsize=(11.5, 6.8 * len(cases)), layout='constrained')
    grid = fig.add_gridspec(len(cases), 1)
    for i, case in enumerate(cases):
        panel(fig, grid[i], case)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / (name + '.' + suffix), dpi=170, facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / 'sample_manifest.json').read_text())
    groups = defaultdict(dict)
    with (SOURCE / 'per_question.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            groups[row['key']][row['condition']] = row
    eligible = sorted(key for key, rows in groups.items()
                      if float(rows['Native']['score']) == 1.0
                      and float(rows['C3+Middle10']['score']) == 0.0)
    records = []
    cases = []
    for qid, category, crop in CASES:
        key = 'textvqa__' + qid
        assert key in eligible
        sample = manifest[key]
        rows = groups[key]
        image_path = Path(sample['image_path'])
        image_sha = sha(image_path)
        assert image_sha == sample['image_sha256'] == rows['Native']['image_sha256']
        assert image_sha == rows['C3+Middle10']['image_sha256']
        assert rows['Native']['input_sha256'] == rows['C3+Middle10']['input_sha256']
        answers = json.loads(rows['Native']['gold_answers'])
        assert answers == sample['answer'] and len(answers) == 10
        image = Image.open(image_path).convert('RGB')
        assert 0 <= crop[0] < crop[2] <= image.width
        assert 0 <= crop[1] < crop[3] <= image.height
        record = dict(key=key, question_id=qid, category=category,
                      image_path=str(image_path), image_sha256=image_sha,
                      image_size=list(image.size), detail_crop_xyxy=list(crop),
                      question=sample['question'], gold_answers=answers,
                      majority_answer=Counter(answers).most_common(1)[0][0],
                      native_answer=rows['Native']['answer'], native_score=1.0,
                      middle10_answer=rows['C3+Middle10']['answer'], middle10_score=0.0,
                      input_sha256=rows['Native']['input_sha256'])
        records.append(record)
        cases.append(dict(record, image=image))
        render(cases[-1:], 'case_' + qid)
    assert len({c['image_sha256'] for c in records}) == len(records)
    render(cases[:2], 'cases_time_and_jersey')
    render(cases[2:], 'cases_bottle_and_author')
    with (OUT / 'cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False)
                             if isinstance(v, list) else v for k, v in row.items()})
    audit = dict(source_commit='a3727ba0808b183bc8bede4396a86e82855c7d5a',
                 source_csv_sha256=sha(SOURCE / 'per_question.csv'),
                 sample_manifest_sha256=sha(SOURCE / 'sample_manifest.json'),
                 script_sha256=sha(__file__),
                 total_questions=len(groups), eligible_count=len(eligible),
                 eligible_image_clusters=len({groups[k]['Native']['image_sha256']
                                              for k in eligible}),
                 eligible_question_ids=[groups[k]['Native']['question_id'] for k in eligible],
                 selection='Post-hoc illustrative selection for readable text and distinct images; not representative sampling.',
                 image_processing='Original images plus manually selected, explicitly labeled detail crops; no enhancement, retouching or generated pixels.',
                 inference_performed=False, cases=records)
    (OUT / 'selection.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    readme = '''# Native correct / Middle10 incorrect: TextVQA examples

These four cases are selected **post hoc** from the completed 500-question experiment.
The strict filter is Native official ten-answer soft score = 1.0 and Middle10 score = 0.0.
There are 48 eligible questions on 45 distinct images. Cases are chosen for readable
source text and distinct images, not to estimate the overall distribution of errors.
Original images are shown alongside explicitly labeled detail crops. No enhancement,
generated visual evidence, new inference, or changes to the frozen experiment are used.
These are output comparisons, not attention maps or evidence of a specific causal mechanism.

| ID | Question | Human majority | Native | Middle10 |
|---|---|---|---|---|
'''
    for c in records:
        readme += (f"|{c['question_id']}|{c['question']}|{c['majority_answer']}|"
                   f"{c['native_answer']} (1.0)|{c['middle10_answer']} (0.0)|\n")
    readme += '\n![Time and jersey](cases_time_and_jersey.png)\n\n'
    readme += '![Bottle and author](cases_bottle_and_author.png)\n\n'
    readme += ('`cases.csv` preserves all ten human answers. `selection.json` includes\n'
               'the full eligible pool, image/input hashes, crop coordinates and source hashes.\n\n'
               'Reproduce from this worktree with:\n\n'
               '```bash\n/data/lk/深度学习/.envs/ate-vlm-research/bin/python '
               'vlm_research/visualize_middle10_failures.py\n```\n')
    (OUT / 'README.md').write_text(readme)
    print(json.dumps(dict(output=str(OUT), eligible=len(eligible), selected=len(records))))


if __name__ == '__main__':
    main()
