"""Author CSVs are mechanism cohorts, not the official GQA benchmark."""
import ast
import csv
from pathlib import Path

from .dataset_registry import digest, validate_sample

GQA_FILES = {
    'gqa_chooseattr': 'GQA_val_correct_question_with_choose_ChooseAttr.csv',
    'gqa_queryattr': 'GQA_val_correct_question_with_positionQuery_QueryAttr.csv',
    'gqa_logicalobj': 'GQA_val_correct_question_with_existThatOr_LogicalObj.csv',
}


def read_gqa(path, dataset, revision, image_root):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    result = []
    for r in rows:
        candidates = None
        if dataset == 'gqa_chooseattr':
            choices = [op['argument'].split('|') for op in ast.literal_eval(r['operations'])
                       if op['operation'].startswith('choose')]
            if len(choices) != 1 or set(choices[0]) != {r['true option'], r['false option']} or r['answer'] != r['true option']:
                raise ValueError(f'Author candidate fields inconsistent: {r["question_id"]}')
            candidates = choices[0]
        elif dataset == 'gqa_logicalobj':
            if r['answer'] not in ('yes', 'no'):
                raise ValueError('Unexpected logical label')
            candidates = ['no', 'yes']
        filename = f'image_{r["img_id"]}.jpg'
        sample = dict(dataset=dataset, subset='author_preselected', split='GQA_val_author_filtered',
                      question_id=r['question_id'], image_id=r['img_id'], image_path=str(Path(image_root)/filename),
                      question=r['question'], answer=r['answer'], candidates=candidates,
                      answer_type='closed' if candidates else 'open', source_revision=revision,
                      source_row_sha256=digest(r), image_sha256=None,
                      metadata={'raw_row': r, 'resource_relative_path': filename,
                                'expected_image_size': [int(r['image_w']), int(r['image_h'])],
                                'selection_bias': 'Author correct_question prefilter; not unbiased GQA'})
        result.append(validate_sample(sample))
    return result
