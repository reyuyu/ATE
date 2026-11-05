"""TextVQA v0.5.1 validation keeps all 10 votes and OCR as metadata only."""
from pathlib import Path

from .dataset_registry import digest, validate_sample


def textvqa_rows(annotation, revision, image_root):
    if str(annotation.get('dataset_version')) != '0.5.1':
        raise ValueError('Require official TextVQA v0.5.1')
    result = []
    for row in annotation['data']:
        if row['set_name'] != 'val':
            raise ValueError('TextVQA validation only')
        relative = row['image_id'] + '.jpg'
        answers = row['answers']
        if len(answers) != 10 or not all(isinstance(a, str) for a in answers):
            raise ValueError('Keep all 10 human answers')
        result.append(validate_sample(dict(
            dataset='textvqa', subset='v0.5.1', split='validation', question_id=str(row['question_id']),
            image_id=row['image_id'], image_path=str(Path(image_root)/relative), question=row['question'],
            answer=answers, candidates=None, answer_type='vqa_soft', source_revision=revision,
            source_row_sha256=digest(row), image_sha256=None,
            metadata={'raw_row': row, 'expected_image_size': [row['image_width'], row['image_height']],
                      'resource_relative_path': relative, 'ocr_policy': 'Metadata only; never inserted into prompt'})))
    return result
