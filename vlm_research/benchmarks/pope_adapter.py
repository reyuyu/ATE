"""Original COCO POPE JSONL, preserving complete image groups and file year."""
import json
from pathlib import Path
import re

from .dataset_registry import audit_rows, digest, validate_sample


def read_pope(path, strategy, revision, image_root):
    result = []
    with Path(path).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        filename = row['image']
        if not re.fullmatch(r'COCO_val2014_\d{12}\.jpg', filename):
            raise ValueError('POPE must use its actual COCO val2014 filename')
        if row['label'] not in ('yes', 'no'):
            raise ValueError('Invalid POPE label')
        result.append(validate_sample(dict(
            dataset='pope', subset=strategy, split='COCO_val2014', question_id=str(row['question_id']),
            image_id=filename, image_path=str(Path(image_root)/filename), question=row['text'],
            answer=row['label'], candidates=['no', 'yes'], answer_type='closed', source_revision=revision,
            source_row_sha256=digest(row), image_sha256=None,
            metadata={'raw_row': row, 'resource_relative_path': filename, 'cluster_id': filename})))
    audit_rows(result)
    return result
