"""Official Test IDs and real image entries only; choices never reordered."""
from pathlib import Path

from .dataset_registry import digest, validate_sample

MATCH_FIELDS = ['question', 'choices', 'answer', 'hint', 'task', 'grade', 'subject', 'topic', 'category', 'skill', 'lecture', 'solution']


def verify_publisher_order(problems, test_ids, mirror_rows):
    """Publisher builder iterates problems.items(), filtered by row['split']."""
    builder_ids = [pid for pid, row in problems.items() if row['split'] == 'test']
    if builder_ids != test_ids or len(test_ids) != len(mirror_rows):
        raise ValueError('Official split / publisher builder order differs')
    for pid, mirror in zip(builder_ids, mirror_rows):
        original = problems[pid]
        if any(original[k] != mirror[k] for k in MATCH_FIELDS):
            raise ValueError(f'Official/mirror content mismatch at PID {pid}')
        if bool(original['image']) != bool(mirror['image']):
            raise ValueError('Official image-presence mismatch')
    return dict(zip(builder_ids, mirror_rows))


def scienceqa_rows(problems, splits, revision, image_root):
    result = []
    if len(splits['test']) != len(set(splits['test'])):
        raise ValueError('Repeated ScienceQA Test IDs')
    for pid in splits['test']:
        row = problems[pid]
        if not row['image']:
            continue
        choices = row['choices']
        gold = row['answer']
        if not isinstance(gold, int) or not 0 <= gold < len(choices) or not 2 <= len(choices) <= 5:
            raise ValueError('Invalid ScienceQA choices/index')
        # Canonical choice identity is the original letter, not possibly duplicated text.
        letters = list('ABCDE'[:len(choices)])
        relative = str(Path('test') / str(pid) / row['image'])
        result.append(validate_sample(dict(
            dataset='scienceqa_img', subset='IMG', split='test', question_id=str(pid), image_id=str(pid),
            image_path=str(Path(image_root)/relative), question=row['question'], answer=letters[gold],
            candidates=letters, answer_type='closed', source_revision=revision,
            source_row_sha256=digest(row), image_sha256=None,
            metadata={'raw_row': row, 'choices': choices, 'gold_answer_index': gold,
                      'diagnostic_choice_texts': choices, 'resource_relative_path': relative,
                      'visual_necessity': 'Has image; not proof that solving requires image',
                      **{k: row.get(k) for k in ('subject', 'topic', 'category', 'skill', 'grade', 'task', 'hint')}})))
    return result
