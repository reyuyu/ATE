"""Isolate official scorer mutations without weakening checkpoint integrity."""
from copy import deepcopy

from ..fastv_svh.inputs import score
from .common import digest


def score_immutable(sample, text):
    return score(deepcopy(sample), text)


def audit_sample(record, sample, allow_legacy_scoring_mutation=False):
    canonical = digest(sample)
    if record['sample_sha256'] == canonical:
        return None
    assert allow_legacy_scoring_mutation, (sample['key'], 'Sample binding mismatch')
    assert sample['benchmark'] == 'MMBench', sample['key']
    replay = deepcopy(sample)
    rescored = score(replay, record['generation']['text'])
    assert rescored == record['scoring'], sample['key']
    changed = {key for key in sample if sample[key] != replay[key]}
    assert changed == {'options'}, (sample['key'], changed)
    assert replay['options'] == {key: str(value).lower() for key, value in sample['options'].items()}
    assert digest(replay) == record['sample_sha256'], sample['key']
    assert digest(sample) == canonical
    return dict(key=sample['key'], canonical_sample_sha256=canonical,
        stored_postscoring_sample_sha256=record['sample_sha256'],
        reproduced_postscoring_sample_sha256=digest(replay), changed_fields=['options'],
        reason='Frozen official can_infer_text lowercases its choices argument in place',
        unchanged=['Prompt', 'Input IDs', 'Image SHA256', 'Gold', 'Generation', 'Original record'],
        raw_record_checksum_still_required=True, exact_reproduction=True)
