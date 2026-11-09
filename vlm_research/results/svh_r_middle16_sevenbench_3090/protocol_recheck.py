"""CPU-only comparison of frozen cohorts and protocols; no new inference."""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PAPER = Path('/data/lk/ATE-worktrees/cvpr2026-information-horizon-alignment/vlm_research/results/cvpr2026_information_horizon_alignment')
GQA_DATA = Path('/data/lk/ATE-resources/data/gqa-testdev-balanced-official')
PAPER_REF = '0414e553ba6fd1e21626ecf741489d225b44279c'
GQA_REF = '71a5762cd8ccb8cc99efe9b6b0267d8abef70c46'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def git_json(ref, path):
    return json.loads(subprocess.check_output(['git', 'show', ref + ':' + path], cwd=ROOT))


def main():
    current = read(OUT / 'protocol.json')
    paper = git_json(PAPER_REF, 'vlm_research/results/featurize_new3090_c3v2_reaccess_sevenbench/paper_protocol.json')
    gqa = git_json(GQA_REF, 'vlm_research/results/featurize_4090_gqa_official_aligned_threeway/protocol.json')
    old_run = git_json(GQA_REF, 'vlm_research/results/featurize_4090_gqa_official_aligned_threeway/run_metadata.json')
    with gzip.open(OUT / 'sample_manifest.jsonl.gz', 'rt') as f:
        samples = [json.loads(line) for line in f]
    with gzip.open(PAPER / 'sample_manifest.jsonl.gz', 'rt') as f:
        historical = {s['key']: s for s in map(json.loads, f)}
    artifact = sha(PAPER / 'sample_manifest.jsonl.gz')
    assert artifact == paper['artifacts']['sample_manifest.jsonl.gz']
    assert sha(OUT / 'sample_manifest.jsonl.gz') == current['manifest_sha256']
    for ref in current['references']:
        assert sha(Path(ref['path'])) == ref['sha256']
    for path, expected in current['sources'].items():
        assert sha(ROOT / path) == expected, path
    shared_source_checks = {}
    for name in ['vlm_research/information_horizon_alignment/engine.py',
                 'vlm_research/information_horizon_alignment/score.py',
                 'vlm_research/c3v2_middle_reaccess/engine.py']:
        assert current['sources'][name] == paper['sources'][name]
        shared_source_checks[name] = current['sources'][name]
    assert current['model_revision'] == paper['model_revision'] == gqa['revision']
    assert current['dtype'] == paper['dtype'] == gqa['generation']['dtype']
    assert current['attention'] == paper['backend'] == gqa['generation']['attention']
    assert current['batch_size'] == paper['batch_size'] == gqa['generation']['batch_size'] == 1
    assert current['logits_to_keep'] == paper['logits_to_keep'] == 1
    assert current['greedy'] == paper['greedy'] == gqa['generation']['greedy'] is True
    assert current['torch'] == paper['pytorch']
    assert current['transformers'] == paper['transformers']
    datasets = {}
    image_hashes = {}
    for bench in ['ScienceQA-IMG', 'TextVQA', 'POPE', 'MMBench']:
        subset = [s for s in samples if s['benchmark'] == bench]
        expected_keys = {k for k, s in historical.items() if s['benchmark'] == bench}
        assert {s['key'] for s in subset} == expected_keys
        assert len(subset) == paper['counts'][bench]
        assert all(s == historical[s['key']] for s in subset)
        assert all(s['max_new_tokens'] == paper['max_new_tokens'][bench] for s in subset)
        datasets[bench] = dict(rows=len(subset), exact_historical_manifest_rows=True,
                              cap=paper['max_new_tokens'][bench])
        for s in subset:
            image_hashes[s['image_path']] = s['image_sha256']
    with zipfile.ZipFile('/data/lk/ATE-resources/information-horizon-alignment/eval.zip') as z:
        prepared = z.read('gqa/llava_gqa_testdev_balanced.jsonl')
    assert hashlib.sha256(prepared).hexdigest() == gqa['official_prepared_sha256']
    official = {str(s['question_id']): s for s in map(json.loads, prepared.splitlines())}
    gold_path = GQA_DATA / 'testdev_balanced_questions.json'
    assert sha(gold_path) == gqa['official_Gold_sha256']
    gold = read(gold_path)
    subset = [s for s in samples if s['benchmark'] == 'GQA']
    assert {s['question_id'] for s in subset} == set(official) == set(gold)
    for s in subset:
        q = s['question_id']
        assert s['user_text'] == official[q]['text']
        assert s['question'] == gold[q]['question'] and s['answer'] == gold[q]['answer']
        assert str(s['image_id']) == str(gold[q]['imageId']) == Path(official[q]['image']).stem
        assert s['max_new_tokens'] == gqa['generation']['max_new_tokens'] == 32
        assert s['prompt'].endswith(' USER: <image>\n' + official[q]['text'] + ' ASSISTANT:')
        image_hashes[s['image_path']] = s['image_sha256']
    datasets['GQA'] = dict(rows=len(subset), exact_official_question_id_question_gold_image_mapping=True,
                           exact_official_prepared_user_text=True, cap=32,
                           protocol_sha256=current['references'][0]['sha256'],
                           old_gpu=old_run['generation_progress']['GPU_before_load'],
                           same_hardware=False)
    mismatches = [str(path) for path, expected in image_hashes.items() if sha(Path(path)) != expected]
    assert not mismatches, mismatches
    report = dict(status='PASS_FROZEN_DATA_AND_DECLARED_PROTOCOL_WITH_LIMITATIONS',
                  at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  CPU_only=True, new_gpu_work=False,
                  paper_source_manifest_sha256=artifact, datasets=datasets,
                  unique_current_images_rehashed=len(image_hashes), image_hash_mismatches=mismatches,
                  common_frozen_source_hashes=shared_source_checks,
                  new_source_unchanged=True,
                  historical_complete_predictions_restored=False,
                  historical_per_question_input_tensor_hashes_restored=False,
                  limitations=['GQA historical comparator used RTX4090; current inference uses RTX3090.',
                               'No same-hardware full historical control replay in this run.',
                               'Missing bound historical predictions prevent paired harm/rescue/CI.',
                               'MME and OCRBench are BLOCKED, not accepted as matching datasets.',
                               'No blanket claim of equivalence to all external paper methods.'])
    (OUT / 'protocol_recheck.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
