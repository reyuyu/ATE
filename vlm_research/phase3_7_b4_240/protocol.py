"""Freeze the existing 240 images, discovery split and actual input mappings."""
import csv
import json
from pathlib import Path
import subprocess

from phase3_7.protocol import PINS, atomic, cpu_input, digest, packages, read, sha

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/phase3_7_b4_240'
OLD = ROOT / 'results/phase3_7_l0_l1_pathways'
HISTORY = ROOT / 'results/phase3_2_fixed_k_exit'
BASELINE = 'f1e661fd228ec600e5a63b3016d68bd2fde64940'
GPU_UUID = 'GPU-a63c6362-dcfd-4a26-c44b-014b40127c89'
CONDITIONS = ['C0', 'C1']
GROUPS = ['image', 'question', 'bos', 'other_text']
RULES = dict(
    conditions=CONDITIONS, layers=[0, 1], gpu_physical=0, gpu_uuid=GPU_UUID,
    mask='Unmodified phase3_7.pathways.PathwayMask(condition=B4); P-1 Query -> all real Question Keys, additive -inf only in L0/L1.',
    score='Original complete candidate sequence logP, no EOS; left/right order, one token each; P-1 is the final prompt query, never the appended answer; margin>0 correct, ties wrong.',
    split='All original Phase3.2 240 distinct images; string question_id order. Primary: exact complement of original Phase3.7 16 image IDs (224); discovery16 and all240 supplementary. No new sampling or output filtering.',
    bootstrap=dict(unit='paired image', resamples=20000, seed=20261029, confidence=.95,
                   interval='percentile, numpy quantile linear', strata='none within each explicitly reported cohort/subset'),
    mcnemar='Two-sided exact binomial on discordant pairs; p=min(1,2*sum(comb(b+c,k),k=0..min(b,c))/2**(b+c)); p=1 if no discordance.',
    subsets=['all', 'left', 'right', 'full_correct', 'full_wrong'],
    primary='Accuracy difference and two-sided exact McNemar in confirmation224; other intervals descriptive, no multiplicity-adjusted claims.',
    mass='Per-head P-1 softmax mass partition: Image, real Question, actual BOS, remaining valid text. Compare same-image global Full/B4 and separately local native replay at identical actual layer inputs; L1 replay is not global Full.',
    correctness='No-op all valid-text logits exact; Question decoder states all32 layers and final RMSNorm exact; blocked weights zero, non-target local queries exact; original causal mask and all visual tokens retained; existing ROW_ATOL=.002 unchanged.',
    scope='Only Full vs B4. No training, decode, KV cache, exit, layer/strength scan or additional dataset. Do not infer joint necessity, text irrelevance or stronger visual fusion from mask definition.')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_hashes():
    paths = list(ROOT.glob('*.py')) + list((ROOT / 'phase3_7').glob('*.py'))
    paths += list((ROOT / 'phase3_7_b4_240').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def split_samples(samples, discovery_ids):
    if len(samples) != 240 or len({s['question_id'] for s in samples}) != 240 or len({s['img_id'] for s in samples}) != 240:
        raise ValueError('Require exactly the original 240 unique images/questions')
    if len(discovery_ids) != 16 or not set(discovery_ids) <= {s['img_id'] for s in samples}:
        raise ValueError('Discovery group must be the original 16 image IDs')
    result = [dict(s, cohort='discovery16' if s['img_id'] in discovery_ids else 'confirmation224')
              for s in sorted(samples, key=lambda s: s['question_id'])]
    if sum(s['cohort'] == 'confirmation224' for s in result) != 224:
        raise ValueError('Confirmation split is not 224')
    return result


def check_old_paths():
    paths = git('diff', '--name-only', BASELINE, '--', 'vlm_research').splitlines()
    allowed = ('vlm_research/phase3_7_b4_240/', 'vlm_research/results/phase3_7_b4_240/')
    if any(not p.startswith(allowed) and p != 'vlm_research/test_phase37_b4_240.py' for p in paths):
        raise ValueError('A historical/shared tracked file changed')


def prepare():
    from PIL import Image
    from transformers import AutoProcessor
    from prepare_fixed_k_data import digest as source_digest
    packages()
    if OUTPUT.exists():
        raise FileExistsError('Never overwrite a frozen manifest/result directory')
    if git('diff', 'HEAD', '--', '*.py') or git('ls-files', '--others', '--exclude-standard', '--', '*.py'):
        raise ValueError('Commit batch implementation and tests before freezing the manifest')
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASELINE, 'HEAD'], cwd=ROOT, check=True)
    check_old_paths()
    old = read(OLD / 'manifest.json')['config']
    historical = read(HISTORY / 'manifest.json')
    processor = AutoProcessor.from_pretrained(old['model_path'], local_files_only=True)
    image_id = read(Path(old['model_path']) / 'config.json')['image_token_index']
    source = ROOT / 'configs/phase3_1_data/GQA_val_correct_question_with_relChooser_ChooseRel.csv'
    if sha(source) != historical['source_csv_sha256']:
        raise ValueError('Frozen annotation source changed')
    lookup = {r['question_id']: r for r in csv.DictReader(source.open())}
    margins = {r['question_id']: float(r['full_margin']) for r in
               csv.DictReader((HISTORY / 'per_sample_results.csv').open()) if r['cutoff'] == '24'}
    discovery_ids = [s['img_id'] for s in old['samples']]
    split = split_samples(historical['samples'], discovery_ids)
    originals = [source, HISTORY / 'manifest.json', HISTORY / 'per_sample_results.csv',
                 OLD / 'manifest.json', OLD / 'per_sample_b.csv', OLD / 'publication_validation.json', OLD / 'pilot/validation.json',
                 ROOT / 'configs/model.json', ROOT / 'requirements.txt']
    samples = []
    for s in split:
        path = Path(old['image_root']) / f"image_{s['img_id']}.jpg"
        if sha(path) != s['image_sha256'] or source_digest(lookup[s['question_id']]) != s['source_row_sha256']:
            raise ValueError('Image or original annotation row changed')
        with Image.open(path) as image:
            if list(image.size) != s['source_image_size']:
                raise ValueError('Image dimensions changed')
            image.load()
        mapped = dict(question_id=s['question_id'], img_id=s['img_id'], question=s['question'],
                      answer=s['answer'], candidates=s['candidates'], cohort=s['cohort'],
                      image_path=str(path), original_image_path=s['image_path'], image_sha256=s['image_sha256'],
                      source_row_sha256=s['source_row_sha256'], historical_full_margin=margins[s['question_id']])
        mapped['input_mapping'] = cpu_input(processor, mapped, path, image_id)
        m = mapped['input_mapping']
        bos_id = processor.tokenizer.bos_token_id
        m['bos_positions'] = [i for i, token in enumerate(m['prompt_ids']) if token == bos_id]
        if len(m['visual_indices']) != 576 or m['bos_positions'] != [0]:
            raise ValueError('Unexpected visual or BOS layout')
        checkpoint = HISTORY / 'checkpoints/run' / (s['question_id'] + '.json')
        originals.append(checkpoint)
        data = read(checkpoint)['body']['data']
        if m['prompt_ids'] != data['prompt_ids'] or m['question_mapping'] != data['question_mapping']:
            raise ValueError('Original prompt/question mapping differs')
        for new, saved in zip(m['candidate_mapping'], data['candidate_mapping']):
            if new['input_ids'] != saved['input_ids'] or new['answer_records'] != saved['answer_records']:
                raise ValueError('Original complete-candidate scoring mapping differs')
        samples.append(mapped)
    for name, expected in old['model_files_sha256'].items():
        if sha(Path(old['model_path']) / name) != expected:
            raise ValueError('Canonical model file changed')
    smoke_ids = [min(s['question_id'] for s in samples if s['cohort'] == 'discovery16' and s['answer'] == label)
                 for label in ('left', 'right')]
    config = dict(baseline=BASELINE, code_commit=git('rev-parse', 'HEAD'), rules=RULES,
                  source_sha256=source_hashes(), packages=PINS, samples=samples,
                  model_path=old['model_path'], model_files_sha256=old['model_files_sha256'],
                  model_runtime_state_sha256=read(OLD / 'pilot/validation.json')['weights_sha256_before'],
                  image_root=old['image_root'], smoke_ids=smoke_ids,
                  discovery_question_ids=[s['question_id'] for s in samples if s['cohort'] == 'discovery16'],
                  confirmation_question_ids=[s['question_id'] for s in samples if s['cohort'] == 'confirmation224'],
                  original_files_sha256={str(p.relative_to(ROOT)): sha(p) for p in originals})
    OUTPUT.mkdir(parents=True)
    atomic(OUTPUT / 'manifest.json', dict(binding=digest(config), config=config))
    print('FROZEN', digest(config), 'confirmation=224', 'discovery=16', 'smoke=' + ','.join(smoke_ids), flush=True)


def protocol(execution=True):
    envelope = read(OUTPUT / 'manifest.json')
    c = envelope['config']
    if digest(c) != envelope['binding'] or c['rules'] != RULES or c['source_sha256'] != source_hashes():
        raise ValueError('Frozen configuration/source hash mismatch')
    if execution and git('rev-parse', 'HEAD') != c['code_commit']:
        raise ValueError('Execution requires the frozen code commit, not a later result commit')
    subprocess.run(['git', 'merge-base', '--is-ancestor', c['code_commit'], 'HEAD'], cwd=ROOT, check=True)
    for path, expected in c['original_files_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Original input/result changed: ' + path)
    check_old_paths()
    return c, envelope['binding']
