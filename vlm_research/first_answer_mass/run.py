import argparse
import fcntl
import io
import json
from pathlib import Path
import subprocess
import textwrap
import time
import traceback
import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor
from attention_probe import SUM_ATOL
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6.run import prepared, tensor_hashes, atomic_json
from paper_rq1_early_joint_attention.protocol import ROOT, MODEL_PATH
from paper_rq1_early_joint_attention.run import preflight
from figure3_decode_windows.run import cohort, TASKS, ORIGINAL
from first_answer_attention.run import verify as verify_previous, baseline
from roi_sink_common import geometry
from vlm_common import load_model, gpu_memory
from .engine import true_question_positions, groups_for, observe

OUT = ROOT / 'results/first_answer_attention_mass'
SEED = 20261031
BASE = 'e495adf4fd93337a3638981bbb66c2bbac81242d'
ORIGINAL_COMMIT = '221cdde56dde53908aaa953fe26672fd33334f84'


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    names = ['first_answer_mass/' + n for n in ['__init__.py', 'engine.py', 'run.py', 'analyze.py']]
    names += ['test_first_answer_mass.py', 'attention_position_probe.py', 'attention_probe.py',
        'roi_sink_common.py', 'paper_rq1_early_joint_attention/statistics.py']
    return {n: sha256(ROOT / n) for n in names}


def prepare():
    p = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    samples = cohort()
    mapping = []
    for sample in samples:
        positions = true_question_positions(p, sample)
        groups_for(sample, positions)
        mapping.append(dict(task=sample['task'], question_id=sample['question_id'],
            image_id=sample['image_id'], image_sha256=sample['image_sha256'],
            sample_sha256=digest(sample), query=sample['input_mapping']['p_minus_one'],
            question_positions=positions))
    freeze_json(OUT / 'input_partitions.json', mapping)
    pool, images = [], set()
    for task in TASKS:
        candidates = sorted((s for s in samples if s['task'] == task),
            key=lambda s: digest(['first-answer-mass-cases', SEED, s['task'], s['question_id']]))
        chosen = []
        for sample in candidates:
            if sample['image_sha256'] not in images:
                chosen.append(sample)
                images.add(sample['image_sha256'])
            if len(chosen) == 2: break
        assert len(chosen) == 2
        pool.extend(chosen)
    freeze_json(OUT / 'case_pool.json', dict(selection='Per-task ascending SHA256 of fixed seed/task/qid; two unique images each; no outputs inspected.',
        seed=SEED, samples=[dict(task=s['task'], question_id=s['question_id'], image_id=s['image_id'],
            image_sha256=s['image_sha256'], question=s['question'], image_path=s['image_path']) for s in pool]))
    canvas = Image.new('RGB', (1200, 1440), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    for i, s in enumerate(pool):
        x, y = (i % 3) * 400, (i // 3) * 360
        image = Image.open(s['image_path']).convert('RGB')
        image.thumbnail((390, 240))
        canvas.paste(image, (x + (400-image.width)//2, y))
        label = f"{s['task']} / {s['question_id']}\n" + '\n'.join(textwrap.wrap(s['question'], 46))
        draw.multiline_text((x+8, y+245), label, fill='black', font=font, spacing=4)
    canvas.save(OUT / 'candidate_contact_sheet.png')
    print('Frozen3000 partitions and12-case pool before GPU inference', flush=True)


def cpu():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('test_first_answer_mass')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    freeze_json(OUT / 'cpu_validation.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        source=sources(), output=stream.getvalue()))
    print(stream.getvalue(), flush=True)
    assert result.wasSuccessful()


def freeze():
    previous = verify_previous()
    tested = read(OUT / 'cpu_validation.json')
    assert tested['passed'] and tested['source'] == sources()
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *sources()], cwd=ROOT, text=True).strip()
    selected = read(OUT / 'showcase_selection.json')
    pool = read(OUT / 'case_pool.json')['samples']
    assert len(selected['question_ids']) == 4 and len(set(selected['question_ids'])) == 4
    assert set(selected['question_ids']) <= {s['question_id'] for s in pool}
    manifest_path = 'vlm_research/results/paper_rq1_single_layer_gqa6/sample_manifest.json'
    blob = subprocess.check_output(['git', 'rev-parse', f'{ORIGINAL_COMMIT}:{manifest_path}'], cwd=ROOT, text=True).strip()
    assert blob == subprocess.check_output(['git', 'hash-object', ORIGINAL / 'sample_manifest.json'], cwd=ROOT, text=True).strip()
    freeze_json(OUT / 'execution_manifest.json', dict(base_commit=BASE, original_commit=ORIGINAL_COMMIT,
        original_manifest_blob=blob, original_manifest_sha256=sha256(ORIGINAL / 'sample_manifest.json'),
        previous_binding=digest(previous), model_revision=previous['model_revision'], source=sources(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_partitions_sha256=sha256(OUT / 'input_partitions.json'),
        case_pool_sha256=sha256(OUT / 'case_pool.json'), showcase_sha256=sha256(OUT / 'showcase_selection.json'),
        cpu_sha256=sha256(OUT / 'cpu_validation.json'), questions=3000, images=2409,
        protocol=dict(physical_gpu=1, native_full_only=True, query='Last valid original Prompt P-1, before first answer Token',
            cache=False, decode=False, teacher_forcing=False, row_sum_atol=SUM_ATOL,
            mass='Head mean of native post-softmax probabilities; four disjoint exhaustive groups.',
            question='Real question only, offset-verified; answer instructions belong to prompt_template.',
            prompt='question + prompt_template, excludes BOS and Image; includes P-1 if template.',
            case_maps='Head-mean raw patch probabilities;24x24 row-major; actual Processor resize/center-crop inverse mapping.',
            display_layers=[0,3,15,31], case_choice='Frozen before GPU inference from original image/question clarity, no Attention or accuracy selection.'),
        statistics=dict(seed=SEED, bootstrap_repetitions=10000, unit='Whole-image SHA256 paired clusters, question-weighted mean',
            ci='Ordinary pointwise95%; not simultaneous across layers/sources/tasks', sd='Across-question sample SD, ddof1; heads/layers not independent samples'),
        historical_csv={name: sha256(ORIGINAL / name) for name in ['first_token_per_question.csv','bos_mass_per_question.csv']}))
    print('Frozen native3000 singleGPU1 protocol', flush=True)


def verify():
    c = read(OUT / 'execution_manifest.json')
    assert c['source'] == sources() and c['previous_binding'] == digest(verify_previous())
    for name, field in [('input_partitions.json','input_partitions_sha256'), ('case_pool.json','case_pool_sha256'),
        ('showcase_selection.json','showcase_sha256'), ('cpu_validation.json','cpu_sha256')]:
        assert sha256(OUT / name) == c[field]
    return c


@torch.inference_mode()
def smoke(model, processor, partitions):
    checks = []
    for task in list(TASKS)[:2]:
        sample = next(s for s in cohort() if s['task'] == task)
        inputs = prepared(model, processor, sample)
        old_inputs, footprint = tensor_hashes(inputs), hook_footprint(model)
        native = model(**inputs, output_attentions=False, use_cache=False, return_dict=True).logits.detach()
        rows, maps, logits = observe(model, inputs, sample, partitions[sample['question_id']], spatial=True)
        with torch.inference_mode():
            output = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
        assert torch.equal(output.logits, native), 'Attention observation altered effective native logits'
        assert torch.equal(logits, native[0,-1].float().cpu())
        direct_rows = [a[0,:,-1].float().cpu().numpy() for a in output.attentions]
        from .engine import reduce_row, groups_for
        groups = groups_for(sample, partitions[sample['question_id']])
        for k, row in enumerate(direct_rows):
            expected, _, visual = reduce_row(row, groups)
            assert all(expected[m] == rows[k][m] for m in expected)
            assert np.array_equal(np.asarray(maps[k]).ravel(), visual)
        assert footprint == hook_footprint(model) and old_inputs == tensor_hashes(inputs)
        checks.append(dict(task=task, question_id=sample['question_id'], query=sample['input_mapping']['p_minus_one'],
            full_logits_exact=True, max_logit_difference=0., all32_reductions_and_maps_exact=True,
            hooks_clean=True, inputs_unchanged=True, first_token=int(logits.argmax())))
    return dict(passed=True, checks=checks)


def worker():
    c = verify()
    binding = digest(c)
    lock = (OUT / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'completion.json').exists():
        assert read(OUT / 'completion.json')['binding'] == binding
        print('Already complete; no GPU allocated', flush=True)
        return
    pre = preflight(0)
    model, processor, environment = load_model(MODEL_PATH)
    assert environment['transformers'] == '4.49.0' and environment['parameter_dtypes'] == ['torch.float16']
    assert environment['model_revision'] == c['model_revision'] and not model.training
    versions = {n: p._version for n,p in model.named_parameters()}
    footprint = hook_footprint(model)
    partitions = {r['question_id']: r['question_positions'] for r in read(OUT / 'input_partitions.json')}
    if not (OUT / 'gpu_smoke.json').exists():
        validation = smoke(model, processor, partitions)
        atomic_json(OUT / 'gpu_smoke.json', dict(binding=binding, passed=True, validation=validation,
            environment=environment, preflight=pre))
    assert read(OUT / 'gpu_smoke.json')['binding'] == binding
    assert read(OUT / 'gpu_smoke.json')['passed']
    pool = {s['question_id'] for s in read(OUT / 'case_pool.json')['samples']}
    baseline_binding = verify_previous()['previous_binding']
    start, resumed = time.perf_counter(), 0
    for i, sample in enumerate(cohort()):
        path = OUT / 'checkpoints' / f"{sample['task']}__{sample['question_id']}.json"
        if path.exists():
            record = read(path)
            assert record['binding'] == binding and record['sample_sha256'] == digest(sample)
            assert record['result_sha256'] == digest(record['result'])
            resumed += 1
            continue
        inputs = prepared(model, processor, sample)
        inputs_sha = tensor_hashes(inputs)
        rows, maps, logits = observe(model, inputs, sample, partitions[sample['question_id']], sample['question_id'] in pool)
        full = baseline(sample, baseline_binding)
        assert int(logits.argmax()) == full['token_ids'][0]
        assert inputs_sha == tensor_hashes(inputs) and hook_footprint(model) == footprint
        assert versions == {n:p._version for n,p in model.named_parameters()}
        result = dict(rows=rows, spatial_maps=maps, full_text=full['text'], full_first_token=int(logits.argmax()),
            first_logits_sha256=digest(logits.tolist()))
        if sample['question_id'] in pool:
            image = Image.open(sample['image_path']).convert('RGB')
            g, cropped = geometry(image, processor.image_processor)
            result['geometry'] = g
            assert sha256(sample['image_path']) == sample['image_sha256']
            folder = OUT / 'case_images'
            folder.mkdir(exist_ok=True)
            image.save(folder / f"{sample['question_id']}.png")
        atomic_json(path, dict(binding=binding, sample_sha256=digest(sample), task=sample['task'],
            question_id=sample['question_id'], image_id=sample['image_id'], image_sha256=sample['image_sha256'],
            result=result, result_sha256=digest(result)))
        if (i+1) % 100 == 0 or i+1 == 3000:
            print(json.dumps(dict(questions=i+1,total=3000,seconds=time.perf_counter()-start)),flush=True)
    verify()
    value = dict(status='complete', binding=binding, questions=3000, rows=96000, resumed=resumed,
        seconds=time.perf_counter()-start, memory=gpu_memory(), hooks_clean=hook_footprint(model)==footprint,
        parameters_unchanged=versions=={n:p._version for n,p in model.named_parameters()})
    atomic_json(OUT / 'completion.json', value)
    print('Complete', value, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','cpu','freeze','worker'])
    args=parser.parse_args()
    try: globals()[args.stage]()
    except BaseException:
        if args.stage=='worker':
            freeze_json(OUT/f'failures/{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
