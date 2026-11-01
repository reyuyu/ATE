import argparse
from collections import Counter
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch

from ..qwen3vl_early_windows.experiment import MODEL, REVISION, digest, prepare, sha256, write
from .engine import ImageOff, Prefill, metrics, tree_hash

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/qwen3vl_image_source_240'
SOURCE = Path('/data/lk/ATE-worktrees/rq2-single-layer-source-knockout/vlm_research/results/rq2_single_layer_source_240/sample_manifest.json')
GPUS = (0, 2, 3)


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT / f'vlm_research/{p}' for p in
        ('qwen3vl_early_windows/experiment.py', 'qwen3vl_smoke/common.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def environment():
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
    return dict(versions={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers', 'Pillow')},
                official_modeling_sha256=sha256(inspect.getfile(modeling)), executable=os.sys.executable)


def freeze():
    from transformers import AutoProcessor
    assert not (OUT / 'protocol.json').exists()
    raw = read(SOURCE)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    samples = []
    for row in raw['samples']:
        sample = {k: row[k] for k in ('task', 'question_id', 'image_id', 'image_path', 'image_sha256', 'question', 'answer')}
        _, layout = prepare(processor, sample)
        sample.update(layout=layout, input_binding=digest(layout))
        samples.append(sample)
    assert len(samples) == 240 and set(Counter(s['task'] for s in samples).values()) == {40}
    assert len({(s['task'], s['question_id']) for s in samples}) == 240
    # Equal 80-question shards; group by exact image SHA, never split an image.
    groups = {}
    for sample in samples:
        groups.setdefault(sample['image_sha256'], []).append(sample)
    bins = [[] for _ in GPUS]
    for image, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        index = min(range(3), key=lambda i: (len(bins[i]), i))
        bins[index].extend(rows)
    assert [len(b) for b in bins] == [80, 80, 80]
    for index, rows in enumerate(bins):
        for sample in rows:
            sample['shard'], sample['physical_gpu'] = index, GPUS[index]
    samples.sort(key=lambda s: (s['task'], s['question_id']))
    manifest = dict(source=str(SOURCE), source_sha256=sha256(SOURCE),
        selection='Exact original LLaVA RQ2.1 240-question cohort; no selection using Qwen output.',
        llava_question_image_overlap=240, samples=samples, shard_counts=[80, 80, 80])
    write(OUT / 'sample_manifest.json', manifest)
    write(OUT / 'environment.json', environment())
    model_manifest = read(ROOT / 'vlm_research/results/qwen3vl_early_attention_windows_240/model_manifest.json')
    for file in model_manifest['files']:
        assert sha256(MODEL / file['path']) == file['sha256']
    write(OUT / 'model_manifest.json', model_manifest)
    write(OUT / 'protocol.json', dict(model_repo='Qwen/Qwen3-VL-8B-Instruct', model_revision=REVISION,
        dtype='bfloat16', attention='eager', transformers='4.57.6', layer_count=36,
        conditions='Native Full then single-layer L0-L35 P-1 -> all actual Image Keys -inf additive mask',
        generation=False, use_cache=False, frozen_visual_replay='Per-sample native final image features AND all three DeepStack feature arrays; exact replay smoke required',
        gpu_ids=GPUS, cohort_count=240, shard_counts=[80, 80, 80],
        metrics=['mean signed logP loss', 'mean per-question absolute logP change', 'top1 flip rate'],
        target='Same sample native Full first top1 token; full-vocabulary FP32 log_softmax; not accuracy',
        code_sha256=sources(), manifest_sha256=sha256(OUT / 'sample_manifest.json'),
        environment_sha256=sha256(OUT / 'environment.json'), model_manifest_sha256=sha256(OUT / 'model_manifest.json')))
    print(json.dumps(dict(prepared=240, image_tokens=[min(s['layout']['image_tokens'] for s in samples),
         max(s['layout']['image_tokens'] for s in samples)], shard_counts=[80, 80, 80])), flush=True)


def verify():
    protocol = read(OUT / 'protocol.json')
    assert protocol['code_sha256'] == sources()
    assert protocol['manifest_sha256'] == sha256(OUT / 'sample_manifest.json')
    assert protocol['environment_sha256'] == sha256(OUT / 'environment.json')
    assert environment() == read(OUT / 'environment.json')
    return protocol


def load(gpu):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    assert gpu in GPUS and os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    assert torch.cuda.device_count() == 1
    external = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,used_gpu_memory',
                                       '--format=csv,noheader,nounits'], text=True).strip()
    assert not external, f'GPU{gpu} occupied: {external}'
    write(OUT / f'shard_{GPUS.index(gpu)}/preflight.json', dict(pid=os.getpid(), physical_gpu=gpu,
        inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                           '--format=csv,noheader'], text=True)))
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager', device_map={'': 'cuda:0'}, use_safetensors=True).eval()
    assert len(model.model.language_model.layers) == 36
    assert model.config.text_config._attn_implementation == model.config.vision_config._attn_implementation == 'eager'
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert {str(p.device) for p in model.parameters()} == {'cuda:0'}
    return model, processor


def checked_inputs(processor, sample):
    inputs, layout = prepare(processor, sample)
    assert layout == sample['layout'] and digest(layout) == sample['input_binding']
    return inputs.to('cuda:0')


def smoke(model, processor, sample):
    inputs = checked_inputs(processor, sample)
    runner = Prefill(model, inputs, sample['layout'])
    weights = [(p._version, p.data_ptr()) for p in model.parameters()]
    plain, _, full_trace = runner.forward(native=True, all_logits=True, trace=True)
    replay, _, _ = runner.forward(all_logits=True)
    noop, _, _ = runner.forward(layer=0, noop=True, all_logits=True)
    assert torch.equal(plain, replay) and torch.equal(plain, noop)
    repeated, _, _ = runner.forward(native=True)
    assert torch.equal(plain[-1], repeated)
    checks = []
    for layer in (0, 2, 17, 35):
        logits, audit, trace = runner.forward(layer=layer, trace=True, all_logits=True, native=True)
        assert torch.equal(plain[:-1], logits[:-1]), 'Other prefill query logits changed'
        for i in range(36):
            for name in ('attention', 'output'):
                assert torch.equal(full_trace.records[i][name][:, :-1], trace.records[i][name][:, :-1])
            if i < layer:
                assert all(torch.equal(full_trace.records[i][name], trace.records[i][name]) for name in full_trace.records[i])
            if i < 3:
                assert torch.equal(full_trace.features[i], trace.features[i])
        checks.append(dict(layer=layer, audit=audit, other_prefill_queries_exact=True,
                           earlier_layers_exact=True, deepstack_native_exact=True))
        del trace, logits
    try:
        with ImageOff(model, 0, sample['layout']):
            raise RuntimeError('cleanup fixture')
    except RuntimeError as exc:
        assert str(exc) == 'cleanup fixture'
    runner.verify()
    assert weights == [(p._version, p.data_ptr()) for p in model.parameters()]
    return dict(passed=True, question_id=sample['question_id'], image_tokens=sample['layout']['image_tokens'],
        grid=sample['layout']['image_grid_thw'], noop_all_valid_logits_max_abs=0,
        native_visual_replay_all_valid_logits_max_abs=0, checks=checks,
        visual_feature_hashes=tree_hash(runner.visual), deepstack_count=len(runner.visual[1]),
        inputs_and_weights_unchanged=True, hooks_clean=True, no_nan_inf_oom=True)


def worker(shard):
    verify()
    gpu = GPUS[shard]
    samples = [s for s in read(OUT / 'sample_manifest.json')['samples'] if s['shard'] == shard]
    assert len(samples) == 80
    folder = OUT / f'shard_{shard}'
    model, processor = load(gpu)
    weights = [(p._version, p.data_ptr()) for p in model.parameters()]
    smoke_started = time.perf_counter()
    validation = smoke(model, processor, max(samples, key=lambda s: s['layout']['input_length']))
    validation.update(physical_gpu=gpu, seconds=time.perf_counter()-smoke_started,
        protocol_sha256=sha256(OUT / 'protocol.json'), code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    write(folder / 'validation.json', validation)
    print(f'GPU{gpu} smoke passed in {validation["seconds"]:.1f}s', flush=True)
    # All workers must pass real smoke before any worker enters formal collection.
    while not all((OUT / f'shard_{s}/validation.json').exists() for s in range(3)):
        if any((OUT / f'shard_{s}/error.json').exists() for s in range(3)):
            raise RuntimeError('Peer failed smoke')
        time.sleep(1)
    for s in range(3):
        peer = read(OUT / f'shard_{s}/validation.json')
        assert peer['passed'] and peer['protocol_sha256'] == validation['protocol_sha256']
        assert peer['code_commit'] == validation['code_commit']
    started = time.perf_counter()
    completed, skipped = 0, 0
    for index, sample in enumerate(samples):
        key = f'{sample["task"]}__{sample["question_id"]}'
        path = folder / 'checkpoints' / f'{key}.json'
        if path.exists():
            record = read(path)
            assert record['protocol_sha256'] == validation['protocol_sha256'] and record['input_binding'] == sample['input_binding']
            assert len(record['rows']) == 36
            skipped += 1
            continue
        runner = Prefill(model, checked_inputs(processor, sample), sample['layout'])
        full, _, _ = runner.forward(native=True)
        rows = []
        for layer in range(36):
            changed, audit, _ = runner.forward(layer=layer)
            row = dict(task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], physical_gpu=gpu, layer=layer,
                image_tokens=sample['layout']['image_tokens'], p_minus_1=sample['layout']['p_minus_1'],
                **metrics(full, changed), **audit)
            rows.append(row)
        runner.verify()
        write(path, dict(protocol_sha256=validation['protocol_sha256'], input_binding=sample['input_binding'],
            code_commit=validation['code_commit'], full_logits_sha256=__import__('hashlib').sha256(full.view(torch.uint8).numpy().tobytes()).hexdigest(),
            full_token_text=processor.tokenizer.decode([int(full.argmax())]), rows=rows))
        del runner
        completed += 1
        write(folder / 'progress.json', dict(done=index+1, total=80, seconds=time.perf_counter()-started,
            completed_new=completed, skipped=skipped, physical_gpu=gpu))
        print(f'GPU{gpu} {index+1}/80 {sample["task"]} {sample["question_id"]} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    assert weights == [(p._version, p.data_ptr()) for p in model.parameters()]
    write(folder / 'complete.json', dict(passed=True, count=80, completed_new=completed, skipped=skipped,
        seconds=time.perf_counter()-started, smoke_seconds=validation['seconds'],
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        protocol_sha256=validation['protocol_sha256'], code_commit=validation['code_commit']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'worker', 'aggregate'])
    parser.add_argument('--shard', type=int, choices=range(3))
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'aggregate':
        from .analyze import aggregate
        aggregate()
    else:
        try:
            worker(args.shard)
        except BaseException:
            write(OUT / f'shard_{args.shard}/error.json', dict(traceback=traceback.format_exc()))
            raise
