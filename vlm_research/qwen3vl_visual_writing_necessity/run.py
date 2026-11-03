"""CPU freeze, two strict GPU Smokes, then resumable paired generation."""
import argparse
import ast
from collections import Counter, defaultdict
import fcntl
import gc
import inspect
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import unittest

import torch

from .common import (ROOT, OUT, HISTORY, WRITER240, MODEL, REVISION, CONDITIONS, ORDER,
    SEED, LLAVA, LLAVA_COMMIT, read, seal, key, command, atomic, source_hashes, writer_layers,
    hardware, assert_idle, verify, checkpoint, digest, sha256)
from .engine import LayerWriterMask, generate
from ..qwen3vl_semantic_visual_writers.engine import roles
from ..qwen3vl_early_windows.experiment import prepare, tensor_digest, hook_counts, gqa_prediction, HISTORICAL
from ..qwen3vl_query_role.engine import generate as native_generate, feature_hashes
from ..qwen3vl_query_role.run import environment, check_models


def cpu():
    from . import tests
    from ..qwen3vl_semantic_visual_writers import tests as inherited
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),
        unittest.defaultTestLoader.loadTestsFromModule(inherited)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and not torch.cuda.is_initialized()
    seal(OUT / 'cpu_tests.json', dict(passed=True, tests=result.testsRun,
        sources=source_hashes(), cuda_initialized=False))


def freeze():
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
    import transformers.masking_utils as masks
    import transformers.generation.utils as generation
    assert not (OUT / 'protocol.json').exists(), 'Protocol already frozen'
    assert read(OUT / 'cpu_tests.json')['sources'] == source_hashes()
    assert command('git', 'status', '--porcelain', '--', 'vlm_research/qwen3vl_visual_writing_necessity') == ''
    old = read(HISTORY / 'qwen_input_manifest.json')
    samples = read(HISTORY / 'shared_3000_manifest.json')['samples']
    assert len(samples) == 3000 and len({key(s) for s in samples}) == 3000
    assert len({s['image_sha256'] for s in samples}) == 2409
    assert sorted(Counter(s['task'] for s in samples).values()) == [500]*6
    assert sha256(HISTORY / 'shared_3000_manifest.json') == read(HISTORY / 'protocol.json')['shared_manifest_sha256']
    seal(OUT / 'sample_manifest.json', read(HISTORY / 'shared_3000_manifest.json'))
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    source = {key(s): s for s in old['samples']}
    assert set(source) == {key(s) for s in samples}
    specs = {}
    for index, sample in enumerate(samples):
        assert sha256(sample['image_path']) == sample['image_sha256']
        layout = source[key(sample)]['layout']
        mapping = roles(layout, processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'))
        ids = torch.tensor([layout['input_ids']])
        positions, delta = Qwen3VLModel.get_rope_index(SimpleNamespace(config=config), ids,
            torch.tensor([layout['image_grid_thw']]), attention_mask=torch.ones_like(ids))
        spec = dict(layout=layout, roles=mapping,
            mrope=dict(position_ids_sha256=tensor_digest(positions), delta=int(delta.item())),
            historical_input_binding=source[key(sample)]['input_binding'])
        spec['binding'] = digest(spec)
        specs[key(sample)] = spec
        if (index+1) % 500 == 0:
            print('Bound historical native inputs', index+1, '/3000', flush=True)
    ordered = sorted(samples, key=lambda s: (specs[key(s)]['layout']['input_length'], key(s)))
    smoke_keys = [key(ordered[i]) for i in (0, 1500, 2999)]
    # Real Processor parity is checked before GPU use and again for every formal sample.
    by = {key(s): s for s in samples}
    for identifier in smoke_keys:
        _, layout = prepare(processor, by[identifier])
        assert layout == specs[identifier]['layout'], 'Pinned historical Qwen input changed'
    seal(OUT / 'input_bindings.json', specs)
    model_manifest = read(ROOT / 'vlm_research/results/qwen3vl_query_role_l1_l6_240/model_manifest.json')
    check_models(model_manifest)
    seal(OUT / 'model_manifest.json', model_manifest)
    env = environment()
    assert env['versions']['transformers'] == '4.57.6' and env['versions']['torch'] == '2.6.0+cu124'
    assert env['official_model_source_sha256'] == 'dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da'
    seal(OUT / 'environment.json', env)
    scorer_node = next(n for n in ast.parse(HISTORICAL.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == 'gqa_prediction')
    scope = {}
    exec(compile(ast.Module(body=[scorer_node], type_ignores=[]), str(HISTORICAL), 'exec'), scope)
    for text in ('Dog.', 'yes...', '', 'a cat', 'blue!'):
        assert gqa_prediction(text) == scope['gqa_prediction'](text)
    seal(OUT / 'official_gqa_prediction.json', dict(path=str(HISTORICAL), sha256=sha256(HISTORICAL),
        function_source=ast.get_source_segment(HISTORICAL.read_text(), scorer_node)))
    history_images = {s['image_sha256'] for s in read(WRITER240 / 'sample_manifest.json')['samples']}
    seal(OUT / 'historical_overlap.json', dict(history_manifest=str(WRITER240 / 'sample_manifest.json'),
        history_sha256=sha256(WRITER240 / 'sample_manifest.json'), historical_images=sorted(history_images),
        overlapping_questions=sum(s['image_sha256'] in history_images for s in samples),
        nonoverlapping_questions=sum(s['image_sha256'] not in history_images for s in samples),
        interpretation='Internal image-disjoint subset, not an independent external dataset'))
    clusters = defaultdict(list)
    for s in samples:
        clusters[s['image_sha256']].append(key(s))
    shards, counts = [[], []], [0, 0]
    for image in sorted(clusters, key=lambda im: (-len(clusters[im]), digest([SEED, im]))):
        i = min(range(2), key=lambda j: (counts[j], j))
        shards[i].extend(sorted(clusters[image]))
        counts[i] += len(clusters[image])
    assert set(shards[0]).isdisjoint(shards[1]) and sum(counts) == 3000
    seal(OUT / 'shards.json', dict(keys=shards, questions=counts, image_cluster_exclusive=True))
    snapshot = hardware()
    candidates = []
    for index in (0, 3, 2, 1):
        gpu = next(g for g in snapshot['gpus'] if int(g['index']) == index)
        try:
            assert_idle(gpu['uuid'], snapshot)
            candidates.append(gpu)
        except AssertionError:
            pass
    assert len(candidates) >= 2, 'Two genuinely idle GPUs are required'
    seal(OUT / 'resource_preflight_resumed.json', dict(timestamp=time.time(), snapshot=snapshot, selected=candidates[:2]))
    seal(OUT / 'llava_reference_summary.json', dict(commit=LLAVA_COMMIT,
        rows=list(__import__('csv').DictReader((LLAVA / 'condition_summary.csv').open())),
        sha256=sha256(LLAVA / 'condition_summary.csv')))
    artifacts = ('sample_manifest.json', 'input_bindings.json', 'model_manifest.json', 'environment.json',
        'official_gqa_prediction.json', 'historical_overlap.json', 'shards.json', 'llava_reference_summary.json', 'cpu_tests.json')
    p = dict(title='Qwen3-VL GQA3000 Layer-wise Visual Writing Necessity',
        code_commit=command('git', 'rev-parse', 'HEAD'), sources=source_hashes(),
        model_revision=REVISION, dtype='bfloat16', attention='eager', transformers='4.57.6',
        native_Processor_mRoPE_DeepStack=True, conditions=CONDITIONS, execution_order=ORDER,
        questions=3000, image_clusters=2409, generations=114000, gpus=candidates[:2],
        R_definition='All positions after last actual image token and before original P-1, including vision_end and template specials',
        P1_Decode_image_off_all_36_layers=True, independent_KV_cache=True,
        greedy=True, max_new_tokens=32, logits_to_keep=1, Gold_NLL=False,
        bootstrap_repetitions=20000, bootstrap_seed=SEED, question_weighted_image_cluster_pairs=True,
        ordinary_alpha=.05, Bonferroni_comparisons=36, smoke_keys=smoke_keys,
        historical_native_inputs_sha256=sha256(HISTORY / 'qwen_input_manifest.json'),
        official_sources={str(Path(inspect.getfile(module))): sha256(inspect.getfile(module))
            for module in (modeling, masks, generation)},
        artifacts={name: sha256(OUT / name) for name in artifacts},
        limitations='Single-drop mechanism analysis on historical cohort; not joint deletion safety or measured acceleration')
    seal(OUT / 'protocol.json', p)
    print('FROZEN', digest(p), 'GPU', [g['index'] for g in p['gpus']], 'shards', counts, 'smoke', smoke_keys, flush=True)


def checked_inputs(processor, sample, spec):
    inputs, layout = prepare(processor, sample)
    assert layout == spec['layout'], 'Actual Processor input differs from frozen native Qwen input'
    return inputs.to('cuda:0')


def cache_hash(cache):
    return digest([[tensor_digest(layer.keys), tensor_digest(layer.values)] for layer in cache.layers])


def smoke(model, processor, samples, specs, p, shard):
    by = {key(s): s for s in samples}
    cases = []
    for identifier in p['smoke_keys']:
        sample, spec = by[identifier], specs[identifier]
        inputs = checked_inputs(processor, sample, spec)
        frozen = {k: tensor_digest(v) for k, v in inputs.items()}
        weights = [(n, v._version, v.data_ptr()) for n, v in model.named_parameters()]
        native, native_scores, native_trace, parent = native_generate(model, processor, inputs, spec['roles'], native=True, audit=True, trace=True)
        feature_ref = feature_hashes(native_trace)
        parent_hash = cache_hash(parent)
        checks = []
        for condition in ('All-Writer', 'No-Writer', 'Drop-L0', 'Drop-L17', 'Drop-L35'):
            noop, scores, _, own = generate(model, processor, inputs, spec, condition, noop=True, audit=True)
            assert noop['token_ids'] == native['token_ids'] and len(scores) == len(native_scores)
            assert all(torch.equal(a, b) for a, b in zip(scores, native_scores)), 'Native/No-op valid logits differ'
            assert all(a.keys.data_ptr() != b.keys.data_ptr() and a.values.data_ptr() != b.values.data_ptr()
                for a, b in zip(parent.layers, own.layers))
            del own, scores
        full, full_scores, full_trace, full_cache = generate(model, processor, inputs, spec, 'All-Writer', audit=True, trace=True)
        for condition in ('No-Writer', 'Drop-L0', 'Drop-L17', 'Drop-L35'):
            changed, scores, trace, own = generate(model, processor, inputs, spec, condition, audit=True, trace=True)
            assert trace.validate() and trace.mlp_calls == [len(changed['token_ids'])]*36
            assert feature_hashes(trace) == feature_ref
            assert cache_hash(parent) == parent_hash
            assert all(a.keys.data_ptr() != b.keys.data_ptr() and a.values.data_ptr() != b.values.data_ptr()
                for a, b in zip(full_cache.layers, own.layers))
            if condition.startswith('Drop-'):
                layer = int(condition.removeprefix('Drop-L'))
                assert torch.equal(full_trace.records[layer]['input'], trace.records[layer]['input'])
                keep = [i for i in range(spec['layout']['input_length']) if i not in spec['roles']['R']]
                assert torch.equal(full_trace.records[layer]['attention'][:, keep], trace.records[layer]['attention'][:, keep])
            if condition == 'Drop-L35':
                assert changed['token_ids'] == full['token_ids'] and len(scores) == len(full_scores)
                assert all(torch.equal(a, b) for a, b in zip(scores, full_scores)), 'L35 structural negative control failed'
            checks.append(dict(condition=condition, passed=True, actual_mask_zero=True,
                cache_storage_independent=True, DeepStack_mRoPE_MLP_native=True,
                row_sum_max_error=changed['source_audit']['row_sum_max_error']))
            del own, scores, trace
        counts = hook_counts(model)
        try:
            with LayerWriterMask(model, spec, 'No-Writer'):
                raise RuntimeError('intentional cleanup')
        except RuntimeError as error:
            assert str(error) == 'intentional cleanup'
        assert hook_counts(model) == counts
        assert frozen == {k: tensor_digest(v) for k, v in inputs.items()}
        assert weights == [(n, v._version, v.data_ptr()) for n, v in model.named_parameters()]
        cases.append(dict(key=identifier, passed=True, native_noop_all_valid_logits_torch_equal=True,
            Drop_L35_all_valid_logits_torch_equal=True, non_target_Query_at_drop_layer_exact=True,
            input_weights_unchanged=True, exception_cleanup=True, checks=checks))
        del inputs, native_scores, native_trace, parent, full_scores, full_trace, full_cache
        gc.collect()
        torch.cuda.empty_cache()
        print('SMOKE PASS', identifier, flush=True)
    result = dict(passed=True, shard=shard, binding=digest(p), cases=cases, strict_torch_equal=True)
    seal(OUT / f'shard{shard}/smoke.json', result)
    return result


def load_record(path, p, sample, spec, shard, condition):
    r = read(path)
    assert r['binding'] == digest(p) and r['sample_sha256'] == digest(sample) and r['input_binding'] == spec['binding']
    assert r['condition'] == condition and r['gpu_uuid'] == p['gpus'][shard]['uuid']
    assert r['generation_sha256'] == digest(r['generation'])
    return r


def execute(shard, smoke_only=False):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    p = verify()
    folder = OUT / f'shard{shard}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert environment() == read(OUT / 'environment.json')
    uuid = p['gpus'][shard]['uuid']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == uuid
    gpu = assert_idle(uuid)
    attempt = str(time.time_ns())
    seal(folder / f'preflight_{attempt}.json', dict(gpu=gpu, timestamp=time.time(), binding=digest(p), pid=os.getpid()))
    check_models(read(OUT / 'model_manifest.json'))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager', device_map={'': 'cuda:0'}, use_safetensors=True, low_cpu_mem_usage=True).eval()
    assert len(model.model.language_model.layers) == 36 and next(model.parameters()).dtype == torch.bfloat16
    assert model.config.text_config._attn_implementation == model.config.vision_config._attn_implementation == 'eager'
    samples = read(OUT / 'sample_manifest.json')['samples']
    specs = read(OUT / 'input_bindings.json')
    with torch.inference_mode():
        smoke(model, processor, samples, specs, p, shard)
        if smoke_only:
            return
        started_wait = time.monotonic()
        while not all((OUT / f'shard{i}/smoke.json').exists() for i in range(2)):
            assert not list(OUT.glob('shard*/failure*.json')), 'Peer worker failed; formal matrix not started'
            assert time.monotonic()-started_wait < 900, 'Peer GPU Smoke barrier timed out'
            time.sleep(2)
        assert all(read(OUT / f'shard{i}/smoke.json')['passed'] and read(OUT / f'shard{i}/smoke.json')['binding'] == digest(p) for i in range(2))
        selected = set(read(OUT / 'shards.json')['keys'][shard])
        work = [s for s in samples if key(s) in selected]
        weights = [(n, v._version, v.data_ptr()) for n, v in model.named_parameters()]
        started, done = time.monotonic(), 0
        seal(folder / f'execution_{attempt}.json', dict(started_unix=time.time(), pid=os.getpid(),
            binding=digest(p), gpu_uuid=uuid, code_commit=p['code_commit'], questions=len(work)))
        for index, sample in enumerate(work):
            identifier, spec = key(sample), specs[key(sample)]
            paths = {c: checkpoint(shard, identifier, c) for c in CONDITIONS}
            records = {c: load_record(path, p, sample, spec, shard, c) for c, path in paths.items() if path.exists()}
            if len(records) == 38:
                assert records['Drop-L35']['structural_negative_control_exact']
                done += 38
                continue
            inputs = checked_inputs(processor, sample, spec)
            input_hashes = {k: tensor_digest(v) for k, v in inputs.items()}
            baseline_scores = None
            for condition in ORDER:
                if condition in records and not (condition == 'All-Writer' and 'Drop-L35' not in records):
                    continue
                capture = condition in ('All-Writer', 'Drop-L35')
                g, scores, _, cache = generate(model, processor, inputs, spec, condition, audit=capture)
                assert input_hashes == {k: tensor_digest(v) for k, v in inputs.items()}
                assert weights == [(n, v._version, v.data_ptr()) for n, v in model.named_parameters()]
                hashes = [tensor_digest(v) for v in scores]
                if condition == 'All-Writer':
                    baseline_scores = scores
                full = g if condition == 'All-Writer' else records['All-Writer']['generation']
                g.update(correct=int(g['prediction'] == sample['answer'].lower()),
                    first_token_flip=int(g['first_token_id'] != full['first_token_id']),
                    first_logits_sha256=hashes[0])
                negative_exact = False
                if condition == 'Drop-L35':
                    assert baseline_scores is not None and len(scores) == len(baseline_scores)
                    assert g['token_ids'] == full['token_ids']
                    assert all(torch.equal(a, b) for a, b in zip(scores, baseline_scores)), 'Formal L35 structural control failed'
                    negative_exact = True
                    baseline_scores = None
                row = dict(binding=digest(p), code_commit=p['code_commit'], condition=condition,
                    task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                    image_sha256=sample['image_sha256'], gold=sample['answer'], gpu_uuid=uuid,
                    sample_sha256=digest(sample), input_binding=spec['binding'],
                    generation=g, generation_sha256=digest(g), effective_logits_sha256=hashes,
                    structural_negative_control_exact=negative_exact)
                if condition in records:
                    assert row == records[condition], 'Resumed paired baseline differs'
                else:
                    seal(paths[condition], row)
                    records[condition] = row
                del cache, scores
            assert len(records) == 38
            del inputs, records
            done += 38
            elapsed = time.monotonic()-started
            atomic(folder / 'progress.json', dict(status='RUNNING', binding=digest(p), completed_generations=done,
                total_generations=len(work)*38, completed_questions=index+1, total_questions=len(work),
                elapsed_seconds=elapsed, generated_per_second=done/elapsed, last_key=identifier, timestamp=time.time()))
            if (index+1) % 10 == 0:
                print('PROGRESS', index+1, '/', len(work), 'generations', done, 'seconds', round(elapsed, 2), flush=True)
        seal(folder / 'complete.json', dict(passed=True, binding=digest(p), questions=len(work),
            generations=done, elapsed_seconds=time.monotonic()-started, gpu_uuid=uuid, finished_unix=time.time()))
        print('COMPLETE shard', shard, done, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    seal(folder / f'release_{attempt}.json', dict(own_cuda_memory_allocated=torch.cuda.memory_allocated(),
        timestamp=time.time(), gpu_uuid=uuid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cpu', 'freeze', 'worker', 'smoke'])
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    try:
        if args.stage == 'cpu':
            cpu()
        elif args.stage == 'freeze':
            freeze()
        else:
            execute(args.shard, args.stage == 'smoke')
    except BaseException:
        if args.stage in ('worker', 'smoke'):
            seal(OUT / f'shard{args.shard}/failure_{time.time_ns()}.json', dict(timestamp=time.time(),
                pid=os.getpid(), traceback=traceback.format_exc(), stage=args.stage))
        raise


if __name__ == '__main__':
    main()
