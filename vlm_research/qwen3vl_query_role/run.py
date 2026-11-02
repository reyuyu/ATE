"""Freeze, strict Smoke, then one GPU/one model/1200 serial generations."""
import argparse
import ast
from collections import Counter
import fcntl
import gc
import importlib.metadata
import inspect
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import unittest

import numpy as np
import torch

from .common import (ROOT, OUT, RAW, SOURCE, SOURCE_SHA, LLAVA_IA, MODEL_MANIFEST,
    MODEL, REVISION, CONDITIONS, LAYERS, GPU_UUID, SEED, COMPARISONS, LLAVA_FORMAL, read, seal,
    sources, roles, key, checkpoint, verify, sha256, digest)
from .engine import RoleBypass, generate, feature_hashes
from ..qwen3vl_early_windows.experiment import (prepare, tensor_digest, hook_counts,
    gqa_prediction, generate as historical_generate)


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def environment():
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
    return dict(python=platform.python_version(), executable=sys.executable,
        versions={p: importlib.metadata.version(p) for p in
          ('torch', 'transformers', 'accelerate', 'huggingface-hub', 'tokenizers', 'Pillow', 'numpy')},
        official_model_source=str(inspect.getfile(modeling)),
        official_model_source_sha256=sha256(inspect.getfile(modeling)), torch_cuda=torch.version.cuda)


def smi():
    return command('nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
                   '--format=csv,noheader,nounits')


def check_models(manifest):
    assert manifest['revision'] == REVISION
    for f in manifest['files']:
        path = MODEL / f['path']
        assert path.stat().st_size == f['size'] and sha256(path) == f['sha256'], str(path)


def freeze():
    from transformers import AutoProcessor
    assert not (OUT / 'protocol.json').exists(), 'Already frozen'
    assert sha256(SOURCE) == SOURCE_SHA
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, OUT / 'sample_manifest.json')
    shutil.copyfile(LLAVA_IA, OUT / 'llava_reference.csv')
    shutil.copyfile(LLAVA_FORMAL, OUT / 'llava_formal_reference.csv')
    samples = read(SOURCE)['samples']
    assert len(samples) == 240 and len({s['image_sha256'] for s in samples}) == 240
    assert sorted(Counter(s['task'] for s in samples).values()) == [40]*6
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    vision_end = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    prepared = []
    for i, s in enumerate(samples):
        _, layout = prepare(processor, s)
        mapping = roles(layout, vision_end)
        prepared.append(dict(task=s['task'], question_id=s['question_id'], image_sha256=s['image_sha256'],
            layout=layout, roles=mapping, binding=digest([layout, mapping])))
        if (i+1) % 40 == 0:
            print(f'Prepared {i+1}/240 unchanged images', flush=True)
    seal(OUT / 'qwen_input_manifest.json', dict(vision_end_id=vision_end, samples=prepared))
    model_manifest = read(MODEL_MANIFEST)
    check_models(model_manifest)
    seal(OUT / 'model_manifest.json', model_manifest)
    env = environment()
    assert env['versions']['transformers'] == '4.57.6'
    assert env['versions']['torch'] == '2.6.0+cu124'
    assert env['official_model_source_sha256'] == 'dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da'
    seal(OUT / 'environment.json', env)
    analysis_env = command('/data/lk/深度学习/.envs/ate-vlm-research/bin/python', '-c',
        'import json,sys,platform,importlib.metadata as m; print(json.dumps(dict(executable=sys.executable,'
        'python=platform.python_version(),versions={p:m.version(p) for p in ("torch","numpy","matplotlib")})))')
    import json
    seal(OUT / 'analysis_environment.json', json.loads(analysis_env))
    scorer = Path('/data/lk/ATE-worktrees/first-answer-early-attention/vlm_research/paper_rq1/scorers.py')
    node = next(n for n in ast.parse(scorer.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'gqa_prediction')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(scorer), 'exec'), scope)
    for text in ('Dog.', 'red!', 'A cat', 'yes...', ''):
        assert gqa_prediction(text) == scope['gqa_prediction'](text)
    ordered = sorted(prepared, key=lambda s: (s['layout']['input_length'], s['question_id']))
    smoke_keys = [key(ordered[0]), key(ordered[len(ordered)//2]), key(ordered[-1])]
    protocol = dict(title='Qwen3-VL Query-Role L1-L6 / frozen 240',
        base_commit='4272f244752f4abc15e75d57a14c37119bb2b840',
        llava_role_commit='8d12f8c05f2b246d029ec4d9ffe45d1f857fa947',
        llava_ia_commit='8019c6009bb1d8c9f2dc3ffd8c18bf4dfb8c1fd1',
        model_revision=REVISION, model_path=str(MODEL), dtype='bfloat16', attention='eager',
        processor='Unmodified pinned native processor, chat template, default image resolution; historical Qwen prompt',
        prompt_suffix='\nAnswer using a single word or short phrase.',
        sample_count=240, unique_images=240, conditions=CONDITIONS, target_layers=LAYERS,
        expected_generations=1200, max_new_tokens=32, do_sample=False, num_beams=1,
        use_cache=True, logits_to_keep=1, gpu_uuid=GPU_UUID, physical_gpu=3,
        role_definition=dict(I='Actual processor image-pad token positions, variable length',
            R='Strictly after vision_end special token and strictly before original P-1, including post-image native template text',
            A='Original prompt P-1 plus each actual incremental decode query including EOS prediction'),
        intervention='Only selected language self_attn.o_proj output rows zero, after projection; no compute skipping, no mask edits',
        preserved=['DeepStack', 'mRoPE', 'MLP', 'Residual', 'KV updates', 'native precision', 'native resolution'],
        smoke_keys=smoke_keys, smoke_selection='Input-length shortest, median, longest; no prediction selection',
        no_op='torch.equal for all returned valid per-step native logits with identical logits_to_keep=1; no tolerance',
        scorer=dict(path=str(scorer), sha256=sha256(scorer), rule='decoded.strip().rstrip(".").lower() == gold.lower()'),
        bootstrap=dict(seed=SEED, repetitions=20000, cluster='image_sha256', paired=True,
            comparisons=COMPARISONS, exploratory=True, correction='Also Bonferroni across six contrasts and IA interaction (7)'),
        sources=sources(), artifacts={name: sha256(OUT / name) for name in
          ('sample_manifest.json', 'qwen_input_manifest.json', 'model_manifest.json', 'environment.json', 'analysis_environment.json',
           'llava_reference.csv', 'llava_formal_reference.csv')})
    seal(OUT / 'protocol.json', protocol)
    print('Frozen protocol', sha256(OUT / 'protocol.json'), 'Smoke', smoke_keys, flush=True)


def cpu():
    from . import tests
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    seal(OUT / 'cpu_tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), sources=sources()))
    assert result.wasSuccessful()


def checked_inputs(processor, sample, prepared):
    inputs, layout = prepare(processor, sample)
    assert layout == prepared['layout'] and digest([layout, prepared['roles']]) == prepared['binding']
    return inputs.to('cuda:0')


def smoke(model, processor, samples, prepared):
    audit = dict(passed=False, protocol_sha256=sha256(OUT/'protocol.json'), cases=[],
                 equality='torch.equal, all returned per-step logits, same native LM-head shape')
    for identifier in read(OUT/'protocol.json')['smoke_keys']:
        sample = next(s for s in samples if key(s) == identifier)
        spec = prepared[identifier]
        inputs = checked_inputs(processor, sample, spec)
        mapping = spec['roles']
        frozen_inputs = {k: tensor_digest(v) for k, v in inputs.items()}
        weights = [(n, p._version, p.data_ptr()) for n, p in model.named_parameters()]
        native, native_scores, _, native_cache = generate(model, processor, inputs, mapping, native=True, audit=True)
        full, scores, observer, full_cache = generate(model, processor, inputs, mapping, trace=True, audit=True)
        assert native['token_ids'] == full['token_ids']
        assert all(torch.equal(a, b) for a, b in zip(native_scores, scores))
        feature_ref = feature_hashes(observer)
        del observer
        repeat, repeated_scores, _, repeat_cache = generate(model, processor, inputs, mapping, audit=True)
        assert repeat['token_ids'] == full['token_ids']
        assert len(repeated_scores) == len(scores) and all(torch.equal(a, b) for a, b in zip(repeated_scores, scores))
        caches = [native_cache, full_cache, repeat_cache]
        checks = []
        for condition in CONDITIONS[1:]:
            noop, noop_scores, _, cache = generate(model, processor, inputs, mapping, condition, noop=True, audit=True)
            caches.append(cache)
            assert noop['token_ids'] == full['token_ids'] and len(noop_scores) == len(scores)
            assert all(torch.equal(a, b) for a, b in zip(noop_scores, scores))
            changed, _, trace, cache = generate(model, processor, inputs, mapping, condition, trace=True, audit=True)
            caches.append(cache)
            assert feature_hashes(trace) == feature_ref
            checks.append(dict(condition=condition, noop_all_logits_equal=True,
                rows_and_layers_exact=True, residual_deepstack_mlp_exact=True,
                actual_prefill_queries=changed['target_prefill_queries'],
                decode_queries=changed['target_decode_queries'], token_ids=changed['token_ids']))
            del trace
        assert len({id(c) for c in caches}) == len(caches)
        # Cache storage is condition-local, not just a distinct Python wrapper.
        ptrs = [c.layers[0].keys.data_ptr() for c in caches]
        assert len(set(ptrs)) == len(ptrs)
        # A-only semantics must exactly match the already validated historical hook.
        old, old_first, _ = historical_generate(model, processor, inputs, LAYERS)
        changed, new_first, _, _ = generate(model, processor, inputs, mapping, 'A-Bypass')
        assert old['token_ids'] == changed['token_ids'] and torch.equal(old_first, new_first[0])
        before = hook_counts(model)
        try:
            with RoleBypass(model, 'I+A-Bypass', mapping, len(spec['layout']['input_ids'])):
                raise RuntimeError('deliberate cleanup check')
        except RuntimeError as exc:
            assert str(exc) == 'deliberate cleanup check'
        assert hook_counts(model) == before
        assert [(n, p._version, p.data_ptr()) for n, p in model.named_parameters()] == weights
        assert {k: tensor_digest(v) for k, v in inputs.items()} == frozen_inputs
        audit['cases'].append(dict(key=identifier, input_binding=spec['binding'],
            native_full_repeat_exact=True, independent_cache_objects_and_storage=True,
            historical_A_exact=True, inputs_weights_unchanged=True,
            exception_hook_cleanup=True, checks=checks))
        print('SMOKE PASS', identifier, flush=True)
        del caches, native_cache, full_cache, repeat_cache, cache, inputs
        gc.collect()
        torch.cuda.empty_cache()
    audit['passed'] = True
    seal(OUT/'validation.json', audit)


def execute():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    protocol = verify()
    assert read(OUT/'cpu_tests.json')['passed']
    assert read(OUT/'cpu_tests.json')['sources'] == sources()
    assert environment() == read(OUT/'environment.json')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == GPU_UUID
    assert command('git', '-C', str(ROOT), 'status', '--porcelain', '--', 'vlm_research/qwen3vl_query_role') == ''
    code_commit = command('git', '-C', str(ROOT), 'rev-parse', 'HEAD')
    compute = command('nvidia-smi', '-i', GPU_UUID, '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits')
    assert not compute, 'Selected GPU is occupied; do not wait/poll or interfere: ' + compute
    seal(OUT/('gpu_preload_'+str(time.time_ns())+'.json'), dict(gpus=smi(), selected_uuid=GPU_UUID, visible_devices=GPU_UUID,
        pid=os.getpid(), code_commit=code_commit, protocol_sha256=sha256(OUT/'protocol.json')))
    check_models(read(OUT/'model_manifest.json'))
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager', device_map={'':'cuda:0'},
        use_safetensors=True, low_cpu_mem_usage=True).eval()
    assert torch.cuda.device_count() == 1
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert {str(p.device) for p in model.parameters()} == {'cuda:0'}
    assert len(model.model.language_model.layers) == 36
    assert model.config.text_config._attn_implementation == model.config.vision_config._attn_implementation == 'eager'
    samples = read(OUT/'sample_manifest.json')['samples']
    prepared = {key(s): s for s in read(OUT/'qwen_input_manifest.json')['samples']}
    weights = [(n, p._version, p.data_ptr()) for n, p in model.named_parameters()]
    before_hooks = hook_counts(model)
    started = time.perf_counter()
    smoke(model, processor, samples, prepared)
    assert read(OUT/'validation.json')['passed']
    formal_start = time.perf_counter()
    count = 0
    for sample in samples:
        spec = prepared[key(sample)]
        inputs = checked_inputs(processor, sample, spec)
        frozen_inputs = {k: tensor_digest(v) for k, v in inputs.items()}
        for condition in CONDITIONS:
            path = checkpoint(sample, condition)
            if path.exists():
                old = read(path)
                assert old['input_binding'] == spec['binding'] and old['protocol_sha256'] == sha256(OUT/'protocol.json')
                assert sha256(old['raw_first_logits_path']) == old['raw_first_logits_sha256']
                count += 1
                continue
            result, first, _, _ = generate(model, processor, inputs, spec['roles'], condition)
            assert {k: tensor_digest(v) for k, v in inputs.items()} == frozen_inputs
            result.update(question_id=sample['question_id'], task=sample['task'], image_sha256=sample['image_sha256'],
                gold=sample['answer'], correct=int(result['prediction'] == sample['answer'].lower()),
                condition=condition, input_binding=spec['binding'], code_commit=code_commit,
                protocol_sha256=sha256(OUT/'protocol.json'))
            raw_path = RAW / key(sample) / (condition + '.npy')
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            assert not raw_path.exists(), 'Uncommitted raw output exists; retain for diagnosis'
            np.save(raw_path, first[0].numpy(), allow_pickle=False)
            result.update(raw_first_logits_path=str(raw_path), raw_first_logits_sha256=sha256(raw_path))
            seal(path, result)
            count += 1
        del inputs
        if count % 60 == 0:
            print(f'FORMAL {count}/1200 elapsed={time.perf_counter()-formal_start:.1f}s '
                  f'allocated={torch.cuda.memory_allocated()/2**20:.0f}MiB', flush=True)
    assert count == 1200
    assert [(n, p._version, p.data_ptr()) for n, p in model.named_parameters()] == weights
    assert hook_counts(model) == before_hooks
    check_models(read(OUT/'model_manifest.json'))
    verify()
    seal(OUT/'generation_completion.json', dict(status='GENERATIONS_COMPLETE', count=count,
        code_commit=code_commit, formal_seconds=time.perf_counter()-formal_start,
        total_seconds=time.perf_counter()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        weights_inputs_unchanged=True, hooks_clean=True, model_content_sha256_verified=True))
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print('All 1200 generations saved; process will exit and release GPU', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'cpu', 'execute'])
    action = parser.parse_args().action
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        with (OUT/'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            {'freeze': freeze, 'cpu': cpu, 'execute': execute}[action]()
    except BaseException:
        seal(OUT / ('failure_' + str(time.time_ns()) + '.json'), dict(action=action,
             error=traceback.format_exc(), sources=sources(), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
