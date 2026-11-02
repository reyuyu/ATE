import argparse
import fcntl
import inspect
import io
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import unittest

import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoProcessor
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

from benchmarks.dataset_registry import freeze_json
from benchmarks.smoke import hook_footprint
from first_answer_attention.engine import supplement_generate, first_metrics, CONDITIONS as OLD_CONDITIONS
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared, tensor_hashes
from vlm_common import load_model, gpu_memory
from .common import (ROOT, OUT, RAW, BASE, REVISION, MODEL, UUID, SEED, TASKS, CONDITIONS,
                     COMPARISONS, LAYERS, POOL, read, write, sources, verify, preflight, selected,
                     sha256, digest)
from .engine import roles, generate


def prepare():
    assert not (OUT/'sample_manifest.json').exists(), 'Never silently refreeze a cohort'
    pool = read(POOL)['samples']
    cohort = selected(pool)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    processor.patch_size = 14
    processor.num_additional_image_tokens = 1
    processor.vision_feature_select_strategy = 'default'
    processor.tokenizer.padding_side = 'left'
    mapped = []
    for s in cohort:
        assert sha256(s['image_path']) == s['image_sha256']
        with Image.open(s['image_path']) as original:
            image = original.convert('RGB')
        chat = processor.apply_chat_template([dict(role='user', content=[dict(type='image'),
            dict(type='text', text=s['prompt_text'])])], tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=chat, return_tensors='pt')
        assert inputs['input_ids'][0].tolist() == s['input_mapping']['prompt_ids']
        assert digest(chat) == s['input_mapping']['chat_prompt_sha256']
        layout = roles(inputs)
        assert layout['A'] == [s['input_mapping']['p_minus_one']]
        assert layout['I'] == s['input_mapping']['visual_positions']
        cpu_half = {k:v.half() if v.is_floating_point() else v for k,v in inputs.items()}
        mapped.append(dict(question_id=s['question_id'], task=s['task'], role_layout=layout,
                           image_size=list(image.size), input_hashes=tensor_hashes(cpu_half)))
    ordered = sorted(range(240), key=lambda i:(mapped[i]['role_layout']['prompt_length'],
                                             cohort[i]['question_id']))
    smoke = [ordered[0], ordered[-1], ordered[len(ordered)//2]]
    assert len(set(smoke)) == 3
    freeze_json(OUT/'sample_manifest.json', dict(samples=cohort, count=240,
        unique_image_sha256=len({s['image_sha256'] for s in cohort}), task_counts={t:40 for t in TASKS},
        seed=SEED, pool_path=str(POOL), pool_sha256=sha256(POOL),
        pool_commit='221cdde56dde53908aaa953fe26672fd33334f84',
        selection='Per-task ascending SHA256(seed, query-role-input-v1, task, question_id, image_sha256, prompt_ids); prefer globally unique image SHA; no outputs used',
        smoke_indices=smoke, smoke_selection='Shortest, longest, median prompt; before any GPU output',
        exploratory=True, independent_confirmation=False))
    freeze_json(OUT/'input_mappings.json', mapped)
    print('Frozen 240 unique images, 40/task; smoke', [cohort[i]['question_id'] for i in smoke], flush=True)


def cpu():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('vlm_research.query_role_attention.tests')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write(OUT/'cpu_validation.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        sources=sources(), output=stream.getvalue(), cuda_used=False))
    print(stream.getvalue(), flush=True)
    assert result.wasSuccessful()


def freeze():
    assert not (OUT/'protocol.json').exists()
    acceptance = read(OUT/'cpu_validation.json')
    assert acceptance['passed'] and acceptance['sources'] == sources()
    assert read(OUT/'sample_manifest.json')['unique_image_sha256'] == 240
    paths = ['vlm_research/'+p for p in sources()]
    tracked = subprocess.check_output(['git', 'ls-files', '--', *paths], cwd=ROOT.parent, text=True).splitlines()
    assert set(tracked) == set(paths)
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *paths], cwd=ROOT.parent, text=True).strip()
    old = read(ROOT/'results/paper_rq1_single_layer_gqa6/execution_manifest.json')['model_files']
    files = sorted([*MODEL.glob('*.safetensors'), *MODEL.glob('*.json'), MODEL/'tokenizer.model', MODEL/'chat_template.jinja'])
    weights = {}
    for path in files:
        h = sha256(path)
        if str(path) in old:
            assert h == old[str(path)]['sha256']
        st = path.stat()
        weights[str(path)] = dict(sha256=h, bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    scorer = read(ROOT/'results/paper_rq1_single_layer_gqa6/scorer_validation.json')['source_sha256']
    assert all(sha256(path) == h for path,h in scorer.items())
    from transformers.cache_utils import DynamicCache
    from transformers.generation.utils import GenerationMixin
    tfiles = {str(Path(inspect.getfile(cls))):sha256(inspect.getfile(cls)) for cls in
              [LlamaAttention, LlamaDecoderLayer, LlavaForConditionalGeneration, DynamicCache, GenerationMixin]}
    write(OUT/'protocol.json', dict(base_commit=BASE,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip(),
        model_revision=REVISION, model_path=str(MODEL), dtype='float16', attention='eager',
        transformers='4.49.0', conditions=CONDITIONS, layers=LAYERS,
        intervention='Only selected query rows of self_attn.o_proj OUTPUT are exactly zero; no source mask, no physical skipping',
        roles=dict(I='All 576 actual prefill visual positions',
            R='All nonvisual positions after last image and strictly before original P-1',
            A='Original P-1 and every actual incremental Decode query, including EOS prediction'),
        generation=dict(greedy=True, max_new_tokens=32, use_cache=True, fresh_cache=True, own_prefix=True),
        gpu=dict(physical_index=1, uuid=UUID, process_device='cuda:0', one_model_replica=True),
        bootstrap=dict(seed=SEED, repetitions=20000, cluster='Image SHA256', question_weighted=True,
                       intervals='ordinary paired exploratory 95%', comparisons=COMPARISONS),
        no_noninferiority_or_acceleration_claim=True, sources=sources(),
        bindings={n:sha256(OUT/n) for n in ['sample_manifest.json','input_mappings.json','cpu_validation.json']},
        model_files=weights, scorer_sources=scorer, transformers_sources=tfiles,
        scoring="Pinned official GQA exact equality; LLaVA conversion text.rstrip('.').lower(); no new parser"))
    print('Protocol and model SHA256 frozen', flush=True)


def native(model, prompt):
    trace = []
    def observe(module, args, output):
        assert torch.isfinite(output.logits).all()
        trace.append(output.logits[0].detach().float().cpu().clone())
    handle = model.register_forward_hook(observe)
    try:
        with torch.inference_mode():
            ids = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32, use_cache=True)
        return ids[0, prompt['input_ids'].shape[1]:].tolist(), trace
    finally:
        handle.remove()


def exact(a, b):
    assert len(a) == len(b) and all(torch.equal(x,y) for x,y in zip(a,b)), 'Effective logits differ'


def smoke(model, processor, cohort, mapping):
    rows = []
    for i in read(OUT/'sample_manifest.json')['smoke_indices']:
        s = cohort[i]; inputs = prepared(model, processor, s)
        before = tensor_hashes(inputs); footprint = hook_footprint(model)
        assert before == mapping[i]['input_hashes']
        versions = {n:p._version for n,p in model.named_parameters()}
        reference, native_trace = native(model, inputs)
        full, first, full_trace, parent = generate(model, processor.tokenizer, inputs, 'Full', capture=True, retain_cache=True)
        assert full['token_ids'] == reference
        # HF generate sets logits_to_keep=1; only its prediction query is returned.
        exact([x[-1] for x in native_trace], [x[-1] for x in full_trace])
        replay, replay_first, replay_trace, _ = generate(model, processor.tokenizer, inputs, 'Full', capture=True)
        assert replay['token_ids'] == reference and torch.equal(first, replay_first)
        exact(full_trace, replay_trace)
        parent_hash = cache_fingerprint(parent)
        role_audit = {}
        for condition in CONDITIONS[1:]:
            noop, _, noop_trace, _ = generate(model, processor.tokenizer, inputs, condition, noop=True, capture=True)
            assert noop['token_ids'] == reference
            exact(full_trace, noop_trace)
            counts = [0]*32
            handles = []
            for k, layer in enumerate(model_layers(model)):
                def mlp(module, args, output, k=k):
                    assert torch.isfinite(output).all()
                    counts[k] += 1
                handles.append(layer.mlp.register_forward_hook(mlp))
            try:
                changed, current, changed_trace, owned = generate(model, processor.tokenizer, inputs,
                    condition, capture=True, retain_cache=True)
            finally:
                for h in handles: h.remove()
            assert counts == [changed['length']]*32
            assert cache_fingerprint(parent) == parent_hash
            assert all(x.data_ptr() != y.data_ptr() for a,b in zip(parent.to_legacy_cache(), owned.to_legacy_cache())
                       for x,y in zip(a,b))
            if condition == 'A-Bypass':
                old, old_first, old_trace = supplement_generate(model, inputs, OLD_CONDITIONS['E'], collect=True)
                assert old['token_ids'] == changed['token_ids'] and torch.equal(old_first, current)
                exact([x[-1] for x in changed_trace], old_trace)
            role_audit[condition] = dict(noop_all_valid_logits_exact=True, noop_tokens_exact=True,
                target_rows_exact_zero=True, direct_nontarget_rows_exact=True,
                independent_cache_nonalias=True, original_cache_hash_unchanged=True,
                mlp_calls_per_layer=counts, cache_lengths=changed['cache_lengths'], events=changed['hook_events'],
                historical_E_exact=condition == 'A-Bypass')
            del owned, changed_trace
            assert hook_footprint(model) == footprint
        assert versions == {n:p._version for n,p in model.named_parameters()}
        assert tensor_hashes(inputs) == before and hook_footprint(model) == footprint
        rows.append(dict(question_id=s['question_id'], task=s['task'], image_size=mapping[i]['image_size'],
            layout=mapping[i]['role_layout'], full_generation=full,
            native_effective_prediction_logits_exact=True, full_replay_all_valid_logits_exact=True,
            hf_logit_shapes=[list(x.shape) for x in native_trace],
            full_logit_shapes=[list(x.shape) for x in full_trace],
            full_and_native_greedy_tokens_exact=True, inputs_unchanged=True, parameters_unchanged=True,
            hooks_clean=True, roles=role_audit))
        print('GPU SMOKE PASS', s['question_id'], flush=True)
        del parent, full_trace, native_trace, replay_trace
    return rows


def save_raw(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as old:
            assert np.array_equal(old['first_logits'], value.numpy())
    else:
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        with tmp.open('xb') as f:
            np.savez_compressed(f, first_logits=value.numpy())
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def gpu():
    started = time.perf_counter()
    p = verify(); binding = digest(p)
    lock = (OUT/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check = preflight()
    assert transformers.__version__ == '4.49.0'
    model, processor, environment = load_model(MODEL)
    assert environment['parameter_dtypes'] == ['torch.float16'] and torch.cuda.device_count() == 1
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    assert any(line.split(',')[0].strip() == UUID and int(line.split(',')[1]) == os.getpid()
               for line in apps.splitlines())
    environment.update(physical_gpu=1, physical_uuid=UUID, logical_device='cuda:0', pid=os.getpid(),
        python_executable=os.sys.executable, python=platform.python_version(), preflight=check,
        dependencies=subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines())
    write(OUT/f'environment_{os.getpid()}.json', environment)
    cohort = read(OUT/'sample_manifest.json')['samples']
    mapping = read(OUT/'input_mappings.json')
    versions = {n:v._version for n,v in model.named_parameters()}
    footprint = hook_footprint(model)
    smoke_path = OUT/'gpu_smoke.json'
    if smoke_path.exists():
        assert read(smoke_path)['binding'] == binding and read(smoke_path)['passed']
        # A resumed process must re-pass smoke before accepting its own new outputs.
        rows = smoke(model, processor, cohort, mapping)
        write(OUT/f'gpu_resume_smoke_{os.getpid()}.json', dict(passed=True, binding=binding, rows=rows))
    else:
        rows = smoke(model, processor, cohort, mapping)
        write(smoke_path, dict(passed=True, binding=binding, rows=rows))
    assert hook_footprint(model) == footprint
    resumed, computed = 0, 0
    for i, s in enumerate(cohort):
        prompt = None
        for condition in CONDITIONS:
            checkpoint = OUT/f"checkpoints/{s['task']}__{s['question_id']}/{condition}.json"
            if checkpoint.exists():
                row = read(checkpoint)
                assert row['binding'] == binding and row['sample_sha256'] == digest(s)
                assert row['result_sha256'] == digest(row['generation'])
                assert row['condition'] == condition and sha256(row['raw']['path']) == row['raw']['sha256']
                resumed += 1
                if condition == 'Full':
                    with np.load(row['raw']['path']) as data:
                        full_logits = torch.from_numpy(data['first_logits'].copy())
                    full = row['generation']
                continue
            if prompt is None:
                prompt = prepared(model, processor, s)
                assert tensor_hashes(prompt) == mapping[i]['input_hashes']
                assert roles(prompt) == mapping[i]['role_layout']
            tick = time.perf_counter()
            generation, first, _, _ = generate(model, processor.tokenizer, prompt, condition)
            if condition == 'Full':
                full_logits = first
                full = generation
            generation.update(first_metrics(full_logits, first), correct=int(generation['prediction'] == s['answer']),
                full_correct_to_wrong=int(full['prediction'] == s['answer'] and generation['prediction'] != s['answer']),
                full_wrong_to_correct=int(full['prediction'] != s['answer'] and generation['prediction'] == s['answer']),
                sequence_changed=generation['token_ids'] != full['token_ids'],
                full_first_logits_sha256=digest(full_logits.tolist()), first_logits_sha256=digest(first.tolist()))
            raw = save_raw(RAW/f"{s['task']}__{s['question_id']}/{condition}.npz", first)
            assert hook_footprint(model) == footprint and tensor_hashes(prompt) == mapping[i]['input_hashes']
            assert versions == {n:v._version for n,v in model.named_parameters()}
            write(checkpoint, dict(binding=binding, code_commit=p['code_commit'], sample_sha256=digest(s),
                condition=condition, question_id=s['question_id'], task=s['task'], image_sha256=s['image_sha256'],
                generation=generation, result_sha256=digest(generation), raw=raw, seconds=time.perf_counter()-tick))
            computed += 1
        if (i+1) % 10 == 0:
            verify()
            print('PROGRESS', i+1, '/240', 'computed', computed, 'resumed', resumed,
                  'seconds', round(time.perf_counter()-started, 1), flush=True)
    verify()
    write(OUT/'gpu_completion.json', dict(passed=True, binding=binding, questions=240, conditions=4,
        checkpoints=960, computed=computed, resumed=resumed, physical_gpu=1, uuid=UUID,
        seconds_including_load_smoke=time.perf_counter()-started, memory=gpu_memory(),
        hooks_clean=hook_footprint(model) == footprint,
        parameters_unchanged=versions == {n:v._version for n,v in model.named_parameters()}))
    print('COMPLETE 240 x 4; GPU process exits now', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'cpu', 'freeze', 'gpu'])
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except BaseException:
        write(OUT/f'failures/{args.stage}_{time.time_ns()}.json',
              dict(stage=args.stage, traceback=traceback.format_exc(), partial=True))
        raise
