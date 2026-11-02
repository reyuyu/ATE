"""Single-GPU preflight, immutable data/protocol, smoke, and resumable real generation."""
import argparse
from collections import Counter
import fcntl
import gc
import inspect
import io
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch

from benchmarks.smoke import hook_footprint
from first_answer_attention.engine import first_metrics, supplement_generate, CONDITIONS as OLD
from knockout_probe import tensor_hash
from paper_rq1.scorers import gqa_prediction, GQA_EVALUATOR_SHA, GQA_CONVERTER_SHA
from paper_rq1_single_layer_gqa6.run import prepared, tensor_hashes
from rq2_decode_visual.engine import generate as historical_visual_generate
from vlm_common import load_model, gpu_memory
from .common import (ROOT, OUT, RAW, BASE, MODEL, REVISION, SEED, TASKS, LAYERS,
                     CONDITIONS, GPU_UUID, read, write, sources, sha256, digest,
                     preflight, device_binding, verify)
from .engine import generate


def samples():
    return read(OUT / 'sample_manifest.json')['samples']


def git_blob(path):
    rel = str(path.relative_to(ROOT.parent))
    expected = subprocess.check_output(['git', 'rev-parse', f'{BASE}:{rel}'], cwd=ROOT, text=True).strip()
    actual = subprocess.check_output(['git', 'hash-object', str(path)], cwd=ROOT, text=True).strip()
    assert expected == actual, rel
    return expected


def prepare():
    from transformers import AutoProcessor
    import transformers.models.llama.modeling_llama as llama
    import transformers.models.llava.modeling_llava as llava
    import transformers.generation.utils as hf_generation
    OUT.mkdir(parents=True, exist_ok=True)
    experiment_one = Path('/data/lk/ATE-worktrees/query-role-attention-boundary-240/vlm_research/results/query_role_attention_boundary_240/sample_manifest.json')
    fallback = ROOT / 'results/rq2_visual_text_handoff_240/sample_manifest.json'
    source = experiment_one if experiment_one.exists() else fallback
    source_manifest = read(source)
    ss = source_manifest['samples']
    assert len(ss) == len({s['question_id'] for s in ss}) == len({s['image_sha256'] for s in ss}) == 240
    assert Counter(s['task'] for s in ss) == Counter({t:40 for t in TASKS})
    pool_path = ROOT / 'results/paper_rq1_single_layer_gqa6/sample_manifest.json'
    pool = {s['question_id']: s for s in read(pool_path)['samples']}
    assert len(pool) == 3000
    git_blob(pool_path)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    checks = []
    for s in ss:
        old = pool[s['question_id']]
        for key in ['task', 'question_id', 'question', 'answer', 'image_sha256', 'prompt_text', 'input_mapping']:
            assert s[key] == old[key], (s['question_id'], key)
        assert sha256(s['image_path']) == s['image_sha256']
        ids = s['input_mapping']['prompt_ids']
        message = [{'role':'user', 'content':[{'type':'image'}, {'type':'text', 'text':s['prompt_text']}]}]
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        expanded = [v for token in processor.tokenizer(text)['input_ids'] for v in [token]*(576 if token == 32000 else 1)]
        assert expanded == ids and ids.count(32000) == 576 and ids.count(1) == 1
        assert s['input_mapping']['p_minus_one'] == len(ids)-1
        checks.append(dict(question_id=s['question_id'], task=s['task'], image_sha256=s['image_sha256'],
            prompt_ids_sha256=digest(ids), answer=s['answer'], original_pool_bound=True))
    if source == fallback:
        git_blob(source)
    write(OUT / 'sample_manifest.json', dict(samples=ss, count=240, images=240,
        selection='Exact existing frozen manifest, no resampling or model-output filtering',
        source_manifest=str(source), source_manifest_sha256=sha256(source),
        experiment_one_manifest=str(experiment_one), experiment_one_available_at_freeze=source == experiment_one,
        same_samples_as_experiment_one=True if source == experiment_one else None,
        fallback_reason=None if source == experiment_one else 'Experiment1 manifest not generated at selection; use permitted frozen handoff240 immediately, do not wait or later replace cohort.',
        original_pool_sha256=sha256(pool_path), exploratory_historical_source=True))
    write(OUT / 'input_binding_audit.json', dict(passed=True, questions=240, unique_images=240,
        six_tasks_each40=True, processor_prompt_ids_exact=True, checks=checks))
    old = read(ROOT / 'results/rq2_visual_text_handoff_240/protocol.json')
    model_files = old['model_files']
    for p, b in model_files.items():
        assert sha256(p) == b['sha256'], p
        assert Path(p).stat().st_size == b['bytes'] and Path(p).stat().st_mtime_ns == b['mtime_ns']
    # The pinned official conversion and exact equality are the existing scorer.
    from paper_rq1.scorers import OLD as OLD_ROOT, RESOURCES
    evaluator = RESOURCES / 'annotations/gqa_eval.py'
    converter = OLD_ROOT / 'sources/llava/scripts/convert_gqa_for_eval.py'
    assert sha256(evaluator) == GQA_EVALUATOR_SHA and sha256(converter) == GQA_CONVERTER_SHA
    write(OUT / 'preparation.json', dict(base_commit=BASE, model_revision=REVISION, model_files=model_files,
        transformers_sources={inspect.getfile(m):sha256(inspect.getfile(m)) for m in [llama, llava, hf_generation]},
        scorer_sources={str(p):sha256(p) for p in [evaluator, converter]},
        sample_manifest_sha256=sha256(OUT / 'sample_manifest.json'),
        hypotheses=['A-Skip and Text-Only separately low harm', 'Joint may or may not have excess harm'],
        comparisons=['Joint-Full', 'Joint-A-Skip', 'Joint-Text-Only', 'Joint-Text-Only-A-Skip+Full'],
        joint_unique_harm='Full correct and BOTH single interventions correct, Joint wrong',
        statistics=dict(seed=SEED, repetitions=20000, unit='Image SHA256 whole cluster, paired, question weighted',
            confidence='Ordinary two-sided exploratory95% percentile; no NI/equivalence certification',
            reference_boundary_pp=-1),
        conditions={k:dict(attention_skip=a, visual_off=v) for k,(a,v) in CONDITIONS.items()},
        target_layers=LAYERS, image_mask_layers=list(range(32)),
        timing='Original P-1 and every true incremental Decode including EOS-predicting Query; all other Prefill queries native',
        generation=dict(do_sample=False, num_beams=1, use_cache=True, max_new_tokens=32, eos_token_id=2),
        physical_gpu=0, gpu_uuid=GPU_UUID, limit='Exactly960 formal new complete generations; no other research conditions'))
    print('Prepared240 fixed questions /240 distinct image SHA256; experiment1 available:', source == experiment_one, flush=True)


def cpu():
    stream = io.StringIO()
    modules = ['attention_relay_joint.tests', 'test_first_answer_attention',
               'test_rq2_decode_visual', 'test_rq2_source_knockout']
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = dict(passed=result.wasSuccessful(), tests=result.testsRun, source=sources(), output=stream.getvalue())
    write(OUT / 'cpu_validation.json', record)
    print(stream.getvalue(), flush=True)
    assert result.wasSuccessful(), 'CPU tests failed; no GPU run permitted'


def guard(model):
    return hook_footprint(model), {n:p._version for n,p in model.named_parameters()}


def validate_generation(g, tokenizer, sample):
    text = tokenizer.decode(g['token_ids'], skip_special_tokens=True).strip()
    unexpected = sorted(set(g['token_ids']) & (set(tokenizer.all_special_ids)-{2}))
    assert not unexpected, ('Unexpected special token', unexpected)
    words = text.lower().split()
    repeated = len(words) >= 6 and len(set(words)) < len(words)/2
    g.update(text=text, prediction=gqa_prediction(text), score=int(gqa_prediction(text) == sample['answer']),
        empty=not text, repeated_output=repeated, unexpected_special_tokens=unexpected)
    return g


def smoke():
    test = read(OUT / 'cpu_validation.json')
    assert test['passed'] and test['source'] == sources()
    pre = preflight()
    model, processor, environment = load_model(MODEL)
    assert environment['transformers'] == '4.49.0'
    logical = device_binding()
    initial = guard(model)
    checks = []
    # First sample of three frozen task groups, never selected by predictions.
    selected = [next(s for s in samples() if s['task'] == t) for t in TASKS[:3]]
    for s in selected:
        prompt = prepared(model, processor, s)
        before = tensor_hashes(prompt)
        full, ff, ft, fd = generate(model, prompt, 'Full', audit=True, collect=True)
        noop, nf, nt, nd = generate(model, prompt, 'Joint', noop=True, audit=True, collect=True)
        assert full['token_ids'] == noop['token_ids']
        assert torch.equal(fd['prefill_logits'], nd['prefill_logits']) and torch.equal(ff, nf)
        assert len(ft) == len(nt) and all(torch.equal(a, b) for a, b in zip(ft, nt))
        assert fd['initial_cache_sha256'] == nd['initial_cache_sha256']
        with torch.inference_mode():
            hf = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
                use_cache=True, eos_token_id=2, return_dict_in_generate=True, output_scores=True)
        assert hf.sequences[0, prompt['input_ids'].shape[1]:].tolist() == full['token_ids']
        default_differences = [float((a[0].float().cpu()-b).abs().max()) for a,b in zip(hf.scores,ft)]
        # HF4.49 defaults logits_to_keep=1 while historical forwards use0.
        # Compare exact logits only with identical LM-head matrix shapes; keep
        # the default HF token comparison and record its raw numerical difference.
        with torch.inference_mode():
            aligned = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
                use_cache=True, eos_token_id=2, logits_to_keep=0,
                return_dict_in_generate=True, output_scores=True)
        assert aligned.sequences[0, prompt['input_ids'].shape[1]:].tolist() == full['token_ids']
        assert len(aligned.scores) == len(ft)
        assert all(torch.equal(a[0].float().cpu(), b) for a,b in zip(aligned.scores,ft))
        conditions = []
        for condition in ['A-Skip', 'Text-Only', 'Joint']:
            g, first, trace, debug = generate(model, prompt, condition, audit=True, collect=True)
            assert torch.equal(fd['prefill_logits'][:, :-1], debug['prefill_logits'][:, :-1])
            assert fd['other_query_states'] == debug['other_query_states']
            assert fd['other_query_attention'] == debug['other_query_attention']
            if condition == 'A-Skip':
                old, old_first, old_trace = supplement_generate(model, prompt, OLD['E'], audit=True, collect=True)
                assert old['token_ids'] == g['token_ids'] and torch.equal(first, old_first)
                assert len(trace) == len(old_trace) and all(torch.equal(a,b) for a,b in zip(trace,old_trace))
            elif condition == 'Text-Only':
                old, old_first, old_trace, _ = historical_visual_generate(model, prompt, 1,
                    blocked=list(range(32)), audit=True, collect=True)
                assert old['token_ids'] == g['token_ids'] and torch.equal(first, old_first)
                assert len(trace) == len(old_trace) and all(torch.equal(a,b) for a,b in zip(trace,old_trace))
            validate_generation(g, processor.tokenizer, s)
            conditions.append(dict(condition=condition, other_prefill_attention_states_logits_exact=True,
                steps=g['steps'], attention_output_events=g['attention_output_events'],
                simultaneous_hooks_verified=condition == 'Joint', token_ids=g['token_ids']))
        try:
            from unittest.mock import patch
            target = model.language_model.model.layers[3].self_attn.q_proj
            with patch.object(target, 'forward', side_effect=RuntimeError('deliberate smoke cleanup')):
                generate(model, prompt, 'Joint')
        except RuntimeError as e:
            assert str(e) == 'deliberate smoke cleanup'
        else:
            raise AssertionError('Missing deliberate exception')
        restored, rf, rt, _ = generate(model, prompt, 'Full', collect=True)
        assert restored['token_ids'] == full['token_ids'] and torch.equal(ff, rf)
        assert len(rt) == len(ft) and all(torch.equal(a,b) for a,b in zip(rt,ft))
        assert tensor_hashes(prompt) == before and guard(model) == initial
        checks.append(dict(question_id=s['question_id'], task=s['task'], p_minus_one=prompt['input_ids'].shape[1]-1,
            noop_all_valid_logits_tokens_and_initial_cache_exact=True, native_hf_greedy_tokens_logits_exact=True,
            native_default_hf_greedy_tokens_exact=True, aligned_hf_logits_to_keep=0,
            default_hf_logit_max_abs_differences=default_differences,
            historical_single_hook_implementations_exact=True, exception_cleanup_and_native_restore=True,
            input_unchanged=True, conditions=conditions))
        print('GPU smoke passed', s['task'], s['question_id'], flush=True)
    write(OUT / 'smoke.json', dict(passed=True, source=sources(), sample_manifest_sha256=sha256(OUT/'sample_manifest.json'),
        cpu_validation_sha256=sha256(OUT/'cpu_validation.json'), preflight=pre, device=logical,
        environment=environment, process_id=os.getpid(), questions=3, checks=checks,
        hooks_clean=True, parameter_versions_unchanged=True))


def freeze():
    test, sm = read(OUT/'cpu_validation.json'), read(OUT/'smoke.json')
    assert test['passed'] and sm['passed'] and test['source'] == sm['source'] == sources()
    assert sm['sample_manifest_sha256'] == sha256(OUT/'sample_manifest.json')
    paths = ['vlm_research/'+n for n in sources()]
    assert set(subprocess.check_output(['git','ls-files','--',*paths],text=True).splitlines()) == set(paths)
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],text=True).strip()
    preparation = read(OUT/'preparation.json')
    bindings = {n:sha256(OUT/n) for n in ['preparation.json', 'sample_manifest.json',
        'input_binding_audit.json', 'cpu_validation.json', 'smoke.json']}
    write(OUT/'protocol.json', dict(**preparation, source=sources(), bindings=bindings,
        status='frozen_after_CPU_and_real_GPU_smoke_before_formal',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    verify()
    print('Protocol and successful source frozen before960 formal generations', flush=True)


def checkpoint(s, condition, binding):
    path = OUT / f"checkpoints/{s['task']}__{s['question_id']}/{condition}.json"
    if not path.exists():
        return None
    r = read(path)
    assert r['binding'] == binding and r['sample_sha256'] == digest(s) and r['condition'] == condition
    assert digest(r['generation']) == r['generation_sha256']
    assert sha256(r['first_logits_path']) == r['first_logits_sha256']
    return r


def worker():
    p = verify()
    binding = digest(p)
    lock = (OUT/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT/'completion.json').exists():
        assert read(OUT/'completion.json')['binding'] == binding
        print('Already complete; no GPU loaded', flush=True)
        return
    pre = preflight()
    tick = time.perf_counter()
    model, processor, environment = load_model(MODEL)
    logical = device_binding()
    sm = read(OUT/'smoke.json')
    for key in ['python','torch','transformers','cuda_runtime','gpu','model_revision','parameter_dtypes']:
        assert environment[key] == sm['environment'][key], key
    initial = guard(model)
    resumed = fresh = 0
    for i, s in enumerate(samples()):
        prompt = prepared(model, processor, s)
        before = tensor_hashes(prompt)
        for condition in CONDITIONS:
            if checkpoint(s, condition, binding):
                resumed += 1
                continue
            g, first, _, _ = generate(model, prompt, condition)
            validate_generation(g, processor.tokenizer, s)
            assert tensor_hashes(prompt) == before and guard(model) == initial
            directory = RAW / f"{s['task']}__{s['question_id']}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'{condition}.npy'
            # Local raw logits are bound by digest; checkpoints, never raw files,
            # are the completion marker. Incomplete prior raw writes may be replaced.
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('xb') as f:
                np.save(f, first.numpy(), allow_pickle=False)
                f.flush(); os.fsync(f.fileno())
            os.replace(temporary, path)
            full = first if condition == 'Full' else torch.from_numpy(np.load(directory/'Full.npy',allow_pickle=False))
            g.update(first_metrics(full, first))
            record = dict(binding=binding, condition=condition, sample_sha256=digest(s),
                question_id=s['question_id'], task=s['task'], image_sha256=s['image_sha256'], gold=s['answer'],
                prompt_ids_sha256=digest(s['input_mapping']['prompt_ids']), input_sha256=before,
                generation=g, generation_sha256=digest(g), first_logits_path=str(path),
                first_logits_sha256=sha256(path), first_logits_tensor_sha256=tensor_hash(first),
                hooks_clean=True, input_and_parameter_versions_unchanged=True)
            write(OUT/f"checkpoints/{s['task']}__{s['question_id']}/{condition}.json", record)
            fresh += 1
        if i % 10 == 0 or i == 239:
            print(f'Completed {i+1}/240 questions; new={fresh}, resumed={resumed}; seconds={time.perf_counter()-tick:.1f}',flush=True)
    assert fresh + resumed == 960
    assert all(checkpoint(s,c,binding) for s in samples() for c in CONDITIONS)
    write(OUT/'completion.json', dict(status='complete',binding=binding,new_generations=fresh,
        resumed_checkpoints=resumed,formal_generations=960,questions=240,conditions=4,
        process_id=os.getpid(),preflight=pre,device=logical,environment=environment,
        seconds=time.perf_counter()-tick,memory=gpu_memory(),hooks_clean=True,weights_unchanged=True))
    del prompt, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print('Complete960; model released; no other experiments scheduled', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','cpu','smoke','freeze','worker','analyze','verify'])
    args = parser.parse_args()
    try:
        if args.stage in ('analyze','verify'):
            from .analyze import analyze, audit
            (analyze if args.stage == 'analyze' else audit)()
        else:
            globals()[args.stage]()
    except BaseException as exc:
        write(OUT/f'failures/{args.stage}_{time.time_ns()}.json',dict(stage=args.stage,
            error=repr(exc), traceback=traceback.format_exc(), source=sources()))
        raise


if __name__ == '__main__':
    main()
