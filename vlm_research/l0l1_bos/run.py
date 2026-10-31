"""One GPU2 worker; existing cached generation engine and scoring, no Full rerun."""
import argparse
from contextlib import ExitStack
import fcntl
import gc
import io
import json
from pathlib import Path
import time
import traceback
import unittest

import numpy as np
from PIL import Image
import torch

from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from decode_five_vs_six import ENV_KEYS
from first_answer_attention.engine import supplement_generate
from knockout_probe import model_layers
from paper_rq1.scorers import gqa_prediction
from paper_rq1_early_joint_attention.engine import JointAttention, generate
from paper_rq1_single_layer_gqa6.run import prepared, tensor_hashes, atomic_json
from vlm_common import load_model, gpu_memory
from .capture import prefill_capture
from .protocol import (ROOT, OUT, CONDITION, old, read, identity, prior, sources,
    prepare, freeze, verify, execution, preflight)


def first_metrics_from_history(reference, changed):
    values = changed.float()
    assert values.ndim == 1 and bool(torch.isfinite(values).all())
    token = reference['full_first_token']
    logp = float(values.log_softmax(-1)[token])
    predicted = int(values.argmax())
    return dict(full_first_token=token, first_token=predicted, first_top1_flip=predicted != token,
        full_first_token_logp=reference['full_first_token_logp'], intervened_full_token_logp=logp,
        first_logp_loss=reference['full_first_token_logp'] - logp,
        full_first_logits_sha256=reference['full_first_logits_sha256'],
        first_logits_sha256=digest(values.tolist()))


def cpu():
    stream = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ['test_early_joint_attention', 'test_first_answer_attention', 'test_l0l1_bos'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    atomic_json(OUT / 'cpu_validation.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        source_sha256=sources(), output=stream.getvalue()))
    print(stream.getvalue(), flush=True)
    assert result.wasSuccessful()


def keys_for(model, processor, prompt, sample):
    ids = prompt['input_ids'][0].tolist()
    image = [i for i, token in enumerate(ids) if token == model.config.image_token_index]
    bos = [i for i, token in enumerate(ids) if token == processor.tokenizer.bos_token_id]
    assert len(image) == 576 and bos == sample['input_mapping']['bos_positions'] == [0]
    assert sample['input_mapping']['p_minus_one'] == len(ids) - 1
    others = sorted(set(range(len(ids))) - set(image) - set(bos))
    assert others and len(image) + len(bos) + len(others) == len(ids)
    return dict(bos=bos, image=image, prompt=others)


def capture_pair(model, processor, sample, previous, generation=None, save=False, label='preselected'):
    prompt = prepared(model, processor, sample)
    before = tensor_hashes(prompt)
    keys = keys_for(model, processor, prompt, sample)
    full_logits, full, full_stats = prefill_capture(model, prompt, keys)
    off_logits, off, off_stats = prefill_capture(model, prompt, keys, off=True)
    reference = previous['L0L5_OFF']
    assert digest(full_logits.tolist()) == reference['full_first_logits_sha256']
    assert int(full_logits.argmax()) == previous['Full']['token_ids'][0] != 2
    assert float(full_logits.log_softmax(-1)[reference['full_first_token']]) == reference['full_first_token_logp']
    if generation is not None:
        assert digest(off_logits.tolist()) == generation['first_logits_sha256']
        assert int(off_logits.argmax()) == generation['token_ids'][0]
    assert np.array_equal(full['attention_row_0'], off['attention_row_0']), 'L0 attention occurs before output bypass'
    assert all(off_stats[k]['attention_output_norm'] == 0. for k in (0, 1))
    assert all(np.array_equal(full[f'bos_{name}_{layer}'], off[f'bos_{name}_{layer}'])
        for name in ('key', 'value') for layer in range(2, 6)), 'Causal BOS K/V invariant violated'
    assert tensor_hashes(prompt) == before
    record = dict(task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
        question=sample['question'], answer=sample['answer'], image_path=sample['image_path'],
        sample_sha256=digest(sample), label=label, query=sample['input_mapping']['p_minus_one'], keys=keys,
        capture_step='Original cached Prefill P-1 predicts FIRST answer token, never EOS Decode step',
        full_first_token=int(full_logits.argmax()), off_first_token=int(off_logits.argmax()),
        full_first_logits_sha256=digest(full_logits.tolist()), off_first_logits_sha256=digest(off_logits.tolist()),
        native_and_saved_logits_exact=True, l0_weights_exact=True, effective_l0_l1_update_zero=True,
        bos_kv_l2_l5_exact=True, full_stats=full_stats, off_stats=off_stats,
        full_answer=previous['Full']['text'], full_score=previous['Full']['score'],
        off_answer=generation['text'] if generation else None, off_score=generation['score'] if generation else None)
    if save:
        folder = OUT / 'cases' / identity(sample)
        folder.mkdir(parents=True, exist_ok=True)
        artifact = folder / 'p_minus_one.npz'
        assert not artifact.exists()
        np.savez_compressed(artifact, **{f'full_{k}': v for k, v in full.items()}, **{f'off_{k}': v for k, v in off.items()})
        pixels = prompt['pixel_values'][0].float().cpu().numpy().transpose(1, 2, 0)
        image_processor = processor.image_processor
        pixels = pixels * np.asarray(image_processor.image_std) + np.asarray(image_processor.image_mean)
        Image.fromarray((np.clip(pixels, 0, 1)*255).round().astype(np.uint8)).save(folder / 'clip_input.png')
        record.update(npz_sha256=sha256(artifact), clip_png_sha256=sha256(folder / 'clip_input.png'))
        atomic_json(folder / 'metadata.json', record)
    return record


def smoke(model, processor):
    selected = [next(s for s in old.cohort() if s['task'] == task) for task in (old.TASKS[0], old.TASKS[-1])]
    footprint = hook_footprint(model)
    records = []
    for sample in selected:
        previous = prior(sample)
        prompt = prepared(model, processor, sample)
        before = tensor_hashes(prompt)
        n = prompt['input_ids'].shape[1]
        full, full_logits = generate(model, prompt, dict(layers=[], mode='Full'), audit=True, collect=True)
        assert full['token_ids'] == previous['Full']['token_ids']
        assert digest(full_logits[0].tolist()) == previous['L0L5_OFF']['full_first_logits_sha256']
        noop, first, noop_logits = supplement_generate(model, prompt, CONDITION, noop=True, audit=True, collect=True)
        assert noop['token_ids'] == full['token_ids'] and torch.equal(first, full_logits[0])
        assert len(noop_logits) == len(full_logits) and all(torch.equal(a, b) for a, b in zip(noop_logits, full_logits))
        assert noop['initial_cache_sha256'] == full['initial_cache_sha256']
        untouched, handles, calls = {}, [], [0]*32
        try:
            for k, block in enumerate(model_layers(model)):
                def capture(module, args, output, k=k):
                    untouched[k] = output[0][:, :-1].detach().clone()
                handles.append(block.register_forward_hook(capture))
            old.reference(model, prompt)
        finally:
            for handle in handles:
                handle.remove()
        handles = []
        try:
            for k, block in enumerate(model_layers(model)):
                def causal(module, args, output, k=k):
                    if output[0].shape[1] == n:
                        assert torch.equal(output[0][:, :-1], untouched[k])
                def mlp(module, args, output, k=k):
                    calls[k] += 1
                    assert torch.isfinite(output).all()
                handles += [block.register_forward_hook(causal), block.mlp.register_forward_hook(mlp)]
            changed, first, changed_logits = supplement_generate(model, prompt, CONDITION, audit=True, collect=True)
        finally:
            for handle in handles:
                handle.remove()
        assert all(count == changed['length'] for count in calls)
        assert torch.equal(first, changed_logits[0]) and int(first.argmax()) == changed['token_ids'][0]
        assert all(e['target'] == 'original_P-1' for e in changed['events'] if e['active'])
        assert all(e['active'] == e['decode'] for e in changed['decode_events'])
        with ExitStack() as stack:
            stack.enter_context(JointAttention(model, [0, 1], 'P-1-only', n))
            stack.enter_context(JointAttention(model, [0, 1], 'Decode-only', n))
            with torch.inference_mode():
                native = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
                    use_cache=True, eos_token_id=2)[0, n:].tolist()
        assert native == changed['token_ids']
        check = capture_pair(model, processor, sample, previous)
        assert check['off_first_logits_sha256'] == digest(first.tolist())
        assert torch.equal(old.reference(model, prompt), full_logits[0])
        assert tensor_hashes(prompt) == before and hook_footprint(model) == footprint
        records.append(dict(task=sample['task'], question_id=sample['question_id'], full_tokens=full['token_ids'],
            off_tokens=changed['token_ids'], noop_all_effective_logits_and_cache_exact=True,
            all32_other_prefill_queries_exact=True, all32_mlp_normal=True, hf_native_off_tokens_exact=True,
            capture_logits_exact=True, hook_cleanup=True, native_restoration_exact=True, bos_kv_exact=True))
    return dict(passed=True, records=records)


def worker():
    binding = execution()
    manifest = verify()
    folder = OUT
    lock = (folder / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pre = preflight()
    model, processor, environment = load_model(old.MODEL_PATH)
    expected = read(old.OUT / 'shard1/smoke.json')['environment']
    assert all(environment[k] == expected[k] for k in ENV_KEYS if k != 'cuda_visible_devices')
    assert environment['transformers'] == '4.49.0' and environment['parameter_dtypes'] == ['torch.float16']
    footprint = hook_footprint(model)
    versions = {n: p._version for n, p in model.named_parameters()}
    code = digest(binding)
    try:
        if (OUT / 'smoke.json').exists():
            saved = read(OUT / 'smoke.json')
            assert saved['binding'] == code and saved['validation']['passed']
        else:
            validation = smoke(model, processor)
            atomic_json(OUT / 'smoke.json', dict(binding=code, environment=environment, preflight=pre, validation=validation))
        print('GPU2 Smoke passed; only L0L1_OFF starts', flush=True)
        references = {(r['task'], r['question_id']): r for r in read(OUT / 'reuse_bindings.json')}
        samples = old.cohort()
        started = time.perf_counter()
        resumed = 0
        for index, sample in enumerate(samples):
            previous = prior(sample, references[sample['task'], sample['question_id']])
            checkpoint = OUT / 'checkpoints' / f'{identity(sample)}.json'
            if checkpoint.exists():
                saved = read(checkpoint)
                assert saved['binding'] == code and saved['sample_sha256'] == digest(sample)
                assert saved['result_sha256'] == digest(saved['generation'])
                resumed += 1
            else:
                prompt = prepared(model, processor, sample)
                before = tensor_hashes(prompt)
                tick = time.perf_counter()
                generation, first, _ = supplement_generate(model, prompt, CONDITION)
                generation.update(first_metrics_from_history(previous['L0L5_OFF'], first))
                text = processor.tokenizer.decode(generation['token_ids'], skip_special_tokens=True).strip()
                generation.update(text=text, prediction=gqa_prediction(text), score=int(gqa_prediction(text) == sample['answer']))
                assert generation['first_token'] == generation['token_ids'][0]
                assert generation['edits'] == 2 * generation['length']
                assert hook_footprint(model) == footprint and tensor_hashes(prompt) == before
                atomic_json(checkpoint, dict(binding=code, sample_sha256=digest(sample), condition='L0L1_OFF',
                    generation=generation, result_sha256=digest(generation), seconds=time.perf_counter()-tick))
            if (index+1) % 100 == 0 or index+1 == len(samples):
                print(json.dumps(dict(questions=index+1, total=len(samples), seconds=time.perf_counter()-started,
                    memory=gpu_memory())), flush=True)
        atomic_json(OUT / 'accuracy_completion.json', dict(status='complete', binding=code, questions=3000,
            new_generations=3000, resumed=resumed, seconds=time.perf_counter()-started))
        chosen = {(r['task'], r['question_id']) for r in manifest['cases']}
        illustrated, tasks, images = [], set(), set()
        for sample in sorted(samples, key=lambda s: (s['task'], s['question_id'])):
            previous = prior(sample)
            g = read(OUT / 'checkpoints' / f'{identity(sample)}.json')['generation']
            if previous['Full']['score'] == 1 and g['score'] == 0 and sample['task'] not in tasks and sample['image_sha256'] not in images:
                illustrated.append(dict(task=sample['task'], question_id=sample['question_id']))
                tasks.add(sample['task'])
                images.add(sample['image_sha256'])
                if len(illustrated) == 3:
                    break
        atomic_json(OUT / 'posthoc_cases.json', dict(selection=manifest['illustrative'], cases=illustrated))
        extra = {(r['task'], r['question_id']) for r in illustrated}
        for sample in samples:
            key = sample['task'], sample['question_id']
            if key not in chosen | extra:
                continue
            path = OUT / 'cases' / identity(sample) / 'metadata.json'
            if path.exists():
                saved = read(path)
                assert saved['sample_sha256'] == digest(sample)
                assert saved['npz_sha256'] == sha256(path.parent / 'p_minus_one.npz')
            else:
                previous = prior(sample)
                g = read(OUT / 'checkpoints' / f'{identity(sample)}.json')['generation']
                label = 'preselected' if key in chosen else 'posthoc_illustrative_degradation'
                capture_pair(model, processor, sample, previous, g, save=True, label=label)
            assert hook_footprint(model) == footprint
            print('Captured P-1', identity(sample), 'preselected' if key in chosen else 'posthoc', flush=True)
        assert versions == {n: p._version for n, p in model.named_parameters()}
        execution()
        atomic_json(OUT / 'completion.json', dict(status='complete', binding=code, accuracy_questions=3000,
            preselected_cases=12, total_captures=len(chosen | extra), posthoc_cases=len(extra),
            hooks_clean=hook_footprint(model) == footprint, parameters_unchanged=True, memory=gpu_memory()))
        print('GPU2 complete:3000 new generations and targeted mechanism captures', flush=True)
    finally:
        assert hook_footprint(model) == footprint
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'cpu', 'freeze', 'worker'))
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except BaseException:
        if args.stage == 'worker':
            atomic_json(OUT / f'failures/{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
