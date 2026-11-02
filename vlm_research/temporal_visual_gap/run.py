import argparse
import fcntl
import io
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest
import uuid

import numpy as np
import torch

from benchmarks.dataset_registry import digest, sha256
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6.run import tensor_hashes
from paper_rq1_token_role_decode.labels import align
from vlm_common import load_model, prepare_inputs, gpu_memory
from .common import (ROOT, OUT, RAW, HROOT, HCOCO, UUID, SEED, TASKS, CONDITIONS,
                     read, write, sources, verify, preflight, history, gate)
from .engine import forward, native_counterfactual, temporal_generate


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    original = read(HROOT/'results/paper_rq1_single_layer_gqa6/sample_manifest.json')['samples']
    assert len(original) == 3000 and all(sum(s['task'] == t for s in original) == 500 for t in TASKS)
    selected, used = [], set()
    for task in TASKS:
        ordered = sorted([s for s in original if s['task'] == task],
            key=lambda s: digest([SEED, 'gqa-coco-gap-120', s['question_id'], s['image_sha256'], s['input_mapping']['prompt_ids']]))
        chosen = []
        for s in ordered:
            if s['image_sha256'] in used: continue
            chosen.append(s); used.add(s['image_sha256'])
            if len(chosen) == 20: break
        assert len(chosen) == 20
        selected.extend(chosen)
    coco = read(HCOCO/'sample_manifest.json')['samples']
    assert len(coco) == len({s['image_sha256'] for s in coco}) == 60
    bindings = {}
    for s in coco+selected:
        assert sha256(s['image_path']) == s['image_sha256']
        assert s['input_mapping']['prompt_ids'].count(32000) == 576
    for s in coco:
        bindings[s['question_id']] = {c: history(s, c)[1] for c in ['Full', 'None']}
    write(OUT/'sample_manifest.json', dict(COCO=coco, GQA=selected, seed=SEED,
        selection='Per task fixed canonical SHA256 of seed/tag/QID/imageSHA/promptIDs; greedily prefer unused image hashes across all six tasks. No outputs read.',
        original_gqa_manifest_sha256=sha256(HROOT/'results/paper_rq1_single_layer_gqa6/sample_manifest.json'),
        original_coco_manifest_sha256=sha256(HCOCO/'sample_manifest.json')))
    write(OUT/'history_bindings.json', bindings)
    write(OUT/'frozen_gates.json', {s['image_sha256']: {c: [gate(c, s['image_sha256'], t) for t in range(1, 96)]
        for c in CONDITIONS} for s in coco})
    torch.set_num_threads(1)
    from . import tests
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write(OUT/'cpu_validation.json', dict(passed=result.wasSuccessful(), tests=result.testsRun, output=stream.getvalue()))
    assert result.wasSuccessful(), stream.getvalue()
    hp = read(HCOCO/'protocol.json')
    for n, b in hp['model_files'].items(): assert sha256(n) == b['sha256']
    write(OUT/'protocol.json', dict(version=1, branch='codex/gqa-coco-temporal-visual-gap',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources=sources(), model_revision=hp['model_revision'], model_files=hp['model_files'],
        transformers_source=hp['transformers_source'], scorer_source=hp['scorer_source'],
        bindings={n:sha256(OUT/n) for n in ['sample_manifest.json', 'history_bindings.json', 'frozen_gates.json', 'cpu_validation.json']},
        gpu=dict(physical=1, uuid=UUID), dtype='float16', attention='eager', transformers='4.49.0',
        generation=dict(COCO_max_new_tokens=96, GQA_max_new_tokens=32, do_sample=False, use_cache=True, eos=2),
        A='Native trajectory. Each Full/OFF step deep-clones the identical unmodified pre-step native cache; discard OFF cache. t0 P-1 only, t>=1 real incremental query. All32 original Image keys pre-softmax -inf. Native cache never passed to forward directly.',
        B=dict(conditions=CONDITIONS, prefill='Entirely native including P-1',
               periodic='t1 ON; Periodic25 (t-1)%4==0; Periodic50 (t-1)%2==0',
               random='canonical dataset_registry.digest([temporal-image-gate-v1,imageSHA,t,seed]) uint256 < 2^256/4 or /2. No tokens/results used.',
               generations=360, historical_reuse=['Full', 'None']),
        large_loss_nats=.5, concentration='Positive-part LogP loss at non-EOS t>=1; top1 and ceil(20% positions); undefined if zero denominator. Report signed distribution separately.',
        labels='Reuse frozen labels.align; map Ambiguous/Special/Unaligned to Unknown, preserve all subwords; word-primary first subword.',
        statistics=dict(seed=SEED, bootstrap=20000, unit='Image SHA256 clusters; ordinary paired percentile 95% exploratory intervals',
                        role_model='Image fixed-effect OLS: role dummies + generation position + relative length + native logP + entropy; image-cluster bootstrap, descriptive only'),
        limits='Historical exploratory cohorts. Direct Image access only; indirect visual state remains. Hard Mask not physical deletion/speed. Single-step losses are not cumulative CIDEr loss. No extra conditions/models.'))
    print('Frozen: COCO60 + GQA120; 5 CPU tests passed; 360 new temporal generations.', flush=True)


def inputs_for(model, processor, sample):
    assert sha256(sample['image_path']) == sample['image_sha256']
    inputs, count = prepare_inputs(model, processor, sample['image_path'], sample['prompt_text'])
    assert count['visual_tokens'] == 576
    assert inputs['input_ids'][0].tolist() == sample['input_mapping']['prompt_ids']
    return inputs


def guard(model, inputs, foot, versions, hashes):
    assert foot == hook_footprint(model) and hashes == tensor_hashes(inputs)
    assert versions == {n:t._version for n,t in model.named_parameters()}


def smoke(model, processor, samples):
    audits = []
    for sample in samples:
        inputs = inputs_for(model, processor, sample)
        foot, before = hook_footprint(model), tensor_hashes(inputs)
        versions = {n:t._version for n,t in model.named_parameters()}
        coco = sample.get('dataset') == 'coco_caption'
        limit = 96 if coco else 32
        base = temporal_generate(model, processor.tokenizer, inputs, sample['image_sha256'], 'Full', limit, audit=True)
        noop = temporal_generate(model, processor.tokenizer, inputs, sample['image_sha256'], 'None', limit, audit=True, noop=True)
        assert base[0]['token_ids'] == noop[0]['token_ids']
        assert all(torch.equal(a,b) for a,b in zip(base[3], noop[3])) and len(base[3]) == len(noop[3])
        with torch.inference_mode():
            official = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=limit, use_cache=True)
        assert official[0, inputs['input_ids'].shape[1]:].tolist() == base[0]['token_ids']
        g, rows, fl, ol, audit = native_counterfactual(model, processor.tokenizer, inputs, limit, audit=True)
        assert g['token_ids'] == base[0]['token_ids']
        assert np.array_equal(fl, np.stack([x.numpy() for x in base[3]]))
        checks = {}
        for c in CONDITIONS if coco else ['None']:
            result = temporal_generate(model, processor.tokenizer, inputs, sample['image_sha256'], c, limit=4, audit=True)
            assert torch.equal(base[4]['full_logits'], result[4]['full_logits'])
            assert result[0]['token_ids'][0] == base[0]['token_ids'][0]
            checks[c] = dict(prefill_all_logits_exact=True, first_token_exact=True, gates=result[2])
        if coco:
            for c in ['Full', 'None']:
                hg, binding = history(sample, c)
                with np.load(binding['raw']) as data: hf = torch.from_numpy(data['first_logits'].copy())
                current = base if c == 'Full' else temporal_generate(model, processor.tokenizer, inputs, sample['image_sha256'], c, limit)
                assert torch.equal(current[1], hf), 'Historical first-logit exact check failed; stop'
                assert current[0]['token_ids'] == hg['token_ids'], 'Historical tokens differ; stop'
        guard(model, inputs, foot, versions, before)
        audits.append(dict(question_id=sample['question_id'], native_noop_all_logits_tokens_exact=True,
            official_HF_tokens_exact=True, native_prefix_replay_all_logits_exact=True,
            parent_cache_SHA256_unchanged=True, non_P1_prefill_states_logits_attention_exact=True,
            all32_actual_attention_checks=audit, conditions=checks, historical_reuse_exact=coco))
    write(OUT/'gpu_smoke.json', dict(passed=True, samples=audits, hooks_clean=True))


def checkpoint(path, p, sample, stage, condition):
    r = read(path)
    assert r['protocol'] == digest(p) and r['sample'] == digest(sample)
    assert r['stage'] == stage and r['condition'] == condition
    if r.get('raw_file'): assert sha256(r['raw_file']) == r['raw_sha256']
    return r


def save_raw(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


@torch.no_grad()
def worker():
    p = verify(); OUT.mkdir(exist_ok=True)
    lock = (OUT/'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT/'completion.json').exists():
        assert read(OUT/'completion.json')['protocol'] == digest(p)
        print('Already complete', flush=True); return
    write(OUT/f'preflight_{time.time_ns()}.json', preflight())
    started = time.time()
    model, processor, env = load_model(str(Path(next(iter(p['model_files']))).parent))
    assert env['model_revision'] == p['model_revision'] and env['transformers'] == '4.49.0'
    assert env['parameter_dtypes'] == ['torch.float16']
    assert uuid.UUID(str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')) == uuid.UUID(UUID.removeprefix('GPU-'))
    write(OUT/f'environment_{time.time_ns()}.json', env)
    manifest = read(OUT/'sample_manifest.json')
    if not (OUT/'gpu_smoke.json').exists():
        smoke(model, processor, [manifest['COCO'][0], manifest['GQA'][0], next(s for s in manifest['GQA'] if s['task'] == 'CompareAttr')])
    assert read(OUT/'gpu_smoke.json')['passed']
    print('GPU1 smoke passed; frozen formal experiment starting.', flush=True)
    versions, foot = {n:t._version for n,t in model.named_parameters()}, hook_footprint(model)
    for idx, sample in enumerate(manifest['COCO']+manifest['GQA']):
        q = sample['question_id'].replace(':','_'); path = OUT/f'checkpoints/A/{q}.json'
        if path.exists(): checkpoint(path, p, sample, 'A', 'Native-ImageOFF'); continue
        tick = time.time(); inputs = inputs_for(model, processor, sample); hashes = tensor_hashes(inputs)
        coco = sample.get('dataset') == 'coco_caption'
        gen, rows, fl, ol, audits = native_counterfactual(model, processor.tokenizer, inputs, 96 if coco else 32)
        annotation = align(processor.tokenizer, gen['token_ids'])
        assert annotation['roundtrip_exact'], 'Token-word alignment failed; no silent relabel/drop'
        if coco:
            hg, b = history(sample, 'Full')
            assert gen['token_ids'] == hg['token_ids']
            with np.load(b['raw']) as data: assert np.array_equal(fl[0], data['first_logits'])
        guard(model, inputs, foot, versions, hashes)
        raw = RAW/f'A/{q}.npz'; save_raw(raw, full_logits=fl, imageoff_logits=ol)
        write(path, dict(protocol=digest(p), sample=digest(sample), stage='A', condition='Native-ImageOFF',
            generation=gen, token_metrics=rows, word_alignment=annotation, probability_audits=audits,
            image_keys=[i for i,t in enumerate(sample['input_mapping']['prompt_ids']) if t == 32000],
            inputs_sha256=hashes, raw_file=str(raw), raw_sha256=sha256(raw), seconds=time.time()-tick,
            hooks_clean=True, weights_inputs_unchanged=True))
        print(f'A {idx+1}/180 {q} tokens={len(rows)} seconds={time.time()-tick:.2f}', flush=True)
    for idx, sample in enumerate(manifest['COCO']):
        q = sample['question_id'].replace(':','_')
        inputs = inputs_for(model, processor, sample); hashes = tensor_hashes(inputs)
        a = checkpoint(OUT/f'checkpoints/A/{q}.json', p, sample, 'A', 'Native-ImageOFF')
        with np.load(a['raw_file']) as data: first = data['full_logits'][0].copy()
        for c in CONDITIONS:
            path = OUT/f'checkpoints/B/{q}/{c}.json'
            if path.exists(): checkpoint(path, p, sample, 'B', c); continue
            tick = time.time()
            if c in ['Full', 'None']:
                gen, binding = history(sample, c)
                gen = dict(gen, visual_open_steps=gen['decode_steps'] if c == 'Full' else 0,
                    visual_open_fraction=1. if c == 'Full' else 0., empty=not gen['text'])
                gates = [dict(t=t, on=c == 'Full', historical_verified=True) for t in range(1, len(gen['token_ids']))]
            else:
                gen, f, gates, _, _ = temporal_generate(model, processor.tokenizer, inputs, sample['image_sha256'], c)
                assert np.array_equal(f.numpy(), first), 'Prefill changed during temporal intervention'
                binding = None
            guard(model, inputs, foot, versions, hashes)
            write(path, dict(protocol=digest(p), sample=digest(sample), stage='B', condition=c,
                generation=gen, gates=gates, historical_binding=binding, first_logits_exact=True,
                seconds=time.time()-tick, hooks_clean=True, weights_inputs_unchanged=True))
        print(f'B {idx+1}/60 {q} all8 conditions checkpointed', flush=True)
    verify()
    write(OUT/'completion.json', dict(passed=True, protocol=digest(p), samples_A=180,
        conditions_B=8, images_B=60, new_generations_B=360, historical_reused_B=120,
        seconds=time.time()-started, peak=gpu_memory(), pid=os.getpid(), hooks_clean=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'worker'])
    args = parser.parse_args()
    try:
        prepare() if args.action == 'prepare' else worker()
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT/f'failure_{time.time_ns()}.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
