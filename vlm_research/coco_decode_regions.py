"""Twelve frozen decode-only visual-access conditions; reuse audited mask primitives."""
import argparse
from contextlib import nullcontext
import fcntl
import io
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch

import rq2_sparse_caption as old
from benchmarks.dataset_registry import digest, sha256, freeze_json
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import step_inputs
from paper_rq1_single_layer_gqa6.run import atomic_json, tensor_hashes
from rq2_decode_visual.engine import DecodeImageKnockout, DecodeProbabilities
from rq2_source_knockout.engine import Observer
from vlm_common import load_model, gpu_memory

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/coco_decode_visual_regions_60'
GPUS = [0, 2, 3]
read = old.read


def conditions():
    sets = {'Full': list(range(32)), 'Decode None': [], 'L0 Only': [0],
            'Early': list(range(11)), 'Middle': [0]+list(range(11, 21)),
            'Late': [0]+list(range(21, 32))}
    for size in [10, 11]:
        for seed in [20261102, 20261103, 20261104]:
            sets[f'Random{size}-{seed}'] = [0]+sorted(random.Random(seed).sample(range(1, 32), size))
    return sets


@torch.inference_mode()
def generate(model, inputs, bos, retained, limit=96, noop=False, audit=False,
             collect=False, expected_visual=576):
    if limit < 1 or not bool(inputs['attention_mask'].all()):
        raise ValueError('Positive budget and unpadded batch1 prompt required')
    assert len(model_layers(model)) == 32
    if len(set(retained)) != len(retained) or not set(retained) <= set(range(32)):
        raise ValueError('Invalid retained layer set')
    blocked = sorted(set(range(32))-set(retained))
    tokens, cache, trace, rows, debug = [], None, [], [], {}
    n = inputs['input_ids'].shape[1]
    while len(tokens) < limit and (not tokens or tokens[-1] != 2):
        step = len(tokens)
        mask = DecodeImageKnockout(model, inputs, blocked, step, noop, expected_visual) if step else None
        obs = DecodeProbabilities(model, mask) if step else Observer(model, inputs, bos, audit, expected_visual)
        with mask if mask is not None else nullcontext(), obs:
            output = model(**step_inputs(inputs, tokens), past_key_values=cache,
                           use_cache=True, output_attentions=True, return_dict=True)
        assert not obs.handles
        if mask is not None:
            mask.assert_one_forward()
            assert not mask.handles
            assert [r['layer'] for r in mask.records if r['applied']] == ([] if noop else blocked)
        cache = output.past_key_values
        assert all(cache.get_seq_length(k) == n+step for k in range(32))
        assert [r['layer'] for r in obs.rows] == list(range(32))
        for row in obs.rows:
            maximum = row['image_probability_max'] if step else row['Image_max']
            active = bool(step and not noop and row['layer'] in blocked)
            assert maximum == 0. if active else maximum > 0.
            rows.append(dict(step=step, layer=row['layer'], blocked=active,
                             image_probability_max=maximum,
                             image_mass=row['image_mass'] if step else row['Image'],
                             row_error=row['row_sum_max_error'] if step else row['row_error']))
        logits = output.logits[0, -1].detach().float().cpu().clone()
        assert torch.isfinite(logits).all()
        if collect:
            trace.append(logits)
        if not step:
            first = logits
            if audit:
                debug = dict(prefill_logits=output.logits.detach().float().cpu().clone(),
                             states=obs.states, prefill_rows=obs.rows)
        tokens.append(int(logits.argmax()))
    return dict(token_ids=tokens, eos=tokens[-1] == 2,
                truncated=len(tokens) == limit and tokens[-1] != 2,
                decode_steps=len(tokens)-1, prefill_blocked=[],
                decode_blocked=[] if noop else blocked,
                own_prefix=True, own_cache=True), first, rows, trace, debug


def source_hashes():
    return dict(old.sources(), **{n: sha256(ROOT/n) for n in
                ['coco_decode_regions.py', 'test_coco_decode_regions.py']})


def history_record(sample, key):
    if key == 'Full':
        return old.historical(sample['question_id'], 'A')
    p = old.verify()
    r = old.checkpoint(old.OUT/f"checkpoints/{sample['question_id'].replace(':','_')}/DN.json",
                       digest(p), sample, 'DN')
    assert r['first_logits_exact_historical_Full']
    assert r['generation']['prefill_blocked'] == []
    assert r['generation']['decode_blocked'] == list(range(32))
    with np.load(old.OUT/r['raw_file']) as data:
        logits = torch.from_numpy(data['first_logits'].copy())
    return r['generation'], logits


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Protocol already frozen'
    tick = time.time()
    historical = old.verify()
    samples = read(old.OUT/'sample_manifest.json')['samples']
    assert len(samples) == len({s['image_sha256'] for s in samples}) == 60
    ordered = sorted(samples, key=lambda s: s['image_sha256'])
    assignment = {s['question_id']: i % 3 for i, s in enumerate(ordered)}
    for s in samples:
        assert sha256(s['image_path']) == s['image_sha256']
        a, af = history_record(s, 'Full'); d, df = history_record(s, 'Decode None')
        assert torch.equal(af, df) and a['token_ids'][0] == d['token_ids'][0]
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_coco_decode_regions'))
    atomic_json(OUT/'cpu_validation.json', dict(passed=result.wasSuccessful(),
                tests=result.testsRun, output=stream.getvalue()))
    assert result.wasSuccessful(), stream.getvalue()
    freeze_json(OUT/'sample_manifest.json', dict(samples=samples, assignment=assignment,
                old_manifest_sha256=sha256(old.OUT/'sample_manifest.json')))
    # Bind both historical result chains, including raw first-token logits and generation hashes.
    paths = [old.OUT/'protocol.json', old.OUT/'history_bindings.json']
    for s in samples:
        q = s['question_id']
        b = read(old.OUT/'history_bindings.json')[q]['A']
        paths += [ROOT/b['path'], ROOT/b['raw_file']]
        d = old.OUT/f"checkpoints/{q.replace(':','_')}/DN.json"
        paths += [d, old.OUT/read(d)['raw_file']]
    bindings = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    freeze_json(OUT/'history_bindings.json', bindings)
    table, apps = old.history.snapshot()
    gpus = []
    for gpu in GPUS:
        fields = next(r for r in table.splitlines() if r.startswith(f'{gpu}, ')).split(', ')
        assert int(fields[4]) >= 18000 and int(fields[5]) == 0
        assert not any(r.startswith(fields[1]) for r in apps.splitlines())
        gpus.append(dict(physical_gpu=gpu, uuid=fields[1]))
    freeze_json(OUT/'protocol.json', dict(
        base_commit='9edcbceb43', remote_main='e89c8acce6e674ecfbd2c8acad57faf71bf4d547',
        source=source_hashes(), model_revision=historical['model_revision'],
        model_files=historical['model_files'], transformers_source=historical['transformers_source'],
        scorer_source=historical['scorer_source'], conditions=conditions(), gpus=gpus,
        generation=dict(max_new_tokens=96, do_sample=False, eos_token_id=2, use_cache=True),
        dtype='float16', attention='eager', transformers='4.49.0',
        mask='Original Image keys only; softmax-before -inf; true incremental Decode only. Entire Prefill native.',
        bindings={n: sha256(OUT/n) for n in ['sample_manifest.json', 'history_bindings.json', 'cpu_validation.json']},
        new_generations=600, historical_generations=120,
        random=dict(algorithm='Python random.Random(seed).sample(range(1,32),k); sorted; prepend L0',
                    seeds=[20261102, 20261103, 20261104], fixed_across_images_and_steps=True),
        bootstrap=dict(repetitions=20000, seed=20261102, unit='image_sha256 paired',
                       metric='Official per-image contributions; CIDEr conditioned on original 60-image reference corpus IDF, identical to historical protocol'),
        qualitative='Posthoc descriptive: lowest and highest Early/Late SPICE changes, Middle gain and random counterexample; unique images, stable SHA tie breaks. No aggregate filtering.',
        scope='60 historical exploratory COCO images only; no specialization, independent confirmation, NI or acceleration claims',
        preflight=dict(inventory=table, apps=apps), started_epoch=tick))
    print('Frozen 60 images / 12 conditions; 600 new, 120 audited historical generations', flush=True)


def verify():
    p = read(OUT/'protocol.json')
    assert p['source'] == source_hashes() and p['conditions'] == conditions()
    for name, h in p['bindings'].items():
        assert sha256(OUT/name) == h
    for name, h in read(OUT/'history_bindings.json').items():
        assert sha256(ROOT/name) == h
    old.verify()
    return p


def checkpoint(path, sample, key, p):
    r = read(path)
    assert r['protocol_sha256'] == digest(p) and r['sample_sha256'] == digest(sample)
    assert r['condition'] == key and r['generation_sha256'] == digest(r['generation'])
    assert r['first_logits_sha256'] == sha256(path.with_suffix('.npz'))
    return r


def smoke(model, processor, sample, rank):
    inputs = old.history.prompt(model, processor, sample)
    bos = processor.tokenizer.bos_token_id
    foot = hook_footprint(model); versions = {n: t._version for n, t in model.named_parameters()}
    before = tensor_hashes(inputs)
    full, first, _, trace, debug = generate(model, inputs, bos, list(range(32)), collect=True, audit=True)
    hfull, hf = history_record(sample, 'Full')
    assert torch.equal(first, hf) and full['token_ids'] == hfull['token_ids']
    noop, nf, _, nt, nd = generate(model, inputs, bos, [], noop=True, collect=True, audit=True)
    assert torch.equal(debug['prefill_logits'], nd['prefill_logits'])
    assert len(trace) == len(nt) and all(torch.equal(a, b) for a, b in zip(trace, nt))
    native = model.generate(**inputs, do_sample=False, max_new_tokens=96, use_cache=True)
    assert native[0, inputs['input_ids'].shape[1]:].tolist() == full['token_ids']
    none, nfirst, nr, _, _ = generate(model, inputs, bos, [])
    hn, hnf = history_record(sample, 'Decode None')
    assert torch.equal(nfirst, hnf) and none['token_ids'] == hn['token_ids']
    checks = {}
    for key, retained in conditions().items():
        g, f, rows, _, d = generate(model, inputs, bos, retained, limit=3, audit=True)
        assert torch.equal(debug['prefill_logits'], d['prefill_logits'])
        assert debug['states'] == d['states'] and debug['prefill_rows'] == d['prefill_rows']
        assert torch.equal(first, f) and g['decode_steps'] >= 1
        checks[key] = dict(all_prefill_logits_states_attention_exact=True, probability_rows=rows)
    old.history.guard(model, inputs, foot, versions, before)
    atomic_json(OUT/f'shard{rank}/smoke.json', dict(passed=True, sample=sample['question_id'],
        historical_Full_DecodeNone_full_tokens_and_logits_exact=True, native_HF_tokens_exact=True,
        noop_all_generation_logits_exact=True, checks=checks, hooks_clean=True))


def worker(rank):
    p = verify(); folder = OUT/f'shard{rank}'; folder.mkdir(exist_ok=True)
    lock = (folder/'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        assert read(folder/'completion.json')['protocol_sha256'] == digest(p)
        print('Already complete; no GPU load', flush=True); return
    gpu = p['gpus'][rank]; assert gpu['physical_gpu'] in GPUS
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu['physical_gpu'])
    table, apps = old.history.snapshot()
    fields = next(r for r in table.splitlines() if r.startswith(f"{gpu['physical_gpu']}, ")).split(', ')
    assert fields[1] == gpu['uuid'] and int(fields[4]) >= 18000 and int(fields[5]) == 0
    assert not any(r.startswith(gpu['uuid']) for r in apps.splitlines())
    atomic_json(folder/'preflight.json', dict(gpu=gpu, inventory=table, apps=apps, pid=os.getpid()))
    start = time.time()
    model, processor, env = load_model(old.history.MODEL_PATH)
    assert env['model_revision'] == p['model_revision'] and env['transformers'] == '4.49.0'
    assert env['parameter_dtypes'] == ['torch.float16'] and not model.training
    manifest = read(OUT/'sample_manifest.json')
    samples = [s for s in manifest['samples'] if manifest['assignment'][s['question_id']] == rank]
    assert len(samples) == 20
    smoke(model, processor, samples[0], rank)
    print(f'GPU{gpu["physical_gpu"]}: smoke passed; waiting for other GPUs', flush=True)
    wait_start = time.time()
    while not all((OUT/f'shard{i}/smoke.json').exists() for i in range(3)):
        assert not list(OUT.glob('shard*/failure.json')), 'Another GPU smoke failed'
        assert time.time()-wait_start < 600, 'Smoke barrier timeout'
        time.sleep(2)
    for i in range(3): assert read(OUT/f'shard{i}/smoke.json')['passed']
    formal_start = time.time(); count = resumed = 0
    foot = hook_footprint(model); versions = {n: t._version for n, t in model.named_parameters()}
    for sample in samples:
        inputs = old.history.prompt(model, processor, sample); before = tensor_hashes(inputs)
        _, historical_first = history_record(sample, 'Full')
        for key, retained in p['conditions'].items():
            if key in ['Full', 'Decode None']: continue
            path = folder/f"checkpoints/{sample['question_id'].replace(':','_')}/{key}.json"
            if path.exists():
                checkpoint(path, sample, key, p); count += 1; resumed += 1; continue
            tick = time.time()
            g, first, rows, _, _ = generate(model, inputs, processor.tokenizer.bos_token_id, retained)
            assert torch.equal(first, historical_first), 'Historical Full exact logits mismatch; stop'
            g['text'] = processor.tokenizer.decode(g['token_ids'], skip_special_tokens=True).strip()
            old.history.guard(model, inputs, foot, versions, before)
            path.parent.mkdir(parents=True, exist_ok=True)
            old.history.raw_save(path.with_suffix('.npz'), dict(first_logits=first.numpy()))
            audit = dict(rows=len(rows), prefill_rows=sum(r['step'] == 0 for r in rows),
                decode_rows=sum(r['step'] > 0 for r in rows),
                blocked_rows=sum(r['blocked'] for r in rows),
                blocked_max=max([r['image_probability_max'] for r in rows if r['blocked']], default=0.),
                retained_min=min(r['image_probability_max'] for r in rows if not r['blocked']),
                max_row_error=max(r['row_error'] for r in rows), all_cache_positions_lengths_checked=True)
            atomic_json(path, dict(protocol_sha256=digest(p), sample_sha256=digest(sample), condition=key,
                generation=g, generation_sha256=digest(g), first_logits_sha256=sha256(path.with_suffix('.npz')),
                first_logits_exact_historical=True, audit=audit, hooks_inputs_weights_unchanged=True,
                seconds=time.time()-tick))
            count += 1
        print(f'GPU{gpu["physical_gpu"]}: {count}/200 generations; {time.time()-formal_start:.1f}s', flush=True)
    verify()
    atomic_json(folder/'completion.json', dict(protocol_sha256=digest(p), generations=count,
        resumed=resumed, formal_seconds=time.time()-formal_start, worker_seconds=time.time()-start,
        environment=env, gpu=gpu, memory=gpu_memory(), finished_epoch=time.time()))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'worker'])
    parser.add_argument('--rank', type=int, choices=range(3))
    args = parser.parse_args()
    if args.command == 'prepare': prepare(); return
    try: worker(args.rank)
    except BaseException as exc:
        atomic_json(OUT/f'shard{args.rank}/failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__': main()
