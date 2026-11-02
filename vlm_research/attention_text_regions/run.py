import argparse
from collections import Counter
import fcntl
import io
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch

import rq2_sparse_caption as history
from benchmarks.dataset_registry import digest, sha256, freeze_json
from benchmarks.smoke import hook_footprint
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import atomic_json, tensor_hashes, prepared
from vlm_common import load_model, gpu_memory
from .engine import CONDITIONS, REGIONS, generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/attention_text_regions_660'
POOL = ROOT/'results/paper_rq1_single_layer_gqa6/sample_manifest.json'
GPUS = [0, 2, 3]
SEED = 20261102
read = history.read


def uid(sample):
    return f"{sample['dataset']}/{sample.get('task', 'caption')}/{sample['question_id']}"


def input_digest(sample):
    return digest(dict(seed=SEED, task=sample['task'], question_id=sample['question_id'],
                       image_sha256=sample['image_sha256'], prompt=sample['prompt_text'],
                       input_mapping=sample['input_mapping']))


def select(pool, captions):
    assert len(pool) == 3000 and set(Counter(s['task'] for s in pool).values()) == {500}
    selected = []; used = {s['image_sha256'] for s in captions}
    for task in sorted({s['task'] for s in pool}):
        ordered = sorted((s for s in pool if s['task'] == task), key=input_digest)
        unique = [s for s in ordered if s['image_sha256'] not in used]
        chosen = []
        for s in unique:
            if s['image_sha256'] in used: continue
            chosen.append(s); used.add(s['image_sha256'])
            if len(chosen) == 100: break
        if len(chosen) < 100:
            chosen += [s for s in ordered if s not in chosen][:100-len(chosen)]
            used.update(s['image_sha256'] for s in chosen)
        selected.extend(chosen)
    assert len(selected) == 600 and set(Counter(s['task'] for s in selected).values()) == {100}
    return selected


def assign(samples):
    groups = {}
    for s in samples: groups.setdefault(s['image_sha256'], []).append(s)
    loads = [0]*3; result = {}
    for image in sorted(groups, key=lambda h: (-len(groups[h]), h)):
        rank = min(range(3), key=lambda r: (loads[r], r))
        for s in groups[image]: result[uid(s)] = rank
        loads[rank] += len(groups[image])
    return result


def sources():
    paths = [ROOT/'attention_text_regions'/name for name in ['__init__.py', 'engine.py', 'run.py']]
    paths += [ROOT/'test_attention_text_regions.py', ROOT/'vlm_common.py', ROOT/'knockout_probe.py',
              ROOT/'paper_rq1/engine.py', ROOT/'paper_rq1/scorers.py', ROOT/'paper_rq1_single_layer_gqa6/run.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def prepare():
    OUT.mkdir(exist_ok=True, parents=True); tick = time.time()
    assert not (OUT/'protocol.json').exists()
    old = history.verify()
    captions = read(history.OUT/'sample_manifest.json')['samples']
    assert len(captions) == 60
    gqa = [dict(s, dataset='gqa_balanced', original_sample_sha256=digest(s))
           for s in select(read(POOL)['samples'], captions)]
    samples = gqa+captions
    assert len({uid(s) for s in samples}) == 660
    for s in samples: assert sha256(s['image_path']) == s['image_sha256']
    ranks = assign(samples)
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_attention_text_regions'))
    atomic_json(OUT/'cpu_validation.json', dict(passed=result.wasSuccessful(), tests=result.testsRun, output=stream.getvalue()))
    assert result.wasSuccessful(), stream.getvalue()
    freeze_json(OUT/'sample_manifest.json', dict(samples=samples, assignments=ranks,
        original_gqa_pool_sha256=sha256(POOL), caption_manifest_sha256=sha256(history.OUT/'sample_manifest.json'),
        gqa_images=len({s['image_sha256'] for s in gqa}), total_images=len({s['image_sha256'] for s in samples}),
        task_counts=dict(Counter(s['task'] for s in gqa))))
    table, apps = history.history.snapshot(); gpus = []
    for gpu in GPUS:
        fields = next(line for line in table.splitlines() if line.startswith(f'{gpu}, ')).split(', ')
        assert int(fields[4]) >= 18000 and int(fields[5]) == 0
        assert not any(line.startswith(fields[1]) for line in apps.splitlines())
        gpus.append(dict(physical_gpu=gpu, uuid=fields[1]))
    freeze_json(OUT/'protocol.json', dict(base='7e19ca0ba5171ff77500a00566552788d4d17d50',
        model_revision=old['model_revision'], model_files=old['model_files'], transformers_source=old['transformers_source'],
        scorer_source=old['scorer_source'], source=sources(), conditions=CONDITIONS, regions=REGIONS,
        dtype='float16', attention='eager', transformers='4.49.0', seed=SEED,
        selection='Task alphabetical order. Sort original inputs by SHA256(seed,task,qid,image SHA,prompt,input_mapping). Prefer globally unused image SHA including COCO; deterministic fallback only if necessary. No model outputs.',
        timing=dict(gqa='Original P-1 plus all true Decode queries', coco='All Prefill native; true Decode only'),
        text_keys='All original non-BOS/non-Image keys and generated history strictly before current Self. Original P-1 becomes history in Decode. Preserve BOS/Image/current Self.',
        attention_off='Zero only target Query self_attn.o_proj output, not MLP/residual/QKV/cache or whole block',
        generation=dict(gqa_max_new_tokens=32, coco_max_new_tokens=96, do_sample=False, eos_token_id=2, use_cache=True),
        gpus=gpus, preflight=dict(inventory=table, apps=apps), planned_generations=4620,
        bootstrap=dict(seed=SEED, repetitions=20000, cluster='image_sha256', estimator='paired question-weighted cluster ratio',
            caption='Official per-image CIDEr/SPICE contributions; CIDEr document frequencies fixed to original 60-image references, same historical statistical protocol'),
        qualitative='Six posthoc illustrative COCO cases: strongest SPICE harm and strongest gain/least harm for each region under Attention OFF, unique image SHA, stable tie break. Include all seven captions and manual object/action notes; no aggregate filtering.',
        bindings={n:sha256(OUT/n) for n in ['sample_manifest.json', 'cpu_validation.json']},
        exploratory=True, no_new_data_or_conditions=True, started_epoch=tick))
    print('Frozen: 600 GQA + 60 COCO; 7 conditions; 4620 new paired generations', flush=True)


def verify():
    p = read(OUT/'protocol.json'); assert p['source'] == sources()
    for n, h in p['bindings'].items(): assert sha256(OUT/n) == h
    for n, b in p['model_files'].items():
        stat = Path(n).stat(); assert stat.st_size == b['bytes'] and stat.st_mtime_ns == b['mtime_ns']
    for n, h in p['transformers_source'].items(): assert sha256(n) == h
    return p


def prompt(model, processor, s):
    return history.history.prompt(model, processor, s) if s['dataset'] == 'coco_caption' else prepared(model, processor, s)


def smoke(model, processor, samples, rank):
    footprint = hook_footprint(model); versions = {n:t._version for n,t in model.named_parameters()}
    checks = []
    for caption in [False, True]:
        s = next(s for s in samples if (s['dataset'] == 'coco_caption') == caption)
        data = prompt(model, processor, s); before = tensor_hashes(data); limit = 96 if caption else 32
        full, first, _, _, trace, debug = generate(model, data, 'Full', not caption, limit, audit=True, collect=True)
        native = model.generate(**data, do_sample=False, num_beams=1, max_new_tokens=limit, use_cache=True, eos_token_id=2)
        assert native[0, data['input_ids'].shape[1]:].tolist() == full['token_ids']
        variants = {}
        for condition in CONDITIONS[1:]:
            noop, nf, _, _, nt, nd = generate(model, data, condition, not caption, limit, noop=True, audit=True, collect=True)
            assert torch.equal(debug['prefill_logits'], nd['prefill_logits'])
            assert noop['token_ids'] == full['token_ids'] and len(trace) == len(nt)
            assert all(torch.equal(a,b) for a,b in zip(trace,nt))
            g, f, rows, projections, _, d = generate(model, data, condition, not caption, 3, audit=True)
            assert torch.equal(debug['prefill_logits'][:, :-1], d['prefill_logits'][:, :-1])
            assert debug['states'] == d['states'] and debug['other_query_attention'] == d['other_query_attention']
            if caption: assert torch.equal(debug['prefill_logits'], d['prefill_logits'])
            variants[condition] = dict(non_target_prefill_exact=True, entire_prefill_exact=caption,
                rows=rows, projections=projections, actual_decode_steps=g['decode_steps'])
        history.history.guard(model, data, footprint, versions, before)
        checks.append(dict(dataset=s['dataset'], question_id=s['question_id'], native_tokens_exact=True,
            noop_all_generation_logits_exact=True, checks=variants))
    atomic_json(OUT/f'shard{rank}/smoke.json', dict(passed=True, checks=checks, hooks_clean=True))


def validate_checkpoint(path, s, condition, p):
    r = read(path); assert r['protocol_sha256'] == digest(p) and r['sample_sha256'] == digest(s)
    assert r['condition'] == condition and r['generation_sha256'] == digest(r['generation'])
    assert sha256(path.with_suffix('.npz')) == r['first_logits_file_sha256']
    return r


def worker(rank):
    p = verify(); folder = OUT/f'shard{rank}'; folder.mkdir(exist_ok=True)
    lock = (folder/'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        assert read(folder/'completion.json')['protocol_sha256'] == digest(p); return
    gpu = p['gpus'][rank]; assert gpu['physical_gpu'] in GPUS
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu['physical_gpu'])
    table, apps = history.history.snapshot()
    fields = next(line for line in table.splitlines() if line.startswith(f"{gpu['physical_gpu']}, ")).split(', ')
    assert fields[1] == gpu['uuid'] and int(fields[4]) >= 18000 and int(fields[5]) == 0
    assert not any(line.startswith(gpu['uuid']) for line in apps.splitlines())
    atomic_json(folder/'preflight.json', dict(gpu=gpu, inventory=table, apps=apps, pid=os.getpid()))
    started = time.time(); model, processor, env = load_model(history.history.MODEL_PATH)
    assert env['model_revision'] == p['model_revision'] and env['transformers'] == '4.49.0'
    assert env['parameter_dtypes'] == ['torch.float16'] and not model.training
    m = read(OUT/'sample_manifest.json'); samples = [s for s in m['samples'] if m['assignments'][uid(s)] == rank]
    smoke(model, processor, samples, rank)
    print(f'GPU{gpu["physical_gpu"]}: smoke passed', flush=True)
    wait = time.time()
    while not all((OUT/f'shard{i}/smoke.json').exists() for i in range(3)):
        assert not list(OUT.glob('shard*/failure*.json')), 'Other worker failed; stop'
        assert time.time()-wait < 600; time.sleep(2)
    formal = time.time(); count = resumed = 0
    footprint = hook_footprint(model); versions = {n:t._version for n,t in model.named_parameters()}
    for i, s in enumerate(samples):
        data = prompt(model, processor, s); before = tensor_hashes(data); caption = s['dataset'] == 'coco_caption'
        full_first = None
        for condition in CONDITIONS:
            path = folder/f"checkpoints/{s.get('task', 'caption')}__{s['question_id'].replace(':','_')}/{condition}.json"
            if path.exists():
                r = validate_checkpoint(path, s, condition, p)
                if condition == 'Full':
                    with np.load(path.with_suffix('.npz')) as raw: full_first = torch.from_numpy(raw['first_logits'].copy())
                count += 1; resumed += 1; continue
            g, first, rows, projections, _, _ = generate(model, data, condition, not caption, 96 if caption else 32)
            if condition == 'Full': full_first = first
            assert full_first is not None
            if caption: assert torch.equal(first, full_first)
            g['text'] = processor.tokenizer.decode(g['token_ids'], skip_special_tokens=True).strip()
            if not caption: g['prediction'] = gqa_prediction(g['text']); g['score'] = int(g['prediction'] == s['answer'])
            history.history.guard(model, data, footprint, versions, before)
            path.parent.mkdir(parents=True, exist_ok=True)
            history.history.raw_save(path.with_suffix('.npz'), dict(first_logits=first.numpy()))
            audit = dict(rows=len(rows), masked_text_rows=sum(r['text_masked'] for r in rows),
                text_probability_max_blocked=max([r['text_probability_max'] for r in rows if r['text_masked']], default=0.),
                min_image_mass=min(r['image_mass'] for r in rows), max_row_error=max(r['row_error'] for r in rows),
                attention_off_rows=sum(r['applied'] for r in projections),
                attention_off_max_norm=max([r['output_norm'] for r in projections if r['applied']], default=0.),
                genuine_cache_positions_lengths_checked=True, non_target_keys_unchanged=True)
            atomic_json(path, dict(protocol_sha256=digest(p), sample_sha256=digest(s), condition=condition,
                generation=g, generation_sha256=digest(g), audit=audit,
                first_logits_file_sha256=sha256(path.with_suffix('.npz')), hooks_inputs_weights_unchanged=True))
            count += 1
        if (i+1) % 10 == 0 or i == len(samples)-1:
            print(f'GPU{gpu["physical_gpu"]}: {count}/{7*len(samples)} generations; {time.time()-formal:.1f}s', flush=True)
    verify()
    atomic_json(folder/'completion.json', dict(protocol_sha256=digest(p), generations=count, samples=len(samples),
        resumed=resumed, formal_seconds=time.time()-formal, worker_seconds=time.time()-started,
        environment=env, gpu=gpu, memory=gpu_memory(), finished_epoch=time.time()))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'worker'])
    parser.add_argument('--rank', type=int, choices=range(3)); args = parser.parse_args()
    if args.command == 'prepare': prepare(); return
    try: worker(args.rank)
    except BaseException as exc:
        atomic_json(OUT/f'shard{args.rank}/failure_{time.time_ns()}.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__': main()
