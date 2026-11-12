"""Two independent, resumable, image-clustered quality workers."""
import argparse
import gc
import json
import os
import time
import traceback

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .common import OUT, OLD, GPUS, MODEL, check, disk, freeze, inventory, prepare, sha, digest, write, scorers, TASKS
from .engine import run
from .store import Store
from ..qwen3vl_svh_preflight48.run import resource, strict_equal, validate_semantics
from ..qwen3vl_svh_numerical_repair.run import compare


def small_result(result):
    return {k: v for k, v in result.items() if k != 'audit'}


def reclaim():
    gc.collect()
    torch.cuda.empty_cache()


def smoke(model, processor, shard):
    path = OUT / f'smoke{shard}.json'
    candidates = json.loads((OUT / 'smoke_candidates.json').read_text())
    rows = [next(c['sample'] for c in candidates if c['sample']['benchmark'] == task and c['quantile'] == 'min') for task in TASKS]
    results = []
    for row in rows:
        inputs, layout = prepare(processor, row)
        assert layout == json.loads((OUT / 'known_input_bindings.json').read_text())[row['key']]
        records = {}
        for name, condition, noop in [('Native','Native',False), ('Noop','Native',True), ('Dense','Dense',False), ('SVH','SVH',False)]:
            result, vectors, structure = run(model, processor, inputs, layout, condition,
                audit=True, capture=True, noop=noop, cap=row['max_new_tokens'])
            records[name] = result, vectors
            assert structure['status'] == 'PASS'
        assert strict_equal(records['Native'], records['Noop'])
        structural = validate_semantics(records['SVH'][0], layout)
        numerical = compare(*records['Dense'], *records['SVH'])
        results.append(dict(key=row['key'], structural=structural, native_noop_equal=True,
            strict_numerical='PASS' if all(r['pass'] for r in numerical) else 'FAIL_RETAINED', comparisons=numerical))
        write(path, dict(status='RUNNING', rows=results, mode='RELAXED_NUMERICAL_QUALITY_ONLY'))
        print('STRUCTURAL_SMOKE', shard, row['key'], results[-1]['strict_numerical'], flush=True)
        del records, result, vectors, inputs
        reclaim()
    write(path, dict(status='PASS', rows=results, mode='RELAXED_NUMERICAL_QUALITY_ONLY', pid=os.getpid()))


def request(model, processor, inputs, layout, row, condition, shard, scoring):
    start = time.time()
    try:
        result, _, structure = run(model, processor, inputs, layout, condition, cap=row['max_new_tokens'])
        scored = scoring.score(row, result['text'])
        record = dict(status='COMPLETE', **small_result(result), scoring=scored, structural=structure)
        record['condition'] = condition
    except torch.OutOfMemoryError as exc:
        record = dict(status='SKIPPED_OOM', error=str(exc), traceback=traceback.format_exc(),
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(), operator_context='vision softmax or current traceback; no backend change')
    record.update(key=row['key'], condition=condition, benchmark=row['benchmark'], gold=row['gold'],
        question_id=row['question_id'], question=row['question'], image_sha256=row['image_sha256'],
        prompt_sha256=row['prompt_sha256'], input_binding=layout, input_binding_sha256=digest(layout),
        image_tokens=row['image_tokens'], input_length=row['input_length'], max_new_tokens=row['max_new_tokens'],
        gpu_uuid=GPUS[shard], mode='RELAXED_NUMERICAL_QUALITY_ONLY',
        strict_numerical='NOT_MEASURED_PER_SAMPLE', elapsed_s=time.time()-start,
        strategy=row.get('strategy'), mme_task=row.get('task'), pair_id=row.get('pair_id'), circular_group=row.get('circular_group'),
        protocol_sha256=sha(OUT / 'protocol.json'))
    return record


def canary(model, processor, inputs, layout, row, shard, count):
    path = OUT / f'canaries/shard{shard}_{count:06d}.json'
    try:
        records = {}
        for name in ['Dense', 'SVH']:
            r, v, structure = run(model, processor, inputs, layout, name, audit=True, capture=True, cap=row['max_new_tokens'])
            assert structure['status'] == 'PASS'
            if name == 'SVH':
                validate_semantics(r, layout)
            records[name] = r, v
        comparisons = compare(*records['Dense'], *records['SVH'])
        write(path, dict(key=row['key'], status='PASS' if all(r['pass'] for r in comparisons) else 'FAIL_RETAINED',
            comparisons=comparisons, structural='PASS', mode='RELAXED_NUMERICAL_QUALITY_ONLY',
            note='Different free prefixes after a token flip are not pure same-input numerical error',
            dense_tokens=records['Dense'][0]['token_ids'], physical_tokens=records['SVH'][0]['token_ids']))
    except torch.OutOfMemoryError as exc:
        write(path, dict(key=row['key'], status='SKIPPED_OOM_NUMERICAL_ONLY', error=str(exc)))
    reclaim()


def worker(shard, smoke_only=False):
    protocol = check()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPUS[shard]
    disk()
    before = resource()
    assert GPUS[shard] not in before['processes'], 'GPU no longer idle'
    write(OUT / f'gpu{shard}_before_{time.time_ns()}.json', before)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    model = store = None
    try:
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'}).eval()
        smoke(model, processor, shard)
        if smoke_only:
            return
        # Both same-code instances must pass structure before formal generation.
        while not all((OUT / f'smoke{s}.json').exists() and json.loads((OUT / f'smoke{s}.json').read_text())['status'] == 'PASS' for s in [0, 1]):
            assert not (OUT / 'STOP_STRUCTURE.json').exists()
            time.sleep(2)
        mapping = json.loads((OUT / 'shards.json').read_text())
        rows = [r for r in inventory() if mapping[r['key']]['shard'] == shard]
        # Finish datasets in protocol order; no output-based selection.
        rows.sort(key=lambda r: (TASKS.index(r['benchmark']), r['image_sha256'], r['key']))
        store = Store(OUT / f'shard{shard}.sqlite', sha(OUT / 'protocol.json'))
        store.plan([r['key'] for r in rows], ['Native', 'SVH'])
        scoring = scorers()
        new_pairs = 0
        started = time.time()
        for row in rows:
            assert not (OUT / 'STOP_STRUCTURE.json').exists() and not (OUT / 'STOP_USER').exists()
            if all(store.state(row['key'], c) != 'PENDING' for c in ['Native', 'SVH']):
                continue
            disk()
            inputs, layout = prepare(processor, row)
            assert __import__('hashlib').sha256(layout['prompt'].encode()).hexdigest() == row['prompt_sha256']
            successful = True
            for condition in ['Native', 'SVH']:
                if store.state(row['key'], condition) != 'PENDING':
                    continue
                write(OUT / f'worker{shard}_current.json', dict(key=row['key'], condition=condition, pid=os.getpid()))
                store.begin(row['key'], condition)
                record = request(model, processor, inputs, layout, row, condition, shard, scoring)
                store.finish(row['key'], condition, record)
                successful &= record['status'] == 'COMPLETE'
                if record['status'] == 'SKIPPED_OOM':
                    reclaim()
                    # After OOM, prove a fresh tiny request still has legal state.
                    probe = next(c['sample'] for c in json.loads((OUT/'smoke_candidates.json').read_text()) if c['sample']['benchmark']=='GQA' and c['quantile']=='min')
                    pi, pl = prepare(processor, probe)
                    run(model, processor, pi, pl, condition, audit=True, cap=2)
                    del pi
                    reclaim()
            new_pairs += 1
            if new_pairs % 128 == 0 and successful:
                canary(model, processor, inputs, layout, row, shard, new_pairs)
            if new_pairs % 16 == 0 or not successful:
                state = resource()
                foreign = [line for line in state['processes'].splitlines() if GPUS[shard] in line and f', {os.getpid()},' not in line]
                assert not foreign, ('PAUSED_GPU_EXTERNAL_PROCESS', foreign)
                progress = dict(counts=store.counts(), last=row['key'], pid=os.getpid(), gpu_uuid=GPUS[shard],
                    new_pairs=new_pairs, elapsed_s=time.time()-started, free_disk=disk(), time=time.time())
                write(OUT / f'worker{shard}_progress.json', progress)
                print('PROGRESS', json.dumps(progress), flush=True)
                store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            del inputs
            gc.collect()
        write(OUT / f'worker{shard}_completion.json', dict(status='TERMINAL', counts=store.counts(), planned_questions=len(rows), pid=os.getpid()))
    except BaseException as exc:
        failure = dict(error=repr(exc), traceback=traceback.format_exc(), shard=shard, pid=os.getpid(), time=time.time())
        write(OUT / f'worker{shard}_failure.json', failure)
        # OOM is handled per request above. All other execution failures need investigation.
        write(OUT / 'STOP_STRUCTURE.json', failure)
        raise
    finally:
        if store:
            store.close()
        del model
        reclaim()
        write(OUT / f'gpu{shard}_after_inprocess.json', resource())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'worker', 'smoke'])
    parser.add_argument('--shard', type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    else:
        assert args.shard is not None
        worker(args.shard, args.stage == 'smoke')
