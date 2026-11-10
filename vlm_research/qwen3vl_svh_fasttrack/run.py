"""Two single-device numerical workers. Any numerical failure closes full-run gate."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from contextlib import ExitStack

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .freeze import OUT, ROOT, MODEL, GPUS, prepare, sha, sources, table, write
from ..qwen3vl_svh_preflight48.run import CompactTrace, compare_traces, strict_equal, validate_semantics, resource
from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest
from ..qwen3vl_svh_numerical_repair.run import generate, compare
from ..qwen3vl_svh_l28_repair.engine import generate as fixed_generate
from ..qwen3vl_svh_l28_repair.run import OperatorTrace, compare_records


def log(shard, *message):
    text = ' '.join(map(str, message))
    print(text, flush=True)
    with (OUT / f'worker{shard}.log').open('a') as stream:
        stream.write(text + '\n')


def check_freeze():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sources() == protocol['source_files']
    assert sha(protocol['inventory_path']) == protocol['inventory_sha256']
    for name, expected in protocol['frozen_files'].items():
        assert sha(OUT / name) == expected
    for path, expected in protocol['scorer_sources'].items():
        assert sha(path) == expected
    return protocol


def call(model, processor, inputs, layout, name, cap, trace=None):
    with ExitStack() as stack:
        if trace:
            stack.enter_context(trace.context())
        if name == 'Physical':
            return fixed_generate(model, processor, inputs, layout, 'C', audit=True, capture=True, max_tokens=cap)
        return generate(model, processor, inputs, layout, 'F' if name == 'Dense' else 'N',
            audit=True, capture=True, noop=name == 'Noop', max_tokens=cap)


def localize(model, processor, inputs, layout, folder, cap):
    traces = {}
    for name in ['Dense', 'Physical']:
        trace = OperatorTrace(model, layout)
        result, vectors = call(model, processor, inputs, layout, name, cap, trace)
        traces[name] = trace
        del result, vectors
    rows = compare_records(traces['Dense'], traces['Physical'])
    table(folder / 'operator_comparison.csv', rows)
    first = next((r for r in rows if not r['equal'] and r['operator'] != 'mask'), None)
    write(folder / 'first_operator_difference.json', dict(first=first, diagnostic_max_tokens=cap))
    return first


def numerical(model, processor, shard):
    mapping = json.loads((OUT / 'shards.json').read_text())
    bindings = json.loads((OUT / 'input_bindings.json').read_text())
    cases = json.loads((OUT / 'numerical_cases.json').read_text())
    cases = [c for c in cases if mapping[c['sample']['key']]['shard'] == shard]
    hooks = hook_counts(model)
    weights = [(n, p.data_ptr(), p._version) for n, p in model.named_parameters()]
    completed = []
    for case in cases:
        if (OUT / 'STOP_NUMERICAL.json').exists():
            log(shard, 'STOP_OTHER_WORKER')
            break
        row = case['sample']
        folder = OUT / 'numerical' / row['key'].replace(':', '_').replace('/', '_')
        if (folder / 'validation.json').exists():
            prior = json.loads((folder / 'validation.json').read_text())
            assert prior['status'] == 'PASS'
            completed.append(prior)
            continue
        assert not folder.exists(), 'Partial attempt retained; explicit investigation required'
        write(folder / 'started.json', dict(case=case, shard=shard, pid=os.getpid(), gpu=GPUS[shard], time=time.time()))
        write(OUT / f'worker{shard}_current.json', dict(key=row['key'], phase='NUMERICAL'))
        log(shard, 'CASE', row['key'], case['quantile'], row['input_length'], row['image_tokens'])
        inputs, layout = prepare(processor, row)
        assert layout == bindings[row['key']]
        records, traces = {}, {}
        for name in ['Native', 'Noop', 'Dense', 'Physical']:
            write(OUT / f'worker{shard}_current.json', dict(key=row['key'], phase='NUMERICAL', condition=name))
            trace = CompactTrace(model, layout) if name in ['Dense', 'Physical'] else None
            result, vectors = call(model, processor, inputs, layout, name, row['max_new_tokens'], trace)
            assert hook_counts(model) == hooks
            assert all(bool(torch.isfinite(v).all()) for v in vectors)
            assert {k:tensor_digest(v) for k,v in inputs.items()} == layout['input_tensor_hashes']
            write(folder / (name + '.json'), result)
            np.savez_compressed(folder / (name + '_logits.npz'), logits=torch.stack(vectors).numpy(), token_ids=result['token_ids'])
            records[name] = result, vectors
            if trace:
                traces[name] = trace
            log(shard, name, repr(result['text']), 'peak_MiB', result['peak_allocated'] / 2**20)
        native_equal = strict_equal(records['Native'], records['Noop'])
        comparisons = compare(*records['Dense'], *records['Physical'])
        layer_rows, semantics = compare_traces(traces['Dense'], traces['Physical'])
        table(folder / 'logits_comparison.csv', comparisons)
        table(folder / 'layer_comparison.csv', layer_rows)
        write(folder / 'semantics.json', semantics)
        structural = validate_semantics(records['Physical'][0], layout)
        passed = (native_equal and bool(comparisons) and all(r['pass'] for r in comparisons)
            and semantics['all_layer_steps_captured'] and semantics['all_mrope_positions_equal']
            and semantics['deepstack_equal'] and semantics['cache_positions_equal'])
        entry = dict(status='PASS' if passed else 'FAIL', key=row['key'], task=row['benchmark'],
            quantile=case['quantile'], shard=shard, gpu_uuid=GPUS[shard], native_noop_equal=native_equal,
            mean_abs=max(r['mean_abs'] for r in comparisons), max_abs=max(r['max_abs'] for r in comparisons),
            all_tokens_equal=records['Dense'][0]['token_ids'] == records['Physical'][0]['token_ids'],
            first_hidden_difference=next((r for r in layer_rows if not r['equal']), None),
            peak_allocated=records['Physical'][0]['peak_allocated'], structural=structural,
            AV_extra_flops=records['Physical'][0]['audit']['extra_AV_query_alignment_flops'])
        write(folder / 'validation.json', entry)
        if not passed:
            write(OUT / 'STOP_NUMERICAL.json', entry)
            log(shard, 'NUMERICAL_FAIL', json.dumps(entry))
            first_bad = next((r['step'] for r in comparisons if not r['pass']), 0)
            del records, traces, vectors, result
            gc.collect()
            torch.cuda.empty_cache()
            entry['first_operator_difference'] = localize(model, processor, inputs, layout, folder, first_bad + 1)
            write(folder / 'validation.json', entry)
            raise RuntimeError('FAST_TRACK_NUMERICAL_FAILED; no evaluation unlock')
        completed.append(entry)
        write(OUT / f'worker{shard}_progress.json', completed)
        log(shard, 'PASS', row['key'], entry['mean_abs'], entry['max_abs'])
        assert weights == [(n, p.data_ptr(), p._version) for n, p in model.named_parameters()]
        del records, traces, vectors, result, inputs
        gc.collect()
    write(OUT / f'worker{shard}_completion.json', dict(status='PASS' if len(completed) == len(cases) else 'STOPPED',
        completed=len(completed), expected=len(cases), stage='NUMERICAL', pid=os.getpid()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    shard = args.shard
    check_freeze()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPUS[shard]
    before = resource()
    assert GPUS[shard] not in before['processes'], 'Another task owns this GPU'
    state = subprocess.check_output(['nvidia-smi', '-i', GPUS[shard], '--query-gpu=name,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert '3090' in state[0] and int(state[1]) > 23500 and int(state[2]) == 0
    write(OUT / f'worker{shard}_gpu_before.json', before)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = None
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager', device_map={'':'cuda:0'}).eval()
        numerical(model, processor, shard)
    except BaseException as exc:
        failure = dict(status='STOPPED', error=repr(exc), traceback=traceback.format_exc(),
            shard=shard, gpu_uuid=GPUS[shard], current=json.loads((OUT / f'worker{shard}_current.json').read_text()),
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
        write(OUT / f'worker{shard}_failure.json', failure)
        if not (OUT / 'STOP_NUMERICAL.json').exists():
            write(OUT / 'STOP_NUMERICAL.json', failure)
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        write(OUT / f'worker{shard}_gpu_after_inprocess.json', resource())


if __name__ == '__main__':
    main()
