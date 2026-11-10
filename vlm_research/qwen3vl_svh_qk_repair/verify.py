"""Fresh-process regression of the evidence-backed QK row alignment."""
import argparse
import gc
import json
import os
import time
import traceback
from contextlib import ExitStack

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from .engine import generate as fixed_generate
from .run import OUT, OLD, ROOT, UUID, MODEL, GATE, sources, samples, call
from ..qwen3vl_svh_preflight48.freeze import prepare, sha, write, table
from ..qwen3vl_svh_preflight48.run import CompactTrace, compare_traces, validate_semantics, strict_equal, resource
from ..qwen3vl_svh_l28_repair.run import OperatorTrace, compare_records
from ..qwen3vl_svh_numerical_repair.run import compare
from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest


def freeze():
    assert not torch.cuda.is_initialized()
    assert not (OUT/'repair_protocol.json').exists()
    diagnosis = json.loads((OUT/'diagnosis_completion.json').read_text())
    isolation = json.loads((OUT/'isolation_completion.json').read_text())
    assert diagnosis['status'] == 'REPRODUCED_BOTH_FAILURES' and isolation['status'] == 'PASS'
    write(OUT/'repair_protocol.json', dict(gate=GATE, source_files=sources(), gpu_uuid=UUID,
        base_commit='38ce71b5391bbffb5a1f0031f2cc1ab0c4a31b70',
        evidence_sha256=sha(OUT/'isolation_completion.json'),
        bindings_sha256=sha(OUT/'input_bindings.json'), manifest_sha256=sha(OUT/'sample_manifest.json'),
        numerical_only=True, formal_dtype='bfloat16', attention='eager',
        repair='Transient QK Query zero rows; existing AV alignment retained',
        no_visual_hidden_or_persistent_visual_KV_restored=True,
        regressions=[r['key'] for group in samples() for r in group]))


def verify(model, processor):
    assert not (OUT/'independent_verification').exists()
    previous_pid = json.loads((OUT/'diagnosis_completion.json').read_text())['pid']
    assert os.getpid() != previous_pid
    bindings = json.loads((OUT/'input_bindings.json').read_text())
    hooks = hook_counts(model)
    eager, matmul = qm.eager_attention_forward, torch.matmul
    weights = [(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
    rows = []
    start = time.time()
    for sample in [r for group in samples() for r in group]:
        inputs, layout = prepare(processor, sample)
        assert bindings[sample['key']] == layout
        folder = OUT/'independent_verification'/sample['key'].replace(':','_')
        write(folder/'started.json', dict(key=sample['key'], pid=os.getpid(), time=time.time()))
        records, compact, operators = {}, {}, {}
        for name in ['Native','Noop','Dense','Physical']:
            with ExitStack() as stack:
                if name in ['Dense','Physical']:
                    compact[name] = CompactTrace(model, layout)
                    stack.enter_context(compact[name].context())
                    if sample['key'] in [r['key'] for r in samples()[0]]:
                        operators[name] = OperatorTrace(model, layout)
                        stack.enter_context(operators[name].context())
                if name == 'Physical':
                    result, vectors = fixed_generate(model, processor, inputs, layout, 'C', audit=True,
                        capture=True, max_tokens=sample['max_new_tokens'])
                else:
                    result, vectors = call(model, processor, inputs, layout, name, sample['max_new_tokens'])
            assert hooks == hook_counts(model) and qm.eager_attention_forward is eager and torch.matmul is matmul
            assert all(bool(torch.isfinite(v).all()) for v in vectors)
            assert {k:tensor_digest(v) for k,v in inputs.items()} == layout['input_tensor_hashes']
            records[name] = result, vectors
            write(folder/(name+'.json'), result)
            np.savez_compressed(folder/(name+'_logits.npz'), logits=torch.stack(vectors).numpy(), token_ids=result['token_ids'])
            print('VERIFY', sample['key'], name, result['token_ids'], flush=True)
        assert strict_equal(records['Native'], records['Noop'])
        comparisons = compare(*records['Dense'], *records['Physical'])
        table(folder/'logits_comparison.csv', comparisons)
        layer_rows, semantics = compare_traces(compact['Dense'], compact['Physical'])
        table(folder/'layer_comparison.csv', layer_rows)
        write(folder/'semantics.json', semantics)
        if operators:
            table(folder/'operator_comparison.csv', compare_records(operators['Dense'], operators['Physical']))
        structural = validate_semantics(records['Physical'][0], layout)
        audit = records['Physical'][0]['audit']
        assert len(audit['QK_query_row_alignment']) == len(audit['AV_query_row_alignment']) == 14
        assert all(r['persistent_dummy_KV']==0 and r['image_hidden_rows']==0 for r in audit['QK_query_row_alignment'])
        history = OLD/'numerical'/sample['key'].replace(':','_')
        history_equal = {}
        if history.exists():
            for name in ['Native','Dense']:
                with np.load(history/(name+'_logits.npz')) as old:
                    history_equal[name] = np.array_equal(old['logits'], torch.stack(records[name][1]).numpy())
                assert history_equal[name], 'Frozen baseline changed'
        passed = bool(comparisons) and all(r['pass'] for r in comparisons) and all(
            semantics[k] for k in ['all_layer_steps_captured','all_mrope_positions_equal','deepstack_equal','cache_positions_equal'])
        entry = dict(key=sample['key'], status='PASS' if passed else 'FAIL',
            native_noop_strict=True, history_equal=history_equal, gpu_uuid=UUID, pid=os.getpid(),
            mean_abs=max(r['mean_abs'] for r in comparisons), max_abs=max(r['max_abs'] for r in comparisons),
            tokens_equal=records['Dense'][0]['token_ids']==records['Physical'][0]['token_ids'],
            real_decode_steps=len(comparisons)-1, structural=structural,
            first_hidden_difference=next((r for r in layer_rows if not r['equal']),None),
            peak_allocated=records['Physical'][0]['peak_allocated'],
            extra_QK_flops=audit['extra_QK_query_alignment_flops'], extra_AV_flops=audit['extra_AV_query_alignment_flops'],
            max_temporary_QK_output_bytes=max(r['output_buffer_bytes'] for r in audit['QK_query_row_alignment']))
        write(folder/'validation.json', entry)
        rows.append(entry)
        assert weights == [(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
        if not passed:
            write(OUT/'verification_completion.json',dict(status='FAIL', pid=os.getpid(), rows=rows))
            raise RuntimeError('Scoped QK repair verification failed; no FAST-TRACK unlock')
        print('PASS', sample['key'], entry['mean_abs'], entry['max_abs'], flush=True)
        del records, compact, operators, inputs, result, vectors
        gc.collect()
    write(OUT/'verification_completion.json',dict(status='PASS', expected=9, completed=len(rows),
        pid=os.getpid(), diagnosis_pid=previous_pid, elapsed_seconds=time.time()-start, rows=rows,
        gate=GATE, full59_status='NOT_COMPLETED', fasttrack24_status='NOT_YET_RETESTED'))


def gpu():
    protocol = json.loads((OUT/'repair_protocol.json').read_text())
    assert protocol['source_files'] == sources()
    assert protocol['gate'] == GATE and protocol['bindings_sha256'] == sha(OUT/'input_bindings.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUID
    before = resource()
    assert UUID not in before['processes']
    line = next(l for l in before['gpus'].splitlines() if UUID in l)
    assert int(line.split(',')[-2].split()[0]) > 23500
    write(OUT/'verify_gpu_before.json',before)
    model = None
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    try:
        processor = AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        verify(model,processor)
    except BaseException as exc:
        write(OUT/'verification_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','gpu'])
    args = parser.parse_args()
    freeze() if args.mode == 'freeze' else gpu()
