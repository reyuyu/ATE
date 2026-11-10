"""Stop-on-failure single-GPU acceptance before any 48-question evaluation."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
import numpy as np
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest
from ..qwen3vl_svh_numerical_repair.engine import generate
from ..qwen3vl_svh_numerical_repair.run import GATE, compare, metrics, source_files
from .freeze import MODEL, OUT, ROOT, TASKS, digest, prepare, sha, table, write


def resource():
    def cmd(args):
        return subprocess.check_output(args, text=True).strip()
    return dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
                visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpus=cmd(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu', '--format=csv']),
                processes=cmd(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv']))


class CompactTrace:
    """Retain matched layer-output rows and position/cache metadata, not attention matrices."""
    def __init__(self, model, layout):
        self.model = model
        self.d = model.model.language_model
        self.n = layout['input_length']
        self.images = set(layout['image_positions'])
        self.text = [i for i in range(self.n) if i not in self.images]
        self.step = 0
        self.outputs = {}
        self.positions = {}
        self.caches = {}
        self.deepstack = []

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            def begin(m, args, kw):
                cp = kw['cache_position']
                self.step = 0 if int(cp[0]) == 0 else int(cp[0]) - self.n + 1
            handle = self.model.register_forward_pre_hook(begin, with_kwargs=True)
            stack.callback(handle.remove)
            for layer_index, layer in enumerate(self.d.layers):
                state = {}
                def start(m, args, kw, i=layer_index, state=state):
                    state['cache'] = kw['past_key_values']
                    cp = kw['cache_position'].detach().cpu().tolist()
                    ix = [j for j, pos in enumerate(cp) if pos not in self.images]
                    state['ix'] = ix
                    state['positions'] = [cp[j] for j in ix]
                    cos, sin = kw['position_embeddings']
                    self.positions[(self.step, i)] = dict(
                        logical=state['positions'],
                        cos=tensor_digest(cos[:, ix]), sin=tensor_digest(sin[:, ix]))
                handle = layer.register_forward_pre_hook(start, with_kwargs=True)
                stack.callback(handle.remove)
                def end(m, args, out, i=layer_index, state=state):
                    assert bool(torch.isfinite(out).all()), 'Nonfinite layer output'
                    self.outputs[(self.step, i)] = out[:, state['ix']].detach().cpu().clone()
                    cache = state['cache']
                    if hasattr(cache, 'entries'):
                        pair = cache.entries[i]
                        pair = None if pair is None else [v[..., :cache.used[i], :] for v in pair]
                    else:
                        pair = [cache.layers[i].keys, cache.layers[i].values]
                    if pair is None:
                        self.caches[(self.step, i)] = None
                    else:
                        positions = (list(range(self.n + self.step)) if pair[0].shape[-2] == self.n + self.step
                                     else self.text + list(range(self.n, self.n + self.step)))
                        assert len(positions) == pair[0].shape[-2]
                        ix = [j for j, pos in enumerate(positions) if pos not in self.images]
                        self.caches[(self.step, i)] = dict(positions=[positions[j] for j in ix],
                            k=tensor_digest(pair[0][..., ix, :]), v=tensor_digest(pair[1][..., ix, :]))
                    state.clear()
                handle = layer.register_forward_hook(end)
                stack.callback(handle.remove)
            deep = self.d._deepstack_process
            def observed_deep(hidden, mask, feature):
                self.deepstack.append(dict(feature=tensor_digest(feature), mask=tensor_digest(mask)))
                return deep(hidden, mask, feature)
            stack.enter_context(patch.object(self.d, '_deepstack_process', observed_deep))
            yield self


def compare_traces(dense, physical):
    rows = []
    for key in sorted(dense.outputs.keys() & physical.outputs.keys()):
        rows.append(dict(step=key[0], layer=key[1], **metrics(dense.outputs[key], physical.outputs[key])))
    same_keys = dense.outputs.keys() == physical.outputs.keys()
    same_positions = dense.positions == physical.positions
    same_deepstack = dense.deepstack == physical.deepstack
    cache_rows = []
    for key, p in physical.caches.items():
        if p is not None:
            d = dense.caches[key]
            cache_rows.append(dict(step=key[0], layer=key[1], positions_equal=d['positions'] == p['positions'],
                                   k_equal=d['k'] == p['k'], v_equal=d['v'] == p['v']))
    return rows, dict(all_layer_steps_captured=same_keys, all_mrope_positions_equal=same_positions,
                      deepstack_equal=same_deepstack, cache_rows=cache_rows,
                      cache_positions_equal=all(r['positions_equal'] for r in cache_rows))


def validate_semantics(result, layout):
    audit = result['audit']
    n, ni = layout['input_length'], layout['image_tokens']
    for row in audit['layers']:
        expected = n - ni if row['step'] == 0 and row['layer'] >= 22 else n if row['step'] == 0 else 1
        assert row['hidden_rows'] == expected
        assert len(row['logical_positions']) == expected
    assert audit['deepstack'] == [0, 1, 2]
    for row in audit['probability']:
        assert row['forbidden_probability'] == 0
    for row in audit['decode_temporary_zero_keys']:
        assert row['dummy_probability'] == 0 and row['persistent_dummy_KV'] == 0
    kv = audit['kv']
    for row in kv['layers']:
        if row['layer'] in range(1, 7):
            assert row['live_tokens'] == 0
        else:
            assert row['live_tokens'] == n - ni + result['generated_tokens'] - 1
    assert len(audit['aligned_linears']) == 14 * 7
    assert all(not r['dense_projection_executed'] for r in audit['aligned_linears'])
    assert len(audit['transient_prefill_columns']) == 14
    return dict(status='PASS', image_tokens=ni, persistent_visual_tokens=0,
                live_KV_bytes=kv['live_bytes'], capacity_KV_bytes=kv['capacity_bytes'],
                zero_alignment_decode_extra_flops=sum(r['extra_QK_AV_flops'] for r in audit['decode_temporary_zero_keys']),
                zero_alignment_prefill_extra_flops=14 * 4 * 32 * (n-ni) * ni * 128,
                zero_alignment_peak_live_KV_buffer_bytes=2 * 8 * (n + result['generated_tokens'] - 1) * 128 * 2,
                reported_memory_is_instrumented_not_latency_benchmark=True)


def freeze_check():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert protocol['gate'] == GATE
    current = {**source_files(), **{str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')}}
    assert current == protocol['source_files'], 'Execution source changed after CPU freeze'
    for name, expected in protocol['frozen_files'].items():
        assert sha(OUT / name) == expected, ('Frozen file changed', name)
    return protocol


def strict_equal(a, b):
    return (a[0]['token_ids'] == b[0]['token_ids'] and len(a[1]) == len(b[1])
            and all(torch.equal(x, y) for x, y in zip(a[1], b[1])))


def numerical(model, processor):
    cases = json.loads((OUT / 'numerical_cases.json').read_text())
    bindings = json.loads((OUT / 'input_bindings.json').read_text())
    baseline_hooks = hook_counts(model)
    weights = [(n, p.data_ptr(), p._version, tuple(p.shape), str(p.dtype)) for n, p in model.named_parameters()]
    completed = []
    for index, case in enumerate(cases):
        row = case['sample']
        folder = OUT / 'numerical' / row['key'].replace(':', '_')
        assert not folder.exists(), 'Existing numerical attempt must not be overwritten'
        write(folder / 'started.json', dict(index=index, sample=row, reasons=case['reasons'], time=time.time()))
        inputs, layout = prepare(processor, row)
        assert layout == bindings[row['key']]
        records, traces = {}, {}
        for name, kind in [('Native', 'N'), ('Noop', 'N'), ('Dense', 'F'), ('Physical', 'C')]:
            trace = CompactTrace(model, layout) if name in ('Dense', 'Physical') else None
            write(OUT / 'current.json', dict(stage='NUMERICAL', key=row['key'], condition=name,
                                           image_tokens=layout['image_tokens'], input_length=layout['input_length']))
            with ExitStack() as stack:
                if trace:
                    stack.enter_context(trace.context())
                result, vectors = generate(model, processor, inputs, layout, kind, repaired=name == 'Physical',
                    audit=True, capture=True, noop=name == 'Noop', max_tokens=row['max_new_tokens'])
            assert hook_counts(model) == baseline_hooks, 'Hook leak'
            assert all(bool(torch.isfinite(v).all()) for v in vectors)
            assert {k: tensor_digest(v) for k, v in inputs.items()} == layout['input_tensor_hashes']
            write(folder / (name + '.json'), result)
            np.savez_compressed(folder / (name + '_logits.npz'), logits=torch.stack(vectors).numpy(),
                                token_ids=np.asarray(result['token_ids'], dtype=np.int64))
            records[name] = result, vectors
            if trace:
                traces[name] = trace
            print('NUMERICAL', index + 1, '/', len(cases), row['key'], name,
                  'tokens', len(result['token_ids']), 'peakMiB', round(result['peak_allocated']/2**20, 1), flush=True)
            if name == 'Noop':
                assert strict_equal(records['Native'], records['Noop']), 'Native/Noop strict logits failure'
        with torch.inference_mode():
            hf = model.generate(**{k:v.cuda() for k,v in inputs.items()}, do_sample=False, num_beams=1,
                repetition_penalty=1., max_new_tokens=row['max_new_tokens'], use_cache=True, logits_to_keep=1,
                return_dict_in_generate=True, output_logits=True, eos_token_id=[151645,151643], pad_token_id=151643)
        native = records['Native']
        hf_equal = (hf.sequences[0,layout['input_length']:].tolist() == native[0]['token_ids']
                    and len(hf.logits) == len(native[1])
                    and all(torch.equal(x, y[0].float().cpu()) for x,y in zip(native[1], hf.logits)))
        del hf
        assert hf_equal, 'Native custom-loop/HF generation mismatch'
        comparisons = compare(*records['Dense'], *records['Physical'])
        table(folder / 'logits_comparison.csv', comparisons)
        layer_rows, semantics = compare_traces(traces['Dense'], traces['Physical'])
        table(folder / 'layer_comparison.csv', layer_rows)
        write(folder / 'trace_metadata.json', dict(semantics=semantics,
              dense_positions={str(k):v for k,v in traces['Dense'].positions.items()},
              physical_positions={str(k):v for k,v in traces['Physical'].positions.items()},
              dense_deepstack=traces['Dense'].deepstack, physical_deepstack=traces['Physical'].deepstack))
        structural = validate_semantics(records['Physical'][0], layout)
        passed = (all(r['pass'] for r in comparisons) and bool(comparisons)
                  and semantics['all_layer_steps_captured'] and semantics['all_mrope_positions_equal']
                  and semantics['deepstack_equal'] and semantics['cache_positions_equal'])
        first = next((r for r in layer_rows if not r['equal']), None)
        item = dict(key=row['key'], benchmark=row['benchmark'], pass_gate=passed, hf_native_strict=hf_equal,
                    native_noop_strict=True, first_layer_difference=first, structural=structural,
                    dense_generated_tokens=len(records['Dense'][0]['token_ids']),
                    physical_generated_tokens=len(records['Physical'][0]['token_ids']),
                    worst_max_abs=max(r['max_abs'] for r in comparisons),
                    worst_mean_abs=max(r['mean_abs'] for r in comparisons),
                    peak_allocated=records['Physical'][0]['peak_allocated'],
                    peak_reserved=records['Physical'][0]['peak_reserved'], reasons=case['reasons'])
        write(folder / 'validation.json', item)
        completed.append(item)
        write(OUT / 'numerical_progress.json', completed)
        print('GATE', row['key'], 'PASS' if passed else 'FAIL', flush=True)
        assert passed, 'Enlarged Dense/Physical numerical gate failure; no 48-question evaluations'
        assert weights == [(n,p.data_ptr(),p._version,tuple(p.shape),str(p.dtype)) for n,p in model.named_parameters()]
        del records, traces, vectors, result, inputs
        gc.collect()
    long_steps = max(r['physical_generated_tokens'] - 1 for r in completed)
    complete = len(completed) == len(cases) and long_steps >= 16
    completion = dict(status='PASS' if complete else 'INCOMPLETE_LONG_DECODE_COVERAGE',
                      numerical_cases_completed=len(completed), numerical_cases_required=len(cases),
                      longest_natural_incremental_decode=long_steps, long_decode_required=16,
                      tasks_covered=sorted(set(r['benchmark'] for r in completed)),
                      pretest_allowed=complete, full_evaluation_allowed=False)
    write(OUT / 'numerical_completion.json', completion)
    return completion


def main():
    freeze_check()
    before = resource()
    write(OUT / 'gpu_before.json', before)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible, 'Expose exactly one selected GPU UUID'
    gpu = subprocess.check_output(['nvidia-smi', '-i', visible, '--query-gpu=name,memory.used,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert '3090' in gpu[0] and int(gpu[1]) < 100 and int(gpu[2]) > 23500 and int(gpu[3]) == 0
    procs = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert visible not in procs, 'Selected GPU already has a compute process'
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = None
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation='eager', device_map={'':'cuda:0'}).eval()
        completion = numerical(model, processor)
        if completion['pretest_allowed']:
            from .score import evaluate
            evaluate(model, processor)
    except BaseException as exc:
        write(OUT / 'failure.json', dict(status='STOPPED', error=repr(exc), traceback=traceback.format_exc(),
              current=json.loads((OUT/'current.json').read_text()) if (OUT/'current.json').exists() else None,
              gpu_uuid=visible, full_evaluation_started=False, pretest_started=(OUT/'pretest').exists(),
              cuda_allocated=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0,
              cuda_reserved=torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0))
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        write(OUT / 'gpu_after_in_process.json', resource())


if __name__ == '__main__':
    main()
