"""Single-GPU capture/isolation before changing the frozen BF16 executor."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from ..qwen3vl_svh_numerical_repair.run import GATE, MODEL, compare, metrics, source_files
from ..qwen3vl_svh_numerical_repair.engine import generate
from ..qwen3vl_svh_l28_repair.engine import generate as old_generate, restore_query_rows
from ..qwen3vl_svh_l28_repair.run import OperatorTrace, compare_records, samples as av_samples
from ..qwen3vl_svh_preflight48.freeze import prepare, sha, table, write
from ..qwen3vl_svh_preflight48.run import resource, strict_equal
from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'vlm_research/results/qwen3vl_svh_fasttrack_sixbench'
OUT = ROOT / 'vlm_research/results/qwen3vl_svh_qk_repair'
PRIVATE = Path('/data/lk/ATE-resources/experiments/qwen3vl_svh_qk_repair')
UUID = 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185'
TARGETS = ['TextVQA:36053', 'POPE:adversarial:205']


def sources():
    dirs = ['qwen3vl_svh_l28_repair', 'qwen3vl_svh_qk_repair', 'qwen3vl_svh_preflight48']
    return {**source_files(), **{str(p.relative_to(ROOT)):sha(p) for d in dirs
            for p in (ROOT / 'vlm_research' / d).glob('*.py')}}


def samples():
    cases = json.loads((OLD / 'numerical_cases.json').read_text())
    indexed = {c['sample']['key']:c['sample'] for c in cases}
    targets = [indexed[k] for k in TARGETS]
    regressions = av_samples() + [indexed['ScienceQA-IMG:10391']]
    assert len(targets + regressions) == len({r['key'] for r in targets + regressions}) == 9
    return targets, regressions


def prepare_cpu():
    assert not torch.cuda.is_initialized()
    assert not (OUT / 'diagnostic_protocol.json').exists()
    torch.set_num_threads(1)
    old = json.loads((OLD / 'protocol.json').read_text())
    for name, expected in old['frozen_files'].items():
        assert sha(OLD / name) == expected
    targets, regressions = samples()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    bindings = {}
    for row in targets + regressions:
        inputs, layout = prepare(processor, row)
        bindings[row['key']] = layout
        if row['key'] in TARGETS:
            assert layout == json.loads((OLD / 'input_bindings.json').read_text())[row['key']]
        del inputs
    write(OUT / 'sample_manifest.json', dict(targets=targets, regressions=regressions))
    write(OUT / 'input_bindings.json', bindings)
    write(OUT / 'diagnostic_protocol.json', dict(base_commit='38ce71b5391bbffb5a1f0031f2cc1ab0c4a31b70',
        gate=GATE, dtype='bfloat16', attention='eager', gpu_uuid=UUID, source_files=sources(),
        targets=TARGETS, regressions=[r['key'] for r in regressions],
        manifest_sha256=sha(OUT / 'sample_manifest.json'), bindings_sha256=sha(OUT / 'input_bindings.json'),
        visual_budget_unchanged=True, no_quality_evaluation=True, other_gpu_unused=True))
    print('CPU FREEZE PASS', len(bindings), flush=True)


class QKProbe(OperatorTrace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qk = {}

    @contextmanager
    def context(self):
        with super().context():
            eager = qm.eager_attention_forward

            def observed(module, q, k, v, attention_mask=None, **kwargs):
                layer = getattr(module, 'layer_idx', -1)
                if not (self.step == 0 and layer in [21, 22, 25, 26, 27]):
                    return eager(module, q, k, v, attention_mask, **kwargs)
                native = torch.matmul
                calls = 0

                def matmul(a, b, *args, **kw):
                    nonlocal calls
                    calls += 1
                    out = native(a, b, *args, **kw)
                    if calls == 1:
                        positions = self.logical(a.shape[-2])
                        indices = [j for j,p in enumerate(positions) if p not in self.images]
                        ix = torch.tensor(indices, device=out.device)
                        keys = self.logical(out.shape[-1])
                        self.qk[layer] = dict(q=a.detach().cpu().clone(), kt=b.detach().cpu().clone(),
                            raw_text_qk=out.index_select(-2, ix).detach().cpu().clone(),
                            mask=attention_mask.index_select(-2, ix).detach().cpu().clone(),
                            query_positions=positions, key_positions=keys,
                            text_positions=[positions[j] for j in indices],
                            q_stride=list(a.stride()), kt_stride=list(b.stride()),
                            dtype=str(a.dtype))
                    return out

                with patch.object(torch, 'matmul', matmul):
                    return eager(module, q, k, v, attention_mask, **kwargs)

            with patch.object(qm, 'eager_attention_forward', observed):
                yield self


def call(model, processor, inputs, layout, name, cap, trace=None):
    with ExitStack() as stack:
        if trace:
            stack.enter_context(trace.context())
        if name == 'Physical':
            return old_generate(model, processor, inputs, layout, 'C', audit=True, capture=True, max_tokens=cap)
        return generate(model, processor, inputs, layout, 'F' if name == 'Dense' else 'N',
            audit=True, capture=True, noop=name == 'Noop', max_tokens=cap)


def diagnose(model, processor):
    assert not (OUT / 'diagnosis').exists()
    bindings = json.loads((OUT / 'input_bindings.json').read_text())
    hooks = hook_counts(model)
    eager, matmul = qm.eager_attention_forward, torch.matmul
    weights = [(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
    summary = []
    for row in samples()[0]:
        inputs, layout = prepare(processor, row)
        assert layout == bindings[row['key']]
        folder = OUT / 'diagnosis' / row['key'].replace(':', '_')
        records, traces = {}, {}
        for name in ['Native', 'Noop', 'Dense', 'Physical']:
            trace = QKProbe(model, layout) if name in ['Dense', 'Physical'] else None
            result, vectors = call(model, processor, inputs, layout, name, row['max_new_tokens'], trace)
            assert hooks == hook_counts(model) and qm.eager_attention_forward is eager and torch.matmul is matmul
            assert {k:tensor_digest(v) for k,v in inputs.items()} == layout['input_tensor_hashes']
            write(folder / (name + '.json'), result)
            np.savez_compressed(folder / (name + '_logits.npz'), logits=torch.stack(vectors).numpy(), token_ids=result['token_ids'])
            records[name] = result, vectors
            if trace:
                traces[name] = trace
            print('DIAGNOSE', row['key'], name, result['token_ids'], flush=True)
        assert strict_equal(records['Native'], records['Noop'])
        comparisons = compare(*records['Dense'], *records['Physical'])
        assert not all(r['pass'] for r in comparisons), 'Expected frozen failure did not reproduce'
        # The same old Physical result must reproduce the previous task exactly.
        previous = OLD / 'numerical' / row['key'].replace(':', '_')
        for name in ['Native', 'Dense', 'Physical']:
            with np.load(previous / (name + '_logits.npz')) as old:
                assert np.array_equal(old['logits'], torch.stack(records[name][1]).numpy())
        table(folder / 'logits_comparison.csv', comparisons)
        operators = compare_records(traces['Dense'], traces['Physical'])
        table(folder / 'operator_comparison.csv', operators)
        first = next(r for r in operators if not r['equal'] and r['operator'] != 'mask')
        write(folder / 'first_difference.json', first)
        PRIVATE.mkdir(parents=True, exist_ok=True)
        path = PRIVATE / (row['key'].replace(':', '_') + '_qk.pt')
        assert not path.exists()
        torch.save({name:trace.qk for name,trace in traces.items()}, path)
        write(folder / 'qk_capture.json', dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
        summary.append(dict(key=row['key'], first=first, worst_mean_abs=max(r['mean_abs'] for r in comparisons),
            worst_max_abs=max(r['max_abs'] for r in comparisons), native_noop_strict=True))
        assert weights == [(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
        del records, traces, trace, inputs, vectors, result
        gc.collect()
    write(OUT / 'diagnosis_completion.json', dict(status='REPRODUCED_BOTH_FAILURES', pid=os.getpid(), rows=summary))


def isolate():
    assert not (OUT / 'isolation_completion.json').exists()
    results = []
    for row in samples()[0]:
        folder = OUT / 'diagnosis' / row['key'].replace(':', '_')
        capture = json.loads((folder / 'qk_capture.json').read_text())
        assert sha(capture['path']) == capture['sha256']
        probes = torch.load(capture['path'], map_location='cpu', weights_only=True)
        first = json.loads((folder / 'first_difference.json').read_text())
        for layer in [22, 25, 26]:
            d, p = probes['Dense'][layer], probes['Physical'][layer]
            positions = p['text_positions']
            dix = [d['query_positions'].index(pos) for pos in positions]
            pix = [p['query_positions'].index(pos) for pos in positions]
            text_keys = [j for j,pos in enumerate(p['key_positions']) if pos in positions]
            dq = d['q'].index_select(-2, torch.tensor(dix))
            pq = p['q'].index_select(-2, torch.tensor(pix))
            dk = d['kt'][..., text_keys]
            pk = p['kt'][..., text_keys]
            inputs_equal = torch.equal(dq, pq) and torch.equal(dk, pk)
            if not inputs_equal:
                assert layer > first['layer'], 'Upstream difference predates alleged QK root'
                continue
            q = pq.cuda()
            kt = p['kt'].cuda()
            expanded, ix = restore_query_rows(q, positions, d['q'].shape[-2])
            compact = torch.matmul(q, kt).cpu()
            restored = torch.matmul(expanded, kt).index_select(-2, ix).cpu()
            dense_same_k = torch.matmul(d['q'].cuda(), kt).index_select(-2, ix).cpu()
            allowed = p['mask'][..., text_keys] > -1e20

            def canonical(x):
                return torch.where(allowed, x[..., text_keys], 0.)

            old_d, old_p = canonical(d['raw_text_qk']), canonical(p['raw_text_qk'])
            compact, restored, dense_same_k = canonical(compact), canonical(restored), canonical(dense_same_k)
            fp32 = canonical(torch.matmul(q.float(), kt.float()).cpu())
            item = dict(key=row['key'], layer=layer, q_text_equal=True, k_text_equal=True,
                compact_q_shape=list(q.shape), restored_q_shape=list(expanded.shape), kt_shape=list(kt.shape),
                dense_q_stride=d['q_stride'], physical_q_stride=p['q_stride'],
                dense_kt_stride=d['kt_stride'], physical_kt_stride=p['kt_stride'],
                dtype=str(q.dtype), compact_matches_old_physical=torch.equal(compact, old_p),
                restored_matches_old_dense=torch.equal(restored, old_d),
                dense_same_physical_k_matches_dense=torch.equal(dense_same_k, old_d),
                before=metrics(old_d, compact), after=metrics(old_d, restored),
                compact_vs_fp32_diagnostic=metrics(fp32, compact), dense_vs_fp32_diagnostic=metrics(fp32, old_d))
            assert item['compact_matches_old_physical'] and item['restored_matches_old_dense'] and item['dense_same_physical_k_matches_dense']
            out = OUT / 'isolated_QK' / (row['key'].replace(':', '_') + '_L' + str(layer))
            write(out.with_suffix('.json'), item)
            np.savez_compressed(out.with_suffix('.npz'), dense_qk=old_d.float().numpy(), physical_qk=old_p.float().numpy(),
                compact_replay=compact.float().numpy(), restored_replay=restored.float().numpy(),
                fp32_diagnostic=fp32.float().numpy(), text_q=dq.float().numpy(), text_kt=dk.float().numpy())
            print('ISOLATE', json.dumps(item), flush=True)
            results.append(item)
        del probes
        gc.collect()
    assert all(any(r['key']==row['key'] and r['layer']==json.loads((OUT/'diagnosis'/row['key'].replace(':','_')/'first_difference.json').read_text())['layer'] for r in results) for row in samples()[0])
    write(OUT / 'isolation_completion.json', dict(status='PASS', pid=os.getpid(), model_loaded=False,
        result='Identical real Q/K; compact Query geometry reproduces Physical, full zero Query geometry reproduces Dense', rows=results))


def gpu(mode):
    protocol = json.loads((OUT / 'diagnostic_protocol.json').read_text())
    assert protocol['source_files'] == sources()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUID
    before = resource()
    assert UUID not in before['processes']
    line = next(l for l in before['gpus'].splitlines() if UUID in l)
    assert '3090' in line and int(line.split(',')[-2].split()[0]) > 23500
    write(OUT / (mode + '_gpu_before.json'), before)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    if mode == 'isolate':
        isolate()
        return
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = None
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager', device_map={'':'cuda:0'}).eval()
        diagnose(model, processor)
    except BaseException as exc:
        write(OUT / (mode + '_failure.json'), dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'diagnose', 'isolate'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare_cpu()
    else:
        gpu(args.mode)
