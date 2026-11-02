"""Frozen six-condition visual-source generation; no relay or output bypass."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import gc
import os
from pathlib import Path
import time
import traceback

import torch

from .common import ROOT, TASKS, IMODEL, read, write, sha, digest, assign, preflight, command
from .engine import InternPrefill, masked, metrics
from .scan import load_intern, environment
from ..internvl3_smoke.engine import hooks_count, tensor_hash, official_preprocessor

OUT = ROOT / 'vlm_research/results/qwen_internvl_sparse_visual_3000'
PARENT = ROOT / 'vlm_research/results/visual_pathways_seven_hour'
RECOVERY = ROOT / 'vlm_research/results/internvl_native_replay_recovery'
GPUS = (0, 2, 3)
KEEP = {
    'qwen': {'Sparse K1': [26], 'Sparse K3': [26, 23, 24],
             'Sparse K8': [26, 23, 24, 21, 12, 25, 27, 10]},
    'internvl': {'Sparse K1': [19], 'Sparse K3': [19, 9, 16],
                'Sparse K8': [19, 9, 16, 14, 18, 21, 3, 2]},
}
LOW = {'qwen': [0, 1, 3, 7, 13, 30, 32, 35],
       'internvl': [4, 17, 20, 23, 24, 25, 26, 27]}


def conditions(model):
    layers = 36 if model == 'qwen' else 28
    return {'Full': [], 'Only Text': list(range(layers)),
            **{name: [i for i in range(layers) if i not in keep]
               for name, keep in KEEP[model].items()}, 'Low-8 OFF': LOW[model]}


def sources():
    paths = [Path(__file__), Path(__file__).with_name('test_sparse_accuracy.py')]
    paths += [ROOT/'vlm_research/visual_pathways'/n for n in ('common.py', 'engine.py', 'scan.py')]
    paths += list((ROOT/'vlm_research/internvl3_smoke').glob('*.py'))
    paths += [ROOT/'vlm_research/qwen3vl_early_windows/experiment.py',
              ROOT/'vlm_research/qwen3vl_keep_l26/run.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def freeze():
    assert not (OUT/'protocol.json').exists()
    original = read(PARENT/'sample_manifest.json')['samples']
    fields = ('task', 'question_id', 'image_id', 'image_path', 'image_sha256', 'question', 'answer')
    samples = [{k: s[k] for k in fields} for s in original]
    assert len(samples) == 3000 and len({s['image_sha256'] for s in samples}) == 2409
    assert all(sum(s['task'] == t for s in samples) == 500 for t in TASKS)
    for s in samples:
        assert sha(s['image_path']) == s['image_sha256']
    shards = assign(samples, 3)
    assert [len(s) for s in shards] == [1000]*3
    owners = {}
    for gpu, rows in zip(GPUS, shards):
        for s in rows:
            assert owners.setdefault(s['image_sha256'], gpu) == gpu
    write(OUT/'sample_manifest.json', dict(samples=samples, questions=3000, image_clusters=2409,
          parent_manifest_sha256=sha(PARENT/'sample_manifest.json')))
    write(OUT/'shards.json', {str(gpu): [s['task']+'__'+s['question_id'] for s in rows]
                            for gpu, rows in zip(GPUS, shards)})
    from ..qwen3vl_early_windows.experiment import MODEL
    verified = {}
    for name, modelpath, manifest in [
        ('qwen', MODEL, ROOT/'vlm_research/results/qwen3vl_image_source_240/model_manifest.json'),
        ('internvl', IMODEL, PARENT/'internvl_model_manifest.json')]:
        entries = read(manifest)['files']
        for entry in entries:
            assert sha(modelpath/entry['path']) == entry['sha256'], (name, entry['path'])
        verified[name] = dict(passed=True, manifest_sha256=sha(manifest), files=len(entries))
    write(OUT/'model_validation.json', verified)
    write(OUT/'protocol.json', dict(time=datetime.now().astimezone().isoformat(),
        base_commit=command('git', 'rev-parse', 'HEAD'), gpu_ids=GPUS, prohibited_gpu=1,
        source_sha256=sources(), manifest_sha256=sha(OUT/'sample_manifest.json'),
        shards_sha256=sha(OUT/'shards.json'),
        model_revisions={'qwen': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', 'internvl': IMODEL.name},
        conditions={m: conditions(m) for m in KEEP}, dtype='bfloat16', attention='eager',
        intervention='Only original P-1 and every actual incremental Decode query -> original Image keys; additive -inf before native softmax',
        other_prefill='Native at every layer, including visual tokens; NO relay OFF',
        qwen_deepstack='Native', internvl=dict(max_num=1, image_tokens=256, logits_to_keep=1),
        greedy=True, max_new_tokens=32, cache='Fresh independent DynamicCache per condition',
        full_reuse=False, full_reason='Generate current paired Full for all 3000; historical scan logits binding checked per question',
        scoring='Existing pinned GQA exact comparison: decoded.strip().rstrip(".").lower() == original answer',
        bootstrap=dict(seed=20261102, repetitions=10000, cluster='Image SHA256', estimator='question-weighted paired'),
        exploratory=True, selection_used_same_3000=True, no_independent_noninferiority_claim=True,
        history_exclusions=['P-1-only generation', 'Relay OFF generation']))
    print('FROZEN: 3000 questions, 2409 image clusters, 6 conditions/model', flush=True)


def verify():
    p = read(OUT/'protocol.json')
    assert p['source_sha256'] == sources()
    assert p['manifest_sha256'] == sha(OUT/'sample_manifest.json')
    assert p['shards_sha256'] == sha(OUT/'shards.json')
    return p


class GenerationMask:
    """Keep source hooks active for prefill P-1 AND all cached decode queries."""
    def __init__(self, lm, layers, n, image, off, cache, noop=False, capture=False):
        self.lm, self.layers, self.n, self.image = lm, layers, n, image
        self.off, self.cache, self.noop, self.capture = set(off), cache, noop, capture
        assert self.off <= set(range(len(layers)))
        assert image and len(image) == len(set(image)) and max(image) < n-1
        self.calls, self.audits, self.prefix, self.deepstack = [], {}, {}, None

    def __enter__(self):
        self.stack = ExitStack()
        def begin(module, args, kwargs):
            step = len(self.calls)
            qlen, start = (self.n, 0) if step == 0 else (1, self.n+step-1)
            value = kwargs.get('inputs_embeds')
            if value is None:
                value = kwargs['input_ids']
            assert value.shape[1] == qlen
            assert kwargs['cache_position'].tolist() == list(range(start, start+qlen))
            assert kwargs['past_key_values'] is self.cache
            assert all(self.cache.get_seq_length(i) == start for i in range(len(self.layers)))
            if not step and self.capture and kwargs.get('deepstack_visual_embeds') is not None:
                self.deepstack = [tensor_hash(v) for v in kwargs['deepstack_visual_embeds']]
                assert kwargs['visual_pos_masks'][0].nonzero().flatten().tolist() == self.image
            if step:
                assert kwargs.get('deepstack_visual_embeds') is None
            self.calls.append(dict(step=step, qlen=qlen, cache_start=start))
        self.stack.callback(self.lm.register_forward_pre_hook(begin, with_kwargs=True).remove)
        for i, layer in enumerate(self.layers):
            if i in self.off:
                def before(module, args, kwargs, i=i):
                    step = len(self.calls)-1
                    qlen = self.n if step == 0 else 1
                    query = self.n-1 if step == 0 else 0
                    assert kwargs['hidden_states'].shape[1] == qlen
                    original = kwargs['attention_mask']
                    if self.noop:
                        changed = original
                    else:
                        assert original is not None, 'Expected native eager additive causal mask'
                        changed, count = masked(original, [query], self.image)
                        assert count == len(self.image)
                        assert int((original != changed).sum()) == count
                    audit = self.audits.setdefault(i, dict(prefill_calls=0, decode_calls=0,
                        masked_probability_max=0., row_sum_max_error=0.))
                    audit['prefill_calls' if step == 0 else 'decode_calls'] += 1
                    return args, dict(kwargs, attention_mask=changed)
                def after(module, args, output, i=i):
                    q = self.n-1 if len(self.calls) == 1 else 0
                    weights = output[1]
                    assert weights is not None
                    row = weights[0, :, q, :]
                    assert torch.isfinite(row).all() and (row >= 0).all()
                    maximum = float(row[:, self.image].max())
                    error = float((row.float().sum(-1)-1).abs().max())
                    assert self.noop or maximum == 0
                    assert error <= 2**-8+2e-6
                    audit = self.audits[i]
                    audit['masked_probability_max'] = max(audit['masked_probability_max'], maximum)
                    audit['row_sum_max_error'] = max(audit['row_sum_max_error'], error)
                self.stack.callback(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True).remove)
                self.stack.callback(layer.self_attn.register_forward_hook(after).remove)
            if self.capture:
                def observe(module, args, output, i=i):
                    if len(self.calls) == 1:
                        value = output[0] if isinstance(output, tuple) else output
                        self.prefix[i] = tensor_hash(value[:, :-1])
                self.stack.callback(layer.register_forward_hook(observe).remove)
        return self

    def __exit__(self, *_):
        self.stack.close()


@torch.inference_mode()
def generate(kind, model, tokenizer, inputs, layout, off=(), noop=False, capture=False, native=False):
    from transformers.cache_utils import DynamicCache
    n = layout['input_length']
    image = layout['image_positions'] if kind == 'qwen' else layout['positions']['image']
    lm = model.model.language_model if kind == 'qwen' else model.language_model
    layers = lm.layers if kind == 'qwen' else lm.model.layers
    cache = DynamicCache(config=model.config.text_config) if kind == 'qwen' else DynamicCache()
    baseline = hooks_count(model)
    fingerprints = {k: tensor_hash(v) for k, v in inputs.items()}
    with ExitStack() as stack:
        control = None if native else stack.enter_context(GenerationMask(lm, layers, n, image, off, cache, noop, capture))
        kwargs = dict(past_key_values=cache, use_cache=True, do_sample=False, num_beams=1,
            max_new_tokens=32, logits_to_keep=1, return_dict_in_generate=True, output_scores=True,
            output_attentions=False, output_hidden_states=False)
        if kind == 'internvl':
            eos = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
            result = lm.generate(**inputs, **kwargs, eos_token_id=eos, pad_token_id=eos)
            ids = result.sequences[0].tolist()
            eos = [eos]
        else:
            result = model.generate(**inputs, **kwargs)
            ids = result.sequences[0, n:].tolist()
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
        assert ids and len(ids) == len(result.scores) and result.past_key_values is cache
        assert all(torch.isfinite(s).all() for s in result.scores)
        assert all(cache.get_seq_length(i) == n+len(ids)-1 for i in range(len(layers)))
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if kind == 'internvl':
            text = text.split(model.conv_template.sep.strip())[0]
        text = text.strip()
        first = result.scores[0][0].detach().cpu().clone()
        if control:
            assert len(control.calls) == len(ids) and set(control.audits) == set(off)
            assert all(a['prefill_calls'] == 1 and a['decode_calls'] == len(ids)-1 for a in control.audits.values())
        value = dict(text=text, prediction=text.rstrip('.').lower(), token_ids=ids,
            eos=ids[-1] in eos, truncated=len(ids) == 32 and ids[-1] not in eos, empty=not bool(text),
            generated_tokens=len(ids), mask_audits=control.audits if control else {},
            fresh_cache=True, final_cache_length=n+len(ids)-1,
            all_step_logits_sha256=[tensor_hash(s) for s in result.scores])
        trace = dict(prefix=control.prefix, deepstack=control.deepstack) if control else {}
    assert hooks_count(model) == baseline
    assert fingerprints == {k: tensor_hash(v) for k, v in inputs.items()}
    return value, first, trace


def prepare(kind, model, processor, sample, fast=None, loader=None):
    if kind == 'internvl':
        runner = InternPrefill(model, processor, fast, sample, loader)
        return runner.inputs, runner.layout, runner
    from ..qwen3vl_early_windows.experiment import prepare as qprepare
    inputs, layout = qprepare(processor, sample)
    return inputs.to('cuda:0'), layout, None


@torch.inference_mode()
def smoke(kind, model, processor, fast, loader, sample):
    inputs, layout, runner = prepare(kind, model, processor, sample, fast, loader)
    tokenizer = processor.tokenizer if kind == 'qwen' else processor
    alloff = conditions(kind)['Only Text']
    native, logits, _ = generate(kind, model, tokenizer, inputs, layout, native=True)
    full, full_logits, reference = generate(kind, model, tokenizer, inputs, layout, capture=True)
    noop, noop_logits, _ = generate(kind, model, tokenizer, inputs, layout, alloff, noop=True)
    assert torch.equal(logits, full_logits) and torch.equal(logits, noop_logits)
    assert native['token_ids'] == full['token_ids'] == noop['token_ids']
    assert native['all_step_logits_sha256'] == full['all_step_logits_sha256'] == noop['all_step_logits_sha256']
    assert len(native['token_ids']) >= 2, 'Smoke must exercise a real incremental Decode'
    if kind == 'internvl':
        from ..internvl3_smoke.run import ChatCapture
        with ChatCapture(model) as official:
            text = model.chat(processor, runner.pixels, runner.request,
                dict(do_sample=False, num_beams=1, max_new_tokens=32))
        assert torch.equal(official.first_logits, logits)
        assert text.strip() == native['text']
    checks = []
    for name in ('Only Text', 'Sparse K1', 'Sparse K3', 'Sparse K8', 'Low-8 OFF'):
        value, _, trace = generate(kind, model, tokenizer, inputs, layout, conditions(kind)[name], capture=True)
        assert trace['prefix'] == reference['prefix']
        assert trace['deepstack'] == reference['deepstack']
        assert all(a['masked_probability_max'] == 0 and a['decode_calls'] >= 1 for a in value['mask_audits'].values())
        checks.append(dict(condition=name, non_target_prefill_exact=True, deepstack_native_exact=True,
            target_probabilities_zero=True, real_decode_masked=True, generated_tokens=len(value['token_ids'])))
    repeat, repeat_logits, _ = generate(kind, model, tokenizer, inputs, layout)
    assert repeat['all_step_logits_sha256'] == full['all_step_logits_sha256'] and torch.equal(logits, repeat_logits)
    before = hooks_count(model)
    lm = model.model.language_model if kind == 'qwen' else model.language_model
    layers = lm.layers if kind == 'qwen' else lm.model.layers
    try:
        from transformers.cache_utils import DynamicCache
        with GenerationMask(lm, layers, layout['input_length'],
            layout['image_positions'] if kind == 'qwen' else layout['positions']['image'], alloff, DynamicCache()):
            raise RuntimeError('cleanup fixture')
    except RuntimeError as error:
        assert str(error) == 'cleanup fixture'
    assert hooks_count(model) == before
    return dict(passed=True, question_id=sample['question_id'], noop_all_logits_exact=True,
        native_first_logits_max_abs=0, native_all_tokens_exact=True, cache_isolation_exact=True,
        exception_cleanup=True, checks=checks)


def history(kind, sample):
    root = PARENT if kind == 'qwen' else RECOVERY
    path = root/kind/'checkpoints'/f'{sample["task"]}__{sample["question_id"]}.json'
    record = read(path)
    assert record['complete'] and record['image_sha256'] == sample['image_sha256']
    assert record['protocol_sha256'] == sha(root/'protocol.json')
    return record


@torch.no_grad()
def worker(kind, gpu):
    p = verify()
    folder = OUT/kind/f'gpu{gpu}'
    preflight(gpu, folder)
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write(folder/'environment.json', environment())
    if kind == 'qwen':
        from ..qwen3vl_keep_l26.run import load
        model, processor = load(gpu, folder/'loader')
        fast = loader = None
    else:
        model, processor, fast = load_intern()
        loader, _ = official_preprocessor(IMODEL)
    samples = assign(read(OUT/'sample_manifest.json')['samples'], 3)[GPUS.index(gpu)]
    assert len(samples) == 1000
    started = time.perf_counter()
    footprint = hooks_count(model)
    versions = [(p._version, p.data_ptr()) for p in model.parameters()]
    validation = smoke(kind, model, processor, fast, loader, samples[0])
    validation.update(code_commit=command('git', 'rev-parse', 'HEAD'), protocol_sha256=sha(OUT/'protocol.json'))
    write(folder/'validation.json', validation)
    print('SMOKE PASS', kind, gpu, flush=True)
    while not all((OUT/kind/f'gpu{i}/validation.json').exists() for i in GPUS):
        assert not any((OUT/kind/f'gpu{i}/error.json').exists() for i in GPUS), 'Peer smoke failed'
        time.sleep(1)
    for i in GPUS:
        other = read(OUT/kind/f'gpu{i}/validation.json')
        assert other['passed'] and other['code_commit'] == validation['code_commit'] and other['protocol_sha256'] == validation['protocol_sha256']
    tokenizer = processor.tokenizer if kind == 'qwen' else processor
    formal = time.perf_counter()
    for index, sample in enumerate(samples, 1):
        key = sample['task']+'__'+sample['question_id']
        path = OUT/kind/'checkpoints'/(key+'.json')
        record = read(path) if path.exists() else dict(complete=False, **sample, sample_sha256=digest(sample),
            protocol_sha256=sha(OUT/'protocol.json'), code_commit=validation['code_commit'], physical_gpu=gpu, conditions={})
        assert record['sample_sha256'] == digest(sample) and record['protocol_sha256'] == validation['protocol_sha256']
        assert record['physical_gpu'] == gpu
        if record['complete']:
            assert set(record['conditions']) == set(conditions(kind))
            continue
        inputs, layout, runner = prepare(kind, model, processor, sample, fast, loader)
        old = history(kind, sample)
        assert digest(layout) == old['input_binding']
        native, full, _ = generate(kind, model, tokenizer, inputs, layout)
        assert tensor_hash(full.to(torch.bfloat16)) == old['full_logits_sha256'], 'Current Full differs from bound native scan'
        if 'Full' in record['conditions']:
            assert native['token_ids'] == record['conditions']['Full']['token_ids']
        else:
            native.update(metrics(full, full), correct=int(native['prediction'] == sample['answer']))
            record['conditions']['Full'] = native
        record.update(input_binding=digest(layout), full_logits_sha256=tensor_hash(full),
            historical_scan_sha256=sha((PARENT if kind == 'qwen' else RECOVERY)/kind/'checkpoints'/(key+'.json')))
        write(path, record)
        for name, off in conditions(kind).items():
            if name in record['conditions']:
                continue
            tick = time.perf_counter()
            value, first, _ = generate(kind, model, tokenizer, inputs, layout, off)
            value.update(metrics(full, first), correct=int(value['prediction'] == sample['answer']), seconds=time.perf_counter()-tick)
            record['conditions'][name] = value
            write(path, record)
        if runner:
            runner.verify()
        record['complete'] = True
        write(path, record)
        assert hooks_count(model) == footprint
        write(folder/'progress.json', dict(done=index, total=1000, seconds=time.perf_counter()-formal,
            last_sample=key, physical_gpu=gpu))
        if index % 25 == 0 or index == 1:
            print('PROGRESS', kind, gpu, index, 1000, round(time.perf_counter()-formal, 1), flush=True)
    assert versions == [(p._version, p.data_ptr()) for p in model.parameters()]
    assert hooks_count(model) == footprint
    write(folder/'complete.json', dict(passed=True, questions=1000, generations=6000,
        formal_seconds=time.perf_counter()-formal, total_seconds=time.perf_counter()-started,
        weights_unchanged=True, hooks_clean=True, no_nan_inf_oom=True,
        code_commit=validation['code_commit'], peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'worker'])
    parser.add_argument('--model', choices=['qwen', 'internvl'])
    parser.add_argument('--gpu', type=int, choices=GPUS)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        try:
            worker(args.model, args.gpu)
        except BaseException:
            write(OUT/args.model/f'gpu{args.gpu}/error.json', dict(traceback=traceback.format_exc(), time=time.time()))
            raise
