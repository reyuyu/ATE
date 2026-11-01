import argparse
from collections import Counter
from pathlib import Path
import os
import subprocess
import time
import traceback

import torch

from ..qwen3vl_early_windows.experiment import MODEL, REVISION, prepare, digest, tensor_digest, hook_counts
from ..qwen3vl_image_source.run import environment, read, sources as prior_sources, sha256, write, GPUS
from .engine import CONDITIONS, PrefillOnly, generate

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / 'vlm_research/results/qwen3vl_image_source_240'
OUT = ROOT / 'vlm_research/results/qwen3vl_keep_l26_generation_240'


def sources():
    return {**prior_sources(), 'vlm_research/qwen3vl_early_windows/analyze.py':sha256(ROOT/'vlm_research/qwen3vl_early_windows/analyze.py'),
            **{str(p.relative_to(ROOT)): sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}


def freeze():
    assert not (OUT / 'protocol.json').exists()
    manifest = read(HISTORY / 'sample_manifest.json')
    assert len(manifest['samples']) == 240
    assert set(Counter(s['task'] for s in manifest['samples']).values()) == {40}
    assert environment() == read(HISTORY / 'environment.json')
    write(OUT / 'sample_manifest.json', manifest)
    previous = read(ROOT / 'vlm_research/results/qwen3vl_early_attention_windows_240/protocol.json')
    write(OUT / 'protocol.json', dict(base_commit='f437e82f2ff887ab4d8bddc9399a892f4d7eb8f6',
        model_repo='Qwen/Qwen3-VL-8B-Instruct', revision=REVISION, model_path=str(MODEL),
        dtype='bfloat16', attention='eager', conditions=CONDITIONS,
        intervention='Original prefill P-1 only: native additive -inf Image-source mask; removed before all decode; DeepStack unchanged',
        greedy=True, max_new_tokens=32, use_cache=True, physical_gpus=GPUS, shard_counts=[80,80,80],
        sample_manifest_sha256=sha256(OUT / 'sample_manifest.json'), code_sha256=sources(), environment=environment(),
        frozen_model_manifest_sha256=sha256(HISTORY / 'model_manifest.json'),
        scorer_bindings=previous['scorer_bindings'], scoring=previous['scoring'],
        bootstrap_seed=20261101, bootstrap_repetitions=100000, cluster='Image SHA256, paired, question-weighted',
        primary_comparison='Keep L26 - Full', secondary_comparison='All OFF - Full (descriptive)',
        caveat='Same exploratory cohort used to nominate L26; not an independent confirmatory/NI experiment'))
    print('Frozen: exact 240 questions; 3 conditions; 720 full generations.', flush=True)


def verify():
    p = read(OUT / 'protocol.json')
    assert p['code_sha256'] == sources() and p['environment'] == environment()
    assert p['sample_manifest_sha256'] == sha256(OUT / 'sample_manifest.json')
    assert sha256(HISTORY / 'sample_manifest.json') == p['sample_manifest_sha256']
    for path, value in p['scorer_bindings'].items():
        assert sha256(path) == value
    return p


def load(gpu, folder):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    assert gpu in GPUS and os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu) and torch.cuda.device_count() == 1
    busy = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not busy, f'GPU{gpu} occupied: {busy}'
    write(folder / 'preflight.json', dict(pid=os.getpid(), physical_gpu=gpu,
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)))
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager', device_map={'':'cuda:0'}, use_safetensors=True).eval()
    assert len(model.model.language_model.layers) == 36
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert {str(p.device) for p in model.parameters()} == {'cuda:0'}
    assert model.config.text_config._attn_implementation == model.config.vision_config._attn_implementation == 'eager'
    return model, processor


def inputs_for(processor, sample):
    inputs, layout = prepare(processor, sample)
    assert layout == sample['layout'] and digest(layout) == sample['input_binding']
    return inputs.to('cuda:0')


def smoke(model, processor, sample):
    inputs = inputs_for(processor, sample)
    native, native_first, _ = generate(model, processor, inputs, sample['layout'], native=True)
    full, full_first, full_trace = generate(model, processor, inputs, sample['layout'], trace=True)
    noop, noop_first, _ = generate(model, processor, inputs, sample['layout'], CONDITIONS['All OFF'], noop=True)
    assert native['token_ids'] == full['token_ids'] == noop['token_ids']
    assert torch.equal(native_first, full_first) and torch.equal(full_first, noop_first)
    checks = []
    for condition in ('Keep L26', 'All OFF'):
        result, _, trace = generate(model, processor, inputs, sample['layout'], CONDITIONS[condition], trace=True)
        for i in range(36):
            for name in ('attention','input','mid','mlp','output'):
                assert torch.equal(full_trace.records[i][name][:,:-1], trace.records[i][name][:,:-1])
            if i < 3:
                assert torch.equal(full_trace.features[i], trace.features[i])
        checks.append(dict(condition=condition, masked_layers=result['masked_layers'],
            all_target_probabilities_zero=all(a['image_probability_max']==0 for a in result['mask_audits'].values()),
            other_prefill_queries_exact=True, deepstack_native_exact=True, decode_hooks_removed=True,
            fresh_cache=True, eos=result['eos']))
        del trace
    from transformers.cache_utils import DynamicCache
    before = hook_counts(model)
    try:
        with PrefillOnly(model, sample['layout'], CONDITIONS['All OFF'], DynamicCache(config=model.config.text_config)):
            raise RuntimeError('cleanup fixture')
    except RuntimeError as exc:
        assert str(exc) == 'cleanup fixture'
    assert hook_counts(model) == before
    return dict(passed=True, question_id=sample['question_id'], native_noop_first_logits_max_abs=0,
        native_noop_full_generation_tokens_exact=True, checks=checks, hooks_clean=True,
        input_length=sample['layout']['input_length'], image_tokens=sample['layout']['image_tokens'])


def worker(shard):
    verify()
    folder, gpu = OUT / f'shard_{shard}', GPUS[shard]
    samples = [s for s in read(OUT / 'sample_manifest.json')['samples'] if s['shard'] == shard]
    assert len(samples) == 80
    model, processor = load(gpu, folder)
    weights = [(p._version,p.data_ptr()) for p in model.parameters()]
    hooks = hook_counts(model)
    start = time.perf_counter()
    validation = smoke(model, processor, max(samples,key=lambda s:s['layout']['input_length']))
    validation.update(seconds=time.perf_counter()-start, physical_gpu=gpu,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), protocol_sha256=sha256(OUT/'protocol.json'))
    write(folder/'validation.json',validation)
    print(f'GPU{gpu} smoke passed ({validation["seconds"]:.1f}s)',flush=True)
    while not all((OUT/f'shard_{i}/validation.json').exists() for i in range(3)):
        assert not any((OUT/f'shard_{i}/error.json').exists() for i in range(3)), 'Peer smoke failed'
        time.sleep(1)
    for i in range(3):
        peer=read(OUT/f'shard_{i}/validation.json')
        assert peer['passed'] and peer['code_commit']==validation['code_commit'] and peer['protocol_sha256']==validation['protocol_sha256']
    start=time.perf_counter()
    for n,sample in enumerate(samples,1):
        path=folder/'checkpoints'/f'{sample["task"]}__{sample["question_id"]}.json'
        if path.exists():
            record=read(path)
            assert record['protocol_sha256']==validation['protocol_sha256'] and record['input_binding']==sample['input_binding']
            assert list(record['conditions'])==list(CONDITIONS)
            continue
        inputs=inputs_for(processor,sample)
        record={k:sample[k] for k in ('task','question_id','image_id','image_sha256','question','answer','input_binding')}
        record.update(protocol_sha256=validation['protocol_sha256'],code_commit=validation['code_commit'],physical_gpu=gpu,conditions={})
        for condition,layers in CONDITIONS.items():
            before=time.perf_counter()
            result,first,_=generate(model,processor,inputs,sample['layout'],layers)
            if condition=='Full':
                full=first
            token=int(full.argmax())
            result.update(correct=int(result['prediction']==sample['answer']),
                full_first_token_id=token,first_top1_flip=int(int(first.argmax())!=token),
                first_logp_loss=float(full.log_softmax(-1)[token]-first.log_softmax(-1)[token]),
                seconds=time.perf_counter()-before)
            record['conditions'][condition]=result
        assert hook_counts(model)==hooks
        assert {k:tensor_digest(v) for k,v in inputs.items()}==sample['layout']['input_tensor_hashes']
        write(path,record)
        del inputs
        write(folder/'progress.json',dict(done=n,total=80,seconds=time.perf_counter()-start,physical_gpu=gpu))
        print(f'GPU{gpu} {n}/80 elapsed={time.perf_counter()-start:.1f}s',flush=True)
    assert weights==[(p._version,p.data_ptr()) for p in model.parameters()]
    write(folder/'complete.json',dict(passed=True,count=80,generations=240,seconds=time.perf_counter()-start,
        smoke_seconds=validation['seconds'],code_commit=validation['code_commit'],protocol_sha256=validation['protocol_sha256'],
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','worker','aggregate'])
    parser.add_argument('--shard',type=int,choices=range(3))
    args=parser.parse_args()
    if args.stage=='freeze':
        freeze()
    elif args.stage=='aggregate':
        from .analyze import aggregate
        aggregate()
    else:
        try:
            worker(args.shard)
        except BaseException:
            write(OUT/f'shard_{args.shard}/error.json',dict(traceback=traceback.format_exc()))
            raise
