"""Immutable code/data bindings, atomic per-question checkpoints, isolated workers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
import transformers

from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.run import preflight
from paper_rq1.scorers import gqa_prediction
from vlm_common import load_model, prepare_inputs, gpu_memory
from .engine import first_token, forward, generate, positions
from .protocol import GPUS, MODEL_PATH, OUTPUT, PROTOCOL, RESOURCES, ROOT, SEED


def atomic_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Never overwrite a different completed checkpoint')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temp.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def save_logits(path, full, changed):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temp.open('xb') as f:
            np.savez_compressed(f, full_logits=full, ablated_logits=changed)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return dict(logits_path=str(path), logits_sha256=sha256(path))


def source_hashes():
    paths = list((ROOT/'paper_rq1_single_layer_gqa6').glob('*.py'))
    paths += list((ROOT/'paper_rq1').glob('*.py'))
    paths += list((ROOT/'benchmarks').glob('*.py'))
    paths += [ROOT/p for p in ('vlm_common.py', 'knockout_probe.py', 'attention_probe.py',
        'attention_position_probe.py', 'benchmarks/_vendor/m4c_evaluator.py',
        'test_paper_rq1_single_layer_gqa6.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


def freeze():
    data = json.loads((OUTPUT/'data_manifest.json').read_text())
    cpu = json.loads((OUTPUT/'cpu_validation.json').read_text())
    if data['status'] != 'ready' or data['total_questions'] != 3000 or not cpu['passed'] or cpu['source_sha256'] != source_hashes():
        raise ValueError('Exact-source CPU tests and full frozen data required')
    paths = ['vlm_research/'+p for p in source_hashes()]
    tracked = subprocess.check_output(['git','ls-files','--',*paths],text=True,cwd=ROOT.parent).splitlines()
    if set(tracked) != set(paths) or subprocess.check_output(['git','diff','HEAD','--name-only','--',*paths],text=True,cwd=ROOT.parent).strip():
        raise ValueError('Commit all final sources before GPU execution')
    model = Path(MODEL_PATH)
    files = sorted([*model.glob('*.safetensors'), *model.glob('*.json'), *model.glob('tokenizer.model')])
    weights = {str(p):dict(sha256=sha256(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files}
    freeze_json(OUTPUT/'execution_manifest.json',dict(protocol=PROTOCOL,source_sha256=source_hashes(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT.parent).strip(),
        data_manifest_sha256=sha256(OUTPUT/'data_manifest.json'), cpu_validation_sha256=sha256(OUTPUT/'cpu_validation.json'),
        model_files=weights, stages=['smoke','first_token','accuracy']))


def verify():
    c = json.loads((OUTPUT/'execution_manifest.json').read_text())
    if c['source_sha256'] != source_hashes() or digest(c['protocol']) != digest(PROTOCOL):
        raise ValueError('Frozen source/protocol changed')
    if sha256(OUTPUT/'data_manifest.json') != c['data_manifest_sha256'] or sha256(OUTPUT/'cpu_validation.json') != c['cpu_validation_sha256']:
        raise ValueError('Data/CPU acceptance changed')
    d = json.loads((OUTPUT/'data_manifest.json').read_text())
    if sha256(OUTPUT/'sample_manifest.json') != d['sample_manifest_sha256'] or sha256(OUTPUT/'scorer_validation.json') != d['scorer_validation_sha256']:
        raise ValueError('Samples/scorer changed')
    for name, h in json.loads((OUTPUT/'scorer_validation.json').read_text())['source_sha256'].items():
        if sha256(name) != h:
            raise ValueError('Official scoring source changed')
    for name, record in c['model_files'].items():
        stat = Path(name).stat()
        if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError('Model artifact changed since SHA256 freeze')
    return c


def samples(rank):
    d = json.loads((OUTPUT/'data_manifest.json').read_text())
    path = OUTPUT/f'manifests/shard{rank}.json'
    if sha256(path) != d['shard_sha256'][str(rank)]:
        raise ValueError('Worker shard changed')
    return json.loads(path.read_text())['samples']


def prepared(model, processor, sample):
    if sha256(sample['image_path']) != sample['image_sha256']:
        raise ValueError('Selected original image is missing or changed')
    inputs, counts = prepare_inputs(model, processor, sample['image_path'], sample['prompt_text'])
    m = sample['input_mapping']
    found = positions(inputs, processor.tokenizer.bos_token_id)
    if inputs['input_ids'][0].tolist() != m['prompt_ids'] or found['query'] != m['p_minus_one'] or found['bos'] != m['bos_positions'] or counts['visual_tokens'] != 576:
        raise ValueError('GPU true input mapping differs from frozen processor mapping')
    return inputs


def tensor_hashes(inputs):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in inputs.items()}


def native_reference(model, inputs, layer=None):
    handles, calls = [], []
    if layer is not None:
        def hook(module, args, output):
            calls.append(output.shape[1])
            if len(calls) == 1:
                if output.shape[1] != inputs['input_ids'].shape[1]:
                    raise AssertionError('Reference is not a real Prefill')
                result = output.clone()
                result[0, inputs['input_ids'].shape[1]-1] = 0
                return result
            if output.shape[1] != 1:
                raise AssertionError('Reference Decode query not a single generated token')
            return output
        handles.append(model_layers(model)[layer].self_attn.o_proj.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=32,use_cache=True,eos_token_id=2)
        return out[0,inputs['input_ids'].shape[1]:].tolist()
    finally:
        for h in handles:
            h.remove()


def smoke(model, processor, rank):
    sample = json.loads((OUTPUT/'sample_manifest.json').read_text())['samples'][0]
    inputs = prepared(model,processor,sample)
    before = tensor_hashes(inputs)
    footprint = hook_footprint(model)
    query = sample['input_mapping']['p_minus_one']
    result, full, changed = first_token(model,processor,inputs)
    native,_,_ = forward(model,inputs,query)
    cached,_,_ = forward(model,inputs,query,cache=True)
    noop,_,noop_trace = forward(model,inputs,query,layer=31,noop=True,audit=True)
    if not all(np.array_equal(full,x.numpy()) for x in (native,cached,noop)):
        raise AssertionError('Native, observed, cached or No-op logits differ')
    _,_,scoped = forward(model,inputs,query,layer=0,audit=True)
    if not scoped.edits[0]['target_zero'] or not scoped.edits[0]['other_queries_exact']:
        raise AssertionError('Single-layer query scope is invalid')
    mlp_calls=[]
    handles=[layer.mlp.register_forward_hook(lambda m,a,o,i=i:mlp_calls.append(i)) for i,layer in enumerate(model_layers(model))]
    try:
        forward(model,inputs,query,layer=15)
    finally:
        for h in handles:h.remove()
    if mlp_calls != list(range(32)):
        raise AssertionError('MLP computation changed')
    generations={}
    for layer in (None,2,31):
        manual,initial = generate(model,inputs,layer)
        expected = full if layer is None else changed[layer]
        if not np.array_equal(initial,expected) or manual['token_ids'] != native_reference(model,inputs,layer):
            raise AssertionError('Generation/native cache semantics disagree with independent reference')
        generations[str(layer)] = manual
    path=RESOURCES/f'raw/smoke/shard{rank}/test.npz'
    raw=save_logits(path,full,changed)
    with np.load(path) as saved:
        if not np.array_equal(saved['full_logits'],full) or not np.array_equal(saved['ablated_logits'],changed):
            raise AssertionError('Smoke logits roundtrip failed')
    if tensor_hashes(inputs) != before or hook_footprint(model) != footprint:
        raise AssertionError('Input mutation or leaked hook')
    value=dict(passed=True,question_id=sample['question_id'],sample_sha256=digest(sample),
        result=result,generations=generations,logits=raw,checks=dict(
            native_observed_noop_cached_logits_exact=True,only_one_layer_and_query=True,
            all_32_mlp_calls_native=True,hook_cleanup=True,input_hashes_unchanged=True,
            full_and_intervened_cache_generation_matches_native_reference=True,
            full_32_conditions_roundtrip=True,bos_dynamic_and_rows_valid=True),memory=gpu_memory())
    return value


def gate(stage):
    statuses=[json.loads((OUTPUT/f'{stage}/shard{i}/completion.json').read_text()) for i in range(4)]
    if any(s['status']!='complete' for s in statuses) or len({s['binding'] for s in statuses})!=1 or len({s['code_commit'] for s in statuses})!=1:
        raise ValueError('Require all four complete matching frozen workers')
    environments=[json.loads((OUTPUT/f'{stage}/shard{i}/environment.json').read_text()) for i in range(4)]
    keys=('python','torch','transformers','cuda_runtime','cudnn','gpu','model_revision','parameter_dtypes','binding','code_commit','actual_seed')
    if len({digest({k:e[k] for k in keys}) for e in environments})!=1:
        raise ValueError('GPU environments differ')
    if stage=='smoke':
        values=[json.loads((OUTPUT/f'smoke/shard{i}/smoke.json').read_text()) for i in range(4)]
        with np.load(values[0]['logits']['logits_path']) as base:
            reference={k:base[k].copy() for k in base.files}
        tol=PROTOCOL['tolerances']
        for value in values[1:]:
            with np.load(value['logits']['logits_path']) as other:
                for k in reference:
                    np.testing.assert_allclose(other[k],reference[k],atol=tol['cross_gpu_logits_atol'],rtol=tol['cross_gpu_logits_rtol'])
            np.testing.assert_allclose([r['bos_mass_per_head'] for r in value['result']['bos']],
                [r['bos_mass_per_head'] for r in values[0]['result']['bos']],
                atol=tol['cross_gpu_bos_atol'],rtol=tol['cross_gpu_bos_rtol'])
            if {k:v['token_ids'] for k,v in value['generations'].items()} != {k:v['token_ids'] for k,v in values[0]['generations'].items()}:
                raise AssertionError('Cross-GPU greedy Smoke tokens differ')
        freeze_json(OUTPUT/'cross_gpu_smoke.json',dict(passed=True,question_id=values[0]['question_id'],
            tolerances=tol,full_and_all_32_layers_compared=True,bos_all_heads_compared=True,
            generation_tokens_exact=True))
    return statuses


def worker(stage,rank):
    config=verify()
    if stage!='smoke':gate('smoke')
    if stage=='accuracy':gate('first_token')
    binding=digest(config)
    folder=OUTPUT/f'{stage}/shard{rank}'
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=preflight(rank)
    if transformers.__version__!='4.49.0':raise ValueError('Wrong Transformers version')
    model,processor,env=load_model(MODEL_PATH)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    env.update(preflight=pre,binding=binding,code_commit=config['code_commit'],actual_seed=SEED,
        checkout_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT.parent).strip())
    envpath=folder/'environment.json'
    if envpath.exists():
        old=json.loads(envpath.read_text())
        for k in ('binding','code_commit','torch','transformers','model_revision','parameter_dtypes','actual_seed'):
            if old[k]!=env[k]:raise ValueError('Resume environment differs')
        freeze_json(folder/'attempts'/f'environment_{time.time_ns()}.json',env)
    else:freeze_json(envpath,env)
    versions={n:p._version for n,p in model.named_parameters()}
    footprint=hook_footprint(model)
    started=time.perf_counter()
    if stage=='smoke':
        atomic_json(folder/'smoke.json',smoke(model,processor,rank))
        completed=expected=1
    else:
        rows=[s for s in samples(rank) if stage=='first_token' or s['accuracy_subset']]
        expected=len(rows);completed=0
        for sample in rows:
            name=sample['task']+'__'+sample['question_id']
            path=folder/'checkpoints'/f'{name}.json'
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['binding']!=binding or saved['sample_sha256']!=digest(sample):raise ValueError('Checkpoint binding differs')
                if stage=='first_token' and sha256(saved['result']['logits_path'])!=saved['result']['logits_sha256']:
                    raise ValueError('Raw logits missing or changed')
                completed+=1
                continue
            try:
                inputs=prepared(model,processor,sample)
                if stage=='first_token':
                    result,full,changed=first_token(model,processor,inputs)
                    result.update(save_logits(RESOURCES/f'raw/first_token/shard{rank}/{name}.npz',full,changed))
                    token=int(full.argmax())
                    result.update(full_first_token_id=token,
                        full_first_token_text=processor.tokenizer.decode([token],skip_special_tokens=True),
                        first_token_matches_gold=token==sample['input_mapping']['gold_continuation_ids'][0],
                        gold_is_single_token=len(sample['input_mapping']['gold_continuation_ids'])==1)
                else:
                    main=json.loads((OUTPUT/f'first_token/shard{rank}/checkpoints/{name}.json').read_text())
                    if main['sample_sha256']!=digest(sample) or sha256(main['result']['logits_path'])!=main['result']['logits_sha256']:
                        raise ValueError('Accuracy prerequisite raw results changed')
                    with np.load(main['result']['logits_path']) as raw:
                        vectors={k:raw[k].copy() for k in raw.files}
                    generations=[]
                    for layer in [None,*range(32)]:
                        generated,first=generate(model,inputs,layer)
                        if not np.array_equal(first,vectors['full_logits'] if layer is None else vectors['ablated_logits'][layer]):
                            raise AssertionError('Generation Prefill logits differ from the paired main experiment')
                        text=processor.tokenizer.decode(generated['token_ids'],skip_special_tokens=True).strip()
                        generated.update(layer=layer,text=text,prediction=gqa_prediction(text),
                            score=float(gqa_prediction(text)==sample['answer']))
                        generations.append(generated)
                    result=dict(generations=generations,main_logits_sha256=main['result']['logits_sha256'],
                                all_33_initial_logits_exact=True)
                if hook_footprint(model)!=footprint:raise AssertionError('Hook leak')
                atomic_json(path,dict(binding=binding,sample_sha256=digest(sample),task=sample['task'],
                    question_id=sample['question_id'],image_id=sample['image_id'],result=result))
                completed+=1
                print(stage,rank,completed,'/',expected,name,flush=True)
            except BaseException as exc:
                atomic_json(folder/'failures'/f'{name}_{time.time_ns()}.json',dict(task=sample['task'],
                    question_id=sample['question_id'],binding=binding,error=repr(exc),traceback=traceback.format_exc(),
                    skipped=False,requires_resume_or_resolution=True))
                raise
    if versions!={n:p._version for n,p in model.named_parameters()} or hook_footprint(model)!=footprint:
        raise AssertionError('Weights changed or hooks leaked')
    verify()
    atomic_json(folder/'completion.json',dict(status='complete' if completed==expected else 'partial',
        binding=binding,code_commit=config['code_commit'],rank=rank,completed=completed,expected=expected,
        wall_seconds=time.perf_counter()-started,memory=gpu_memory(),model_versions_unchanged=True,hooks_clean=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['freeze','smoke','first_token','accuracy'])
    p.add_argument('--rank',type=int,choices=range(4))
    args=p.parse_args()
    if args.stage=='freeze':freeze()
    elif args.rank is None:p.error('worker requires --rank')
    else:worker(args.stage,args.rank)


if __name__=='__main__':main()
