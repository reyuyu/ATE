"""Diagnose the exact native-replay contract before resuming InternVL scans."""
import argparse
from contextlib import ExitStack
import gc
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

from .common import ROOT, IMODEL, TASK1_END, DEADLINE, TASKS, read, write, sha, digest, assign, preflight, command
from .engine import InternPrefill, metrics
from .scan import load_intern, intern_smoke, environment
from ..internvl3_smoke.engine import official_preprocessor, tensor_hash, hooks_count, replay
from ..internvl3_smoke.run import ChatCapture

RECOVERY = ROOT / 'vlm_research/results/internvl_native_replay_recovery'
PARENT = ROOT / 'vlm_research/results/visual_pathways_seven_hour'


def difference(a, b):
    return dict(exact=torch.equal(a, b), max_abs=float((a.float()-b.float()).abs().max()),
                changed=int((a != b).sum()), top1_equal=int(a.argmax()) == int(b.argmax()),
                first_sha256=tensor_hash(a), second_sha256=tensor_hash(b))


@torch.no_grad()
def diagnose(gpu):
    preflight(gpu, RECOVERY / 'diagnosis')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    model, tokenizer, fast = load_intern()
    loader, _ = official_preprocessor(IMODEL)
    sample = next(s for s in read(PARENT/'sample_manifest.json')['samples'] if s['pilot'])
    rows = []
    baseline_hooks = hooks_count(model)
    with torch.inference_mode(), ExitStack() as stack:
        def head(module, args, output):
            rows.append(dict(input_shape=list(args[0].shape),
                p1_hidden_sha256=tensor_hash(args[0][:, -1]),
                p1_logits_sha256=tensor_hash(output[:, -1])))
        stack.callback(model.language_model.lm_head.register_forward_hook(head).remove)
        runner = InternPrefill(model, tokenizer, fast, sample, loader)
        old_output = model.language_model(**runner.inputs, use_cache=False,
            past_key_values=None, output_attentions=False, logits_to_keep=0)
        old = old_output.logits[0, -1].detach().cpu().clone()
        with ChatCapture(model) as capture:
            model.chat(tokenizer, runner.pixels, runner.request,
                       dict(do_sample=False, num_beams=1, max_new_tokens=1))
        fixed_output = model.language_model(**runner.inputs, use_cache=False,
            past_key_values=None, output_attentions=False, logits_to_keep=1)
        fixed = fixed_output.logits[0, -1].detach().cpu().clone()
        exact_replay, _, _, _ = replay(model, capture.saved)
        assert capture.input_ids[0].tolist() == runner.layout['input_ids']
        assert torch.equal(capture.saved['inputs_embeds'], runner.inputs['inputs_embeds'].cpu())
        kwargs = {k:dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_hash(v))
                  if isinstance(v, torch.Tensor) else v for k, v in capture.saved.items()}
        report = dict(question_id=sample['question_id'], image_sha256=sample['image_sha256'],
            physical_gpu=gpu, old_vs_official=difference(old, capture.first_logits),
            logits_to_keep_1_vs_official=difference(fixed, capture.first_logits),
            saved_official_replay=difference(exact_replay, capture.first_logits),
            lm_head_calls=rows, official_kwargs=kwargs,
            same_token_ids=True, same_visual_embeddings=True,
            seconds=time.perf_counter()-start,
            model_revision=IMODEL.name,
            installed_qwen2_source_sha256=sha(__import__('transformers.models.qwen2.modeling_qwen2', fromlist=['']).__file__))
    assert hooks_count(model) == baseline_hooks
    write(RECOVERY/'diagnosis.json', report)
    print(report, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def execution_sources():
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    import transformers.generation.utils as generation
    paths = [ROOT/'vlm_research/visual_pathways'/name for name in (
        'internvl_recovery.py','test_internvl_recovery.py','engine.py','scan.py','common.py')]
    paths += sorted((ROOT/'vlm_research/internvl3_smoke').glob('*.py'))
    paths += [Path(qwen2.__file__),Path(generation.__file__)]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in paths}


def freeze():
    assert not (RECOVERY/'protocol.json').exists()
    diagnosis = read(RECOVERY/'diagnosis.json')
    assert diagnosis['logits_to_keep_1_vs_official']['exact']
    assert diagnosis['saved_official_replay']['exact']
    assert len({row['p1_hidden_sha256'] for row in diagnosis['lm_head_calls']}) == 1
    original = read(PARENT/'sample_manifest.json')['samples']
    fields = ('task','question_id','image_id','image_sha256','image_path','question','answer','pilot')
    samples = [dict({key:s[key] for key in fields}, source_sample_sha256=digest(s)) for s in original]
    assert len(samples) == 3000 and sum(s['pilot'] for s in samples) == 240
    assert all(sum(s['task'] == task for s in samples) == 500 for task in TASKS)
    write(RECOVERY/'sample_manifest.json', dict(samples=samples,
        parent_manifest_sha256=sha(PARENT/'sample_manifest.json'), selection='Exact parent 3000 and pilot240; no resampling'))
    model_manifest = read(PARENT/'internvl_model_manifest.json')
    for entry in model_manifest['files']:
        assert sha(IMODEL/entry['path']) == entry['sha256'], entry['path']
    write(RECOVERY/'model_validation.json',dict(passed=True, revision=IMODEL.name,
        parent_manifest_sha256=sha(PARENT/'internvl_model_manifest.json'), files_checked=len(model_manifest['files'])))
    shards = assign(samples, 3)
    write(RECOVERY/'shards.json',dict(gpu_ids=[0,2,3], shards=[dict(index=i, gpu=[0,2,3][i],
        all_count=len(rows), pilot_count=sum(s['pilot'] for s in rows),
        samples=[dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],pilot=s['pilot'])
                 for s in rows]) for i, rows in enumerate(shards)]))
    write(RECOVERY/'protocol.json',dict(
        parent_commit='d92b647309e7fc3dbd1128a669b4a3b1e553ce79',
        parent_protocol_sha256=sha(PARENT/'protocol.json'), original_start=read(PARENT/'protocol.json')['start'],
        task1_end=TASK1_END, total_deadline=DEADLINE, manifest_sha256=sha(RECOVERY/'sample_manifest.json'),
        shards_sha256=sha(RECOVERY/'shards.json'), source_sha256=execution_sources(),
        model_revision=IMODEL.name, dtype='bfloat16', attention='eager', transformers='4.49.0',
        max_num=1, image_tokens=256, layers=28, cache=False,
        conditions='Single target layer: original P-1 -> actual Image keys -inf before native softmax; others native',
        target='Same-image native Full top1; FP32 full-vocabulary log_softmax',
        native_projection='logits_to_keep=1, exact native official Chat LM-head shape',
        numerical_tolerance_change=False, no_op_and_native_comparison='torch.equal, max_abs=0',
        parent_failed_attempts_preserved=True, no_old_internvl_results_reused=True,
        bootstrap=read(PARENT/'protocol.json')['bootstrap'],
        task2_extension=False, task2_reason='Parent LLaVA LogP interaction negative; no InternVL 2x2 extension',
        task3='Not automatically added; finish the originally requested source scan first'))
    print('Frozen recovery: pilot240 first, exact3000 extension, original deadline', TASK1_END, flush=True)


def verify():
    protocol = read(RECOVERY/'protocol.json')
    assert sha(RECOVERY/'sample_manifest.json') == protocol['manifest_sha256']
    assert sha(RECOVERY/'shards.json') == protocol['shards_sha256']
    assert execution_sources() == protocol['source_sha256']
    return protocol


@torch.no_grad()
def worker(gpu, cohort):
    protocol = verify()
    shard = [0,2,3].index(gpu)
    assigned = assign(read(RECOVERY/'sample_manifest.json')['samples'], 3)[shard]
    if cohort == 'all':
        assert read(RECOVERY/'pilot_gate.json')['passed'], 'All 240 pilot samples and three smokes must pass first'
    samples = [s for s in assigned if cohort == 'all' or s['pilot']]
    samples.sort(key=lambda s:(not s['pilot'],s['task'],s['question_id']))
    folder = RECOVERY/'internvl'/f'{cohort}_gpu{gpu}'
    preflight(gpu, folder)
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    write(folder/'environment.json', environment())
    model, tokenizer, fast = load_intern()
    loader, _ = official_preprocessor(IMODEL)
    smoke_samples = [next(s for s in samples if s['task'] == task) for task in TASKS[shard*2:shard*2+2]]
    checks = [intern_smoke(model,tokenizer,fast,s,loader) for s in smoke_samples]
    code_commit = command('git','rev-parse','HEAD')
    write(folder/'validation.json',dict(passed=True, physical_gpu=gpu, checks=checks,
        code_commit=code_commit, protocol_sha256=sha(RECOVERY/'protocol.json'),
        seconds_from_start=time.perf_counter()-started))
    print('SMOKE PASS', gpu, cohort, flush=True)
    versions = [(p._version,p.data_ptr()) for p in model.parameters()]
    baseline = hooks_count(model)
    formal = time.perf_counter()
    done = new = reused = 0
    for sample in samples:
        if datetime.now().astimezone() >= datetime.fromisoformat(TASK1_END):
            print('TIME LIMIT: checkpoint preserved', flush=True)
            break
        key = f'{sample["task"]}__{sample["question_id"]}'
        path = RECOVERY/'internvl/checkpoints'/f'{key}.json'
        if path.exists():
            prior = read(path)
            assert prior['sample_sha256'] == digest(sample) and prior['protocol_sha256'] == sha(RECOVERY/'protocol.json')
            assert prior['image_sha256'] == sample['image_sha256']
            if prior['complete']:
                assert len(prior['rows']) == 28
                done += 1
                reused += 1
                continue
        with torch.inference_mode():
            runner = InternPrefill(model,tokenizer,fast,sample,loader)
            full,_,_ = runner.forward()
            record = dict(complete=False, task=sample['task'], question_id=sample['question_id'],
                image_id=sample['image_id'], image_sha256=sample['image_sha256'],
                sample_sha256=digest(sample), protocol_sha256=sha(RECOVERY/'protocol.json'),
                code_commit=code_commit, physical_gpu=gpu,
                input_layout=runner.layout, input_binding=digest(runner.layout),
                full_logits_sha256=tensor_hash(full), rows=[])
            if path.exists():
                prior = read(path)
                assert prior['input_binding'] == record['input_binding'] and prior['full_logits_sha256'] == record['full_logits_sha256']
                record['rows'] = prior['rows']
            for layer in range(28):
                if any(row['layer'] == layer for row in record['rows']):
                    continue
                changed,audit,_ = runner.forward(layer)
                record['rows'].append(dict(model='internvl', task=sample['task'], question_id=sample['question_id'],
                    image_id=sample['image_id'], image_sha256=sample['image_sha256'], physical_gpu=gpu,
                    layer=layer, image_tokens=256, **metrics(full,changed), **audit))
                write(path,record)
            runner.verify()
            record['complete'] = True
            write(path,record)
            del runner,full
        new += 1
        done += 1
        progress = dict(done=done,total=len(samples),new=new,reused=reused,
            seconds=time.perf_counter()-formal, physical_gpu=gpu, last_sample=key)
        write(folder/'progress.json',progress)
        if new % 20 == 0:
            print('PROGRESS',progress,flush=True)
    assert versions == [(p._version,p.data_ptr()) for p in model.parameters()]
    assert hooks_count(model) == baseline
    write(folder/'complete.json',dict(passed=True, status='complete' if done==len(samples) else 'partial',
        done=done,total=len(samples),new=new,reused=reused,formal_seconds=time.perf_counter()-formal,
        total_seconds=time.perf_counter()-started,weights_unchanged=True,hooks_clean=True,no_nan_inf_oom=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def pilot_gate():
    verify()
    samples = [s for s in read(RECOVERY/'sample_manifest.json')['samples'] if s['pilot']]
    assert len(samples) == 240
    versions = set()
    for gpu in (0,2,3):
        folder = RECOVERY/'internvl'/f'pilot_gpu{gpu}'
        validation = read(folder/'validation.json')
        complete = read(folder/'complete.json')
        assert validation['passed'] and complete['status'] == 'complete' and complete['done'] == complete['total']
        assert complete['weights_unchanged'] and complete['hooks_clean'] and complete['no_nan_inf_oom']
        versions.add(validation['code_commit'])
    assert len(versions) == 1
    for sample in samples:
        key = f'{sample["task"]}__{sample["question_id"]}'
        record = read(RECOVERY/'internvl/checkpoints'/f'{key}.json')
        assert record['complete'] and record['sample_sha256'] == digest(sample)
        assert record['protocol_sha256'] == sha(RECOVERY/'protocol.json')
        assert len(record['rows']) == 28 and sorted(row['layer'] for row in record['rows']) == list(range(28))
        assert all(row['masked_probability_max'] == 0 and row['changed_entries'] == 256 for row in record['rows'])
    write(RECOVERY/'pilot_gate.json',dict(passed=True, questions=240, condition_rows=6720,
        code_commit=versions.pop(),source_sha256=execution_sources(),time=datetime.now().astimezone().isoformat()))
    print('PILOT240 PASS: 6,720 valid rows; original3000 extension permitted',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['diagnose','freeze','worker','pilot-gate'])
    parser.add_argument('--gpu', type=int, choices=[0, 2, 3])
    parser.add_argument('--cohort', choices=['pilot','all'])
    args = parser.parse_args()
    if args.action == 'diagnose':
        diagnose(args.gpu)
    elif args.action == 'freeze':
        freeze()
    elif args.action == 'pilot-gate':
        pilot_gate()
    else:
        assert args.gpu is not None and args.cohort is not None
        try:
            worker(args.gpu,args.cohort)
        except BaseException:
            write(RECOVERY/'internvl'/f'{args.cohort}_gpu{args.gpu}'/'error.json',
                  dict(traceback=traceback.format_exc(), time=datetime.now().astimezone().isoformat()))
            raise
