"""Phase 3.2 fixed-K experiment; immutable binding and atomic sample checkpoints.

Order: prepare_fixed_k_data.py -> CPU tests -> smoke -> validate -> run -> summarize.
Only imports the unchanged Phase 3.1 physical deletion and candidate scoring path.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from vlm_common import ROOT, REVISION, default_model, load_model, prepare_inputs
from attention_position_probe import locate_question_tokens
from knockout_probe import resource_preflight
from physical_exit import TOLERANCES, cosine_distances, equivalence
from physical_exit_pilot import candidate_inputs, pack_forward
from prepare_physical_exit_data import sha256
from prepare_fixed_k_data import OUTPUT, digest, audit_image

CUTOFFS = [0, 8, 12, 16, 20, 24, 28, 32]
CODE_FILES = ['fixed_k_exit.py', 'prepare_fixed_k_data.py', 'physical_exit.py', 'physical_exit_pilot.py',
              'vlm_common.py', 'attention_position_probe.py', 'knockout_probe.py']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def configuration():
    import platform
    import transformers
    import inspect
    from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaAttention
    from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
    model_path = Path(default_model())
    return dict(schema=1, cutoffs=CUTOFFS, tolerances=TOLERANCES, manifest_sha256=sha256(OUTPUT/'manifest.json'),
                code_sha256={name: sha256(ROOT/name) for name in CODE_FILES},
                installed_source_sha256={c.__name__: sha256(inspect.getfile(c)) for c in
                                         (LlamaModel, LlamaDecoderLayer, LlamaAttention, LlavaForConditionalGeneration)},
                python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                cuda_runtime=torch.version.cuda, model_revision=REVISION, model_path=str(model_path),
                model_config_sha256=sha256(model_path/'config.json'),
                model_shards={p.name: {'size_bytes': p.stat().st_size, 'sha256': sha256(p)}
                              for p in sorted(model_path.glob('*.safetensors'))},
                protocol='FP16/eager/eval/inference_mode; batch=1; use_cache=False; native Full paired with physical; unchanged chat template; candidates left,right; full joint logP without EOS; strict margin>0; final RMSNorm Question hidden from first candidate; no decode.',
                timings='Diagnostic only: no independent warmup/repeats. Includes CLIP/projector, observers and position hashes; Full stores all hidden states, physical only last. Not a fair stable speed benchmark.',
                validation='Original 12 Phase3.1 samples; all 8 K; frozen Phase3.1 numerical tolerances and old-result compatibility at shared K.')


def verify_manifest(manifest):
    samples = manifest['samples']
    if len(samples) != 240 or len({s['question_id'] for s in samples}) != 240 or len({s['img_id'] for s in samples}) != 240:
        raise ValueError('Manifest must have 240 unique questions/images')
    if any(sum(s['answer'] == label for s in samples) != 120 for label in ('left', 'right')):
        raise ValueError('Manifest balance changed')
    if set(manifest['excluded_image_ids']) & {s['img_id'] for s in samples}:
        raise ValueError('Pilot images reused')
    for s in samples + manifest['validation_samples']:
        if s['candidates'] != ['left', 'right'] or {s['true option'], s['false option']} != {'left', 'right'} or s['answer'] != s['true option']:
            raise ValueError('Invalid binary candidate mapping')
        audit = audit_image(s)
        if not audit['image_available'] or 'error' in audit or audit.get('image_sha256') != s.get('image_sha256'):
            raise ValueError(f"Frozen image missing/corrupt/changed: {s['question_id']} {audit}")
    ids = manifest['smoke_question_ids']
    smoke = [s for s in samples if s['question_id'] in ids]
    if len(ids) != 24 or len(smoke) != 24 or any(sum(s['answer']==x for s in smoke)!=12 for x in ('left','right')):
        raise ValueError('Smoke subset changed')


def save_checkpoint(path, binding, sample, data):
    if Path(path).exists():
        raise ValueError('Refuse to overwrite committed sample checkpoint')
    body = dict(binding=binding, sample_sha256=digest(sample), sample=sample, data=data)
    atomic_json(path, dict(body=body, body_sha256=digest(body)))


def load_checkpoint(path, binding, sample):
    value = json.loads(Path(path).read_text())
    body = value['body']
    if value['body_sha256'] != digest(body) or body['binding'] != binding or body['sample_sha256'] != digest(sample) or body['sample'] != sample:
        raise ValueError(f'Checkpoint fingerprint/sample mismatch: {path}')
    data = body['data']
    rows = data['rows']
    if len(rows) != len(CUTOFFS) or sorted(r['cutoff'] for r in rows) != CUTOFFS:
        raise ValueError('Duplicate or incomplete K rows in checkpoint')
    if any(r['question_id'] != sample['question_id'] or r['img_id'] != sample['img_id'] or r['answer'] != sample['answer'] for r in rows):
        raise ValueError('Checkpoint row identity mismatch')
    if not data['k32_exact']:
        raise ValueError('Checkpoint failed Full identity')
    return data


def check_directory(directory, samples, binding):
    directory.mkdir(parents=True, exist_ok=True)
    expected = {s['question_id']: s for s in samples}
    existing = {}
    for path in directory.glob('*.json'):
        if path.stem not in expected:
            raise ValueError(f'Unrecognized checkpoint {path}')
        existing[path.stem] = load_checkpoint(path, binding, expected[path.stem])
    return existing


def evaluate(model, processor, sample, validate=False, smoke=False):
    prompt, counts = prepare_inputs(model, processor, sample['image_path'], sample['question'])
    ids = prompt['input_ids'][0].cpu().numpy()
    qrows, qaudit = locate_question_tokens(processor, sample['question'], ids, model.config.image_token_index)
    qpos = [r['sequence_index'] for r in qrows]
    candidates = candidate_inputs(processor, prompt, sample['candidates'])
    positions = [torch.where((x['input_ids'][0] != model.config.image_token_index) & x['attention_mask'][0].bool())[0]
                 for x, _ in candidates]
    full = [pack_forward(model, x, 'full', 32, p, qpos, rec)
            for (x, rec), p in zip(candidates, positions)]
    target = sample['candidates'].index(sample['answer'])
    wrong = 1-target
    baseline_margin = full[target]['logp'] - full[wrong]['logp']
    result = dict(rows=[], details=[], counts=counts, question_mapping=qrows, question_audit=qaudit,
                  prompt_ids=ids.tolist(), full_candidates=[{'candidate': label, 'logp': x['logp'],
                                                           'measurement': x['measurement']} for label,x in zip(sample['candidates'],full)],
                  candidate_mapping=[dict(label=label, answer_records=rec, valid_text_positions=p.cpu().tolist(),
                                          input_ids=x['input_ids'][0].cpu().tolist(), attention_mask=x['attention_mask'][0].cpu().tolist())
                                     for label,(x,rec),p in zip(sample['candidates'],candidates,positions)], k32_exact=False)
    for k in CUTOFFS:
        values = [pack_forward(model, x, 'physical', k, p, qpos, rec)
                  for (x,rec),p in zip(candidates,positions)]
        if k == 32:
            exact = all(torch.equal(a['logits'],b['logits']) and torch.equal(a['hidden'],b['hidden']) for a,b in zip(full,values))
            if not exact:
                raise AssertionError('Physical K32 differs from native Full')
            result['k32_exact'] = exact
        for value,(x,rec),p in zip(values,candidates,positions):
            n = x['input_ids'].shape[1]
            expected = [n if layer<k else n-576 for layer in range(32)]
            if [r['sequence_length'] for r in value['layers']] != expected:
                raise AssertionError('Layer sequence lengths do not reflect physical deletion')
            output_pos = value['output_original_positions']
            expected_pos = list(range(n)) if k==32 else torch.where(x['input_ids'][0] != model.config.image_token_index)[0].cpu().tolist()
            if output_pos != expected_pos or any(r['query_position'] not in output_pos for r in rec):
                raise AssertionError('Original output/answer position mapping changed')
        pair = None
        if validate:
            masks = [pack_forward(model,x,'mask',k,p,qpos,rec) for (x,rec),p in zip(candidates,positions)]
            checks = [equivalence(a['hidden'],b['hidden'],a['logits'],b['logits']) for a,b in zip(masks,values)]
            logp_error=max(abs(a['logp']-b['logp']) for a,b in zip(masks,values))
            margin_error=abs((masks[target]['logp']-masks[wrong]['logp'])-(values[target]['logp']-values[wrong]['logp']))
            pair=dict(candidate_checks=checks,candidate_logp_max_abs_error=logp_error,margin_abs_error=margin_error,
                      passed=all(c['passed'] for c in checks) and logp_error<=TOLERANCES['candidate_joint_logp_atol_nats'] and margin_error<=TOLERANCES['candidate_margin_atol_nats'])
            if not pair['passed']:
                exc=AssertionError('Mask/Physical failed original frozen tolerances; no results interpretation allowed')
                exc.diagnostic=dict(question_id=sample['question_id'],cutoff=k,**pair)
                raise exc
            del masks
        margin = values[target]['logp']-values[wrong]['logp']
        drift = cosine_distances(full[0]['question_hidden'],values[0]['question_hidden'])[0]
        row=dict(question_id=sample['question_id'],img_id=sample['img_id'],answer=sample['answer'],cutoff=k,
                 full_correct=baseline_margin>0,candidate_correct=margin>0, full_margin=baseline_margin,
                 answer_margin=margin,answer_margin_drop=baseline_margin-margin,
                 correct_answer_log_probability=values[target]['logp'], wrong_answer_log_probability=values[wrong]['logp'],
                 correct_answer_sequence_probability=float(np.exp(values[target]['logp'])),
                 correct_probability_within_two_candidates=float(torch.sigmoid(torch.tensor(margin,dtype=torch.float64))),
                 question_mean_cosine_distance=float(drift.mean()),question_max_cosine_distance=float(drift.max()),
                 forward_seconds_sum_two_candidates=sum(v['measurement']['forward_seconds'] for v in values),
                 peak_allocated_mib=max(v['measurement']['gpu_memory']['peak_allocated_mib'] for v in values),
                 peak_reserved_mib=max(v['measurement']['gpu_memory']['peak_reserved_mib'] for v in values),
                 allocated_mib=max(v['measurement']['gpu_memory']['allocated_mib'] for v in values),
                 reserved_mib=max(v['measurement']['gpu_memory']['reserved_mib'] for v in values),physical_deletion=k<32,
                 original_sequence_length=candidates[0][0]['input_ids'].shape[1],
                 post_exit_sequence_length=candidates[0][0]['input_ids'].shape[1]-(576 if k<32 else 0))
        result['rows'].append(row)
        result['details'].append(dict(cutoff=k,question_per_token_cosine_distance=drift.tolist(),equivalence=pair,
                                     candidates=[dict(candidate=label,logp=v['logp'],measurement=v['measurement'],layers=v['layers'],
                                                      output_original_positions=v['output_original_positions'])
                                                 for label,v in zip(sample['candidates'],values)]))
        if smoke and k==0:
            x,rec=candidates[0]
            control=pack_forward(model,dict(x,pixel_values=x['pixel_values'].flip(-1)),'physical',0,positions[0],qpos,rec)
            if not torch.equal(control['hidden'],values[0]['hidden']) or not torch.equal(control['logits'],values[0]['logits']):
                raise AssertionError('K0 still depends on pixel values in language backbone')
            result['k0_pixel_invariance']=True
        del values
    if smoke:
        x,rec=candidates[0]
        after=pack_forward(model,x,'full',32,positions[0],qpos,rec)
        if not torch.equal(full[0]['logits'],after['logits']) or not torch.equal(full[0]['hidden'],after['hidden']):
            raise AssertionError('Subsequent native Full forward contaminated')
        result['native_forward_cleanup_exact']=True
    return result


def historical_check(data, sample):
    import csv
    previous=list(csv.DictReader((ROOT/'results/phase3_1_physical_exit_pilot/per_sample_results.csv').open()))
    checks=[]
    for row in data['rows']:
        old=[r for r in previous if r['question_id']==sample['question_id'] and r['mode']=='physical' and int(r['cutoff'])==row['cutoff']]
        if not old:
            continue
        a=old[0]
        delta=abs(float(a['answer_margin'])-row['answer_margin'])
        logp_delta=abs(float(a['correct_answer_log_probability'])-row['correct_answer_log_probability'])
        ok=delta<=TOLERANCES['candidate_margin_atol_nats'] and logp_delta<=TOLERANCES['candidate_joint_logp_atol_nats']
        checks.append(dict(cutoff=row['cutoff'],margin_abs_error=delta,gold_logp_abs_error=logp_delta,passed=ok))
    if len(checks)!=5 or not all(r['passed'] for r in checks):
        raise AssertionError(f'Phase 3.1 historical compatibility failed: {checks}')
    return checks


def check_gate(stage, binding):
    gate=json.loads((OUTPUT/f'{stage}_complete.json').read_text())
    if gate['binding']!=binding or not gate['passed']:
        raise ValueError('Earlier stage binding/gate invalid')
    return gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','validate','run'])
    args=parser.parse_args()
    # Cross-process locking prevents duplicated or interleaved checkpoint commits.
    with (OUTPUT/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_stage(args.stage)


def run_stage(stage):
    started=time.perf_counter()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logs=OUTPUT/'logs'; logs.mkdir(exist_ok=True)
    try:
        manifest=json.loads((OUTPUT/'manifest.json').read_text()); verify_manifest(manifest)
        config=configuration(); binding=digest(config)
        config_path=OUTPUT/'experiment_config.json'
        if config_path.exists():
            if json.loads(config_path.read_text())!=config:
                raise ValueError('Frozen config/code/model binding changed; refuse resume')
        else:
            atomic_json(config_path,config)
        if stage in ('validate','run'):
            check_gate('smoke',binding)
        if stage=='run':
            check_gate('validate',binding)
        samples=(manifest['validation_samples'] if stage=='validate' else manifest['samples'])
        if stage=='smoke':
            samples=[s for s in samples if s['question_id'] in manifest['smoke_question_ids']]
        directory=OUTPUT/'checkpoints'/stage
        existing=check_directory(directory,samples,binding)
        # Gate is recreated only after validating every checkpoint, never by row counts alone.
        if len(existing)==len(samples):
            atomic_json(OUTPUT/f'{stage}_complete.json',dict(binding=binding,passed=True,count=len(samples),resumed_all=True))
            print(f'{stage}: all {len(samples)} verified checkpoints already complete',flush=True)
            return
        resource=resource_preflight()
        atomic_json(logs/f'{stamp}_{stage}_resource.json',resource)
        if not resource['ready']:
            raise RuntimeError(resource.get('reason', 'GPU preflight failed'))
        model,processor,load_report=load_model()
        if len(model.language_model.model.layers)!=32 or model.config.image_seq_length!=576:
            raise ValueError('Unexpected model structure')
        atomic_json(logs/f'{stamp}_{stage}_environment.json',load_report)
        print(f'{stage}: loaded; {len(existing)}/{len(samples)} verified checkpoints; binding={binding}',flush=True)
        for i,sample in enumerate(samples):
            if sample['question_id'] in existing:
                continue
            print(f'{stage} {i+1}/{len(samples)} qid={sample["question_id"]}',flush=True)
            data=evaluate(model,processor,sample,validate=stage=='validate',smoke=stage=='smoke')
            if stage=='validate':
                data['historical_checks']=historical_check(data,sample)
            save_checkpoint(directory/f'{sample["question_id"]}.json',binding,sample,data)
        verified=check_directory(directory,samples,binding)
        atomic_json(OUTPUT/f'{stage}_complete.json',dict(binding=binding,passed=len(verified)==len(samples),count=len(verified),
                                                      wall_seconds_including_load_hash_audit=time.perf_counter()-started))
        print(f'{stage}: COMPLETE {len(verified)} samples, {time.perf_counter()-started:.3f}s',flush=True)
    except Exception as exc:
        resource_text=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
        atomic_json(logs/f'{stamp}_{stage}_failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                    diagnostic=getattr(exc,'diagnostic',None),nvidia_smi=resource_text,
                    recovery=f'Correct the external failure, preserve frozen files and rerun fixed_k_exit.py {stage}; incompatible bindings fail closed. Never terminate other GPU users.'))
        raise


if __name__=='__main__':
    main()
