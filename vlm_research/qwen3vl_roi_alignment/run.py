import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from ..qwen3vl_early_windows.experiment import MODEL,REVISION,hook_counts,tensor_digest
from ..qwen3vl_image_source.run import environment
from ..qwen3vl_keep_l26.run import load,inputs_for
from .engine import AttentionRows,NativeDeepStack,summarize
from .preflight import OUT,HISTORY,ROOT,dump,sha

GPUS=(0,2,3)


def read(p):return json.loads(Path(p).read_text())


def sources():
    paths=[Path(__file__).parent/n for n in ('engine.py','run.py','annotations.py','processor_check.py','preflight.py')]
    paths += [ROOT/'vlm_research'/n for n in ('qwen3vl_early_windows/experiment.py','qwen3vl_smoke/common.py',
              'qwen3vl_image_source/run.py','qwen3vl_keep_l26/run.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def freeze():
    assert read(OUT/'annotation_resolution.json')['samples']==240
    assert read(OUT/'processor_validation.json')['count']==240
    assert environment()==read(HISTORY/'environment.json')
    mapping=read(OUT/'roi_mapping.json')
    assert len(mapping)==240
    old=read(HISTORY/'model_manifest.json')
    for item in old['files']:
        assert sha(MODEL/item['path'])==item['sha256']
    dump(OUT/'environment.json',environment())
    dump(OUT/'protocol.json',dict(status='frozen_ready_for_gpu_smoke',
        initial_partial_commit='812f3e495',model_repo='Qwen/Qwen3-VL-8B-Instruct',revision=REVISION,
        dtype='bfloat16',attention='eager',layers=36,samples=240,physical_gpus=GPUS,
        sample_manifest_sha256=sha(OUT/'sample_manifest.json'),roi_mapping_sha256=sha(OUT/'roi_mapping.json'),
        annotation_resolution_sha256=sha(OUT/'annotation_resolution.json'),
        model_manifest_sha256=sha(HISTORY/'model_manifest.json'),environment=environment(),code_sha256=sources(),
        queries=['P-1','Question'],question_aggregation='Arithmetic mean of native attention over heads and all true question subtoken queries, then image-normalized ratios',
        roi_definition='Union of positive-area intersecting merged visual cells of exact terminal-semantic target object BBoxes; clipped to actual image bounds with provenance',
        roi_hit_ratio='ROI Mass / Image Mass',roi_enrichment='ROI Hit Ratio / (ROI merged-token count / all merged visual tokens)',
        random_baseline=dict(repetitions=128,algorithm='numpy PCG64; sample K merged tokens uniformly without replacement, allow overlap with actual ROI',
                             same_sets_all_layers_and_queries=True,seed='First 64 SHA256 bits of random_roi_v1:<input_binding>'),
        intervention=False,generation=False,visual_replay=False,deepstack='Native, unchanged',
        primary_sensitivity='absolute_logp_change',secondary_sensitivity='signed_logp_loss',
        sensitivity_csv_sha256=sha(HISTORY/'per_question.csv'),
        smoke_keys=read(OUT/'processor_validation.json')['smoke_keys'],
        bootstrap=dict(seed=20261102,repetitions=2000,cluster='Image SHA256',estimator='Question-weighted, all same-image questions resampled together',interval='Ordinary percentile 95%, descriptive/exploratory, no causal or confirmatory significance claims'),
        primary_alignment='Spearman of 36 layer-mean ROI Enrichment vs mean absolute LogP change; per-question layerwise Spearman; same-layer cross-question L26 Spearman',
        peak_definition='Lowest layer index attaining maximum, no smoothing; ROI Enrichment peak minus absolute LogP change peak',
        native_row_sum_tolerance='Unchanged historical BF16 threshold 2**-8 + 2e-6',
        stop_on_failure=True))
    print('Frozen native Full ROI protocol and exact 240-question mapping.',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    assert p['code_sha256']==sources() and p['environment']==environment()
    assert p['sample_manifest_sha256']==sha(OUT/'sample_manifest.json')==sha(HISTORY/'sample_manifest.json')
    assert p['roi_mapping_sha256']==sha(OUT/'roi_mapping.json')
    assert p['sensitivity_csv_sha256']==sha(HISTORY/'per_question.csv')
    return p


@torch.inference_mode()
def forward(model,inputs,layout,collect=True,deepstack=False):
    hooks=hook_counts(model)
    with ExitStack() as stack:
        rows=stack.enter_context(AttentionRows(model,layout)) if collect else None
        ds=stack.enter_context(NativeDeepStack(model,layout['image_positions'])) if deepstack else None
        result=model(**inputs,use_cache=False,past_key_values=None,
            cache_position=torch.arange(layout['input_length'],device='cuda:0'),
            logits_to_keep=1,output_attentions=False,output_hidden_states=False)
        assert result.past_key_values is None
        logits=result.logits[0,-1].detach().cpu().clone()
        assert torch.isfinite(logits).all()
        visual=rows.arrays() if rows else None
        errors=[rows.records[i]['row_sum_errors'] for i in range(36)] if rows else None
        if ds:ds.validate()
    assert hook_counts(model)==hooks
    return logits,visual,errors


def historical(sample):
    key=f'{sample["task"]}__{sample["question_id"]}'
    old=read(HISTORY/f'shard_{sample["shard"]}'/'checkpoints'/f'{key}.json')
    assert old['input_binding']==sample['input_binding']
    return old


def collect_sample(model,processor,sample,mapping,folder,smoke=False,deepstack=False):
    inputs=inputs_for(processor,sample)
    original={k:tensor_digest(v) for k,v in inputs.items()}
    baseline,_,_=forward(model,inputs,sample['layout'],collect=False) if smoke else (None,None,None)
    logits,visual,errors=forward(model,inputs,sample['layout'],deepstack=deepstack)
    if smoke:assert torch.equal(logits,baseline), 'Read-only hooks changed native logits'
    logit_hash=tensor_digest(logits)
    old=historical(sample)
    assert logit_hash==old['full_logits_sha256'], 'Current native Full differs from same-hardware frozen Full'
    assert original=={k:tensor_digest(v) for k,v in inputs.items()}
    values=summarize(visual,mapping)
    rows=[]
    for layer in range(36):
        oldrow=old['rows'][layer]
        assert oldrow['layer']==layer and oldrow['image_sha256']==sample['image_sha256']
        for query_index,query in enumerate(('P-1','Question')):
            rows.append(dict(task=sample['task'],question_id=sample['question_id'],image_id=sample['image_id'],
                image_sha256=sample['image_sha256'],layer=layer,query=query,
                image_tokens=visual.shape[-1],roi_tokens=len(mapping['roi_token_indices']),
                roi_fraction=mapping['roi_fraction'],question_tokens=len(sample['layout']['question_positions']),
                physical_gpu=sample['physical_gpu'],p_minus_1=sample['layout']['p_minus_1'],
                row_sum_max_error=errors[layer][query_index],
                **{k:float(v[layer,query_index]) for k,v in values.items()},
                absolute_logp_change=oldrow['absolute_logp_change'],signed_logp_loss=oldrow['signed_logp_loss'],
                top1_flip=oldrow['top1_flip']))
    key=f'{sample["task"]}__{sample["question_id"]}'
    data=folder/'rows'/f'{key}.npz';data.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(data,visual_attention=visual)
    record=dict(input_binding=sample['input_binding'],protocol_sha256=sha(OUT/'protocol.json'),
        full_logits_sha256=logit_hash,historical_full_exact=True,full_first_token=int(logits.argmax()),
        row_array_sha256=sha(data),rows=rows,smoke=smoke,native_hook_logits_exact=smoke,
        native_deepstack_addition_exact=deepstack)
    target=folder/'checkpoints'/f'{key}.json';target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix('.tmp');dump(tmp,record);tmp.replace(target)
    return record


def worker(shard):
    p=verify();gpu=GPUS[shard]
    folder=OUT/f'shard_{shard}';folder.mkdir(parents=True,exist_ok=True)
    samples=[s for s in read(OUT/'sample_manifest.json')['samples'] if s['shard']==shard]
    assert len(samples)==80
    model,processor=load(gpu,folder)
    versions=[(v._version,v.data_ptr()) for v in model.parameters()]
    hooks=hook_counts(model);mapping=read(OUT/'roi_mapping.json')
    start=time.perf_counter();smoke=[]
    for s in samples:
        key=f'{s["task"]}__{s["question_id"]}'
        if key in p['smoke_keys']:
            collect_sample(model,processor,s,mapping[key],folder,smoke=True,deepstack=not smoke)
            smoke.append(key)
    assert smoke
    try:
        with AttentionRows(model,samples[0]['layout']):raise RuntimeError('cleanup fixture')
    except RuntimeError as exc:assert str(exc)=='cleanup fixture'
    assert hook_counts(model)==hooks
    validation=dict(passed=True,physical_gpu=gpu,smoke_keys=smoke,smoke_count=len(smoke),
        native_and_historical_logits_exact=True,all_36_layers_captured=True,attention_rows_valid=True,
        native_deepstack_addition_exact=True,normal_and_exception_hooks_clean=True,
        seconds=time.perf_counter()-start,protocol_sha256=sha(OUT/'protocol.json'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    dump(folder/'validation.json',validation)
    print(f'GPU{gpu} smoke passed: {len(smoke)} questions, {validation["seconds"]:.1f}s',flush=True)
    while not all((OUT/f'shard_{i}/validation.json').exists() for i in range(3)):
        assert not any((OUT/f'shard_{i}/error.json').exists() for i in range(3)), 'Peer smoke failed'
        time.sleep(1)
    for i in range(3):
        peer=read(OUT/f'shard_{i}/validation.json')
        assert peer['passed'] and peer['protocol_sha256']==validation['protocol_sha256']
        assert peer['code_commit']==validation['code_commit']
    start=time.perf_counter();reused=0
    for index,s in enumerate(samples):
        key=f'{s["task"]}__{s["question_id"]}'
        path=folder/'checkpoints'/f'{key}.json'
        if path.exists():
            r=read(path)
            assert r['protocol_sha256']==validation['protocol_sha256'] and r['input_binding']==s['input_binding']
            assert len(r['rows'])==72 and r['historical_full_exact']
            assert sha(folder/'rows'/f'{key}.npz')==r['row_array_sha256']
            reused+=1
        else:collect_sample(model,processor,s,mapping[key],folder)
        dump(folder/'progress.json',dict(done=index+1,total=80,seconds=time.perf_counter()-start))
        print(f'GPU{gpu} {index+1}/80 {key} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    assert hook_counts(model)==hooks and versions==[(v._version,v.data_ptr()) for v in model.parameters()]
    dump(folder/'complete.json',dict(passed=True,count=80,reused_smoke_or_checkpoint=reused,
        formal_seconds=time.perf_counter()-start,smoke_seconds=validation['seconds'],
        protocol_sha256=validation['protocol_sha256'],code_commit=validation['code_commit'],
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,hooks_clean=True,weights_unchanged=True))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','worker']);a.add_argument('--shard',type=int,choices=range(3));args=a.parse_args()
    if args.action=='freeze':freeze()
    else:
        try:worker(args.shard)
        except BaseException:
            dump(OUT/f'shard_{args.shard}/error.json',dict(traceback=traceback.format_exc()))
            raise
