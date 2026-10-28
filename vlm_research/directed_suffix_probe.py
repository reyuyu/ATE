"""Phase 3.3 directed suffix experiment; frozen Phase3.2 cohort, no decode."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from vlm_common import ROOT, load_model, prepare_inputs
from knockout_probe import resource_preflight, model_layers
from attention_position_probe import locate_question_tokens
from physical_exit import TOLERANCES, equivalence, cosine_distances
from physical_exit_pilot import candidate_inputs, sequence_score, pack_forward
from fixed_k_exit import atomic_json, configuration as phase32_configuration, load_checkpoint as load_phase32
from prepare_fixed_k_data import digest
from prepare_physical_exit_data import sha256
from fixed_k_exit import verify_manifest
from directed_suffix import MODES, CUTOFFS, query_mapping, directed_forward

OUTPUT=ROOT/'results/phase3_3_directed_suffix'
PREVIOUS=ROOT/'results/phase3_2_fixed_k_exit'
FOCUS=['0216546','02310173','05480434']
KEYS=[('full',32)]+[(mode,k) for k in CUTOFFS for mode in MODES]


def config_and_manifest():
    manifest=json.loads((PREVIOUS/'manifest.json').read_text());verify_manifest(manifest)
    previous=phase32_configuration()
    if previous!=json.loads((PREVIOUS/'experiment_config.json').read_text()):
        raise ValueError('Phase3.2 frozen implementation/model/environment changed')
    code={name:sha256(ROOT/name) for name in ['directed_suffix.py','directed_suffix_probe.py']}
    smoke=[]
    for label in ('left','right'):
        smoke.extend([s['question_id'] for s in manifest['samples'] if s['answer']==label][:3])
    smoke=list(dict.fromkeys(smoke+FOCUS))
    config=dict(phase='3.3',cutoffs=CUTOFFS,modes=MODES,phase32_binding=digest(previous),
                phase32_commit='78f1f4a3d00bb04e43b1c4118495a4a611b31bb8',phase32_configuration=previous,
                code_sha256=code,manifest_sha256=sha256(PREVIOUS/'manifest.json'),
                smoke_question_ids=smoke,smoke_rule='First three per gold label in frozen Phase3.2 order plus the three user-specified historical failures, deduplicated. All 240 retained for production.',
                protocol='Same native Full and complete candidate joint-logP scoring; left/right each one token; Answer query P-1, not input answer P. Masks on suffix K..31; no physical deletion. Final RMSNorm Question hidden. Strict margin>0, ties incorrect.',
                tolerances=TOLERANCES,causal_hidden_checks='Bitwise equality for A-only Question vs Full and Q+A Question vs Q-only; K32 all valid text hidden/logits bitwise equal.',
                timing='Diagnostic forward seconds, including mask checking/clone/hooks, no speedup claim. Smoke enables output_attentions; production does not.')
    return config,manifest


def validate_data(data,sample):
    rows=data['rows']
    if len(rows)!=len(KEYS) or sorted((r['mode'],r['cutoff']) for r in rows)!=sorted(KEYS):
        raise ValueError('Incomplete/duplicate mode,K checkpoint')
    if any(r['question_id']!=sample['question_id'] or r['img_id']!=sample['img_id'] or r['answer']!=sample['answer'] for r in rows):
        raise ValueError('Row identity mismatch')
    if not all(data['checks'][key] for key in ('k32_exact','causal_hidden_exact','global_history_within_tolerance','full_history_exact')):
        raise ValueError('Invalid result correctness flags')
    return data


def read_checkpoint(path,binding,sample):
    envelope=json.loads(Path(path).read_text());body=envelope['body']
    if digest(body)!=envelope['body_sha256'] or body['binding']!=binding or body['sample']!=sample:
        raise ValueError('Checkpoint checksum/config/sample mismatch')
    return validate_data(body['data'],sample)


def checkpoint(path,binding,sample,data):
    if path.exists():raise ValueError('Refuse checkpoint overwrite')
    validate_data(data,sample)
    body=dict(binding=binding,sample=sample,data=data)
    atomic_json(path,dict(body=body,body_sha256=digest(body)))


def existing_results(stage,samples,binding):
    directory=OUTPUT/'checkpoints'/stage;directory.mkdir(parents=True,exist_ok=True)
    lookup={s['question_id']:s for s in samples}; existing={}
    for path in directory.glob('*.json'):
        if path.stem not in lookup:raise ValueError('Unexpected checkpoint ID')
        existing[path.stem]=read_checkpoint(path,binding,lookup[path.stem])
    return existing


def packed(model,inputs,mode,k,mapping,records,audit=False):
    out=directed_forward(model,inputs,mode,k,mapping,audit=audit)
    valid=torch.where((inputs['input_ids'][0]!=model.config.image_token_index)&inputs['attention_mask'][0].bool())[0]
    query=torch.tensor([r['query_position'] for r in records],device=valid.device)
    qpos=torch.tensor(mapping['question_queries'],device=valid.device)
    result=dict(hidden=out['hidden'].index_select(1,valid).cpu(),logits=out['logits'].index_select(1,valid).cpu(),
                question_hidden=out['hidden'].index_select(1,qpos).cpu(),
                logp=sequence_score(out['logits'].index_select(1,query)[0].cpu(),[r['token_id'] for r in records]),
                measurement=out['measurement'],hook_audit=out['hook_audit'],attention_audit=out['attention_audit'])
    del out
    return result,valid


def evaluate(model,processor,sample,history,smoke=False):
    prompt,counts=prepare_inputs(model,processor,sample['image_path'],sample['question'])
    ids=prompt['input_ids'][0].cpu().numpy();P=len(ids)
    qrows,qaudit=locate_question_tokens(processor,sample['question'],ids,model.config.image_token_index)
    qpos=[r['sequence_index'] for r in qrows]
    candidates=candidate_inputs(processor,prompt,sample['candidates'])
    maps=[query_mapping(x,model.config.image_token_index,qpos,rec,P) for x,rec in candidates]
    if maps[0]!=maps[1]:raise AssertionError('Candidate query/key mapping differs')
    if ids.tolist()!=history['prompt_ids'] or qrows!=history['question_mapping']:
        raise AssertionError('Prompt/real Question mapping differs from frozen Phase3.2')
    for (x,rec),old in zip(candidates,history['candidate_mapping']):
        if x['input_ids'][0].cpu().tolist()!=old['input_ids'] or x['attention_mask'][0].cpu().tolist()!=old['attention_mask'] or rec!=old['answer_records']:
            raise AssertionError('Candidate IDs/masks/answer positions changed')
    checks=dict(k32_exact=True,causal_hidden_exact=True,global_history_within_tolerance=True,full_history_exact=True)
    full=[];valid=[]
    for (x,rec),mapping,old in zip(candidates,maps,history['full_candidates']):
        value,p=packed(model,x,'full',32,mapping,rec)
        if value['logp']!=old['logp']:raise AssertionError('Native Full score changed since Phase3.2')
        full.append(value);valid.append(p)
    target=sample['candidates'].index(sample['answer']);wrong=1-target
    full_margin=full[target]['logp']-full[wrong]['logp']
    data=dict(rows=[],details=[],counts=counts,prompt_ids=ids.tolist(),query_mapping=maps[0],question_token_records=qrows,
              question_audit=qaudit,candidate_records=[rec for x,rec in candidates],checks=checks)
    def record(mode,k,values,extra=None):
        margin=values[target]['logp']-values[wrong]['logp']
        drift=cosine_distances(full[0]['question_hidden'],values[0]['question_hidden'])[0]
        row=dict(question_id=sample['question_id'],img_id=sample['img_id'],answer=sample['answer'],mode=mode,cutoff=k,
                 full_correct=full_margin>0,candidate_correct=margin>0,full_margin=full_margin,answer_margin=margin,
                 answer_margin_drop=full_margin-margin,correct_answer_log_probability=values[target]['logp'],
                 wrong_answer_log_probability=values[wrong]['logp'],correct_answer_sequence_probability=float(np.exp(values[target]['logp'])),
                 correct_probability_within_two_candidates=float(torch.sigmoid(torch.tensor(margin,dtype=torch.float64))),
                 question_mean_cosine_distance=float(drift.mean()),question_max_cosine_distance=float(drift.max()),
                 sequence_length=candidates[0][0]['input_ids'].shape[1],physical_deletion=False,
                 forward_seconds_sum_two_candidates=sum(v['measurement']['forward_seconds'] for v in values),
                 peak_allocated_mib=max(v['measurement']['gpu_memory']['peak_allocated_mib'] for v in values),
                 peak_reserved_mib=max(v['measurement']['gpu_memory']['peak_reserved_mib'] for v in values))
        data['rows'].append(row)
        data['details'].append(dict(mode=mode,cutoff=k,question_per_token_cosine_distance=drift.tolist(),extra_checks=extra,
                                   candidates=[dict(label=label,logp=v['logp'],measurement=v['measurement'],
                                                    hook_audit=v['hook_audit'],attention_audit=v['attention_audit'])
                                               for label,v in zip(sample['candidates'],values)]))
    record('full',32,full)
    for k in CUTOFFS:
        q_only_hidden=None
        for mode in MODES:
            values=[packed(model,x,mode,k,mapping,rec,audit=smoke)[0] for (x,rec),mapping in zip(candidates,maps)]
            if k==32 and not all(torch.equal(a['hidden'],b['hidden']) and torch.equal(a['logits'],b['logits']) for a,b in zip(full,values)):
                raise AssertionError('K32 not bitwise native Full')
            if mode=='question_only':q_only_hidden=[v['question_hidden'] for v in values]
            if mode=='answer_only' and not all(torch.equal(a['question_hidden'],b['question_hidden']) for a,b in zip(full,values)):
                raise AssertionError('Answer-only modified causally earlier Question states')
            if mode=='question_answer' and not all(torch.equal(a,b['question_hidden']) for a,b in zip(q_only_hidden,values)):
                raise AssertionError('Q+A Question states differ from Q-only')
            extra=None
            if mode=='global':
                old=next(d for d in history['details'] if d['cutoff']==k)
                old_margin=old['candidates'][target]['logp']-old['candidates'][wrong]['logp']
                logp_error=max(abs(a['logp']-b['logp']) for a,b in zip(values,old['candidates']))
                error=abs(values[target]['logp']-values[wrong]['logp']-old_margin)
                extra=dict(history_physical_margin=old_margin,candidate_logp_max_abs_error=logp_error,
                           margin_abs_error=error,history_physical_correct=old_margin>0,
                           global_vs_historical_classification_changed=(values[target]['logp']-values[wrong]['logp']>0)!=(old_margin>0))
                if logp_error>TOLERANCES['candidate_joint_logp_atol_nats'] or error>TOLERANCES['candidate_margin_atol_nats']:
                    exc=AssertionError('Global vs historical Physical exceeds frozen tolerances')
                    exc.diagnostic=dict(question_id=sample['question_id'],cutoff=k,**extra);raise exc
                if smoke:
                    physical=[pack_forward(model,x,'physical',k,p,qpos,rec) for (x,rec),p in zip(candidates,valid)]
                    eq=[equivalence(a['hidden'],b['hidden'],a['logits'],b['logits']) for a,b in zip(values,physical)]
                    if not all(c['passed'] for c in eq):
                        exc=AssertionError('GPU Global/Physical hidden-logit equivalence failed')
                        exc.diagnostic=dict(question_id=sample['question_id'],cutoff=k,checks=eq);raise exc
                    extra['current_physical_equivalence']=eq
                    del physical
            record(mode,k,values,extra)
            del values
    if smoke:
        for (x,rec),mapping,baseline in zip(candidates,maps,full):
            after,_=packed(model,x,'full',32,mapping,rec)
            if not torch.equal(after['logits'],baseline['logits']) or not torch.equal(after['hidden'],baseline['hidden']):
                raise AssertionError('Post-hook native Full contaminated')
        checks['post_cleanup_native_full_exact']=True
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['smoke','run']);args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.stage)


def run(stage):
    started=time.perf_counter();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logs=OUTPUT/'logs';logs.mkdir(exist_ok=True)
    try:
        config,manifest=config_and_manifest();binding=digest(config)
        dest=OUTPUT/'manifest.json'
        if dest.exists() and sha256(dest)!=config['manifest_sha256']:raise ValueError('Manifest copy changed')
        if not dest.exists():dest.write_bytes((PREVIOUS/'manifest.json').read_bytes())
        cp=OUTPUT/'experiment_config.json'
        if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Frozen binding changed; refuse resume')
        if not cp.exists():atomic_json(cp,config)
        smoke_samples=[s for s in manifest['samples'] if s['question_id'] in config['smoke_question_ids']]
        if stage=='run':
            gate=json.loads((OUTPUT/'smoke_complete.json').read_text())
            verified=existing_results('smoke',smoke_samples,binding)
            if gate['binding']!=binding or not gate['passed'] or len(verified)!=len(smoke_samples):raise ValueError('Smoke gate/checkpoints invalid')
        samples=smoke_samples if stage=='smoke' else manifest['samples']
        existing=existing_results(stage,samples,binding)
        pending=[s for s in samples if s['question_id'] not in existing]
        if not pending:
            print(f'{stage}: all {len(samples)} checkpoints verified; no GPU work',flush=True);return
        resource=resource_preflight();atomic_json(logs/f'{stamp}_{stage}_resource.json',resource)
        if not resource['ready']:raise RuntimeError(resource.get('reason','GPU not available'))
        model,processor,load_report=load_model();atomic_json(logs/f'{stamp}_{stage}_environment.json',load_report)
        if len(model_layers(model))!=32:raise ValueError('Unexpected model structure')
        for i,sample in enumerate(samples):
            if sample['question_id'] in existing:continue
            print(f'{stage} {i+1}/{len(samples)} qid={sample["question_id"]}',flush=True)
            history=load_phase32(PREVIOUS/'checkpoints/run'/f'{sample["question_id"]}.json',config['phase32_binding'],sample)
            data=evaluate(model,processor,sample,history,smoke=stage=='smoke')
            checkpoint(OUTPUT/'checkpoints'/stage/f'{sample["question_id"]}.json',binding,sample,data)
        all_results=existing_results(stage,samples,binding)
        atomic_json(OUTPUT/f'{stage}_complete.json',dict(binding=binding,passed=len(all_results)==len(samples),count=len(all_results),
                                                      wall_seconds_including_audit_load=time.perf_counter()-started))
        print(f'{stage} COMPLETE {len(all_results)}, wall={time.perf_counter()-started:.3f}s',flush=True)
    except Exception as exc:
        atomic_json(logs/f'{stamp}_{stage}_failure.json',dict(error=str(exc),traceback=traceback.format_exc(),diagnostic=getattr(exc,'diagnostic',None),
                   nvidia_smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,
                   recovery=f'Preserve frozen inputs/code; rerun directed_suffix_probe.py {stage}. No other processes are terminated.'))
        raise


if __name__=='__main__':main()
