"""CPU acceptance, real-model smoke, then the frozen single-GPU A/B pilot."""
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import unittest
import uuid

import numpy as np
from .protocol import (ROOT, OUTPUT, CONDITIONS, read, digest, sha, atomic, sources, packages,
                       prepare, protocol)


def acceptance(model_path):
    import vlm_common
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU gate must hide all GPUs')
    vlm_common.default_model=lambda:str(model_path)
    config,binding=protocol();packages()
    folder=OUTPUT/'acceptance'
    folder.mkdir(exist_ok=False)
    results={}
    for label,suite in [('focused',unittest.defaultTestLoader.loadTestsFromName('test_phase37')),
                        ('all',unittest.defaultTestLoader.discover(str(ROOT),pattern='test_*.py'))]:
        expected=suite.countTestCases()
        with (folder/f'{label}.txt').open('x') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        results[label]=dict(passed=result.wasSuccessful() and not result.skipped and result.testsRun==expected,
                            tests=result.testsRun,expected=expected,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped))
    for label,args in [('pip_check',['check']),('pip_freeze',['freeze'])]:
        result=subprocess.run([sys.executable,'-m','pip',*args],capture_output=True,text=True)
        (folder/f'{label}.txt').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('Package consistency check failed')
    passed=all(v['passed'] for v in results.values())
    atomic(folder/'validation.json',dict(passed=passed,tests=results,binding=binding,code_commit=config['code_commit'],
        packages=packages(),python=platform.python_version(),gpu_forward=False))
    print('CPU_ACCEPTANCE',results,flush=True)
    if not passed:raise RuntimeError('CPU acceptance failed; GPU execution prohibited')


def installed_sources():
    from transformers.models.llama.modeling_llama import LlamaAttention,LlamaDecoderLayer,LlamaModel
    from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
    return {str(Path(inspect.getfile(c))):sha(inspect.getfile(c)) for c in
            (LlamaAttention,LlamaDecoderLayer,LlamaModel,LlavaForConditionalGeneration)}


def inputs(model,processor,sample,parent):
    from vlm_common import prepare_inputs
    from attention_position_probe import locate_question_tokens
    from physical_exit_pilot import candidate_inputs
    if sha(parent['image_path'])!=parent['image_sha256']:raise ValueError('Frozen image changed')
    prompt,counts=prepare_inputs(model,processor,parent['image_path'],sample['question'])
    frozen=sample['input_mapping'];ids=prompt['input_ids'][0].cpu().tolist()
    q,audit=locate_question_tokens(processor,sample['question'],ids,model.config.image_token_index)
    if ids!=frozen['prompt_ids'] or q!=frozen['question_mapping'] or len(ids)-1!=frozen['answer_query']:
        raise ValueError('Actual CPU-frozen prompt/Question mapping changed')
    candidates=candidate_inputs(processor,prompt,['left','right'])
    for (x,records),old in zip(candidates,frozen['candidate_mapping']):
        if x['input_ids'][0].cpu().tolist()!=old['input_ids'] or records!=old['answer_records']:
            raise ValueError('Frozen complete candidate mapping changed')
    return prompt,candidates,[r['sequence_index'] for r in q],len(ids)-1,counts


def save_observation(path,value):
    np.savez_compressed(path,**value['arrays'],prediction_logits=value['prediction_logits'].numpy())


def one_b(model,processor,sample,folder):
    import torch
    from roi_sink_probe import score
    from .pathways import observed_forward,drift
    _,candidates,qpos,apos,counts=inputs(model,processor,sample,sample)
    def evaluate(condition,noop=False):
        values=[observed_forward(model,x,qpos,apos,condition,noop=noop) for x,_ in candidates]
        if not torch.equal(values[0]['prediction_logits'],values[1]['prediction_logits']):
            raise AssertionError('Candidate leaked into prediction query')
        if not np.array_equal(values[0]['arrays']['hidden'],values[1]['arrays']['hidden']):
            raise AssertionError('Candidate leaked into early Question/P-1 hidden')
        logs=[score(v,rec) for v,(_,rec) in zip(values,candidates)]
        return values,logs
    full,baseline=evaluate('B0');gold=['left','right'].index(sample['answer']);wrong=1-gold
    full_margin=baseline[gold]-baseline[wrong]
    noop,_=evaluate('B1',True)
    if not all(torch.equal(a['text_logits'],b['text_logits']) for a,b in zip(full,noop)):
        raise AssertionError('No-op effective-text logits differ from Native Full')
    rows=[];details={}
    for condition in CONDITIONS:
        values,logs=(full,baseline) if condition=='B0' else evaluate(condition)
        value=values[0];arrays=value['arrays'];base=full[0]['arrays'];margin=logs[gold]-logs[wrong]
        if not np.isfinite(logs).all():raise AssertionError('Invalid candidate logP')
        row=dict(question_id=sample['question_id'],img_id=sample['img_id'],answer=sample['answer'],condition=condition,
                 historical_full_margin=sample['historical_full_margin'],full_margin=full_margin,margin=margin,gain=margin-full_margin,
                 correct_logp=logs[gold],wrong_logp=logs[wrong],full_correct_logp=baseline[gold],
                 absolute_correct_logp_gain=logs[gold]-baseline[gold],full_correct=full_margin>0,correct=margin>0,tie=margin==0,
                 full_correct_to_wrong=full_margin>0 and margin<=0,full_wrong_to_correct=full_margin<=0 and margin>0,
                 forward_seconds=sum(v['measurement']['seconds'] for v in values))
        vector=value['prediction_logits'].double();ids=[rec[0]['token_id'] for _,rec in candidates]
        row.update(correct_logit=float(vector[ids[gold]]),wrong_logit=float(vector[ids[wrong]]),
                   full_vocabulary_logsumexp=float(torch.logsumexp(vector,-1)))
        for layer in (0,1):
            for group,index in [('question',slice(None,-1)),('answer',slice(-1,None))]:
                prefix=f'l{layer}_{group}'
                d=drift(base['hidden'][layer,index],arrays['hidden'][layer,index])
                row.update({f'{prefix}_{k}':v for k,v in d.items()})
                for j,key in enumerate(['image','question','other_text']):
                    current=float(arrays['mass'][layer,:,index,j].mean());original=float(base['mass'][layer,:,index,j].mean())
                    row[f'{prefix}_{key}_mass']=current;row[f'{prefix}_full_{key}_mass']=original
            local=value['audit']['local_redistribution'][layer]
            row[f'l{layer}_masked_local_native_mass_mean']=float(np.mean(local['masked_mass_before'])) if local['query_indices'] else 0.
            if local['query_indices']:
                a,b=np.asarray(local['before_group_mass']),np.asarray(local['after_group_mass'])
                for j,key in enumerate(['image','question','other_text']):
                    row[f'l{layer}_local_redistributed_{key}_mass']=float((b[:,:,j]-a[:,:,j]).mean())
            else:
                for key in ['image','question','other_text']:row[f'l{layer}_local_redistributed_{key}_mass']=0.
        final=drift(base['final_hidden'],arrays['final_hidden'])
        row.update({f'final_question_answer_{k}':v for k,v in final.items()})
        if condition in ('B2','B4'):
            if not np.array_equal(base['final_hidden'][:-1],arrays['final_hidden'][:-1]):
                raise AssertionError('Later prediction-only intervention changed earlier Question')
        save_observation(folder/f'{condition}.npz',value)
        details[condition]=dict(audits=[v['audit'] for v in values],measurements=[v['measurement'] for v in values])
        rows.append(row)
    from .pathways import observed_forward
    restored=observed_forward(model,candidates[0][0],qpos,apos)
    if not torch.equal(restored['text_logits'],full[0]['text_logits']):
        raise AssertionError('Final Native Full restoration failed')
    return dict(rows=rows,audits=details,noop_effective_text_bitwise_exact=True,post_cleanup_native_bitwise_exact=True,
                counts=counts,input_mapping=sample['input_mapping'],
                historical_full_margin_difference=full_margin-sample['historical_full_margin'])


def one_a(model,processor,sample,folder):
    from .pathways import observed_forward,spatial,js_cosine,drift
    questions=[sample,sample['pair']];values=[];rows=[]
    for number,s in enumerate(questions,1):
        prompt,_,qpos,apos,_=inputs(model,processor,s,sample)
        v=observed_forward(model,prompt,qpos,apos,probe=True)
        save_observation(folder/f'A_Q{number}.npz',v);values.append(v)
    matched=sample['matched_pair']
    if matched:
        for key in ['l0_answer_input_sha256','l0_visual_input_sha256']:
            if values[0]['audit'][key]!=values[1]['audit'][key]:raise AssertionError('Matched L0 inputs differ')
        for key in ['l0_visual_scores','l0_visual_only_conditional']:
            if not np.array_equal(values[0]['arrays'][key],values[1]['arrays'][key]):
                raise AssertionError('Matched L0 P-1 visual QK invariance failed; do not loosen')
    for layer in (0,1):
        for group in ('question','answer'):
            selector=slice(None,-1) if group=='question' else slice(-1,None)
            metrics=[spatial(v['arrays']['attention'][layer,:,selector,:]) for v in values]
            if not all(m['available'] for m in metrics):raise AssertionError('Native A visual Mass is zero')
            maps=[m['distribution'] for m in metrics]
            compared=js_cosine(*maps)
            h=[v['arrays']['hidden'][layer,selector].mean(0) for v in values]
            row=dict(question_id=sample['question_id'],paired_question_id=sample['pair']['question_id'],img_id=sample['img_id'],
                     layer=layer,query_group=group,matched_prompt_length_position_token=matched,
                     l0_p1_qk_invariant=True if matched else None,
                     q1_image_mass=metrics[0]['mass'],q2_image_mass=metrics[1]['mass'],
                     q1_normalized_entropy=metrics[0]['normalized_entropy'],q2_normalized_entropy=metrics[1]['normalized_entropy'],
                     hidden_pooled_cosine_distance=drift(h[0],h[1])['cosine_distance_mean'],
                     hidden_pooled_change_norm=drift(h[0],h[1])['change_norm_mean'],**compared)
            targets=[s['objects'][1]['patch_indices'] for s in questions]
            for i in (0,1):
                for name,keys in [('own_target',targets[i]),('other_question_target',targets[1-i]),
                                  ('anchor',questions[i]['objects'][0]['patch_indices']),
                                  ('background',sorted(set(range(576))-set(targets[0])-set(targets[1])))]:
                    row[f'q{i+1}_{name}_enrichment']=float(maps[i][keys].sum()/(len(keys)/576)) if keys else None
            rows.append(row)
    return dict(rows=rows,matched_l0_qk_invariance_passed=True if matched else None,
                audits=[v['audit'] for v in values],measurements=[v['measurement'] for v in values],
                question_ids=[s['question_id'] for s in questions])


def checkpoint(path,binding,sample,stage_root):
    envelope=read(path);body=envelope['body']
    if digest(body)!=envelope['sha256'] or body['binding']!=binding or body['sample_sha256']!=digest(sample) or not body['passed']:
        raise ValueError('Checkpoint identity/binding/hash failed')
    if [r['condition'] for r in body['b']['rows']]!=CONDITIONS:
        raise ValueError('Incomplete/duplicate B conditions')
    if body['question_id']!=sample['question_id'] or any(r['question_id']!=sample['question_id'] for r in body['b']['rows']):
        raise ValueError('Checkpoint/score row question identity mismatch')
    if bool(body['a'])!=bool(sample['pair']):raise ValueError('A coverage mismatch')
    for name,h in body['artifacts'].items():
        p=(stage_root/name).resolve()
        if not p.is_relative_to(stage_root.resolve()) or sha(p)!=h:raise ValueError('Artifact mismatch/escape')
    return body


def execute(stage):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Expose only the preflight-checked GPU0')
    config,binding=protocol();packages()
    cpu=read(OUTPUT/'acceptance/validation.json')
    if not cpu['passed'] or cpu['binding']!=binding:raise ValueError('CPU acceptance gate failed')
    if stage=='pilot':
        smoke=read(OUTPUT/'smoke/validation.json')
        if not smoke['passed'] or smoke['binding']!=binding:raise ValueError('GPU smoke gate failed')
    samples=[s for s in config['samples'] if stage=='pilot' or s['question_id'] in config['smoke_ids']]
    root=OUTPUT/stage;root.mkdir(exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        points=root/'checkpoints';points.mkdir(exist_ok=True)
        allowed={s['question_id']:s for s in samples};done={}
        for p in points.glob('*.json'):
            if p.stem not in allowed:raise ValueError('Foreign checkpoint')
            done[p.stem]=checkpoint(p,binding,allowed[p.stem],root)
        from vlm_common import load_model
        from knockout_probe import resource_preflight
        from .pathways import state_hash
        resource=resource_preflight()
        if not resource['ready']:raise RuntimeError(f'GPU preflight failed: {resource}')
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        model,processor,environment=load_model(model_path=config['model_path'])
        atomic(root/f'environment_{stamp}.json',dict(environment=environment,resource=resource,binding=binding,
               code_commit=config['code_commit'],packages=packages(),python=platform.python_version(),installed_sources=installed_sources()))
        original_weights=state_hash(model);libraries=installed_sources();started=time.perf_counter()
        try:
            for s in samples:
                if s['question_id'] in done:continue
                if config['sources']!=sources():raise ValueError('Source changed while running')
                folder=root/'samples'/f"{s['question_id']}_{stamp}";folder.mkdir(parents=True,exist_ok=False)
                try:
                    b=one_b(model,processor,s,folder)
                    a=one_a(model,processor,s,folder) if s['pair'] else None
                    with gzip.open(folder/'audit.json.gz','wt') as f:json.dump(dict(b=b,a=a),f,allow_nan=False)
                    artifacts={str(p.relative_to(root)):sha(p) for p in folder.iterdir() if p.is_file()}
                    body=dict(binding=binding,sample_sha256=digest(s),passed=True,question_id=s['question_id'],
                              b=dict(rows=b['rows'],historical_full_margin_difference=b['historical_full_margin_difference']),
                              a=dict(rows=a['rows'],matched_l0_qk_invariance_passed=a['matched_l0_qk_invariance_passed']) if a else None,
                              artifacts=artifacts)
                    path=points/f"{s['question_id']}.json"
                    if path.exists():raise FileExistsError('Never overwrite completed sample')
                    atomic(path,dict(body=body,sha256=digest(body)));done[s['question_id']]=checkpoint(path,binding,s,root)
                    print(json.dumps(dict(stage=stage,complete=len(done),expected=len(samples),question_id=s['question_id'],a_pair=bool(a))),flush=True)
                except BaseException as error:
                    atomic(folder/'failure.json',dict(status='partial',error=repr(error),traceback=traceback.format_exc()));raise
            final_weights=state_hash(model)
            if original_weights!=final_weights or installed_sources()!=libraries:raise AssertionError('Weights/third-party source changed')
            if stage=='smoke':
                import torch
                from .pathways import observed_forward
                s=samples[0];_,cc,qpos,apos,_=inputs(model,processor,s,s)
                with torch.inference_mode():plain=model(**cc[0][0],output_attentions=False,use_cache=False,return_dict=True)
                observed=observed_forward(model,cc[0][0],qpos,apos)
                if not torch.equal(plain.logits[0,apos].cpu(),observed['prediction_logits']):raise AssertionError('Observer changed Native prediction')
            atomic(root/'validation.json',dict(passed=True,status='completed',binding=binding,code_commit=config['code_commit'],
                count=len(done),expected=len(samples),question_ids=[s['question_id'] for s in samples],
                a_pairs=sum(bool(v['a']) for v in done.values()),weights_sha256_before=original_weights,
                weights_sha256_after=final_weights,third_party_source_unchanged=True,
                checkpoints_sha256={q:sha(points/f'{q}.json') for q in done},wall_seconds=time.perf_counter()-started))
            print('COMPLETE',stage,len(done),flush=True)
        except BaseException as error:
            atomic(root/f'failure_{stamp}.json',dict(status='partial',complete=len(done),error=repr(error),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','acceptance','smoke','pilot'])
    p.add_argument('--model-path',type=Path);p.add_argument('--image-root',type=Path);args=p.parse_args()
    if args.stage=='prepare':
        if not args.model_path or not args.image_root:p.error('prepare needs fixed local model and image roots')
        prepare(args.model_path.resolve(),args.image_root.resolve())
    elif args.stage=='acceptance':acceptance(args.model_path or Path(read(OUTPUT/'manifest.json')['config']['model_path']))
    else:execute(args.stage)


if __name__=='__main__':main()
