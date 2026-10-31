"""One idle-GPU worker, frozen CPU/GPU acceptance and atomic per-condition resume."""
import argparse
from contextlib import ExitStack
import fcntl
import gc
import io
import json
import os
import subprocess
import time
import traceback
import unittest

import numpy as np
from PIL import Image
import torch
from benchmarks.dataset_registry import digest,sha256
from benchmarks.smoke import hook_footprint
from decode_five_vs_six import ENV_KEYS
from knockout_probe import model_layers
from l0l1_bos.protocol import preflight
from l0l1_bos.run import first_metrics_from_history
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from paper_rq1_early_joint_attention.engine import JointAttention
from vlm_common import load_model,gpu_memory
from .protocol import (ROOT,OUT,GPU,CONDITIONS,old,read,identity,sources,prepare,verify,entries,
    check_reuse,freeze,execution,atomic_json,true_mapping)
from .engine import observed,measurements,SourceMask


def cpu():
    stream=io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ['test_early_source_delay','test_early_joint_attention','test_first_answer_attention'])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    atomic_json(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        source_sha256=sources(),output=stream.getvalue()))
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful()


def check_first(logits,previous,early=False):
    expected=previous['first_logits_sha256'] if early else previous['L0L5_OFF']['full_first_logits_sha256']
    assert digest(logits.tolist())==expected,'Native/history first logits not exact'


def smoke(model,processor):
    chosen=entries(); selected=[chosen[0],next(e for e in chosen if e['sample']['task']==list(old.TASKS)[-1])]
    result=[];footprint=hook_footprint(model)
    for entry in selected:
        s=entry['sample']; mapping=entry['mapping']; previous,early=check_reuse(entry)
        prompt=prepared(model,processor,s); before=tensor_hashes(prompt)
        assert true_mapping(processor,s)==mapping
        full,first,arrays,stats,au,trace=observed(model,prompt,mapping,{},audit=True,collect=True)
        check_first(first,previous)
        assert full['token_ids']==previous['Full']['token_ids']
        checks=[]
        for name,c in CONDITIONS.items():
            noop,nf,na,ns,nu,nt=observed(model,prompt,mapping,c,noop=True,audit=True,collect=True)
            assert noop['token_ids']==full['token_ids'] and torch.equal(first,nf)
            assert noop['initial_cache_sha256']==full['initial_cache_sha256']
            assert len(trace)==len(nt) and all(torch.equal(a,b) for a,b in zip(trace,nt))
            calls=[0]*32;handles=[]
            try:
                for k,block in enumerate(model_layers(model)):
                    def mlp(module,args,out,k=k):
                        calls[k]+=1; assert torch.isfinite(out).all()
                    handles.append(block.mlp.register_forward_hook(mlp))
                g,logits,a,st,audit,gt=observed(model,prompt,mapping,c,audit=True,collect=True)
            finally:
                for h in handles:h.remove()
            assert all(v==g['length'] for v in calls)
            assert g['edits']==(2 if c.get('early_off') else 0)
            assert torch.equal(logits,gt[0])
            if name=='EarlyOFF':check_first(logits,early,early=True)
            measured=measurements(a,st,mapping,arrays)
            with ExitStack() as stack:
                stack.enter_context(SourceMask(model,prompt,mapping,c,audit=True))
                layers=[0,1] if c.get('early_off') else []
                stack.enter_context(JointAttention(model,layers,'P-1-only' if layers else 'Full',mapping['query']+1))
                with torch.inference_mode():
                    expected=model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,eos_token_id=2)[0,mapping['query']+1:].tolist()
            assert expected==g['token_ids'],'Independent HF generation mismatch'
            assert tensor_hashes(prompt)==before and hook_footprint(model)==footprint
            checks.append(dict(condition=name,all_noop_logits_and_initial_cache_exact=True,
                local_mask_audit=audit,all_mlp_calls_normal=True,independent_hf_tokens_exact=True,
                bos_kv_exact=measured['bos_kv_exact'],early_prefill_edits=g['edits'],decode_o_proj_edits=0))
        _,restored,_,_,_,_=observed(model,prompt,mapping,{},generate_answer=False)
        assert torch.equal(first,restored)
        result.append(dict(task=s['task'],question_id=s['question_id'],historical_full_tokens_and_logits_exact=True,
            native_restoration_exact=True,hooks_clean=True,inputs_unchanged=True,conditions=checks))
    return dict(passed=True,samples=result)


def artifact_arrays(entry,name,arrays,prompt,processor):
    folder=OUT/'cases'/identity(entry['sample']);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}.npz'
    np.savez_compressed(path,**arrays)
    if not (folder/'clip_input.png').exists():
        image=prompt['pixel_values'][0].detach().float().cpu().numpy().transpose(1,2,0)
        image=image*np.asarray(processor.image_processor.image_std)+np.asarray(processor.image_processor.image_mean)
        Image.fromarray(np.clip(image*255,0,255).round().astype('uint8')).save(folder/'clip_input.png')
    return dict(path=str(path.relative_to(OUT)),sha256=sha256(path))


def valid_checkpoint(path,entry,name,binding):
    r=read(path)
    assert r['binding']==digest(binding) and r['sample_sha256']==entry['sample_sha256'] and r['condition']==name
    assert r['body_sha256']==digest(r['body'])
    if r['body'].get('arrays'):
        a=r['body']['arrays']; assert sha256(OUT/a['path'])==a['sha256']
    return r['body']


def save_checkpoint(path,entry,name,binding,body):
    atomic_json(path,dict(binding=digest(binding),sample_sha256=entry['sample_sha256'],condition=name,
        body_sha256=digest(body),body=body))


def worker():
    binding=execution(); protocol=verify()
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=preflight()
    model,processor,environment=load_model(old.MODEL_PATH)
    expected=read(ROOT/'results/gpu2_l0l1_bos_mechanism/smoke.json')['environment']
    assert all(environment[k]==expected[k] for k in ENV_KEYS if k!='cuda_visible_devices')
    assert environment['transformers']=='4.49.0' and environment['parameter_dtypes']==['torch.float16']
    footprint=hook_footprint(model); versions={n:p._version for n,p in model.named_parameters()}
    try:
        path=OUT/'smoke.json'
        if path.exists():
            accepted=read(path);assert accepted['binding']==digest(binding) and accepted['validation']['passed']
        else:
            validation=smoke(model,processor)
            atomic_json(path,dict(binding=digest(binding),environment=environment,preflight=pre,validation=validation))
        print('GPU2 Smoke passed;11 Prefill-only conditions; Full complete answers reused',flush=True)
        start=time.perf_counter();new=0;resumed=0
        for i,entry in enumerate(entries()):
            s=entry['sample'];name=identity(s);mapping=entry['mapping']
            previous,early=check_reuse(entry)
            folder=OUT/'checkpoints'/name;folder.mkdir(parents=True,exist_ok=True)
            if all((folder/f'{c}.json').exists() for c in ['Full',*CONDITIONS]):
                for c in ['Full',*CONDITIONS]:valid_checkpoint(folder/f'{c}.json',entry,c,binding)
                resumed+=11;continue
            prompt=prepared(model,processor,s);before=tensor_hashes(prompt)
            assert true_mapping(processor,s)==mapping
            # Only a single native Prefill for source/hidden measurements, not free Full generation.
            _,full_logits,full,full_stats,audit,_=observed(model,prompt,mapping,{},generate_answer=False)
            check_first(full_logits,previous)
            keep=name in protocol['cases']
            native_generation=dict(previous['Full'])
            native_generation.update(first_metrics_from_history(previous['L0L5_OFF'],full_logits))
            body=dict(generation=native_generation,measurements=measurements(full,full_stats,mapping,full),
                reused_generation=True,audit=audit,first_logits_sha256=digest(full_logits.tolist()))
            if keep:body['arrays']=artifact_arrays(entry,'Full',full,prompt,processor)
            path=folder/'Full.json'
            if path.exists():
                existing=valid_checkpoint(path,entry,'Full',binding)
                assert digest(existing)==digest(body)
            else:save_checkpoint(path,entry,'Full',binding,body)
            for c,condition in CONDITIONS.items():
                path=folder/f'{c}.json'
                if path.exists():valid_checkpoint(path,entry,c,binding);resumed+=1;continue
                tick=time.perf_counter()
                g,first,arrays,stats,audit,_=observed(model,prompt,mapping,condition)
                g.update(first_metrics_from_history(previous['L0L5_OFF'],first))
                if c=='EarlyOFF':check_first(first,early,early=True)
                text=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()
                g.update(text=text,prediction=gqa_prediction(text),score=int(gqa_prediction(text)==s['answer']))
                assert g['first_token']==g['token_ids'][0] and g['edits']==(2 if condition.get('early_off') else 0)
                body=dict(generation=g,measurements=measurements(arrays,stats,mapping,full),audit=audit,
                    reused_generation=False,seconds=time.perf_counter()-tick)
                if keep:body['arrays']=artifact_arrays(entry,c,arrays,prompt,processor)
                assert tensor_hashes(prompt)==before and hook_footprint(model)==footprint
                save_checkpoint(path,entry,c,binding,body);new+=1
            print(json.dumps(dict(questions=i+1,total=240,new_generations=new,resumed=resumed,
                elapsed_seconds=time.perf_counter()-start,memory=gpu_memory())),flush=True)
        assert versions=={n:p._version for n,p in model.named_parameters()}
        execution()
        atomic_json(OUT/'completion.json',dict(status='complete',binding=digest(binding),questions=240,
            conditions=12,expected_new_generations=2640,new_generations_this_run=new,resumed=resumed,
            full_generations_reused=240,primary_case_count=6,seconds=time.perf_counter()-start,
            hooks_clean=hook_footprint(model)==footprint,parameters_unchanged=True))
        print('COMPLETE240; GPU2 worker exiting and releasing model',flush=True)
    finally:
        assert hook_footprint(model)==footprint
        del model;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','cpu','freeze','worker'])
    args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        atomic_json(OUT/f'failures/{args.stage}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
