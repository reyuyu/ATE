import argparse
from contextlib import AbstractContextManager
import fcntl
import inspect
import io
import os
import shutil
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch
import transformers
from benchmarks.smoke import hook_footprint
from first_answer_attention.engine import supplement_generate, first_metrics, CONDITIONS as LEGACY
from paper_rq1_single_layer_gqa6.run import prepared, tensor_hashes
from query_role_attention.engine import generate, roles
from query_role_attention.run import save_raw
import query_role_smoke_fix.run as repaired
from vlm_common import load_model, gpu_memory
from .common import (ROOT, OUT, OLD, REPAIR, RAW, MODEL, UUID, BASE, REVISION,
    TASKS, CONDITIONS, COMPARISONS, SEED, LAYERS, MANIFEST_SHA, TASK_TEXT_SHA,
    read, write, sha256, digest, preflight, sources, historical, verify, divergence, repeated_trigram)


class LegacyLMHeadShape(AbstractContextManager):
    """Align only the historical generator's supported forward argument."""
    def __init__(self,model):
        self.model,self.handle=model,None

    def __enter__(self):
        assert 'logits_to_keep' in inspect.signature(self.model.forward).parameters
        def align(module,args,kwargs):
            assert kwargs.get('logits_to_keep',1)==1, 'Conflicting explicit LM-head shape'
            return args,{**kwargs,'logits_to_keep':1}
        self.handle=self.model.register_forward_pre_hook(align,with_kwargs=True)
        return self

    def __exit__(self,*exc):
        if self.handle is not None:self.handle.remove()
        self.handle=None


def cpu():
    assert sha256(OLD/'sample_manifest.json')==MANIFEST_SHA
    assert read(REPAIR/'gpu_smoke_validation.json')['passed']
    stream=io.StringIO()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(name) for name in [
        'vlm_research.query_role_attention.tests','vlm_research.query_role_smoke_fix.tests',
        'vlm_research.query_role_formal.tests']])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        output=stream.getvalue(),sources=sources(),cuda_initialized=torch.cuda.is_initialized(),
        original_manifest_sha256=MANIFEST_SHA))
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful() and not torch.cuda.is_initialized()


def freeze():
    acceptance=read(OUT/'cpu_validation.json')
    assert acceptance['passed'] and acceptance['sources']==sources()
    paths=['vlm_research/'+p for p in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    assert sha256(OLD/'sample_manifest.json')==MANIFEST_SHA
    old=read(OLD/'protocol.json')
    for path,info in old['model_files'].items():assert sha256(path)==info['sha256'],path
    for path,h in {**old['transformers_sources'],**old['scorer_sources']}.items():assert sha256(path)==h
    for name in ('sample_manifest.json','input_mappings.json'):
        assert not (OUT/name).exists()
        shutil.copyfile(OLD/name,OUT/name)
        assert sha256(OLD/name)==sha256(OUT/name)
    manifest=read(OUT/'sample_manifest.json')
    assert len(manifest['samples'])==240 and manifest['unique_image_sha256']==240
    assert all(sum(s['task']==t for s in manifest['samples'])==40 for t in TASKS)
    assert [manifest['samples'][i]['question_id'] for i in manifest['smoke_indices']]==['00150746','11697551','11127723']
    write(OUT/'protocol.json',dict(base_commit=BASE,scientific_base=old['base_commit'],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        sources=sources(),historical_files=historical(),model_files=old['model_files'],
        transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],
        bindings={n:sha256(OUT/n) for n in ('sample_manifest.json','input_mappings.json','cpu_validation.json')},
        model_revision=REVISION,model_path=str(MODEL),dtype='float16',attention='eager',transformers='4.49.0',
        manifest_note=dict(task_text_sha256=TASK_TEXT_SHA,task_text_hash_length=len(TASK_TEXT_SHA),
            verified_original_sha256=MANIFEST_SHA,action='Preserve exact bytes from requested repair commit; transcription discrepancy recorded'),
        conditions=CONDITIONS,layers=LAYERS,roles=old['roles'],intervention=old['intervention'],
        generation=dict(greedy=True,max_new_tokens=32,use_cache=True,logits_to_keep=1,independent_prefix_and_cache=True),
        legacy_A_validation='Unchanged historical E controller/generator, supported logits_to_keep=1 injected call-locally; exact raw effective logits and complete tokens',
        smoke='Three original frozen questions, repaired same-shape checks plus historical A parity, before any formal checkpoint',
        bootstrap=dict(seed=SEED,repetitions=20000,cluster='Image SHA256',question_weighted=True,
            comparisons=COMPARISONS,interval='Ordinary paired exploratory 95%; no noninferiority claim'),
        generation_diagnostics=dict(divergence='0-based first differing generated-token index; length boundary if only suffix differs',
            repetition='Any duplicate token trigram among non-EOS generated tokens; descriptive flag only'),
        gpu=dict(physical_index=1,uuid=UUID,logical_device='cuda:0',one_model_replica=True),
        scoring=old['scoring'],planned_questions=240,planned_formal_generations=960,
        no_physical_skipping_or_speedup=True,exploratory=True))
    print('FORMAL protocol frozen; original manifest unchanged',flush=True)


@torch.inference_mode()
def smoke(model,processor,manifest,mapping,binding):
    # Reuse the repaired validation unchanged, with a new output destination only.
    repaired.OUT=OUT
    rows=repaired.smoke(model,processor,manifest,mapping,binding)
    legacy_rows=[]
    for i in manifest['smoke_indices']:
        s=manifest['samples'][i];qid=s['question_id']
        inputs=prepared(model,processor,s);before=tensor_hashes(inputs);footprint=hook_footprint(model)
        changed,current=repaired.custom(model,processor,inputs,'A-Bypass')
        with LegacyLMHeadShape(model),repaired.Recorder(model) as reference:
            old,old_first,old_trace=supplement_generate(model,inputs,LEGACY['E'],audit=True,collect=True)
        comparison=repaired.compare('historical_A_same_shape',qid,current,reference,binding)
        assert changed[0]['token_ids']==old['token_ids']
        assert torch.equal(changed[1],old_first)
        assert len(old_trace)==len(current.logits)
        assert all(torch.equal(a[-1].float(),b) for a,b in zip(current.logits,old_trace))
        assert all(call['logits_to_keep']==1 for call in reference.calls)
        assert hook_footprint(model)==footprint and tensor_hashes(inputs)==before
        legacy_rows.append(dict(question_id=qid,passed=True,complete_tokens_exact=True,
            raw_effective_logits_exact=comparison['all_exact'],all_legacy_steps_explicit_keep1=True,
            hooks_clean=True))
        print('HISTORICAL A EXACT PASS',qid,flush=True)
    return dict(passed=True,binding=binding,questions=3,rows=rows,historical_A=legacy_rows,
                no_tolerance_relaxation=True,formal_generations_counted=0)


def gpu():
    started=time.perf_counter();p=verify();binding=digest(p)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check=preflight();write(OUT/f'gpu_preflight_{os.getpid()}.json',check)
    assert transformers.__version__=='4.49.0'
    model,processor,environment=load_model(MODEL)
    assert torch.cuda.device_count()==1 and environment['parameter_dtypes']==['torch.float16']
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'],text=True)
    assert any(line.split(',')[0].strip()==UUID and int(line.split(',')[1])==os.getpid() for line in apps.splitlines())
    environment.update(pid=os.getpid(),physical_gpu=1,uuid=UUID,logical_device='cuda:0',preflight=check,
        process_mapping=apps,python_executable=os.sys.executable,
        dependencies=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines())
    write(OUT/f'environment_{os.getpid()}.json',environment)
    manifest=read(OUT/'sample_manifest.json');mapping=read(OUT/'input_mappings.json')
    versions={n:v._version for n,v in model.named_parameters()};footprint=hook_footprint(model)
    acceptance=smoke(model,processor,manifest,mapping,binding)
    smoke_path=OUT/'gpu_smoke_validation.json'
    if smoke_path.exists():
        assert read(smoke_path)['passed'] and read(smoke_path)['binding']==binding
        write(OUT/f'gpu_resume_smoke_{os.getpid()}.json',acceptance)
    else:write(smoke_path,acceptance)
    verify();formal_started=time.perf_counter();computed=resumed=0
    print('GATE PASS: starting 240 x 4 FORMAL generations',flush=True)
    for i,s in enumerate(manifest['samples']):
        prompt=prepared(model,processor,s)
        assert tensor_hashes(prompt)==mapping[i]['input_hashes'] and roles(prompt)==mapping[i]['role_layout']
        for condition in CONDITIONS:
            checkpoint=OUT/f"checkpoints/{s['task']}__{s['question_id']}/{condition}.json"
            if checkpoint.exists():
                row=read(checkpoint)
                assert row['binding']==binding and row['sample_sha256']==digest(s)
                assert row['condition']==condition and row['input_hashes']==mapping[i]['input_hashes']
                assert row['result_sha256']==digest(row['generation']) and sha256(row['raw']['path'])==row['raw']['sha256']
                if condition=='Full':
                    with np.load(row['raw']['path']) as raw:full_logits=torch.from_numpy(raw['first_logits'].copy())
                    full=row['generation']
                resumed+=1
                continue
            tick=time.perf_counter()
            generation,first,_,_=generate(model,processor.tokenizer,prompt,condition,logits_to_keep=1)
            if condition=='Full':full_logits=first;full=generation
            generation.update(first_metrics(full_logits,first),correct=int(generation['prediction']==s['answer']),
                full_correct_to_wrong=int(full['prediction']==s['answer'] and generation['prediction']!=s['answer']),
                full_wrong_to_correct=int(full['prediction']!=s['answer'] and generation['prediction']==s['answer']),
                sequence_changed=generation['token_ids']!=full['token_ids'],
                first_divergence=divergence(full['token_ids'],generation['token_ids']),
                repeated_trigram=repeated_trigram(generation['token_ids']),
                full_first_logits_sha256=digest(full_logits.tolist()),first_logits_sha256=digest(first.tolist()))
            raw=save_raw(RAW/f"{s['task']}__{s['question_id']}/{condition}.npz",first)
            assert tensor_hashes(prompt)==mapping[i]['input_hashes'] and hook_footprint(model)==footprint
            assert versions=={n:v._version for n,v in model.named_parameters()}
            write(checkpoint,dict(binding=binding,code_commit=p['code_commit'],sample_sha256=digest(s),
                question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],
                prompt_sha256=s['input_mapping']['chat_prompt_sha256'],input_hashes=mapping[i]['input_hashes'],
                gold=s['answer'],condition=condition,generation=generation,result_sha256=digest(generation),
                raw=raw,seconds=time.perf_counter()-tick))
            computed+=1
        if (i+1)%10==0:
            verify()
            print('PROGRESS',i+1,'/240',computed+resumed,'/960','formal_seconds',round(time.perf_counter()-formal_started,1),flush=True)
    verify()
    assert len(list((OUT/'checkpoints').glob('*/*.json')))==960
    write(OUT/'gpu_completion.json',dict(passed=True,binding=binding,questions=240,conditions=4,
        checkpoints=960,computed=computed,resumed=resumed,formal_seconds=time.perf_counter()-formal_started,
        seconds_including_load_smoke=time.perf_counter()-started,memory=gpu_memory(),
        hooks_clean=hook_footprint(model)==footprint,parameters_unchanged=True,
        physical_gpu=1,uuid=UUID,model_replicas=1))
    print('COMPLETE 960/960 formal generations; GPU process exits',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','gpu'])
    args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        count=len(list((OUT/'checkpoints').glob('*/*.json')))
        status=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
        write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json',dict(stage=args.stage,
            traceback=traceback.format_exc(),status='PARTIAL',formal_checkpoints=count,gpu_status=status.stdout))
        raise
