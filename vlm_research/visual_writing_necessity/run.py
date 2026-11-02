import argparse
import fcntl
import io
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import unittest

import torch
import transformers
from transformers import AutoProcessor
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from query_role_attention.engine import roles
from relay_source_causality.run import frozen_inputs
from vlm_common import load_model,gpu_memory
import query_role_smoke_fix.run as repaired
from visual_write_window.engine import generate as prior_generate
from .common import *
from .engine import WritingMask,evaluate,teacher_reference


def cpu():
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('vlm_research.visual_writing_necessity.tests')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        output=stream.getvalue(),sources=source_hashes(),cuda_initialized=torch.cuda.is_initialized()))
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful() and not torch.cuda.is_initialized()


def freeze():
    assert not (OUT/'protocol.json').exists()
    tested=read(OUT/'cpu_validation.json');assert tested['passed'] and tested['sources']==source_hashes()
    paths=['vlm_research/'+n for n in source_hashes()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    selection=hardware();busy={r.split(',')[0].strip() for r in selection['compute_processes'].splitlines()}
    gpus=[]
    for row in selection['table'].splitlines():
        index,uuid,name,used,free,util=[x.strip() for x in row.split(',')]
        if uuid not in busy and int(used)<200 and int(free)>22000 and int(util)==0:
            gpus.append(dict(physical_index=int(index),uuid=uuid,name=name))
    gpus.sort(key=lambda g:([2,3,0,1].index(g['physical_index'])))
    write(OUT/'gpu_selection.json',dict(**selection,selected=gpus[:2]))
    assert len(gpus)>=2,'Need two idle GPUs; stop without polling or stealing resources'
    shutil.copyfile(POOL,OUT/'sample_manifest.json');manifest=read(POOL);samples=manifest['samples']
    assert len(samples)==len({s['question_id'] for s in samples})==3000
    assert all(sum(s['task']==t for s in samples)==500 for t in TASKS)
    assert len({s['image_sha256'] for s in samples})==2409
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    processor.patch_size=14;processor.num_additional_image_tokens=1
    processor.vision_feature_select_strategy='default';processor.tokenizer.padding_side='left'
    mappings={};gold={}
    for i,s in enumerate(samples):
        mappings[s['question_id']]=frozen_inputs(processor,s)
        assert mappings[s['question_id']]['image_sha256']==s['image_sha256']
        gold[s['question_id']]=gold_tokens(processor,s)
        if (i+1)%500==0:print('FREEZE INPUTS',i+1,'/3000',flush=True)
    split,owners=shards(samples)
    write(OUT/'input_bindings.json',mappings);write(OUT/'gold_sequences.json',gold)
    write(OUT/'shards.json',dict(question_ids=split,image_owners=owners,gpus=gpus[:2]))
    old=read(HISTORY/'sample_manifest.json')['samples'];old_images={s['image_sha256'] for s in old};old_q={s['question_id'] for s in old}
    overlap=dict(historical_questions=len(old),question_overlap=sum(s['question_id'] in old_q for s in samples),
        image_overlap_questions=sum(s['image_sha256'] in old_images for s in samples),
        image_overlap_clusters=len(old_images&{s['image_sha256'] for s in samples}),
        image_disjoint_question_ids=[s['question_id'] for s in samples if s['image_sha256'] not in old_images],
        scope='Existing research cohort expansion, not independent external validation')
    write(OUT/'overlap_audit.json',overlap)
    historical=read(HISTORY/'protocol.json')
    for path,info in historical['model_files'].items():assert sha256(path)==info['sha256'],path
    smoke=[]
    for qs in split:
        ss=sorted([s for s in samples if s['question_id'] in set(qs)],key=lambda s:s['question_id'])
        chosen=[ss[0],max(ss,key=lambda s:len(gold[s['question_id']]['token_ids']))]
        chosen.append(next(s for s in ss if s not in chosen and len(gold[s['question_id']]['token_ids'])>1))
        smoke.append([s['question_id'] for s in chosen])
    names=['sample_manifest.json','input_bindings.json','gold_sequences.json','shards.json','overlap_audit.json','cpu_validation.json','gpu_selection.json']
    write(OUT/'protocol.json',dict(base_commit=BASE,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        frozen_epoch=time.time(),sources=source_hashes(),bindings={n:sha256(OUT/n) for n in names},
        model_files=historical['model_files'],model_revision=REVISION,transformers_sources=historical['transformers_sources'],
        scorer_sources=historical['scorer_sources'],dependencies=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines(),
        dtype='float16',attention='eager',transformers='4.49.0',conditions={c:writer_layers(c) for c in CONDITIONS},
        generation=dict(greedy=True,max_new_tokens=32,logits_to_keep=1,independent_condition_cache=True),
        Gold_NLL=dict(rule='Official lowercase gold appended with one space; joint tokenization preserves exact original prompt prefix; no additional BOS/EOS',
            loss='Negative per-question mean FP32 full-vocabulary Gold token LogP; signed differences; EOS excluded',
            optimization='Reuse identical first prediction logits; for multi-token gold clone the same-condition prefill cache into disjoint storage, then teacher force every remaining gold token; validate against fresh-cache reference'),
        intervention='Every condition: answer P-1 and all real Decode/GOLD prefix queries cannot read Image Keys in 32 layers. Only R->Image mask layers vary; no o_proj/block/KV changes.',
        roles=historical['relay_rule'],smoke_question_ids=smoke,gpus=gpus[:2],planned_questions=3000,planned_generations=102000,
        history=dict(path=str(HISTORY),protocol_sha256=sha256(HISTORY/'protocol.json'),manifest_sha256=sha256(HISTORY/'sample_manifest.json'),
            All_Writer_not_native_Full=True,reused_generations=0),
        bootstrap=dict(repetitions=20000,seed=SEED,cluster='Image SHA256',question_weighted=True,primary_family=32,
            pointwise='2.5/97.5 percentile',corrected='Bonferroni family 32, alpha=.05/32; 0.078125/99.921875 percentiles; report Monte Carlo tail resolution'),
        concentration='Descriptive cumulative positive mean Delta NLL only; raw signed values retained; K80 is not information proportion or joint sufficiency',
        limitations='Historical cohort; no external confirmation, noninferiority, actual speedup, or new Only-L conditions'))
    print('FROZEN 3000 questions / 2409 images; shards',list(map(len,split)),'GPUs',[g['physical_index'] for g in gpus[:2]],flush=True)


@torch.inference_mode()
def smoke(model,processor,samples,gold,indices,binding,folder):
    rows=[];footprint=hook_footprint(model);versions=[(v._version,v.data_ptr()) for v in model.parameters()]
    for j,qid in enumerate(indices):
        s=next(s for s in samples if s['question_id']==qid);prompt=prepared(model,processor,s);fingerprint=tensor_hashes(prompt)
        ids,rec=repaired.native(model,prompt)
        def observed(condition,noop=False):
            counts=[0]*32;prefix={};handles=[]
            end=roles(prompt,model.config.image_token_index)['R'][0]
            try:
                for layer,block in enumerate(model_layers(model)):
                    def mlp(module,args,out,layer=layer):
                        assert torch.isfinite(out).all();counts[layer]+=1
                    def hidden(module,args,out,layer=layer):
                        value=out[0] if isinstance(out,tuple) else out
                        if value.shape[1]>1:prefix[layer]=tensor_hashes({'prefix':value[:,:end]})['prefix']
                    handles.append(block.mlp.register_forward_hook(mlp))
                    handles.append(block.register_forward_hook(hidden))
                result=evaluate(model,processor.tokenizer,prompt,condition,gold[qid]['token_ids'],noop=noop,capture=True)
                assert counts==[len(result[0]['token_ids'])+len(gold[qid]['token_ids'])-1]*32
                assert len(prefix)==32
                return result,prefix
            finally:
                for h in handles:h.remove()
        (native,first,trace),native_prefix=observed('Native')
        assert ids==native['token_ids'] and len(trace)==len(rec.logits)
        assert all(torch.equal(a,b[-1]) for a,b in zip(trace,rec.logits))
        (noop,_,nt),noop_prefix=observed('No-Writer',noop=True)
        assert noop['token_ids']==ids and len(nt)==len(trace) and all(torch.equal(a,b) for a,b in zip(nt,trace))
        assert noop_prefix==native_prefix
        configs=CONDITIONS if j==0 else ['All-Writer','No-Writer','Drop-L8']
        for c in configs:
            (g,_,ct),prefix=observed(c)
            assert prefix==native_prefix,'Causally preceding visual/other queries changed'
            ref=teacher_reference(model,prompt,c,gold[qid]['token_ids'])
            assert g['gold_token_logp']==ref,'Teacher-forced cache branch differs from fresh Gold reference'
            if c in ['All-Writer','No-Writer']:
                previous,_,pt,_=prior_generate(model,processor.tokenizer,prompt,c,capture=True)
                assert previous['token_ids']==g['token_ids'] and all(torch.equal(a,b) for a,b in zip(ct,pt))
            assert tensor_hashes(prompt)==fingerprint and hook_footprint(model)==footprint
        try:
            with WritingMask(model,prompt,'No-Writer'):raise RuntimeError('cleanup-test')
        except RuntimeError as e:assert str(e)=='cleanup-test'
        assert hook_footprint(model)==footprint and versions==[(v._version,v.data_ptr()) for v in model.parameters()]
        row=dict(passed=True,question_id=qid,gold_token_count=len(gold[qid]['token_ids']),native_noop_all_valid_logits_tokens_exact=True,
            prior_mask_baselines_exact=True,Gold_teacher_reference_exact=True,cache_storage_isolation=True,all_layers_MLP_native=True,
            target_probability_zero=True,non_target_mask_exact=True,inputs_parameters_unchanged=True,hooks_exception_cleanup=True,conditions=len(configs))
        rows.append(row);print('SMOKE PASS',qid,len(configs),'conditions; Gold tokens',len(gold[qid]['token_ids']),flush=True)
    write(folder/'smoke.json',dict(passed=True,binding=binding,rows=rows,no_tolerance_relaxation=True))


def worker(index):
    p=verify();binding=digest(p);folder=OUT/f'shard{index}';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'complete.json').exists():return
    gpu=p['gpus'][index];attempt=folder/'attempts'/f'{time.time_ns()}-{os.getpid()}'
    write(attempt/'preflight.json',dict(pid=os.getpid(),**preflight(gpu)))
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.time();model,processor,environment=load_model(MODEL)
    assert transformers.__version__=='4.49.0' and torch.cuda.device_count()==1 and environment['parameter_dtypes']==['torch.float16']
    write(attempt/'environment.json',{**environment,'pid':os.getpid(),'selected_gpu':gpu})
    samples=read(OUT/'sample_manifest.json')['samples'];gold=read(OUT/'gold_sequences.json')
    smoke(model,processor,samples,gold,p['smoke_question_ids'][index],binding,attempt)
    if not (folder/'smoke.json').exists():write(folder/'smoke.json',read(attempt/'smoke.json'))
    assert read(folder/'smoke.json')['binding']==binding and read(folder/'smoke.json')['passed']
    while not all((OUT/f'shard{i}/smoke.json').exists() for i in range(2)):
        assert not any(OUT.glob('shard*/error.json')),'Peer failed; stop before formal matrix'
        time.sleep(2)
    assert all(read(OUT/f'shard{i}/smoke.json')['binding']==binding for i in range(2))
    split=read(OUT/'shards.json')['question_ids'][index];byid={s['question_id']:s for s in samples}
    chosen=sorted([byid[q] for q in split],key=lambda s:(s['image_sha256'],s['task'],s['question_id']))
    mappings=read(OUT/'input_bindings.json');footprint=hook_footprint(model)
    versions=[(v._version,v.data_ptr()) for v in model.parameters()];formal=time.time();computed=resumed=0
    for k,s in enumerate(chosen,1):
        assert not any(OUT.glob('shard*/error.json')),'Peer failure; stop without changing protocol'
        prompt=prepared(model,processor,s);fingerprint=tensor_hashes(prompt)
        assert fingerprint==mappings[s['question_id']]['input_hashes']
        stem=folder/'checkpoints'/f"{s['task']}__{s['question_id']}";base=None
        for c in CONDITIONS:
            path=stem/(c+'.json')
            if path.exists():
                row=read(path)
                assert row['binding']==binding and row['sample_sha256']==digest(s) and row['condition']==c
                assert row['generation_sha256']==digest(row['generation']) and row['input_hashes']==fingerprint
                assert row['code_commit']==p['code_commit'] and row['gpu']==gpu
                g=row['generation'];resumed+=1
            else:
                assert not any(OUT.glob('shard*/error.json'))
                tick=time.time()
                with torch.inference_mode():
                    g,first,_=evaluate(model,processor.tokenizer,prompt,c,gold[s['question_id']]['token_ids'])
                    logp=first.float().log_softmax(-1)
                if c=='All-Writer':ref_token=g['first_top1'];ref_logp=float(logp[ref_token])
                else:ref_token=base['first_top1'];ref_logp=base['first_top1_logp']
                g.update(correct=int(g['prediction']==s['answer']),first_top1_logp=float(logp[g['first_top1']]),
                    baseline_top1_token=ref_token,baseline_token_logp=float(logp[ref_token]),
                    first_top1_flip=g['first_top1']!=ref_token,first_logp_loss=ref_logp-float(logp[ref_token]))
                assert not path.exists(),'Never overwrite a completed condition'
                write(path,dict(binding=binding,code_commit=p['code_commit'],question_id=s['question_id'],task=s['task'],
                    image_sha256=s['image_sha256'],sample_sha256=digest(s),gold=s['answer'],condition=c,
                    gpu=gpu,input_hashes=fingerprint,generation=g,generation_sha256=digest(g),seconds=time.time()-tick))
                computed+=1
            if c=='All-Writer':base=g
        assert hook_footprint(model)==footprint and tensor_hashes(prompt)==fingerprint
        assert versions==[(v._version,v.data_ptr()) for v in model.parameters()]
        progress_write(folder/'progress.json',dict(questions=k,total_questions=len(chosen),computed=computed,resumed=resumed,
            seconds=time.time()-formal,last_question=s['question_id'],epoch=time.time()))
        if k%10==0:print('PROGRESS',k,'/',len(chosen),'generations',computed+resumed,'seconds',round(time.time()-formal,1),flush=True)
    verify()
    assert len(list((folder/'checkpoints').glob('*/*.json')))==len(chosen)*34
    write(folder/'complete.json',dict(passed=True,binding=binding,questions=len(chosen),conditions=34,
        checkpoints=len(chosen)*34,computed=computed,resumed=resumed,formal_seconds=time.time()-formal,
        total_seconds=time.time()-started,pid=os.getpid(),gpu=gpu,memory=gpu_memory(),hooks_clean=True,weights_inputs_unchanged=True))
    print('WORKER COMPLETE',index,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','worker']);parser.add_argument('--shard',type=int,choices=[0,1]);a=parser.parse_args()
    try:
        if a.stage=='worker':worker(a.shard)
        else:globals()[a.stage]()
    except BaseException:
        folder=OUT/f'shard{a.shard}' if a.stage=='worker' else OUT
        write(folder/'error.json',dict(stage=a.stage,traceback=traceback.format_exc(),epoch=time.time(),hardware=hardware()))
        raise
