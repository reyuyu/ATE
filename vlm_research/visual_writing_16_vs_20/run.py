import argparse
import fcntl
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback
import unittest

import torch
import transformers
from transformers import AutoProcessor
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from paper_rq1.scorers import gqa_prediction
from relay_source_causality.run import frozen_inputs
from visual_writing_necessity.common import verify as verify_history
from vlm_common import load_model,gpu_memory
import query_role_smoke_fix.run as repaired
from .common import *
from .engine import generate,KeepMask


def cpu():
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('vlm_research.visual_writing_16_vs_20.tests')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        output=stream.getvalue(),sources=sources(),CUDA_initialized=torch.cuda.is_initialized()))
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful() and not torch.cuda.is_initialized()


def historical_baseline(samples):
    p=verify_history();binding=digest(p);smap={s['question_id']:s for s in samples}
    assert read(HISTORY/'completion.json')['status']=='COMPLETE'
    a=read(HISTORY/'analysis_complete.json')
    names=['sample_manifest.json','protocol.json','input_bindings.json','raw_archive_manifest.json','checkpoint_hashes.json','condition_summary.csv']
    for n in names:assert sha256(HISTORY/n)==a['hashes'][n],n
    expected={r['path']:r['sha256'] for r in read(HISTORY/'checkpoint_hashes.json')}
    inputs=read(HISTORY/'input_bindings.json');baseline={};evidence=[]
    for archive in read(HISTORY/'raw_archive_manifest.json')['archives']:
        path=HISTORY/archive['path'];assert sha256(path)==archive['sha256']
        with tarfile.open(path,'r:gz') as stream:
            for member in stream:
                if not member.name.endswith('/All-Writer.json'):continue
                raw=stream.extractfile(member).read();h=hashlib.sha256(raw).hexdigest()
                assert h==expected[member.name]
                r=json.loads(raw);qid=r['question_id'];g=r['generation'];s=smap[qid]
                assert qid not in baseline and r['condition']=='All-Writer'
                assert r['binding']==binding and r['code_commit']==p['code_commit']
                assert r['sample_sha256']==digest(s) and r['input_hashes']==inputs[qid]['input_hashes']
                assert r['image_sha256']==s['image_sha256'] and r['task']==s['task'] and r['gold']==s['answer']
                assert digest(g)==r['generation_sha256'] and g['writer_layers']==list(range(32))
                assert g['P1_Decode_image_off'] and g['independent_free_gold_caches']
                assert g['first_top1']==g['token_ids'][0] and len(g['first_logits_sha256'])==64
                assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'])
                assert g['free_audit']['blocked_probability_max']==0
                assert all(e['blocked_connections_per_layer']==[576]*32 for e in g['free_audit']['actual_positions'])
                baseline[qid]=r;evidence.append(dict(question_id=qid,archive=archive['path'],member=member.name,sha256=h))
    assert len(baseline)==3000 and sum(r['generation']['correct'] for r in baseline.values())==2611
    write(OUT/'historical_all_writer.json',baseline)
    write(OUT/'baseline_reuse_audit.json',dict(passed=True,questions=3000,accuracy_pct=2611/30,
        source_commit=BASE,historical_protocol_sha256=sha256(HISTORY/'protocol.json'),
        model_revision=p['model_revision'],raw_records_match_archived_checkpoint_SHA256=True,
        all_question_image_prompt_model_cache_generation_bindings_valid=True,not_native_Full=True,
        evidence=evidence))
    return p,baseline


def freeze():
    assert not (OUT/'protocol.json').exists()
    tested=read(OUT/'cpu_validation.json');assert tested['passed'] and tested['sources']==sources()
    paths=['vlm_research/'+n for n in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    state=hardware();busy={r.split(',')[0].strip() for r in state['compute_processes'].splitlines()};gpus=[]
    for line in state['table'].splitlines():
        index,uuid,name,used,free,util=[x.strip() for x in line.split(',')]
        if uuid not in busy and int(used)<200 and int(free)>22000 and int(util)==0:
            gpus.append(dict(physical_index=int(index),uuid=uuid,name=name))
    gpus.sort(key=lambda g:[2,3,0,1].index(g['physical_index']))
    write(OUT/'gpu_selection.json',dict(**state,selected=gpus[:2]));assert len(gpus)>=2,'Two idle GPUs required; stop'
    shutil.copyfile(HISTORY/'sample_manifest.json',OUT/'sample_manifest.json')
    shutil.copyfile(HISTORY/'input_bindings.json',OUT/'input_bindings.json')
    samples=read(OUT/'sample_manifest.json')['samples']
    assert len(samples)==len({s['question_id'] for s in samples})==3000
    assert len({s['image_sha256'] for s in samples})==2409
    assert all(sum(s['task']==t for s in samples)==500 for t in TASKS)
    old,baseline=historical_baseline(samples)
    for path,info in old['model_files'].items():assert sha256(path)==info['sha256']
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    processor.patch_size=14;processor.num_additional_image_tokens=1;processor.vision_feature_select_strategy='default';processor.tokenizer.padding_side='left'
    inputs=read(OUT/'input_bindings.json')
    for i,s in enumerate(samples):
        assert frozen_inputs(processor,s)==inputs[s['question_id']]
        if (i+1)%500==0:print('INPUT BINDINGS',i+1,'/3000',flush=True)
    historical=read(ROOT/'results/visual_write_window_240/sample_manifest.json')['samples']
    old_images={s['image_sha256'] for s in historical};primary=[s['question_id'] for s in samples if s['image_sha256'] not in old_images]
    previous=read(HISTORY/'overlap_audit.json')
    assert set(primary)==set(previous['image_disjoint_question_ids']) and len(primary)==2640
    write(OUT/'analysis_sets.json',dict(primary_image_disjoint=primary,primary_questions=2640,
        primary_images=len({s['image_sha256'] for s in samples if s['question_id'] in set(primary)}),
        historical_exploration_questions=len(historical),historical_image_clusters=len(old_images),
        overlap_questions=3000-len(primary),all_questions=[s['question_id'] for s in samples],
        selection='Original question/image binding only; no outcome filtering',
        limitation='Disjoint from historical 240-image exploration only; not independent of all prior cohort research'))
    split,owners=shards(samples)
    for s in samples:assert baseline[s['question_id']]['gpu']['uuid']==gpus[owners[s['image_sha256']]]['uuid'],'Historical pairing GPU changed'
    write(OUT/'shards.json',dict(question_ids=split,image_owners=owners,gpus=gpus[:2]))
    smoke=[]
    for ids in split:
        ss=sorted([s for s in samples if s['question_id'] in set(ids)],key=lambda s:s['question_id'])
        chosen=[ss[0],max(ss,key=lambda s:len(s['input_mapping']['prompt_ids']))]
        chosen.append(next(s for s in ss if s not in chosen));assert len({s['question_id'] for s in chosen})==3
        smoke.append([s['question_id'] for s in chosen])
    names=['sample_manifest.json','input_bindings.json','historical_all_writer.json','baseline_reuse_audit.json','analysis_sets.json','shards.json','cpu_validation.json','gpu_selection.json']
    write(OUT/'protocol.json',dict(base_commit=BASE,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        epoch=time.time(),sources=sources(),bindings={n:sha256(OUT/n) for n in names},
        model_revision=REVISION,model_files=old['model_files'],transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],
        model='LLaVA-1.5-7B',dtype='float16',attention='eager',transformers='4.49.0',eval=True,inference_mode=True,
        conditions={c:writers(c) for c in CONDITIONS},gpus=gpus[:2],smoke_question_ids=smoke,
        generation=dict(greedy=True,max_new_tokens=32,EOS=2,logits_to_keep=1,independent_KV_cache=True),
        intervention='P-1 and every real Decode cannot read Image Keys at any layer; only prefill R->Image allowed layers vary. Other prefill/Image queries, MLP/residual/QKV/cache native.',
        R='Original historical post-image, pre-P-1 nonvisual prompt queries; dynamic actual input positions',
        new_generations=6000,baseline_reused_questions=3000,baseline='Historical All-Writer, never native Full',
        primary='2640 questions image-disjoint from historical 240-image exploration',supportive='All 3000 questions',
        comparisons=[['Keep-0-15','All-Writer'],['Keep-0-19','All-Writer'],['Keep-0-19','Keep-0-15']],
        bootstrap=dict(repetitions=20000,seed=SEED,cluster='Image SHA256',question_weighted=True,ordinary_percentiles=[2.5,97.5],
            NI_family_candidates=2,NI_family_CI='Bonferroni two-sided familywise 95% intervals, per-comparison 97.5% (1.25/98.75 percentiles)',
            NI_primary_margin_pp=-1,NI_supplement_margin_pp=-2,NI_rule='Corrected lower bound strictly above margin; primary decision uses 2640 only',
            keep20_vs_keep16='Predeclared secondary comparison, ordinary two-sided paired 95% CI; positive lower bound = detectable benefit'),
        recommendation='Among primary-set candidates passing -1pp, prefer fewer allowed writing layers. If only -2pp passes, label supplemental tolerance, not -1pp NI. If neither passes, recommend neither.',
        limitations='Source masks are not actual acceleration; no added conditions or post hoc sampling'))
    print('FROZEN: 2640 primary / 3000 total; 6000 new generations; GPUs',[g['physical_index'] for g in gpus[:2]],flush=True)


@torch.inference_mode()
def smoke(model,processor,samples,baseline,ids,binding,folder):
    footprint=hook_footprint(model);versions=[(p._version,p.data_ptr()) for p in model.parameters()];rows=[]
    for qid in ids:
        s=next(s for s in samples if s['question_id']==qid);prompt=prepared(model,processor,s);fingerprint=tensor_hashes(prompt)
        ids_native,rec=repaired.native(model,prompt)
        def observed(c,noop=False):
            handles=[];prefix={};calls=[0]*32
            try:
                end=int((prompt['input_ids'][0]==model.config.image_token_index).nonzero()[-1])+1
                for layer,block in enumerate(model_layers(model)):
                    def hidden(module,args,out,layer=layer):
                        h=out[0] if isinstance(out,tuple) else out
                        if h.shape[1]>1:prefix[layer]=tensor_hashes({'prefix':h[:,:end]})['prefix']
                    def mlp(module,args,out,layer=layer):
                        assert torch.isfinite(out).all();calls[layer]+=1
                    handles.append(block.register_forward_hook(hidden));handles.append(block.mlp.register_forward_hook(mlp))
                result=generate(model,processor.tokenizer,prompt,c,noop,capture=True)
                assert len(prefix)==32 and calls==[len(result[0]['token_ids'])]*32
                return result,prefix
            finally:
                for h in handles:h.remove()
        (native,_,trace),prefix=observed('Native')
        assert native['token_ids']==ids_native and len(trace)==len(rec.logits)
        assert all(torch.equal(a,b[-1]) for a,b in zip(trace,rec.logits))
        (noop,_,nt),nprefix=observed(CONDITIONS[0],True)
        assert noop['token_ids']==native['token_ids'] and len(nt)==len(trace) and all(torch.equal(a,b) for a,b in zip(nt,trace))
        assert nprefix==prefix
        (base,first,_),bp=observed('All-Writer');history=baseline[qid]['generation']
        assert bp==prefix and base['token_ids']==history['token_ids'] and base['text']==history['text']
        assert base['first_logits_sha256']==history['first_logits_sha256'],'Historical All-Writer logits mismatch'
        assert float(first.float().log_softmax(-1)[history['first_top1']])==history['first_top1_logp']
        for c in CONDITIONS:
            (g,_,ct),cp=observed(c);assert cp==prefix
            again,_,at=generate(model,processor.tokenizer,prompt,c,capture=True)
            assert again==g and len(ct)==len(at) and all(torch.equal(a,b) for a,b in zip(ct,at)),'Cache/hook contamination'
        try:
            with KeepMask(model,prompt,CONDITIONS[0]):raise RuntimeError('cleanup-smoke')
        except RuntimeError as e:assert str(e)=='cleanup-smoke'
        assert tensor_hashes(prompt)==fingerprint and hook_footprint(model)==footprint
        assert versions==[(p._version,p.data_ptr()) for p in model.parameters()]
        rows.append(dict(question_id=qid,passed=True,HF_noop_all_valid_logits_exact=True,historical_All_Writer_first_logits_tokens_exact=True,
            original_visual_queries_native=True,MLP_native=True,masked_probability_zero=True,cache_prefix_isolation=True,
            hooks_normal_exception_cleanup=True,inputs_parameters_unchanged=True))
        print('SMOKE PASS',qid,flush=True)
    write(folder/'smoke.json',dict(passed=True,binding=binding,rows=rows,no_tolerance_relaxation=True))


def worker(index):
    p=verify();binding=digest(p);folder=OUT/f'shard{index}';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'complete.json').exists():return
    attempt=folder/'attempts'/f'{time.time_ns()}-{os.getpid()}';gpu=p['gpus'][index]
    write(attempt/'preflight.json',dict(pid=os.getpid(),**preflight(gpu)))
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.time();model,processor,env=load_model(MODEL)
    assert transformers.__version__=='4.49.0' and torch.cuda.device_count()==1 and env['parameter_dtypes']==['torch.float16']
    write(attempt/'environment.json',{**env,'pid':os.getpid(),'selected_gpu':gpu})
    samples=read(OUT/'sample_manifest.json')['samples'];baseline=read(OUT/'historical_all_writer.json')
    smoke(model,processor,samples,baseline,p['smoke_question_ids'][index],binding,attempt)
    if not (folder/'smoke.json').exists():write(folder/'smoke.json',read(attempt/'smoke.json'))
    while not all((OUT/f'shard{i}/smoke.json').exists() for i in range(2)):
        assert not any(OUT.glob('shard*/error.json')),'Peer failure; stop before formal matrix'
        time.sleep(2)
    assert all(read(OUT/f'shard{i}/smoke.json')['binding']==binding for i in range(2))
    selected=set(read(OUT/'shards.json')['question_ids'][index]);chosen=sorted([s for s in samples if s['question_id'] in selected],key=lambda s:(s['image_sha256'],s['task'],s['question_id']))
    inputs=read(OUT/'input_bindings.json');footprint=hook_footprint(model);versions=[(p._version,p.data_ptr()) for p in model.parameters()]
    formal=time.time();computed=resumed=0
    for k,s in enumerate(chosen,1):
        assert not any(OUT.glob('shard*/error.json')),'Peer failure; stop'
        qid=s['question_id'];prompt=prepared(model,processor,s);fingerprint=tensor_hashes(prompt)
        assert fingerprint==inputs[qid]['input_hashes'];base=baseline[qid]['generation']
        for c in CONDITIONS:
            path=folder/'checkpoints'/f"{s['task']}__{qid}"/(c+'.json')
            if path.exists():
                row=read(path);assert row['binding']==binding and row['sample_sha256']==digest(s) and row['condition']==c
                assert row['generation_sha256']==digest(row['generation']) and row['input_hashes']==fingerprint and row['gpu']==gpu
                resumed+=1
            else:
                tick=time.time();g,first,_=generate(model,processor.tokenizer,prompt,c)
                g.update(correct=int(g['prediction']==s['answer']),baseline_top1_token=base['first_top1'],first_top1_flip=g['first_top1']!=base['first_top1'])
                assert not path.exists()
                write(path,dict(binding=binding,code_commit=p['code_commit'],question_id=qid,task=s['task'],image_sha256=s['image_sha256'],
                    sample_sha256=digest(s),gold=s['answer'],condition=c,gpu=gpu,input_hashes=fingerprint,generation=g,
                    generation_sha256=digest(g),seconds=time.time()-tick));computed+=1
        assert tensor_hashes(prompt)==fingerprint and hook_footprint(model)==footprint
        assert versions==[(p._version,p.data_ptr()) for p in model.parameters()]
        progress_write(folder/'progress.json',dict(questions=k,total_questions=len(chosen),computed=computed,resumed=resumed,seconds=time.time()-formal,epoch=time.time()))
        if k%50==0:print('PROGRESS',k,'/',len(chosen),'generations',computed+resumed,'seconds',round(time.time()-formal,1),flush=True)
    verify();assert len(list((folder/'checkpoints').glob('*/*.json')))==len(chosen)*2
    write(folder/'complete.json',dict(passed=True,binding=binding,questions=len(chosen),conditions=2,generations=len(chosen)*2,
        computed=computed,resumed=resumed,formal_seconds=time.time()-formal,total_seconds=time.time()-start,gpu=gpu,pid=os.getpid(),
        hooks_clean=True,weights_inputs_unchanged=True,memory=gpu_memory()))
    print('COMPLETE',index,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','worker']);parser.add_argument('--shard',type=int,choices=[0,1]);a=parser.parse_args()
    try:
        if a.stage=='worker':worker(a.shard)
        else:globals()[a.stage]()
    except BaseException:
        folder=OUT/f'shard{a.shard}' if a.stage=='worker' else OUT
        write(folder/'error.json',dict(stage=a.stage,epoch=time.time(),traceback=traceback.format_exc(),hardware=hardware()));raise
