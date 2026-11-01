"""Frozen holdout selection, strict historical Full reuse and isolated workers."""
import argparse
from collections import Counter,defaultdict
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch

from benchmarks.dataset_registry import digest,freeze_json,sha256
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers,tensor_hash
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import prepared,atomic_json,tensor_hashes
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_early_joint_attention.protocol import ROOT,MODEL_PATH
from figure3_decode_windows.run import cohort
from rq2_source_knockout.engine import forward as native_forward
from rq2_source_knockout import run as discovery
from rq2_visual_full import run as validation
from vlm_common import load_model,gpu_memory
from .engine import S,CONDITIONS,LABELS,MultiPromptKnockout,generate,native_reference

OUT=ROOT/'results/rq2_prompt_sensitive_five_240'
BASE='acc137dc33b02c17cffb9b1a21999d0cb95735c8'
SEED=20261101
TASKS=discovery.TASKS
ALLOWED={0:'GPU-a63c6362-dcfd-4a26-c44b-014b40127c89',
         2:'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6',
         3:'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185'}
SECONDARY=['A-Full','B-Full','D-Full','E-Full','C-A','C-B','C-A-B+Full']


def read(path):return json.loads(Path(path).read_text())


def sources():
    names=set(validation.sources())|{'rq2_visual_full/run.py'}
    names|={'rq2_prompt_five/'+n for n in ['__init__.py','engine.py','run.py','analyze.py']}
    names|={'test_rq2_prompt_five.py'}
    return {n:sha256(ROOT/n) for n in sorted(names)}


def pool():
    prior=read(validation.OUT/'sample_manifest.json')
    samples=[s for s in prior['samples'] if prior['splits'][s['question_id']]['image_disjoint']]
    assert len(samples)==2656
    return samples


def selection():
    return sorted([s for task in TASKS for s in sorted((s for s in pool() if s['task']==task),
        key=lambda s:digest(['rq2.5-prompt-five',SEED,task,s['question_id']]))[:40]],
        key=lambda s:(s['task'],s['question_id']))


def assign(samples,workers):
    groups=defaultdict(list)
    for s in samples:groups[s['image_sha256']].append(s)
    loads=[0]*workers;ranks={}
    for image in sorted(groups,key=lambda h:(-len(groups[h]),digest(['rq2.5-shard',SEED,h]))):
        rank=min(range(workers),key=lambda k:(loads[k],k))
        for s in groups[image]:ranks[s['question_id']]=rank
        loads[rank]+=len(groups[image])
    return ranks


def snapshot():
    table=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    return table,apps


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    table,apps=snapshot();gpus=[]
    for line in table.splitlines():
        index,uuid,name,used,free,util=line.split(', ');index=int(index)
        if index in ALLOWED and uuid==ALLOWED[index] and int(free)>=18000 and int(util)==0 and not any(p.startswith(uuid) for p in apps.splitlines()):
            gpus.append(dict(physical_gpu=index,uuid=uuid))
    if not gpus:raise RuntimeError('No idle allowed GPU. GPU1 is always forbidden.')
    assert 1<=len(gpus)<=3 and all(g['physical_gpu']!=1 for g in gpus)
    ss=selection();ds=read(discovery.OUT/'sample_manifest.json')['samples']
    assert len(ss)==240 and Counter(s['task'] for s in ss)==Counter({t:40 for t in TASKS})
    assert not ({s['image_sha256'] for s in ss}&{s['image_sha256'] for s in ds})
    original={s['question_id']:s for s in cohort()};hist=validation.history();records={}
    blobs={}
    for commit,name in [(BASE,'results/rq2_visual_sufficiency_full3000/sample_manifest.json'),
        (BASE,'results/rq2_visual_sufficiency_full3000/history_bindings.json'),
        ('3a7c498c722dab7344419490372c2af5dbb67a4d','results/rq2_single_layer_source_240/sample_manifest.json')]:
        blob=subprocess.check_output(['git','rev-parse',f'{commit}:vlm_research/{name}'],cwd=ROOT,text=True).strip()
        assert blob==subprocess.check_output(['git','hash-object',ROOT/name],text=True).strip()
        blobs[name]=dict(commit=commit,blob=blob,sha256=sha256(ROOT/name))
    for s in ss:
        assert s==original[s['question_id']] and sha256(s['image_path'])==s['image_sha256']
        old=hist[s['question_id']];path=ROOT/old['full_path'];full=read(path)
        assert sha256(path)==old['full_sha256'] and old['sample_sha256']==digest(s)
        assert old['prompt_sha256']==digest(s['input_mapping']) and full['sample_sha256']==digest(s)
        assert full['binding']==digest(read(validation.previous.FIGURE/'execution_manifest.json'))
        assert full['result_sha256']==digest(full['generation'])
        g=full['generation'];assert g['edits']==0 and g['score']==int(gqa_prediction(g['text'])==s['answer'])
        first=validation.native_logits(s,True);assert int(first.argmax())==g['token_ids'][0]
        records[s['question_id']]=old
    ranks=assign(ss,len(gpus))
    freeze_json(OUT/'sample_manifest.json',dict(seed=SEED,pool_size=2656,selection="Per task take40 smallest SHA256(canonical JSON ['rq2.5-prompt-five',20261101,task,qid]); no outcome filtering.",
        samples=ss,sample_sha256=[digest(s) for s in ss],assignments=ranks,
        images=len({s['image_sha256'] for s in ss}),gpus=gpus,shards=dict(Counter(str(k) for k in ranks.values())),
        discovery_image_overlap=0,reuse_warning='These240 images are consumed for the present Prompt-five validation; future condition searches must not call them unused confirmation.',historical_git_blobs=blobs))
    freeze_json(OUT/'history_bindings.json',records)
    freeze_json(OUT/'gpu_selection.json',dict(snapshot=table,compute_processes=apps,chosen=gpus,forbidden=[1]))
    print('Frozen240:',dict(Counter(ranks.values())),'GPUs',gpus,flush=True)


def cpu():
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(n) for n in ['test_rq2_source_knockout','test_rq2_visual_sufficiency','test_rq2_prompt_five']))
    print(stream.getvalue(),flush=True)
    record=dict(passed=result.wasSuccessful(),tests=result.testsRun,source=sources(),log=stream.getvalue())
    atomic_json(OUT/f'attempts/cpu_{time.time_ns()}.json',record)
    if not result.wasSuccessful():raise AssertionError('CPU regression failed')
    freeze_json(OUT/'cpu_validation.json',record)


def code_check():
    paths=['vlm_research/'+n for n in sources()]
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert read(OUT/'cpu_validation.json')['source']==sources()


def preflight(rank):
    gpu=read(OUT/'sample_manifest.json')['gpus'][rank]
    assert gpu['physical_gpu'] in ALLOWED and gpu['physical_gpu']!=1 and ALLOWED[gpu['physical_gpu']]==gpu['uuid']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid']
    table,apps=snapshot();row=next(line.split(', ') for line in table.splitlines() if gpu['uuid'] in line)
    if int(row[4])<18000 or int(row[5])!=0 or any(line.startswith(gpu['uuid']) for line in apps.splitlines()):
        raise RuntimeError('Selected GPU no longer idle; do not interrupt other processes')
    return dict(**gpu,snapshot=table,compute_processes=apps)


def historical_full(s):
    h=read(OUT/'history_bindings.json')[s['question_id']]
    assert sha256(ROOT/h['full_path'])==h['full_sha256']
    return read(ROOT/h['full_path'])['generation']


def historical_single(s):
    assignments=read(discovery.OUT/'sample_manifest.json')['assignments']
    p=discovery.OUT/f"shard{assignments[s['question_id']]}/checkpoints/{s['task']}__{s['question_id']}.json"
    return next(r for r in read(p)['result']['metrics'] if r['source']=='Prompt' and r['layer']==1)


def smoke(rank):
    code_check();pre=preflight(rank);model,processor,environment=load_model(MODEL_PATH)
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    ds=read(discovery.OUT/'sample_manifest.json')['samples']
    selected=[next(s for s in ds if s['task']=='ChooseRel'),next(s for s in selection() if s['task']=='QueryAttr')]
    checks=[]
    for i,s in enumerate(selected):
        prompt=prepared(model,processor,s);before=tensor_hashes(prompt);bos=processor.tokenizer.bos_token_id;n=prompt['input_ids'].shape[1]
        full,first,trace,debug=generate(model,prompt,bos,[],audit=True,collect=True)
        rawfull=validation.native_logits(s,True)
        previous=read(validation.previous.full_path(s))['generation']
        assert torch.equal(first,rawfull) and full['token_ids']==previous['token_ids']
        with torch.inference_mode():
            hf=model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,eos_token_id=2)[0,n:].tolist()
        assert hf==full['token_ids']
        noop,no_first,no_trace,no_debug=generate(model,prompt,bos,CONDITIONS['C'],noop=True,audit=True,collect=True)
        assert noop['token_ids']==full['token_ids'] and torch.equal(no_first,first)
        assert torch.equal(no_debug['prefill_logits'],debug['prefill_logits'])
        assert len(trace)==len(no_trace) and all(torch.equal(a,b) for a,b in zip(trace,no_trace))
        legacy,_,_=native_forward(model,prompt,bos,1,'Prompt')
        single,single_first,_,_=generate(model,prompt,bos,[1])
        assert torch.equal(single_first,legacy)
        if i==0:
            assert tensor_hash(single_first)==historical_single(s)['logits_sha256']
            assert metric_rows(rawfull,single_first[None])[0]['delta_logp']==historical_single(s)['delta_logp']
        results=[]
        for key,blocked in CONDITIONS.items():
            g,logits,_,d=generate(model,prompt,bos,blocked,audit=True)
            assert torch.equal(d['prefill_logits'][0,:n-1],debug['prefill_logits'][0,:n-1])
            assert d['observer'].states==debug['observer'].states
            assert all(a['other_query_attention_sha256']==b['other_query_attention_sha256'] for a,b in zip(d['rows'],debug['rows']))
            assert all(d['rows'][k]['Prompt_max']==0 for k in blocked)
            assert all(d['rows'][k]==debug['rows'][k] for k in range(min(blocked)))
            again,again_first,_,_=generate(model,prompt,bos,blocked)
            assert g==again and torch.equal(logits,again_first)
            assert native_reference(model,prompt,bos,blocked)==g['token_ids']
            mask=MultiPromptKnockout(model,prompt,bos,blocked)
            with mask,torch.inference_mode():
                cached=model(**prompt,use_cache=True,return_dict=True)
            # Independent cached Prefill must also match the audited driver.
            assert torch.equal(cached.logits[0,-1].float().cpu(),logits)
            assert not mask.handles and len(mask.records)==32
            results.append(dict(condition=key,masked_layers=blocked,prompt_probabilities_zero=True,
                image_bos_self_keys_preserved=True,other_query_logits_attention_states_exact=True,
                cache_and_repeat_exact=True,decode_native=True,first_logits_sha256=tensor_hash(logits),token_ids=g['token_ids']))
        try:
            with MultiPromptKnockout(model,prompt,bos,CONDITIONS['E']):raise RuntimeError('cleanup smoke')
        except RuntimeError:pass
        assert footprint==hook_footprint(model) and before==tensor_hashes(prompt)
        assert versions=={n:p._version for n,p in model.named_parameters()}
        checks.append(dict(question_id=s['question_id'],task=s['task'],p_minus_one=n-1,visual_keys=576,
            historical_full_logits_and_generation_exact=True,noop_all_effective_logits_exact=True,
            historical_single_prompt_L1_exact=i==0,legacy_single_prompt_exact=True,
            normal_exception_hook_cleanup=True,inputs_weights_unchanged=True,conditions=results))
    atomic_json(OUT/f'shard{rank}/smoke.json',dict(passed=True,source=sources(),manifest_sha256=sha256(OUT/'sample_manifest.json'),
        environment=environment,preflight=pre,checks=checks))
    print('Smoke passed GPU',pre['physical_gpu'],flush=True)


def freeze():
    code_check();manifest=read(OUT/'sample_manifest.json')
    smokes=[read(OUT/f'shard{k}/smoke.json') for k in range(len(manifest['gpus']))]
    keys=['python','torch','transformers','cuda_runtime','cudnn','gpu','model_id','model_revision','parameter_dtypes']
    assert all(s['passed'] and s['source']==sources() and s['checks']==smokes[0]['checks'] and s['manifest_sha256']==sha256(OUT/'sample_manifest.json') for s in smokes)
    assert all({k:s['environment'][k] for k in keys}=={k:smokes[0]['environment'][k] for k in keys} for s in smokes)
    prior=read(validation.OUT/'protocol.json')
    for p,b in prior['model_files'].items():assert sha256(p)==b['sha256']
    names=['sample_manifest.json','history_bindings.json','gpu_selection.json','cpu_validation.json',*[f'shard{k}/smoke.json' for k in range(len(smokes))]]
    freeze_json(OUT/'protocol.json',dict(status='frozen',base_commit=BASE,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source=sources(),
        bindings={n:sha256(OUT/n) for n in names},model_revision=prior['model_revision'],model_files=prior['model_files'],
        transformers_source=prior['transformers_source'],transformers='4.49.0',dtype='FP16',attention='eager',
        candidate_layers=S,conditions=CONDITIONS,labels=LABELS,gpus=manifest['gpus'],forbidden_gpu=1,
        questions=240,reused_full_generations=240,new_generations=1200,
        generation=dict(do_sample=False,num_beams=1,max_new_tokens=32,eos_token_id=2,use_cache=True,
            intervention='Only original P-1 Prompt keys before Softmax; remove all hooks before first Decode; own prefix and native cache.'),
        prompt='Exact RQ2.1: valid original text except BOS,576 image keys,P-1 self; no template-only condition.',
        full_validation='Reuse bound historical complete Full answers; one same-GPU Native Full Prefill per question for exact raw-logit and mass checks, not a new Full answer generation.',
        statistics=dict(seed=SEED,repetitions=100000,unit='Paired whole image SHA256 cluster resampling; question-weighted estimator.',
            primary='C-Full Accuracy difference (pp), two-sided95% percentile CI, one prespecified contrast.',
            secondary=SECONDARY,secondary_role='Exploratory; ordinary95% plus Bonferroni7 simultaneous reference intervals; no confirmatory claims.',
            interaction='C-A-B+Full: negative is excess joint Accuracy harm versus additive marginal differences. Not semantic information decomposition.',
            no_equivalence_or_noninferiority=True,taskwise='Descriptive only'),
        limits='Exactly240 holdout questions and five new conditions; no layer search, new samples, decode intervention, training or automatic follow-up.'))
    print('Frozen protocol with',len(smokes),'GPUs; GPU1 forbidden',flush=True)


def verify():
    c=read(OUT/'protocol.json');assert c['source']==sources() and c['conditions']==CONDITIONS and c['candidate_layers']==S
    for p,h in c['bindings'].items():assert sha256(OUT/p)==h
    for p,h in c['transformers_source'].items():assert sha256(p)==h
    for p,b in c['model_files'].items():
        st=Path(p).stat();assert st.st_size==b['bytes'] and st.st_mtime_ns==b['mtime_ns']
    assert all(g['physical_gpu']!=1 for g in c['gpus'])
    return c


def checkpoint_valid(path,binding,s,key):
    r=read(path)
    assert r['binding']==binding and r['sample_sha256']==digest(s) and r['condition']==key
    assert r['result_sha256']==digest(r['generation']) and r['metric_sha256']==digest(r['first_token'])
    assert sha256(OUT/r['logits_file'])==r['logits_sha256']
    return r


def worker(rank):
    c=verify();binding=digest(c);manifest=read(OUT/'sample_manifest.json');folder=OUT/f'shard{rank}';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        assert read(folder/'completion.json')['binding']==binding;print('Already complete; no GPU loaded',flush=True);return
    pre=preflight(rank);start=time.perf_counter();epoch=time.time()
    model,processor,environment=load_model(MODEL_PATH)
    assert environment['model_revision']==c['model_revision'] and environment['transformers']=='4.49.0'
    assert environment['parameter_dtypes']==['torch.float16'] and not model.training
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    ss=[s for s in manifest['samples'] if manifest['assignments'][s['question_id']]==rank];resumed=0
    for i,s in enumerate(ss):
        prompt=None
        for key,blocked in CONDITIONS.items():
            path=folder/f"checkpoints/{s['task']}__{s['question_id']}/{key}.json"
            if path.exists():checkpoint_valid(path,binding,s,key);resumed+=1;continue
            if prompt is None:
                prompt=prepared(model,processor,s);before=tensor_hashes(prompt);full=validation.native_logits(s)
                current,observed,_=native_forward(model,prompt,processor.tokenizer.bos_token_id,observe=True)
                assert torch.equal(current,full),'Current Full differs from frozen historical Full; stop without tolerance relaxation.'
                baseline=dict(full_logits_sha256=tensor_hash(full),full_native_rows=observed.rows,logits_exact_history=True)
                atomic_json(folder/f"baseline/{s['task']}__{s['question_id']}.json",dict(sample_sha256=digest(s),binding=binding,**baseline))
            tick=time.perf_counter()
            g,first,_,debug=generate(model,prompt,processor.tokenizer.bos_token_id,blocked,observe=True)
            text=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()
            g.update(text=text,prediction=gqa_prediction(text),score=int(gqa_prediction(text)==s['answer']))
            metric=metric_rows(full,first[None])[0];metric['logp_loss']=-metric['delta_logp']
            raw=path.with_suffix('.npz');raw.parent.mkdir(parents=True,exist_ok=True);tmp=raw.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,first_logits=first.numpy())
            tmp.replace(raw)
            assert before==tensor_hashes(prompt) and footprint==hook_footprint(model)
            assert versions=={n:p._version for n,p in model.named_parameters()}
            atomic_json(path,dict(binding=binding,sample_sha256=digest(s),task=s['task'],question_id=s['question_id'],image_id=s['image_id'],
                image_sha256=s['image_sha256'],condition=key,generation=g,result_sha256=digest(g),first_token=metric,metric_sha256=digest(metric),
                logits_file=str(raw.relative_to(OUT)),logits_sha256=sha256(raw),native_attention_rows=debug['rows'],
                attention_sha256=digest(debug['rows']),seconds=time.perf_counter()-tick,hooks_clean=True,inputs_unchanged=True,
                parameters_unchanged=True,physical_gpu=pre['physical_gpu']))
        if (i+1)%10==0 or i+1==len(ss):
            print(json.dumps(dict(gpu=pre['physical_gpu'],questions=i+1,total=len(ss),generations=(i+1)*5,seconds=time.perf_counter()-start)),flush=True)
    verify()
    atomic_json(folder/'completion.json',dict(status='complete',binding=binding,questions=len(ss),generations=len(ss)*5,resumed=resumed,
        started_epoch=epoch,finished_epoch=time.time(),seconds=time.perf_counter()-start,environment=environment,preflight=pre,
        memory=gpu_memory(),hooks_clean=footprint==hook_footprint(model),parameters_unchanged=versions=={n:p._version for n,p in model.named_parameters()}))
    print('Complete GPU',pre['physical_gpu'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cpu','smoke','freeze','worker']);p.add_argument('--rank',type=int,default=0);args=p.parse_args()
    try:
        if args.stage in ['smoke','worker']:globals()[args.stage](args.rank)
        else:globals()[args.stage]()
    except BaseException:
        atomic_json(OUT/f'failures/{args.stage}_{args.rank}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
