"""Reuse all240 RQ2.5 inputs and F/M answers; only E/EM/L generate anew."""
import argparse
from collections import Counter
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
from knockout_probe import tensor_hash
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_single_layer_gqa6.run import prepared,atomic_json,tensor_hashes
from rq2_prompt_five import run as old
from rq2_prompt_five.engine import generate,native_reference
from .engine import CONDITIONS,ALL,CHECKPOINTS,MIDDLE,captured_generate,prefill

ROOT=old.ROOT;MODEL_PATH=old.MODEL_PATH;TASKS=old.TASKS;SEED=20261101
OUT=ROOT/'results/rq3_prompt_cross_layer_interaction_240'
BASE='f3f48c4eba457e38e6ec38493698349045b2870a'
SECONDARY=['E-F','M-F','EM-F','EM-M','L-F','EM-E']
read=old.read


def sources():
    names=set(old.sources())|{'rq3_prompt_interaction/'+n for n in ['__init__.py','engine.py','run.py','analyze.py']}
    names|={'test_rq3_prompt_interaction.py','paper_rq1_single_layer_gqa6/analyze.py'}
    return {n:sha256(ROOT/n) for n in sorted(names)}


def old_m_path(s):
    previous=read(old.OUT/'sample_manifest.json');rank=previous['assignments'][s['question_id']]
    return old.OUT/f"shard{rank}/checkpoints/{s['task']}__{s['question_id']}/B.json"


def history(s,key):
    binding=read(OUT/'history_bindings.json')[s['question_id']][key]
    path=ROOT/binding['path'];assert sha256(path)==binding['sha256']
    record=read(path);assert record['sample_sha256']==digest(s) and record['result_sha256']==digest(record['generation'])
    return record


def historic_logits(s,key):
    if key=='F':return old.validation.native_logits(s,True)
    h=read(OUT/'history_bindings.json')[s['question_id']]['M'];p=ROOT/h['logits_path']
    assert sha256(p)==h['logits_sha256']
    with np.load(p,allow_pickle=False) as data:return torch.from_numpy(data['first_logits'].copy())


def prepare():
    prior=old.verify();original=read(old.OUT/'sample_manifest.json');samples=original['samples'];OUT.mkdir(parents=True,exist_ok=True)
    assert len(samples)==240 and len({s['image_sha256'] for s in samples})==235
    table,apps=old.snapshot();gpus=[]
    for line in table.splitlines():
        index,uuid,name,used,free,util=line.split(', ');index=int(index)
        if index in old.ALLOWED and uuid==old.ALLOWED[index] and int(free)>=18000 and int(util)==0 and not any(p.startswith(uuid) for p in apps.splitlines()):
            gpus.append(dict(physical_gpu=index,uuid=uuid))
    if not gpus:raise RuntimeError('No allowed idle GPU; GPU1 excluded')
    blobs={}
    for name in ['sample_manifest.json','protocol.json','history_bindings.json','accuracy_summary.csv']:
        path=old.OUT/name;relative=str(path.relative_to(ROOT.parent))
        blob=subprocess.check_output(['git','rev-parse',f'{BASE}:{relative}'],cwd=ROOT,text=True).strip()
        assert blob==subprocess.check_output(['git','hash-object',str(path)],text=True).strip()
        blobs[relative]=dict(commit=BASE,blob=blob,sha256=sha256(path))
    records={};totals=Counter();previous_binding=digest(prior)
    for s in samples:
        assert sha256(s['image_path'])==s['image_sha256']
        oldfull=read(old.OUT/'history_bindings.json')[s['question_id']];full=read(ROOT/oldfull['full_path'])
        assert sha256(ROOT/oldfull['full_path'])==oldfull['full_sha256'] and full['sample_sha256']==digest(s)
        middle=old.checkpoint_valid(old_m_path(s),previous_binding,s,'B')
        assert middle['generation']['masked_layers']==MIDDLE
        old.validation.native_logits(s,True)
        for key,r in [('F',full),('M',middle)]:
            g=r['generation'];assert r['result_sha256']==digest(g) and g['score']==int(gqa_prediction(g['text'])==s['answer'])
            totals[key]+=g['score']
        records[s['question_id']]=dict(sample_sha256=digest(s),input_mapping_sha256=digest(s['input_mapping']),
            F=dict(path=oldfull['full_path'],sha256=oldfull['full_sha256']),
            M=dict(path=str(old_m_path(s).relative_to(ROOT)),sha256=sha256(old_m_path(s)),
                   logits_path=str((old.OUT/middle['logits_file']).relative_to(ROOT)),logits_sha256=middle['logits_sha256']))
    assert totals==dict(F=214,M=204)
    freeze_json(OUT/'sample_manifest.json',dict(samples=samples,sample_sha256=[digest(s) for s in samples],
        questions=240,image_clusters=235,task_counts=dict(Counter(s['task'] for s in samples)),gpus=gpus,
        assignments=old.assign(samples,len(gpus)),rq25_manifest_sha256=sha256(old.OUT/'sample_manifest.json'),
        historical_git_blobs=blobs,role='Same consumed RQ2.5 cohort: mechanism exploration, not independent confirmation. No resampling or result filtering.'))
    freeze_json(OUT/'history_bindings.json',records)
    freeze_json(OUT/'gpu_selection.json',dict(snapshot=table,compute_processes=apps,chosen=gpus,forbidden=[1]))
    print('Frozen same240/235; Full214/M204; GPUs',gpus,flush=True)


def cpu():
    stream=io.StringIO();tests=['test_rq2_source_knockout','test_rq2_visual_sufficiency','test_rq2_prompt_five','test_rq3_prompt_interaction']
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in tests))
    record=dict(passed=result.wasSuccessful(),tests=result.testsRun,source=sources(),log=stream.getvalue());print(stream.getvalue(),flush=True)
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
    assert gpu['physical_gpu'] in old.ALLOWED and gpu['physical_gpu']!=1 and gpu['uuid']==old.ALLOWED[gpu['physical_gpu']]
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid']
    table,apps=old.snapshot();row=next(line.split(', ') for line in table.splitlines() if gpu['uuid'] in line)
    if int(row[4])<18000 or int(row[5])!=0 or any(line.startswith(gpu['uuid']) for line in apps.splitlines()):raise RuntimeError('GPU no longer idle; do not interrupt external work')
    return dict(**gpu,snapshot=table,compute_processes=apps,pid=os.getpid())


def causal_states(full,changed,key):
    for i,layer in enumerate(CHECKPOINTS):
        if layer<=min(ALL[key],default=32):assert torch.equal(full[i],changed[i]),f'{key}: pre-intervention state changed atL{layer}'


def smoke(rank):
    code_check();pre=preflight(rank);model,processor,environment=old.load_model(MODEL_PATH)
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()};checks=[]
    samples=read(OUT/'sample_manifest.json')['samples'];selected=[next(s for s in samples if s['task']==t) for t in ['ChooseRel','QueryAttr']]
    for s in selected:
        prompt=prepared(model,processor,s);before=tensor_hashes(prompt);bos=processor.tokenizer.bos_token_id
        f,first,trace,debug=generate(model,prompt,bos,[],audit=True,collect=True)
        cf,cf_first,cf_trace,cf_debug,states=captured_generate(model,prompt,bos,[],audit=True,collect=True)
        assert f==cf and torch.equal(first,cf_first) and all(torch.equal(a,b) for a,b in zip(trace,cf_trace))
        assert torch.equal(debug['prefill_logits'],cf_debug['prefill_logits']) and states.shape==(6,4096)
        assert torch.equal(first,historic_logits(s,'F')) and f['token_ids']==history(s,'F')['generation']['token_ids']
        no,nf,nt,nd,ns=captured_generate(model,prompt,bos,CONDITIONS['EM'],noop=True,audit=True,collect=True)
        assert no['token_ids']==f['token_ids'] and torch.equal(nf,first) and torch.equal(ns,states)
        assert torch.equal(nd['prefill_logits'],debug['prefill_logits']) and all(torch.equal(a,b) for a,b in zip(trace,nt))
        results=[]
        for key in ['M',*CONDITIONS]:
            blocked=ALL[key];g,changed,t,d,h=captured_generate(model,prompt,bos,blocked,audit=True,collect=True)
            plain,pf,pt,_=generate(model,prompt,bos,blocked,collect=True)
            assert plain==g and torch.equal(pf,changed) and all(torch.equal(a,b) for a,b in zip(t,pt))
            assert native_reference(model,prompt,bos,blocked)==g['token_ids']
            assert torch.equal(d['prefill_logits'][:,:-1],debug['prefill_logits'][:,:-1])
            assert d['observer'].states==debug['observer'].states
            assert all(a['other_query_attention_sha256']==b['other_query_attention_sha256'] for a,b in zip(d['rows'],debug['rows']))
            assert all(d['rows'][k]['Prompt_max']==0 for k in blocked)
            pf,hp,rows=prefill(model,prompt,bos,blocked)
            assert torch.equal(pf,changed) and torch.equal(h,hp);causal_states(states,h,key)
            if key=='M':
                assert g['token_ids']==history(s,'M')['generation']['token_ids'] and torch.equal(changed,historic_logits(s,'M'))
            results.append(dict(condition=key,token_ids=g['token_ids'],first_logits_sha256=tensor_hash(changed),
                states_sha256=tensor_hash(h),mask_exact=True,other_queries_exact=True,native_hf_decode_exact=True,causal_invariants=True))
        try:
            from .engine import StateCapture
            with StateCapture(model,prompt):raise RuntimeError('state hook cleanup smoke')
        except RuntimeError:pass
        assert footprint==hook_footprint(model) and before==tensor_hashes(prompt) and versions=={n:p._version for n,p in model.named_parameters()}
        checks.append(dict(question_id=s['question_id'],historical_full_and_M_exact=True,noop_all_logits_exact=True,
            read_only_capture_all_logits_and_tokens_exact=True,normal_exception_cleanup=True,conditions=results))
    atomic_json(OUT/f'shard{rank}/smoke.json',dict(passed=True,source=sources(),manifest_sha256=sha256(OUT/'sample_manifest.json'),
        checks=checks,environment=environment,preflight=pre,memory=old.gpu_memory()))
    print('Smoke passed GPU',pre['physical_gpu'],flush=True)


def freeze():
    code_check();manifest=read(OUT/'sample_manifest.json');prior=old.verify()
    smokes=[read(OUT/f'shard{k}/smoke.json') for k in range(len(manifest['gpus']))]
    assert all(s['passed'] and s['source']==sources() and s['checks']==smokes[0]['checks'] and s['manifest_sha256']==sha256(OUT/'sample_manifest.json') for s in smokes)
    keys=['python','torch','transformers','cuda_runtime','cudnn','gpu','model_revision','parameter_dtypes']
    assert all({k:s['environment'][k] for k in keys}=={k:smokes[0]['environment'][k] for k in keys} for s in smokes)
    for p,b in prior['model_files'].items():assert sha256(p)==b['sha256']
    names=['sample_manifest.json','history_bindings.json','gpu_selection.json','cpu_validation.json',*[f'shard{k}/smoke.json' for k in range(len(smokes))]]
    freeze_json(OUT/'protocol.json',dict(status='frozen',base_commit=BASE,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source=sources(),
        bindings={n:sha256(OUT/n) for n in names},model_revision=prior['model_revision'],model_files=prior['model_files'],transformers_source=prior['transformers_source'],
        transformers='4.49.0',dtype='FP16',attention='eager',conditions=ALL,new_conditions=CONDITIONS,questions=240,image_clusters=235,
        new_complete_generations=720,reused_complete_generations=dict(F=240,M=240),gpus=manifest['gpus'],forbidden_gpu=1,
        generation=prior['generation'],prompt=prior['prompt'],
        statistics=dict(seed=SEED,repetitions=100000,primary='EM-E-M+F',secondary=SECONDARY,secondary_family=6,
            unit='Same whole-image cluster weights for all conditions and interaction, question-weighted estimates.',
            scope='Exploratory same previously used240; ordinary95% primary interval is descriptive, not independent confirmation; secondary Bonferroni6 reference intervals.',
            no_equivalence_or_noninferiority=True),
        hidden_states=dict(checkpoints=CHECKPOINTS,position='original P-1',stage='decoder layer input',read_only=True,
            formal_prefills='One F compatibility Prefill and one M hidden-only Prefill per question, not complete regeneration. New E/EM/L captured inside their existing generation.',
            metrics='FP32 cosine distance, signed norm change and relative L2; before-first-mask inputs must be bitwise equal.'),
        anomaly_rules=dict(empty='decoded text strip empty',question_echo='normalized complete original question appears in decoded text',
            repeated_text='three consecutive identical tokens or three repeated normalized words',format='nonempty output without ASCII alphanumeric content or contains role marker USER:/ASSISTANT:',
            interpretation='Deterministic descriptive flags, not an extra scorer or a sample exclusion.'),
        limits='Only E L2-L12, EM L2-L16, L L17-L31 Prompt source OFF at original P-1; no Decode intervention, patching, tuning, training, new samples or automatic follow-up.'))
    print('Frozen interaction and720-generation protocol',flush=True)


def verify():
    c=read(OUT/'protocol.json');assert c['source']==sources() and c['conditions']==ALL and c['new_conditions']==CONDITIONS
    for name,h in c['bindings'].items():assert sha256(OUT/name)==h
    for p,h in c['transformers_source'].items():assert sha256(p)==h
    for p,b in c['model_files'].items():
        st=Path(p).stat();assert st.st_size==b['bytes'] and st.st_mtime_ns==b['mtime_ns']
    assert all(g['physical_gpu']!=1 for g in c['gpus']);return c


def raw_save(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)


def checkpoint(path,binding,s,key):
    r=read(path);assert r['binding']==binding and r['sample_sha256']==digest(s) and r['condition']==key
    assert r['result_sha256']==digest(r['generation']) and r['metric_sha256']==digest(r['first_token'])
    assert sha256(OUT/r['raw_file'])==r['raw_sha256'];return r


def worker(rank):
    c=verify();binding=digest(c);manifest=read(OUT/'sample_manifest.json');folder=OUT/f'shard{rank}';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        assert read(folder/'completion.json')['binding']==binding;print('Complete already; no GPU loaded',flush=True);return
    pre=preflight(rank);start=time.perf_counter();epoch=time.time();model,processor,env=old.load_model(MODEL_PATH)
    assert env['model_revision']==c['model_revision'] and env['transformers']=='4.49.0' and env['parameter_dtypes']==['torch.float16'] and not model.training
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()};resumed=0
    ss=[s for s in manifest['samples'] if manifest['assignments'][s['question_id']]==rank]
    for i,s in enumerate(ss):
        prompt=None;stem=f"{s['task']}__{s['question_id']}"
        for key,blocked in CONDITIONS.items():
            path=folder/f'checkpoints/{stem}/{key}.json'
            if path.exists():checkpoint(path,binding,s,key);resumed+=1;continue
            if prompt is None:
                prompt=prepared(model,processor,s);before=tensor_hashes(prompt);bos=processor.tokenizer.bos_token_id
                full=historic_logits(s,'F');current,full_states,full_rows=prefill(model,prompt,bos,[])
                middle,middle_states,middle_rows=prefill(model,prompt,bos,MIDDLE)
                assert torch.equal(current,full) and torch.equal(middle,historic_logits(s,'M')),'Historical F/M mismatch; stop without relaxing tolerance'
                causal_states(full_states,middle_states,'M')
                baseline_raw=folder/f'baseline/{stem}.npz'
                raw_save(baseline_raw,F=full_states.numpy(),M=middle_states.numpy())
                atomic_json(baseline_raw.with_suffix('.json'),dict(binding=binding,sample_sha256=digest(s),F_logits_exact=True,M_logits_exact=True,
                    F_logits_sha256=tensor_hash(current),M_logits_sha256=tensor_hash(middle),raw_file=str(baseline_raw.relative_to(OUT)),raw_sha256=sha256(baseline_raw),
                    F_states_sha256=tensor_hash(full_states),M_states_sha256=tensor_hash(middle_states),F_rows=full_rows,M_rows=middle_rows))
            tick=time.perf_counter();g,first,_,debug,states=captured_generate(model,prompt,bos,blocked,observe=True)
            causal_states(full_states,states,key)
            text=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()
            g.update(text=text,prediction=gqa_prediction(text),score=int(gqa_prediction(text)==s['answer']))
            metric=metric_rows(full,first[None])[0];metric['logp_loss']=-metric['delta_logp']
            raw=path.with_suffix('.npz');raw_save(raw,first_logits=first.numpy(),states=states.numpy())
            assert footprint==hook_footprint(model) and before==tensor_hashes(prompt) and versions=={n:p._version for n,p in model.named_parameters()}
            atomic_json(path,dict(binding=binding,sample_sha256=digest(s),condition=key,task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],
                generation=g,result_sha256=digest(g),first_token=metric,metric_sha256=digest(metric),raw_file=str(raw.relative_to(OUT)),raw_sha256=sha256(raw),
                states_sha256=tensor_hash(states),attention_rows=debug['rows'],attention_sha256=digest(debug['rows']),
                seconds=time.perf_counter()-tick,physical_gpu=pre['physical_gpu'],hooks_clean=True,inputs_unchanged=True,parameters_unchanged=True))
        if (i+1)%10==0 or i+1==len(ss):print(json.dumps(dict(gpu=pre['physical_gpu'],questions=i+1,total=len(ss),generations=(i+1)*3,seconds=time.perf_counter()-start)),flush=True)
    verify();atomic_json(folder/'completion.json',dict(status='complete',binding=binding,questions=len(ss),generations=len(ss)*3,resumed=resumed,
        started_epoch=epoch,finished_epoch=time.time(),seconds=time.perf_counter()-start,environment=env,preflight=pre,memory=old.gpu_memory(),
        hooks_clean=footprint==hook_footprint(model),parameters_unchanged=versions=={n:p._version for n,p in model.named_parameters()}))
    print('Complete GPU',pre['physical_gpu'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cpu','smoke','freeze','worker']);p.add_argument('--rank',type=int,default=0);args=p.parse_args()
    try:
        if args.stage in ['smoke','worker']:globals()[args.stage](args.rank)
        else:globals()[args.stage]()
    except BaseException:
        atomic_json(OUT/f'failures/{args.stage}_{args.rank}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
