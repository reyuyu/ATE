"""One fixed GPU,240 fixed questions,one new condition; historical A/B are read-only."""
import argparse
from collections import Counter
import fcntl
from functools import lru_cache
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

from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from knockout_probe import tensor_hash
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import prepared, atomic_json, tensor_hashes
from paper_rq1_early_joint_attention.protocol import ROOT, MODEL_PATH
from rq2_visual_full import run as prior
from rq2_visual_sufficiency.engine import generate as old_generate
from vlm_common import load_model, gpu_memory
from .engine import generate, BLOCKED

BASE = 'acc137dc33b02c17cffb9b1a21999d0cb95735c8'
OUT = ROOT/'results/rq2_decode_visual_lightweight_240'
SEED = 20261101
TASKS = list(prior.TASKS)
GPU = 'GPU-b08489af-0a8a-eb48-55ba-8346733af417'
LABELS = dict(Full='Full', B='P-1-only', C='P-1+Decode')


def read(path):
    return json.loads(Path(path).read_text())


def select(samples, seed=SEED):
    selected, used = [], set()
    for task in TASKS:
        candidates = sorted((s for s in samples if s['task']==task),
            key=lambda s:digest(['rq2.4',seed,task,s['question_id'],s['image_sha256']]))
        chosen=[]
        for s in candidates:
            if s['image_sha256'] not in used:
                chosen.append(s);used.add(s['image_sha256'])
                if len(chosen)==40:break
        if len(chosen)!=40:
            raise ValueError('Insufficient image-unique frozen cohort; do not silently resample')
        selected.extend(chosen)
    return sorted(selected,key=lambda s:(s['task'],s['question_id']))


def sources():
    paths = set(prior.sources()) | {'rq2_visual_full/run.py', 'test_rq2_decode_visual.py'}
    paths |= {'rq2_decode_visual/'+n for n in ('__init__.py','engine.py','run.py','analyze.py')}
    return {n:sha256(ROOT/n) for n in sorted(paths)}


@lru_cache(maxsize=1)
def histories():
    return read(OUT/'history_bindings.json')


@lru_cache(maxsize=1)
def samples():
    return read(OUT/'sample_manifest.json')['samples']


def length_group(ids):
    count=sum(t!=2 for t in ids)
    return 'One-token' if count==1 else 'Multi-token' if count>=2 else 'Zero-token'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=read(prior.OUT/'sample_manifest.json')
    pool=[s for s in manifest['samples'] if manifest['splits'][s['question_id']]['image_disjoint']]
    assert len(pool)==2656
    ss=select(pool)
    assert len(ss)==240 and len({s['image_sha256'] for s in ss})==240
    assert Counter(s['task'] for s in ss)==Counter({t:40 for t in TASKS})
    old=read(prior.OUT/'protocol.json');binding=digest(old);records={}
    for name in ['sample_manifest.json','history_bindings.json','protocol.json','validation.json']:
        local=prior.OUT/name
        blob=subprocess.check_output(['git','rev-parse',f'{BASE}:vlm_research/{local.relative_to(ROOT)}'],text=True).strip()
        assert blob==subprocess.check_output(['git','hash-object',local],text=True).strip()
    for s in ss:
        h=prior.history()[s['question_id']]
        full=ROOT/h['full_path'];assert sha256(full)==h['full_sha256']
        f=read(full);assert f['sample_sha256']==digest(s) and digest(f['generation'])==f['result_sha256']
        rank=manifest['assignments'][s['question_id']]
        b=prior.OUT/f"shard{rank}/checkpoints/{s['task']}__{s['question_id']}/C.json"
        rec=prior.checkpoint_valid(b,binding,s,'C')
        assert rec['generation']['decode_native'] and rec['generation']['masked_layers']==BLOCKED
        assert sha256(s['image_path'])==s['image_sha256']
        native=prior.native_logits(s,True)
        assert int(native.argmax())==f['generation']['token_ids'][0]
        for g in (f['generation'],rec['generation']):
            assert g['score']==int(gqa_prediction(g['text'])==s['answer'])
            assert g['eos']==(g['token_ids'][-1]==2)
        records[s['question_id']]=dict(sample_sha256=digest(s),Full=f['generation'],B=rec['generation'],
            full_path=str(full.relative_to(ROOT)),full_sha256=sha256(full),
            b_path=str(b.relative_to(ROOT)),b_sha256=sha256(b),
            b_logits_path=str((prior.OUT/rec['logits_file']).relative_to(ROOT)),b_logits_sha256=rec['logits_sha256'],
            full_first_logits_sha256=tensor_hash(native),historical_source_commit=BASE)
    for p,b in old['model_files'].items():assert sha256(p)==b['sha256']
    for p,h in old['transformers_source'].items():assert sha256(p)==h
    freeze_json(OUT/'sample_manifest.json',dict(seed=SEED,selection='Task-order hash rank; first40 unused image SHA256 per task; no output/score/length filtering',
        source_commit=BASE,source_manifest_sha256=sha256(prior.OUT/'sample_manifest.json'),
        eligible_questions=2656,samples=ss,questions=240,image_clusters=240))
    freeze_json(OUT/'history_bindings.json',records)
    from .analyze import write_csv
    rows=[]
    for task in ['All',*TASKS]:
        subset=[s for s in ss if task=='All' or s['task']==task]
        lengths=Counter(sum(t!=2 for t in records[s['question_id']]['Full']['token_ids']) for s in subset)
        for length,n in sorted(lengths.items()):
            rows.append(dict(task=task,non_eos_answer_tokens=length,questions=n,
                             selection='Frozen Full outputs; not used for sample selection'))
    write_csv(OUT/'generation_length_summary.csv',rows)
    print('Frozen240 questions/240 images; Full non-EOS lengths:',dict(Counter(sum(t!=2 for t in records[s['question_id']]['Full']['token_ids']) for s in ss)),flush=True)


def cpu():
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromName('test_rq2_decode_visual')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful()
    freeze_json(OUT/'cpu_validation.json',dict(passed=True,tests=result.testsRun,source=sources(),output=stream.getvalue()))


def code_check():
    names=['vlm_research/'+n for n in sources()]
    assert set(subprocess.check_output(['git','ls-files','--',*names],cwd=ROOT.parent,text=True).splitlines())==set(names)
    assert not subprocess.check_output(['git','diff','HEAD','--',*names],cwd=ROOT.parent,text=True).strip()
    assert read(OUT/'cpu_validation.json')['source']==sources()


def freeze():
    code_check();old=read(prior.OUT/'protocol.json')
    freeze_json(OUT/'protocol.json',dict(status='frozen_before_any_GPU',base_commit=BASE,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source=sources(),
        bindings={n:sha256(OUT/n) for n in ('sample_manifest.json','history_bindings.json','generation_length_summary.csv','cpu_validation.json')},
        model_revision=old['model_revision'],model_files=old['model_files'],transformers_source=old['transformers_source'],
        dtype='FP16',attention='eager',transformers='4.49.0',gpu_index=1,gpu_uuid=GPU,
        generation=old['generation'],blocked_layers=BLOCKED,retained_layers=[14,19],
        intervention='Original Prefill P-1 only and every later incremental Query -> real576 Image keys; pre-softmax -inf; native other queries/edges/MLP/positions/cache',
        reused=dict(Full=240,B=240),new_generations=240,
        statistics=dict(comparison='C-B',unit='whole Image SHA256 clusters, question-weighted paired bootstrap',
            repetitions=200000,seed=SEED,ci='Exploratory ordinary95% percentile; no confirmatory NI claim',
            groups='Full non-EOS length==1 / >=2; fixed before GPU; do not reselect if few multitoken answers'),
        stop='Only240 new C generations, apart from necessary smoke; no new layers or follow-up experiments'))


def verify():
    c=read(OUT/'protocol.json');assert c['source']==sources()
    for n,h in c['bindings'].items():assert sha256(OUT/n)==h
    for p,h in c['transformers_source'].items():assert sha256(p)==h
    for p,b in c['model_files'].items():
        stat=Path(p).stat();assert stat.st_size==b['bytes'] and stat.st_mtime_ns==b['mtime_ns']
    return c


def preflight():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    table=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    row=next(line.split(', ') for line in table.splitlines() if GPU in line)
    if int(row[0])!=1 or int(row[3])<18000 or GPU in apps:
        raise RuntimeError('Selected GPU1 is occupied/unsuitable; stop, never switch or terminate processes')
    return dict(table=table,compute_apps=apps,selected_gpu=1,uuid=GPU)


def b_logits(s):
    h=histories()[s['question_id']];p=ROOT/h['b_logits_path']
    assert sha256(p)==h['b_logits_sha256']
    with np.load(p) as z:return torch.from_numpy(z['first_logits'].copy())


def smoke():
    c=verify();pre=preflight();tick=time.perf_counter()
    model,processor,environment=load_model(MODEL_PATH)
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    ss=samples();hh=histories();selected=[]
    for group in ('One-token','Multi-token'):
        eligible=[s for s in ss if length_group(hh[s['question_id']]['Full']['token_ids'])==group]
        if eligible:selected.append(eligible[0])
    if len(selected)==1:selected.append(next(s for s in ss if s!=selected[0]))
    checks=[]
    for s in selected:
        prompt=prepared(model,processor,s);before=tensor_hashes(prompt);bos=processor.tokenizer.bos_token_id
        full,ff,ft,fd=old_generate(model,prompt,bos,[],audit=True,collect=True)
        noop,nf,nt,nd=generate(model,prompt,bos,noop=True,audit=True,collect=True)
        assert full['token_ids']==hh[s['question_id']]['Full']['token_ids']==noop['token_ids']
        assert torch.equal(ff,prior.native_logits(s,True)) and torch.equal(ff,nf)
        assert torch.equal(fd['prefill_logits'],nd['prefill_logits'])
        assert len(ft)==len(nt) and all(torch.equal(a,b) for a,b in zip(ft,nt))
        with torch.inference_mode():
            native=model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,eos_token_id=2)[0,prompt['input_ids'].shape[1]:].tolist()
        assert native==full['token_ids']
        old,of,_,od=old_generate(model,prompt,bos,BLOCKED,audit=True)
        new,cf,_,cd=generate(model,prompt,bos,audit=True)
        assert old['token_ids']==hh[s['question_id']]['B']['token_ids']
        assert torch.equal(of,b_logits(s)) and torch.equal(of,cf)
        assert old['token_ids'][0]==new['token_ids'][0]
        assert torch.equal(od['prefill_logits'],cd['prefill_logits'])
        assert torch.equal(fd['prefill_logits'][0,:-1],cd['prefill_logits'][0,:-1])
        assert fd['observer'].states==cd['observer'].states
        assert all(a['other_query_attention_sha256']==b['other_query_attention_sha256'] for a,b in zip(fd['observer'].rows,cd['observer'].rows))
        assert all(r['Image_max']==0 if r['layer'] in BLOCKED else r['Image']>0 for r in cd['observer'].rows)
        assert tensor_hashes(prompt)==before and hook_footprint(model)==footprint
        assert versions=={n:p._version for n,p in model.named_parameters()}
        checks.append(dict(question_id=s['question_id'],task=s['task'],full_length_group=length_group(full['token_ids']),
            p_minus_one=prompt['input_ids'].shape[1]-1,visual_keys=new['image_key_indices'],
            noop_all_prefill_logits_and_all_decode_logits_exact=True,historical_full_and_B_tokens_exact=True,
            B_C_first_logits_max_difference=0.,B_C_all_prefill_logits_exact=True,other_prefill_queries_exact=True,
            first_token_equal=True,decode_steps=new['decode_steps'],decode_audit=new['decode_mask_audit']))
    freeze_json(OUT/'smoke.json',dict(passed=True,protocol_sha256=sha256(OUT/'protocol.json'),
        process_id=os.getpid(),
        checks=checks,preflight=pre,environment=environment,seconds=time.perf_counter()-tick,
        hooks_clean=True,inputs_and_parameters_unchanged=True))
    print('GPU1 smoke passed: exact B/C Prefill, history, No-op and actual Decode probabilities',flush=True)


def checkpoint(s,binding):
    p=OUT/f"checkpoints/{s['task']}__{s['question_id']}.json"
    if not p.exists():return None
    r=read(p);assert r['binding']==binding and r['sample_sha256']==digest(s)
    assert r['result_sha256']==digest(r['generation']) and r['first_logits_exact_B']
    assert sha256(OUT/r['logits_file'])==r['logits_sha256']
    return r


def worker():
    c=verify();binding=digest(c)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'completion.json').exists():
        assert read(OUT/'completion.json')['binding']==binding
        print('Already complete; no GPU loaded',flush=True);return
    sm=read(OUT/'smoke.json');assert sm['passed'] and sm['protocol_sha256']==sha256(OUT/'protocol.json')
    pre=preflight();start=time.time();tick=time.perf_counter()
    model,processor,environment=load_model(MODEL_PATH)
    for k in ('python','torch','transformers','cuda_runtime','cudnn','gpu','model_revision','parameter_dtypes'):
        assert environment[k]==sm['environment'][k]
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()};resumed=0
    for i,s in enumerate(samples()):
        if checkpoint(s,binding):resumed+=1;continue
        before_step=time.perf_counter();prompt=prepared(model,processor,s);before=tensor_hashes(prompt)
        g,first,_,_=generate(model,prompt,processor.tokenizer.bos_token_id)
        assert torch.equal(first,b_logits(s)) and g['token_ids'][0]==histories()[s['question_id']]['B']['token_ids'][0]
        text=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()
        g.update(text=text,prediction=gqa_prediction(text),score=int(gqa_prediction(text)==s['answer']))
        assert tensor_hashes(prompt)==before and hook_footprint(model)==footprint
        assert versions=={n:p._version for n,p in model.named_parameters()}
        path=OUT/f"checkpoints/{s['task']}__{s['question_id']}.json";path.parent.mkdir(exist_ok=True)
        raw=path.with_suffix('.npz');temp=raw.with_suffix('.npz.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,first_logits=first.numpy())
        temp.replace(raw)
        atomic_json(path,dict(binding=binding,condition='C',sample_sha256=digest(s),
            task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],generation=g,
            result_sha256=digest(g),first_logits_sha256=tensor_hash(first),first_logits_exact_B=True,
            logits_file=str(raw.relative_to(OUT)),logits_sha256=sha256(raw),seconds=time.perf_counter()-before_step,
            hooks_clean=True,inputs_unchanged=True,parameters_unchanged=True))
        if (i+1)%20==0:print(f'Complete {i+1}/240; {time.perf_counter()-tick:.1f}s',flush=True)
    verify()
    freeze_json(OUT/'completion.json',dict(status='complete',binding=binding,new_generations=240,
        process_id=os.getpid(),
        resumed=resumed,started_epoch=start,finished_epoch=time.time(),seconds=time.perf_counter()-tick,
        environment=environment,preflight=pre,memory=gpu_memory(),hooks_clean=hook_footprint(model)==footprint,
        parameters_unchanged=versions=={n:p._version for n,p in model.named_parameters()}))
    print('All240 new C generations complete; GPU process exiting',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','cpu','freeze','smoke','worker']);args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        atomic_json(OUT/f'failures/{args.stage}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
