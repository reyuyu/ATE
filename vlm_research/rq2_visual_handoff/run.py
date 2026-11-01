"""Bind existing data, freeze four conditions, then run resumable isolated workers."""
import argparse
from collections import Counter, defaultdict
import fcntl
import io
import itertools
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
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_single_layer_gqa6.run import atomic_json, tensor_hashes
from rq2_decode_visual import run as old
from vlm_common import load_model, prepare_inputs, gpu_memory
from .engine import CONDITIONS, CHECKPOINTS, PrefillMask, generate, distances

ROOT, MODEL_PATH = old.ROOT, old.MODEL_PATH
OUT = ROOT/'results/rq2_visual_text_handoff_240'
BASE = '3ae8c72296f2521577b68e7b5c069ea7ef63d8c1'
SEED = 20261101
ALLOWED = {0:'GPU-a63c6362-dcfd-4a26-c44b-014b40127c89',
           2:'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6',
           3:'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185'}
read = old.read


def snapshot():
    table = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    return table, apps


def sources():
    names = set(old.sources()) | {'rq2_decode_visual/analyze.py','test_rq2_visual_handoff.py'}
    names |= {'rq2_visual_handoff/'+n for n in ['__init__.py','engine.py','run.py','analyze.py']}
    return {n:sha256(ROOT/n) for n in sorted(names)}


def assign(primary, pairs, captions, n):
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    for pair in pairs:
        a,b=[s['image_sha256'] for s in pair['samples']]
        parent[find(b)]=find(a)
    units=defaultdict(list)
    for kind, records in [('main',primary),('caption',captions)]:
        for s in records: units[find(s['image_sha256'])].append((kind,s['question_id']))
    for p in pairs: units[find(p['samples'][0]['image_sha256'])].append(('swap',p['pair_id']))
    loads=[0]*n; assignments={k:{} for k in ['main','swap','caption']}; image_ranks={}
    for cluster, items in sorted(units.items(),key=lambda kv:(-sum(8 if k=='swap' else 4 for k,_ in kv[1]),digest([SEED,kv[0]]))):
        rank=min(range(n),key=lambda k:(loads[k],k));image_ranks[cluster]=rank
        for kind, key in items:assignments[kind][key]=rank
        loads[rank]+=sum(8 if k=='swap' else 4 for k,_ in items)
    for image in parent:image_ranks[image]=image_ranks[find(image)]
    return assignments,image_ranks


def partition_record(processor, sample):
    message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':sample['prompt_text']}]}]
    text=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
    raw=processor.tokenizer(text,return_offsets_mapping=True);ids=[];offsets=[]
    for tid, span in zip(raw['input_ids'],raw['offset_mapping']):
        ids.extend([tid]*(576 if tid==32000 else 1));offsets.extend([span]*(576 if tid==32000 else 1))
    assert ids==sample['input_mapping']['prompt_ids']
    image=[i for i,t in enumerate(ids) if t==32000];p1=len(ids)-1
    assert len(image)==576 and image==list(range(image[0],image[-1]+1))
    relay=[i for i in range(image[-1]+1,p1) if ids[i]!=32000]
    assert ids[0]==1 and ids.count(1)==1 and relay
    q=sample.get('question','');qs=text.find(q) if q else -1
    instruction='Answer the question using a single word or phrase.';ins=text.find(instruction)
    groups={k:[] for k in ['Question','Instruction','Template']}
    for i in relay:
        a,b=offsets[i]
        group='Question' if qs>=0 and a<qs+len(q) and b>qs else 'Instruction' if ins>=0 and a<ins+len(instruction) and b>ins else 'Template'
        groups[group].append(i)
    assert sorted(sum(groups.values(),[]))==relay
    return dict(image_keys=image,p_minus_one=p1,relay_queries=relay,subgroups=groups,
        text_before_image=[i for i in range(image[0]) if ids[i]!=32000],
        question_span_policy='Actual processor tokenizer offsets, expanded IDs verified against frozen inputs; boundary overlaps assigned Question before Instruction. Every post-image pre-P-1 nonvisual position included regardless of subgroup.',
        condition_blocked_connections_per_head_per_layer={k:576*((0 if r else len(relay))+(0 if d else 1)) for k,(r,d) in CONDITIONS.items()},
        prompt_ids_sha256=digest(ids),sample_sha256=digest(sample))


def prepare():
    from transformers import AutoProcessor
    OUT.mkdir(parents=True,exist_ok=True)
    prior=old.verify();manifest=read(old.OUT/'sample_manifest.json');primary=manifest['samples']
    assert len(primary)==240 and len({s['image_sha256'] for s in primary})==240
    assert Counter(s['task'] for s in primary)==Counter({t:40 for t in old.TASKS})
    histories=read(old.OUT/'history_bindings.json');history={}
    for s in primary:
        h=histories[s['question_id']];path=ROOT/h['full_path'];f=read(path)
        assert sha256(path)==h['full_sha256'] and f['sample_sha256']==digest(s)
        assert digest(f['generation'])==f['result_sha256'] and f['generation']==h['Full']
        assert f['generation']['score']==int(gqa_prediction(f['generation']['text'])==s['answer'])
        native=old.prior.native_logits(s,True)
        assert tensor_hash(native)==h['full_first_logits_sha256'] and int(native.argmax())==h['Full']['token_ids'][0]
        history[s['question_id']]=dict(path=h['full_path'],sha256=h['full_sha256'],sample_sha256=digest(s),generation=h['Full'],first_logits_sha256=tensor_hash(native))
    source_path=ROOT/'results/paper_rq1_single_layer_gqa6/sample_manifest.json'
    original=read(source_path)['samples'];groups=defaultdict(list)
    for s in original:groups[s['question']].append(s)
    candidates=[]
    for q,group in groups.items():
        possible=[(a,b) for a,b in itertools.combinations(group,2) if a['image_sha256']!=b['image_sha256'] and a['answer']!=b['answer'] and a['input_mapping']['prompt_ids']==b['input_mapping']['prompt_ids']]
        if possible:candidates.append((q,sorted(possible,key=lambda ab:digest([SEED,ab[0]['question_id'],ab[1]['question_id']]))))
    pairs=[];used=set()
    for q,possible in sorted(candidates,key=lambda row:digest([SEED,'swap',row[0]])):
        eligible=[ab for ab in possible if all(s['image_sha256'] not in used for s in ab)]
        if not eligible:continue
        a,b=eligible[0];pair_id=f'pair{len(pairs):03d}'
        pairs.append(dict(pair_id=pair_id,question=q,samples=[a,b],gold_answers=[a['answer'],b['answer']],
            same_text_ids=True,different_annotation_gold=True,role='Semantic image exchange, real GQA labels, no prediction selection'))
        used.update([a['image_sha256'],b['image_sha256']])
        if len(pairs)==40:break
    if not pairs:raise ValueError('No trustworthy same-question image pairs')
    scorer_path=ROOT/'results/paper_rq1_attention_time/scorer_validation.json';scorer=read(scorer_path)
    caption_ok=scorer['passed'] and bool(scorer['checks'].get('official_caption_identical_reference'))
    for name,h in scorer['source_sha256'].items():
        if not Path(name).exists() or sha256(name)!=h:caption_ok=False
    captions=[];blocked_images=used|{s['image_sha256'] for s in primary}
    if caption_ok:
        pool=[]
        for p in sorted((ROOT/'results/paper_rq1_attention_time/manifests').glob('shard*.json')):
            pool.extend(s for s in read(p)['samples'] if s['dataset']=='coco_caption' and s['split']=='discovery')
        for s in sorted(pool,key=lambda s:digest([SEED,'caption',s['image_sha256']])):
            if s['image_sha256'] in blocked_images:continue
            captions.append(s);blocked_images.add(s['image_sha256'])
            if len(captions)==60:break
        assert len(captions)==60,'Available caption cohort insufficient; do not resample'
    table,apps=snapshot();gpus=[]
    for line in table.splitlines():
        index,uuid,name,used_mb,free,util=line.split(', ');index=int(index)
        if index in ALLOWED and uuid==ALLOWED[index] and int(free)>=18000 and int(util)==0 and not any(row.startswith(uuid) for row in apps.splitlines()):
            gpus.append(dict(physical_gpu=index,uuid=uuid,name=name))
    if not gpus:raise RuntimeError('No allowed idle GPU; GPU1 forbidden')
    assignments,image_ranks=assign(primary,pairs,captions,len(gpus))
    processor=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
    partitions={}
    for s in primary+[s for p in pairs for s in p['samples']]+captions:
        assert sha256(s['image_path'])==s['image_sha256']
        record=partition_record(processor,s);key=s['question_id']
        if key in partitions:assert record==partitions[key]
        partitions[key]=record
    freeze_json(OUT/'sample_manifest.json',dict(samples=primary,questions=240,image_clusters=240,
        source_commit=BASE,source_manifest_sha256=sha256(old.OUT/'sample_manifest.json'),
        seed=SEED,gpus=gpus,assignments=assignments,image_ranks=image_ranks,
        role='Previously consumed RQ2.4 cohort, exploratory mechanism experiment; not an independent confirmation set.'))
    freeze_json(OUT/'history_bindings.json',history)
    freeze_json(OUT/'source_partition_audit.json',partitions)
    freeze_json(OUT/'image_swap_manifest.json',dict(pairs=pairs,candidate_question_groups=len(candidates),
        target=40,actual=len(pairs),source_sha256=sha256(source_path),source_commit='221cdde56dde53908aaa953fe26672fd33334f84',
        rule='Hash-order exact question groups; hash-order differing annotated answers with equal text IDs; no repeated endpoint image; no prediction/score filtering.',hidden_pairs=[p['pair_id'] for p in pairs[:6]]))
    freeze_json(OUT/'caption_manifest.json',dict(samples=captions,enabled=caption_ok,questions=len(captions),
        max_new_tokens=96,scorer_validation_sha256=sha256(scorer_path),scorer_source_sha256=scorer['source_sha256'],
        reason='Existing frozen discovery COCO images and previously validated CIDEr/SPICE; no downloads.' if caption_ok else 'Existing scorer missing or changed; skipped rather than introducing an unaudited metric.'))
    print('Prepared',240,'GQA,',len(pairs),'swap pairs,',len(captions),'captions; GPUs',gpus,flush=True)


def cpu():
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_rq2_visual_handoff','test_rq2_decode_visual','test_rq2_source_knockout'])
    stream=io.StringIO();tick=time.perf_counter();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    freeze_json(OUT/'cpu_validation.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,output=stream.getvalue(),seconds=time.perf_counter()-tick,source=sources()))
    print(stream.getvalue(),flush=True)
    if not r.wasSuccessful():raise AssertionError('CPU regression failure')


def freeze():
    test=read(OUT/'cpu_validation.json');assert test['passed'] and test['source']==sources()
    names=['sample_manifest.json','history_bindings.json','source_partition_audit.json','image_swap_manifest.json','caption_manifest.json','cpu_validation.json']
    tracked=subprocess.check_output(['git','ls-files','--',*['vlm_research/'+p for p in sources()]],text=True).splitlines()
    assert set(tracked)=={'vlm_research/'+p for p in sources()}
    assert not subprocess.check_output(['git','diff','HEAD','--',*['vlm_research/'+p for p in sources()]],text=True).strip()
    previous=read(old.OUT/'protocol.json')
    for path,b in previous['model_files'].items():assert sha256(path)==b['sha256']
    freeze_json(OUT/'protocol.json',dict(status='frozen_before_GPU',base_commit=BASE,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source=sources(),
        bindings={n:sha256(OUT/n) for n in names},model_revision=previous['model_revision'],model_files=previous['model_files'],
        transformers_source=previous['transformers_source'],dtype='float16',attention='eager',transformers='4.49.0',
        conditions={k:dict(relay_on=r,direct_on=d,layers=list(range(32))) for k,(r,d) in CONDITIONS.items()},
        generation=dict(do_sample=False,use_cache=True,eos_token_id=2,gqa_max_new_tokens=32,caption_max_new_tokens=96),
        intervention='Pre-softmax -inf only on dynamically mapped target Query -> Image edges. R includes every post-image nonvisual original Query strictly before P-1. D includes P-1 and all incremental Decode.',
        statistics=dict(seed=SEED,repetitions=100000,clusters='Image SHA256',weights='Question-weighted estimate, whole-image resampling, identical draws across all columns',
            primary=['B-D','B-A'],secondary=['C-D','D-C-B+A'],multiplicity='Both primaries also report Bonferroni-2 intervals; secondary results exploratory.',no_NI_or_equivalence=True),
        hidden_checkpoints=dict(layers=CHECKPOINTS,location='Decoder layer output before final model normalization; actual per-position hook values, FP32 distance statistics'),
        swap_D_acceptance='Exact effective FP32-cast FP16 logits at every real Decode step and all nonvisual Prefill logits; no tolerance relaxation.',
        stop='Only frozen240 GQA, frozen image swaps, optional60 captions. No training, sweeps, activation patching, KV replacement, pruning or acceleration.'))


def verify():
    c=read(OUT/'protocol.json');assert c['source']==sources()
    for n,h in c['bindings'].items():assert sha256(OUT/n)==h
    for p,b in c['model_files'].items():
        st=Path(p).stat();assert st.st_size==b['bytes'] and st.st_mtime_ns==b['mtime_ns']
    for p,h in c['transformers_source'].items():assert sha256(p)==h
    return c


def preflight(rank):
    gpu=read(OUT/'sample_manifest.json')['gpus'][rank];assert gpu['physical_gpu']!=1
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');assert visible==str(gpu['physical_gpu']) or visible==gpu['uuid']
    table,apps=snapshot()
    line=next(row for row in table.splitlines() if row.startswith(str(gpu['physical_gpu'])+', '))
    fields=line.split(', ');assert fields[1]==gpu['uuid'] and int(fields[4])>=18000 and int(fields[5])==0
    assert not any(row.startswith(gpu['uuid']) for row in apps.splitlines())
    return dict(**gpu,snapshot=table,compute_apps=apps,pid=os.getpid())


def prompt(model,processor,s):
    inputs,counts=prepare_inputs(model,processor,s['image_path'],s['prompt_text'])
    assert sha256(s['image_path'])==s['image_sha256'] and counts['visual_tokens']==576
    assert inputs['input_ids'][0].tolist()==s['input_mapping']['prompt_ids']
    return inputs


def guard(model,inputs,footprint,versions,before):
    assert hook_footprint(model)==footprint and tensor_hashes(inputs)==before
    assert versions=={n:p._version for n,p in model.named_parameters()}


def swap_run(model,processor,pair,capture=False):
    data=[prompt(model,processor,s) for s in pair['samples']];bos=processor.tokenizer.bos_token_id
    assert torch.equal(data[0]['input_ids'],data[1]['input_ids'])
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    runs={};result={};hidden=[]
    partition=read(OUT/'source_partition_audit.json')[pair['samples'][0]['question_id']]
    for key in CONDITIONS:
        runs[key]=[]
        for inputs in data:
            before=tensor_hashes(inputs)
            r=generate(model,inputs,bos,key,audit=key=='D',capture=capture,collect=key=='D')
            guard(model,inputs,footprint,versions,before);runs[key].append(r)
        a,b=runs[key];metric=metric_rows(a[1],b[1][None])[0]
        generations=[]
        for endpoint,(g,f,trace,debug) in enumerate(runs[key]):
            text=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()
            g.update(text=text,prediction=gqa_prediction(text),score=int(gqa_prediction(text)==pair['samples'][endpoint]['answer']))
            generations.append(g)
        result[key]=dict(generations=generations,first_metric=metric,first_logits_max_difference=float((a[1]-b[1]).abs().max()),
            first_token_same=a[0]['token_ids'][0]==b[0]['token_ids'][0],tokens_same=a[0]['token_ids']==b[0]['token_ids'],
            first_logits_hashes=[tensor_hash(a[1]),tensor_hash(b[1])],
            all_blocked_probabilities_zero=all(r['image_probability_max']==0 for run in runs[key] for r in run[3]['attention_rows'] if r['blocked']))
    da,db=runs['D'];nonvisual=data[0]['input_ids'][0].cpu()!=model.config.image_token_index
    assert torch.equal(da[1],db[1]),'STOP: D remains image conditional'
    assert da[0]['token_ids']==db[0]['token_ids'] and len(da[2])==len(db[2])
    assert all(torch.equal(a,b) for a,b in zip(da[2],db[2])),'D Decode logits differ'
    assert torch.equal(da[3]['prefill_logits'][0,nonvisual],db[3]['prefill_logits'][0,nonvisual]),'D nonvisual Prefill logits differ'
    audit=dict(D_first_logits_exact=True,D_all_decode_logits_exact=True,D_nonvisual_prefill_logits_exact=True,D_full_tokens_exact=True,
        maximum_absolute_D_logits_difference=0.,mask_probabilities_exact_zero=True)
    if capture:
        positions=da[3]['state_positions']
        for endpoint in range(2):
            for left,right in [('A','B'),('C','D')]:
                for layer in CHECKPOINTS:
                    assert torch.equal(runs[left][endpoint][3]['states'][layer][:-1],runs[right][endpoint][3]['states'][layer][:-1]),'Same-image non-P-1 causal invariant failed'
        for key in CONDITIONS:
            for layer in CHECKPOINTS:
                a=runs[key][0][3]['states'][layer];b=runs[key][1][3]['states'][layer]
                if key in ('C','D'):assert torch.equal(a[:-1],b[:-1]),'Relay states retain an unexpected visual route'
                if key=='D':assert torch.equal(a,b)
                for i,pos in enumerate(positions):
                    group='P-1' if pos==partition['p_minus_one'] else next(k for k,v in partition['subgroups'].items() if pos in v)
                    hidden.append(dict(pair_id=pair['pair_id'],condition=key,layer=layer,position=pos,group=group,comparison='same_text_different_image',**distances(a[i],b[i])))
        audit.update(A_B_other_prefill_states_exact=True,C_D_other_prefill_states_exact=True,C_D_relay_states_image_invariant=True)
    raw={f'{k}_{e}':runs[k][e][1].numpy() for k in CONDITIONS for e in range(2)}
    return result,hidden,audit,raw


def smoke(rank):
    c=verify();pre=preflight(rank);tick=time.perf_counter()
    model,processor,env=load_model(MODEL_PATH);footprint=hook_footprint(model)
    manifest=read(OUT/'sample_manifest.json');s=next(s for s in manifest['samples'] if manifest['assignments']['main'][s['question_id']]==rank)
    inputs=prompt(model,processor,s);bos=processor.tokenizer.bos_token_id;before=tensor_hashes(inputs);versions={n:p._version for n,p in model.named_parameters()}
    a,af,at,ad=generate(model,inputs,bos,'A',audit=True,capture=True,collect=True)
    noop,nf,nt,nd=generate(model,inputs,bos,'D',noop=True,audit=True,collect=True)
    assert torch.equal(af,nf) and torch.equal(ad['prefill_logits'],nd['prefill_logits'])
    assert len(at)==len(nt) and all(torch.equal(x,y) for x,y in zip(at,nt))
    assert torch.equal(af,old.prior.native_logits(s,True))
    assert a['token_ids']==read(OUT/'history_bindings.json')[s['question_id']]['generation']['token_ids']==noop['token_ids']
    with torch.inference_mode():
        native=model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,eos_token_id=2)[0,inputs['input_ids'].shape[1]:].tolist()
    assert native==a['token_ids']
    runs={'A':(a,af,at,ad)}
    for key in 'BCD':runs[key]=generate(model,inputs,bos,key,audit=True,capture=True)
    end=runs['A'][0]['image_indices'][-1]+1
    for key in 'BCD':assert torch.equal(ad['prefill_logits'][0,:end],runs[key][3]['prefill_logits'][0,:end])
    for left,right in [('A','B'),('C','D')]:
        for layer in CHECKPOINTS:assert torch.equal(runs[left][3]['states'][layer][:-1],runs[right][3]['states'][layer][:-1])
    pair=read(OUT/'image_swap_manifest.json')['pairs'][rank]
    swap,hidden,swap_audit,raw=swap_run(model,processor,pair,True)
    guard(model,inputs,footprint,versions,before)
    freeze_json(OUT/f'shard{rank}/smoke.json',dict(passed=True,protocol_sha256=sha256(OUT/'protocol.json'),
        preflight=pre,environment=env,question_id=s['question_id'],seconds=time.perf_counter()-tick,
        noop_all_prefill_and_all_decode_logits_exact=True,HF_native_tokens_exact=True,historical_Full_first_logits_and_tokens_exact=True,
        all_before_image_and_image_query_logits_native=True,all_non_target_mask_edges_native=True,
        A_B_and_C_D_relay_states_exact=True,swap_pair=pair['pair_id'],swap_audit=swap_audit,
        mask_rows={k:v[3]['attention_rows'] for k,v in runs.items()},hooks_clean=True,inputs_and_parameters_unchanged=True))
    print('Smoke passed GPU',pre['physical_gpu'],'D swap exactly invariant',flush=True)


def raw_save(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('xb') as f:np.savez_compressed(f,**values)
    os.replace(temp,path)


def valid_checkpoint(path,binding,s=None,key=None):
    r=read(path);assert r['binding']==binding
    if s is not None:assert r['sample_sha256']==digest(s) and r['condition']==key
    assert digest(r['result'])==r['result_sha256'] and sha256(OUT/r['raw_file'])==r['raw_sha256']
    return r


def worker(rank):
    c=verify();binding=digest(c);folder=OUT/f'shard{rank}';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        assert read(folder/'completion.json')['binding']==binding;print('Already complete, GPU not loaded');return
    for k in range(len(read(OUT/'sample_manifest.json')['gpus'])):
        sm=read(OUT/f'shard{k}/smoke.json');assert sm['passed'] and sm['protocol_sha256']==sha256(OUT/'protocol.json')
    pre=preflight(rank);tick=time.perf_counter();epoch=time.time();model,processor,env=load_model(MODEL_PATH)
    assert env['model_revision']==c['model_revision'] and env['transformers']=='4.49.0' and env['parameter_dtypes']==['torch.float16'] and not model.training
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    manifest=read(OUT/'sample_manifest.json');assignments=manifest['assignments'];counts=Counter();resumed=0
    for s in manifest['samples']:
        if assignments['main'][s['question_id']]!=rank:continue
        inputs=None
        for key in 'BCD':
            path=folder/f"main/{s['task']}__{s['question_id']}/{key}.json"
            if path.exists():valid_checkpoint(path,binding,s,key);resumed+=1;counts['main']+=1;continue
            if inputs is None:
                inputs=prompt(model,processor,s);before=tensor_hashes(inputs)
                with torch.inference_mode():current=model(**inputs,use_cache=False,return_dict=True).logits[0,-1].detach().float().cpu()
                full=old.prior.native_logits(s,True)
                assert torch.equal(current,full),'Historical baseline mismatch; stop, do not relax tolerances'
                atomic_json(folder/f"baseline/{s['question_id']}.json",dict(binding=binding,sample_sha256=digest(s),historical_Full_logits_exact=True,first_logits_sha256=tensor_hash(current)))
            g,first,_,debug=generate(model,inputs,processor.tokenizer.bos_token_id,key)
            text=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip();g.update(text=text,prediction=gqa_prediction(text),score=int(gqa_prediction(text)==s['answer']))
            metric=metric_rows(full,first[None])[0];metric['logp_loss']=-metric['delta_logp']
            result=dict(generation=g,first_token=metric,attention_rows=debug['attention_rows'],mask_records=debug['mask_records'])
            guard(model,inputs,footprint,versions,before)
            raw=path.with_suffix('.npz');raw_save(raw,dict(first_logits=first.numpy()))
            atomic_json(path,dict(binding=binding,sample_sha256=digest(s),condition=key,result=result,result_sha256=digest(result),
                raw_file=str(raw.relative_to(OUT)),raw_sha256=sha256(raw),hooks_clean=True,inputs_unchanged=True,parameters_unchanged=True,physical_gpu=pre['physical_gpu']))
            counts['main']+=1
        if counts['main']%30==0:print(json.dumps(dict(gpu=pre['physical_gpu'],counts=dict(counts),seconds=time.perf_counter()-tick)),flush=True)
    swap_manifest=read(OUT/'image_swap_manifest.json')
    for pair in swap_manifest['pairs']:
        if assignments['swap'][pair['pair_id']]!=rank:continue
        path=folder/f"swap/{pair['pair_id']}.json"
        if path.exists():valid_checkpoint(path,binding);resumed+=1;counts['swap_pairs']+=1;continue
        result,hidden,audit,raw=swap_run(model,processor,pair,pair['pair_id'] in swap_manifest['hidden_pairs'])
        values=dict(conditions=result,hidden=hidden,audit=audit,pair_sha256=digest(pair))
        rawpath=path.with_suffix('.npz');raw_save(rawpath,raw)
        atomic_json(path,dict(binding=binding,result=values,result_sha256=digest(values),raw_file=str(rawpath.relative_to(OUT)),raw_sha256=sha256(rawpath),physical_gpu=pre['physical_gpu']))
        counts['swap_pairs']+=1
        print('GPU',pre['physical_gpu'],'swap',counts['swap_pairs'],flush=True)
    for s in read(OUT/'caption_manifest.json')['samples']:
        if assignments['caption'][s['question_id']]!=rank:continue
        inputs=None
        for key in CONDITIONS:
            stem=s['question_id'].replace(':','_');path=folder/f'caption/{stem}/{key}.json'
            if path.exists():valid_checkpoint(path,binding,s,key);resumed+=1;counts['caption']+=1;continue
            if inputs is None:inputs=prompt(model,processor,s);before=tensor_hashes(inputs)
            g,first,_,debug=generate(model,inputs,processor.tokenizer.bos_token_id,key,limit=96)
            text=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip();g['text']=text
            result=dict(generation=g,attention_rows=debug['attention_rows'],mask_records=debug['mask_records'])
            guard(model,inputs,footprint,versions,before)
            rawpath=path.with_suffix('.npz');raw_save(rawpath,dict(first_logits=first.numpy()))
            atomic_json(path,dict(binding=binding,sample_sha256=digest(s),condition=key,result=result,result_sha256=digest(result),
                raw_file=str(rawpath.relative_to(OUT)),raw_sha256=sha256(rawpath),physical_gpu=pre['physical_gpu'],hooks_clean=True,inputs_unchanged=True,parameters_unchanged=True))
            counts['caption']+=1
        print('GPU',pre['physical_gpu'],'captions',counts['caption'],'seconds',time.perf_counter()-tick,flush=True)
    assert footprint==hook_footprint(model) and versions=={n:p._version for n,p in model.named_parameters()}
    verify();atomic_json(folder/'completion.json',dict(status='complete',binding=binding,counts=dict(counts),resumed=resumed,
        started_epoch=epoch,finished_epoch=time.time(),seconds=time.perf_counter()-tick,preflight=pre,environment=env,memory=gpu_memory(),hooks_clean=True,parameters_unchanged=True))
    print('Complete GPU',pre['physical_gpu'],dict(counts),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','cpu','freeze','smoke','worker']);parser.add_argument('--rank',type=int,default=0);args=parser.parse_args()
    try:
        if args.stage in ('smoke','worker'):globals()[args.stage](args.rank)
        else:globals()[args.stage]()
    except BaseException:
        atomic_json(OUT/f'failures/{args.stage}_{args.rank}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
