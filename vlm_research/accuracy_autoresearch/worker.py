"""One resident model per authorized GPU; immutable, per-round JSON dispatch."""
import argparse
from contextlib import contextmanager
import dataclasses
import os
import time
import traceback

import torch

from .common import *
from .engine import generate, LAYERS
from ..p1_layerwise_accuracy.engine import Adapter, tensor_hash
from ..physical_handoff_fastpath.engine import Layout
from ..physical_visual_handoff.engine import equivalence
from ..fastp2_bypass_writer.run import operators, CompleteCensus
from ..ate_fulljoint_v2.run import observe_visual, compare_visual


def prepare(adapter, sample, cfg):
    s = dict(sample, prompt_text=prompt_text(sample,cfg['prompt_variant']))
    chat = adapter.processor.apply_chat_template([dict(role='user',content=[dict(type='image'),
        dict(type='text',text=s['prompt_text'])])],add_generation_prompt=True)
    expanded = chat.replace('<image>','<image>'*576)
    enc = adapter.processor.tokenizer(expanded,return_offsets_mapping=True)
    if cfg['prompt_variant']=='original':
        assert enc['input_ids'] == sample['input_mapping']['prompt_ids']
    s['input_mapping'] = dict(prompt_ids=enc['input_ids'])
    adapter.prepare(s,{})
    if 'historical_input_hashes' in sample and cfg['prompt_variant']=='original':
        assert adapter.input_hashes == sample['historical_input_hashes']
    layout = dataclasses.replace(Layout.from_adapter(adapter,s,{}),cutoff=cfg['cutoff'])
    start = expanded.rindex('ASSISTANT')
    assistant = [i for i,(a,b) in enumerate(enc['offset_mapping']) if b>start and a<start+9]
    assert len(assistant)==4 and [enc['input_ids'][i] for i in assistant]==[319,1799,9047,13566]
    if cfg['writer'] in ('all','native'):
        writers = list(layout.relay)
    elif cfg['writer']=='qsa':
        writers = sorted(set(layout.question)|set(assistant))
    elif cfg['writer']=='conditional15':
        writers = conditional_writers(s['question'],layout.question,assistant)[0]
    else:
        raise ValueError(cfg['writer'])
    assert set(writers)<=set(layout.relay) and layout.length-1 not in writers
    return layout,writers,dict(prompt_ids=enc['input_ids'],input_hashes=adapter.input_hashes,
        Q=list(layout.question),ASSISTANT=assistant,R=list(layout.relay),writers=writers,
        prompt_text=s['prompt_text'])


def path_for(cfg,k):
    return OUT/'records'/config_id(cfg)/(k+'.json')


def compact(value):
    value.pop('process_memory_snapshot',None)
    value['kv']={k:v for k,v in value['kv'].items() if k!='layers'}
    return value


def read_record(cfg,k,sample):
    path = path_for(cfg,k)
    if not path.exists():
        return None
    blob=read(path); row=blob['payload']
    assert digest(row)==blob['sha256'] and row['config']==cfg
    assert row['sample_sha256']==digest(sample) and row['sources']==source_hashes()
    return row


def smoke(adapter,rank,pool,split):
    chosen = [k for k in split['coarse'] if split['image_shard'][pool[k]['image_sha256']]==rank]
    chosen = [next(k for k in chosen if pool[k]['dataset']=='GQA')]+[
        k for k in chosen if pool[k]['dataset']=='VQAv2'][:2]
    audits=[]
    for ix,k in enumerate(chosen):
        ensure_time(True)
        s=pool[k]
        for head in (0,1):
            c=config('native',head=head)
            layout,writers,spec=prepare(adapter,s,c)
            full=generate(adapter,layout,c,writers,2,True,True)
            noop=generate(adapter,layout,c,writers,2,True,True,True)
            assert len(full[2])==len(noop[2])==2
            assert all(torch.equal(x,y) for x,y in zip(full[2],noop[2]))
            with torch.inference_mode():
                official=adapter.model.generate(**adapter.inputs,max_new_tokens=2,do_sample=False,
                                               use_cache=True,logits_to_keep=head)
            assert official[0,layout.length:].tolist()==full[0]['token_ids']
            del full,noop,official
        # Same implementation, varied physical handoff and answer-mode dependencies.
        c=config('qsa',('both','decode','none')[ix],(20,24,28)[ix])
        layout,writers,spec=prepare(adapter,s,c)
        baseline=dict(c,answer_attention='none')
        with observe_visual(adapter,layout) as baseline_visual:
            b=generate(adapter,layout,baseline,writers,2,True,True)
        with observe_visual(adapter,layout) as visual, operators(adapter) as (calls,mlp,shapes):
            actual=generate(adapter,layout,c,writers,2,True,True)
        with observe_visual(adapter,layout) as ref_visual:
            expected=generate(adapter,layout,c,writers,2,True,True,True)
        e=equivalence('llava',expected[1],actual[1],expected[3],actual[3])
        assert e['passed'] and actual[0]['token_ids']==expected[0]['token_ids'],e
        if c['answer_attention']=='both':
            preservation=compare_visual(k,config_id(c),layout,visual,ref_visual,baseline_visual)
        else:
            preservation=[]
            assert set(visual)==set(ref_visual)==set(baseline_visual)
            for key in visual:
                assert torch.equal(visual[key],ref_visual[key]) and torch.equal(visual[key],baseline_visual[key])
                preservation.append(dict(sample=k,layer=key[0],tensor=key[1],passed=True,
                    whole_prefill_including_Pminus1_exact=True))
        assert all(mlp[i]==2 for i in range(32))
        for i in LAYERS:
            for op in ('q_proj','k_proj','v_proj','o_proj'):
                assert calls[i,op]==(2 if c['answer_attention']=='none' else 1)
        for row in shapes:
            if row['layer'] in LAYERS and c['answer_attention']=='both':
                assert row['shape'][1]==layout.length-(row['op'] in ('q_proj','o_proj'))
        kv=actual[0]['kv']
        assert kv['historical_cache_copies']==kv['append_concatenations']==0
        for row in kv['layers']:
            expected_length=0 if row['layer'] in LAYERS and c['answer_attention']!='none' else len(layout.text)+1
            assert row['used_tokens']==expected_length
            if row['used_tokens']:
                assert set(row['original_positions']).isdisjoint(layout.image)
        record=dict(sample=k,config=c,numerical=e,preservation=preservation,
                    operator_shapes=shapes,MLP=dict(mlp),events=actual[0]['audit'],KV=kv,
                    native_noop_exact_both_head_shapes=True)
        audits.append(record)
        adapter.verify()
        print('SMOKE',rank,ix+1,3,k,'PASS',flush=True)
        del actual,expected,b,visual,ref_visual,baseline_visual
    # Native shape-1 head and original shape-0 head are separate bindings, not assumed equal.
    atomic(OUT/f'worker{rank}/smoke.json',dict(status='PASS',sources=source_hashes(),audits=audits))


def generate_one(adapter,k,s,cfg,job):
    ensure_time(True)
    layout,writers,spec=prepare(adapter,s,cfg)
    result,logits,_,_,_=generate(adapter,layout,cfg,writers,capture=True)
    native_cfg=dict(cfg,writer='native',answer_attention='none',cutoff=20)
    native = None if cfg['writer']=='native' else read_record(native_cfg,k,s)
    assert cfg['writer']=='native' or native is not None,'Native must be evaluated first'
    target=int(logits[0].argmax()) if native is None else native['generation']['token_ids'][0]
    result['native_first_token_logp']=float(logits[0].log_softmax(-1)[0,target])
    result['first_logits_sha256']=tensor_hash(logits[0])
    result['score']=score(s,result['text'])
    result['repeated']=any(result['token_ids'][i:i+3]==result['token_ids'][j:j+3]
        for i in range(len(result['token_ids'])-2) for j in range(i+3,len(result['token_ids'])-2))
    result=compact(result)
    row=dict(sample=k,sample_sha256=digest(s),config=cfg,input_binding=spec,
             generation=result,sources=source_hashes(),first_job=job['id'],at=now())
    path=path_for(cfg,k)
    assert not path.exists()
    atomic(path,dict(payload=row,sha256=digest(row)))
    adapter.verify()
    return row


def performance(adapter,k,s,configs,job):
    result=[]
    prepared={}
    for c in configs:
        layout,writers,spec=prepare(adapter,s,c)
        prepared[config_id(c)]=(layout,writers,spec)
        for _ in range(3):
            ensure_time(True)
            generate(adapter,layout,c,writers)
    for rep in range(5):
        shift=rep%len(configs)
        for c in configs[shift:]+configs[:shift]:
            ensure_time(True)
            layout,writers,spec=prepare(adapter,s,c)
            v=generate(adapter,layout,c,writers)[0]
            assert v['token_ids']==read_record(c,k,s)['generation']['token_ids']
            result.append(dict(sample=k,config_id=config_id(c),rep=rep,**compact(v)))
    shapes=[]
    for c in configs:
        layout,writers,spec=prepare(adapter,s,c)
        census=CompleteCensus(adapter)
        with census.context():
            v=generate(adapter,layout,c,writers)[0]
        assert v['token_ids']==read_record(c,k,s)['generation']['token_ids']
        shapes.append(dict(config_id=config_id(c),rows=census.rows(),n_prompt=layout.length,
                           n_text=len(layout.text),KV=v['kv']))
    atomic(OUT/'performance'/job['id']/(k+'.json'),dict(sample=k,sample_sha256=digest(s),
        configs=configs,measurements=result,flops=shapes,sources=source_hashes(),at=now()))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,required=True); args=ap.parse_args()
    rank=args.rank
    p=read(OUT/'execution_freeze.json')
    assert source_hashes()==p['sources']
    assert os.environ['CUDA_VISIBLE_DEVICES']==p['devices'][rank]['uuid']
    state=hardware(); assert p['devices'][rank]['uuid'] not in state['processes']
    pool,split=read(OUT/'sample_manifest.json'),read(OUT/'split_manifest.json')
    adapter=None
    try:
        adapter=Adapter('llava')
        atomic(OUT/f'worker{rank}/environment.json',adapter.environment)
        smoke(adapter,rank,pool,split)
        while not all((OUT/f'worker{i}/smoke.json').exists() for i in range(len(p['devices']))):
            ensure_time(True)
            assert not any((OUT/f'worker{i}/STOP.json').exists() for i in range(len(p['devices']))), 'Another worker failed Smoke'
            time.sleep(1)
        while time.time()<deadline('stop_new_gpu_work_by') and not (OUT/'SHUTDOWN.json').exists():
            jobs=sorted((OUT/'jobs').glob('*.json'))
            for path in jobs:
                ensure_time(True)
                job=read(path)
                assert job['sources']==p['sources'] and job['samples_sha256']==sha(OUT/'sample_manifest.json')
                done=OUT/f'worker{rank}/jobs'/(job['id']+'.json')
                if done.exists():
                    assert read(done)['job_sha256']==digest(job)
                    continue
                if job['split']=='FINAL':
                    frozen=read(OUT/'final_candidate_freeze.json')
                    assert job['final_freeze_sha256']==digest(frozen)
                    assert set(map(config_id,job['configs']))<=set(map(config_id,frozen['evaluation_configs']))
                keys=[k for k in job['samples'] if split['image_shard'][pool[k]['image_sha256']]==rank]
                started=time.time()
                for ix,k in enumerate(keys):
                    assert split['image_split'][pool[k]['image_sha256']]==job['split']
                    for c in job['configs']:
                        ensure_time(True)
                        if read_record(c,k,pool[k]) is None:
                            generate_one(adapter,k,pool[k],c,job)
                    if job.get('performance') and not (OUT/'performance'/job['id']/(k+'.json')).exists():
                        performance(adapter,k,pool[k],job['configs'],job)
                    if (ix+1)%10==0 or ix+1==len(keys):
                        print('PROGRESS',rank,job['id'],ix+1,len(keys),'seconds',round(time.time()-started,1),flush=True)
                atomic(done,dict(status='COMPLETE',job_sha256=digest(job),samples=len(keys),elapsed_seconds=time.time()-started,at=now()))
            time.sleep(1)
    except BaseException:
        atomic(OUT/f'worker{rank}/STOP.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
    finally:
        del adapter
        torch.cuda.empty_cache()
        atomic(OUT/f'worker{rank}/released.json',dict(at=now(),pid=os.getpid(),reason='shutdown or stop',no_external_process_terminated=True))


if __name__=='__main__':
    main()
