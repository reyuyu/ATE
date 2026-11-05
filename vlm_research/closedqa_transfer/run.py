"""Fixed two-shard transfer evaluation, with no adaptive dispatch or tuning."""
import argparse
import copy
import dataclasses
import os
import time
import traceback

import torch

from .common import *
from .engine import prepare, generate
from ..accuracy_autoresearch.engine import generate as inherited_generate
from ..ate_fulljoint_v2.engine import LAYERS
from ..ate_fulljoint_v2.run import observe_visual, compare_visual
from ..fastp2_bypass_writer.run import operators, CompleteCensus
from ..p1_layerwise_accuracy.engine import Adapter, tensor_hash
from ..physical_visual_handoff.engine import equivalence


def save(path,payload):
    assert not path.exists(),f'Checkpoint overwrite: {path}'
    atomic(path,dict(payload=payload,sha256=digest(payload)))


def compact(g):
    g.pop('process_memory_snapshot',None)
    g['kv']={k:v for k,v in g['kv'].items() if k!='layers'}
    return g


def check_stop():
    assert not list(OUT.glob('worker*/STOP.json')),'A worker failed; preserving checkpoints'


def smoke(a,rank,samples,partition):
    keys=[k for group in partition['smoke'].values() for k in group
          if partition['image_shard'][samples[k]['image_sha256']]==rank]
    checks=[]
    for k in keys:
        check_stop()
        s=samples[k]; l,w,spec=prepare(a,s,'Native Full')
        native=generate(a,l,'Native Full',w,2,True,True)
        noop=generate(a,l,'Native Full',w,2,True,True,True)
        assert len(native[2])==len(noop[2])==2,'Smoke must exercise real Decode'
        assert all(torch.equal(x,y) for x,y in zip(native[2],noop[2]))
        captured=[]
        h=a.model.register_forward_hook(lambda m,args,out:captured.append(out.logits.detach().cpu().clone()))
        try:
            with torch.inference_mode():
                official=a.model.generate(**a.inputs,max_new_tokens=2,do_sample=False,num_beams=1,
                                          use_cache=True,logits_to_keep=1)
        finally:
            h.remove()
        assert official[0,l.length:].tolist()==native[0]['token_ids']
        assert len(captured)==2 and all(torch.equal(x,y) for x,y in zip(captured,native[2]))
        detail=[]
        del noop,native,official,captured
        for label in list(CONFIGS)[1:]:
            l,w,spec=prepare(a,s,label)
            baseline_config=dict(CONFIGS[label],answer_attention='none')
            with observe_visual(a,l) as before:
                b=inherited_generate(a,l,baseline_config,w,2,True,True)
            with observe_visual(a,l) as actual_visual,operators(a) as (calls,mlp,shapes):
                actual=generate(a,l,label,w,2,True,True)
            with observe_visual(a,l) as reference_visual:
                expected=generate(a,l,label,w,2,True,True,True)
            e=equivalence('llava',expected[1],actual[1],expected[3],actual[3])
            e.update(same_tokens=actual[0]['token_ids']==expected[0]['token_ids'])
            assert e['passed'] and e['same_tokens'],(k,label,e)
            assert len(actual[0]['token_ids'])==2
            assert all(mlp[i]==2 for i in range(32))
            mode=CONFIGS[label]['answer_attention']
            if mode=='both':
                preservation=compare_visual(k,label,l,actual_visual,reference_visual,before)
            else:
                preservation=[]
                assert set(actual_visual)==set(reference_visual)==set(before)
                for index,v in actual_visual.items():
                    assert torch.equal(v,reference_visual[index]) and torch.equal(v,before[index]),index
            for row in shapes:
                i=row['layer']; expected_rows=l.length if i<20 else len(l.text)
                if mode=='both' and i in LAYERS and row['op'] in ('q_proj','o_proj'):
                    expected_rows-=1
                assert row['shape'][1] in (1,expected_rows),(label,row,expected_rows)
            if mode!='none':
                assert all(calls[i,op]==1 for i in LAYERS for op in ('q_proj','k_proj','v_proj','o_proj'))
            events=actual[0]['audit']; assert events
            assert all(ev['blocked_probability_max']==0 for ev in events)
            kv=actual[0]['kv']
            assert kv['historical_cache_copies']==kv['append_concatenations']==0
            for row in kv['layers']:
                assert row['used_tokens']==(0 if mode!='none' and row['layer'] in LAYERS else len(l.text)+1)
                if row['used_tokens']:
                    assert row.get('original_positions',list(l.text)+[l.length])==list(l.text)+[l.length]
            # The dense reference keeps original visual rows and masks their inaccessible edges.
            assert all(row['used_tokens']==l.length+1 for row in expected[0]['kv']['layers'])
            assert len(actual[4])==len(expected[4])==2
            for j,(x,y) in enumerate(zip(actual[4],expected[4])):
                assert x.keys()==y.keys()
                for field in x:
                    assert torch.equal(x[field],y[field]),('Original RoPE mismatch',j,field)
            detail.append(dict(condition=label,numerical=e,preservation=preservation,
                operators=shapes,MLP_calls=dict(mlp),events=events,KV=kv,
                non_target_prefill_preserved=True,full_dense_reference=True))
            a.verify()
            del actual,expected,b,actual_visual,reference_visual,before
        # Every call has a fresh request cache, and returning to Native must be exact.
        first=generate(a,l,'Native Full',w,2,True)
        second=generate(a,l,'Native Full',w,2,True)
        assert all(torch.equal(x,y) for x,y in zip(first[1],second[1]))
        del first,second
        a.verify()
        row=dict(sample=k,input_binding=spec,native_noop_all_valid_logits_exact=True,
                 HF_generate_all_valid_logits_exact=True,cache_isolation=True,details=detail)
        checks.append(row)
        atomic(OUT/f'worker{rank}/smoke_partial.json',checks)
        print('SMOKE',rank,k,'PASS',flush=True)
    atomic(OUT/f'worker{rank}/smoke.json',dict(status='PASS',sources=source_hashes(),checks=checks))


def quality(a,rank,samples,partition):
    own=[k for k,s in samples.items() if partition['image_shard'][s['image_sha256']]==rank]
    # Canonical inputs first; aliases never cross image shards.
    keys=sorted(own,key=lambda k:(partition['canonical_input'][k]!=k,k))
    started=time.time(); reused=0; generated=0
    for ix,k in enumerate(keys):
        check_stop(); s=samples[k]; canonical=partition['canonical_input'][k]
        for label in CONFIGS:
            if record(k,label,s) is not None: continue
            l,w,spec=prepare(a,s,label)
            spec_path=OUT/'input_specs'/(k+'.json')
            if spec_path.exists(): assert read(spec_path)['input_hashes']==spec['input_hashes']
            else: atomic(spec_path,spec)
            source=record(canonical,label,samples[canonical]) if canonical!=k else None
            if canonical!=k:
                assert source is not None
                assert source['input_binding']==spec
                g=copy.deepcopy(source['generation']); reused+=1
            else:
                g,logits,_,_,_=generate(a,l,label,w,capture=True)
                baseline=None if label=='Native Full' else record(k,'Native Full',s)
                target=int(logits[0].argmax()) if baseline is None else baseline['generation']['token_ids'][0]
                g['native_first_token_logp']=float(logits[0].log_softmax(-1)[0,target])
                g['first_logits_sha256']=tensor_hash(logits[0])
                g['repeated']=any(g['token_ids'][i:i+3]==g['token_ids'][j:j+3]
                    for i in range(len(g['token_ids'])-2) for j in range(i+3,len(g['token_ids'])-2))
                g=compact(g); generated+=1; del logits
            row=dict(sample=k,sample_sha256=digest(s),config=CONFIGS[label],condition=label,
                generation=g,scoring=scored(s,g['text']),input_binding=spec,sources=source_hashes(),
                reuse_from=canonical if canonical!=k else None,
                reuse_checkpoint_sha256=sha(checkpoint(canonical,label)) if canonical!=k else None,
                at=now(),rank=rank)
            save(checkpoint(k,label),row); a.verify()
        if (ix+1)%10==0 or ix+1==len(keys):
            status=dict(at=now(),questions=ix+1,total=len(keys),generated=generated,reused=reused,
                        elapsed_seconds=time.time()-started)
            atomic(OUT/f'worker{rank}/quality_progress.json',status)
            print('QUALITY',rank,status,flush=True)
    atomic(OUT/f'worker{rank}/quality_complete.json',dict(status='COMPLETE',questions=len(keys),
        generated=generated,reused=reused,elapsed_seconds=time.time()-started))


def performance(a,rank,samples,partition):
    keys=[k for group in partition['performance'].values() for k in group
          if partition['image_shard'][samples[k]['image_sha256']]==rank]
    for ix,k in enumerate(keys):
        path=OUT/'performance'/(k+'.json'); s=samples[k]
        if path.exists():
            b=read(path); assert b['sha256']==digest(b['payload'])
            assert b['payload']['sample_sha256']==digest(s) and b['payload']['sources']==source_hashes()
            continue
        check_stop(); measurements=[]; census_rows=[]
        for label in CONFIGS:
            l,w,_=prepare(a,s,label)
            for _ in range(3):
                g=generate(a,l,label,w)[0]
                assert g['token_ids']==record(k,label,s)['generation']['token_ids']
        labels=list(CONFIGS)
        for rep in range(5):
            shift=(rep+ix)%5
            for label in labels[shift:]+labels[:shift]:
                check_stop(); l,w,_=prepare(a,s,label)
                g=generate(a,l,label,w)[0]
                assert g['token_ids']==record(k,label,s)['generation']['token_ids'],'Nondeterministic generation'
                measurements.append(dict(condition=label,rep=rep,**compact(g)))
        for label in CONFIGS:
            l,w,_=prepare(a,s,label); census=CompleteCensus(a)
            with census.context(): g=generate(a,l,label,w)[0]
            assert g['token_ids']==record(k,label,s)['generation']['token_ids']
            census_rows.append(dict(condition=label,rows=census.rows(),KV=g['kv'],
                                    n_prompt=l.length,n_text=len(l.text)))
        save(path,dict(sample=k,sample_sha256=digest(s),sources=source_hashes(),rank=rank,
                       measurements=measurements,flops=census_rows,at=now()))
        a.verify(); print('PERFORMANCE',rank,ix+1,len(keys),k,flush=True)
    atomic(OUT/f'worker{rank}/performance_complete.json',dict(status='COMPLETE',questions=len(keys),
        timed_requests=len(keys)*25,warmup_requests=len(keys)*15,census_requests=len(keys)*5))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--rank',type=int); args=parser.parse_args()
    if args.freeze: freeze(); return
    rank=args.rank; assert rank in (0,1)
    f=read(OUT/'execution_freeze.json'); assert f['sources']==source_hashes()
    assert sha(OUT/'sample_manifest.json')==f['sample_manifest_sha256']
    assert sha(OUT/'partition.json')==f['partition_sha256']
    assert sha(OUT/'protocol.json')==f['protocol_sha256']
    device=f['devices'][rank]; assert os.environ['CUDA_VISIBLE_DEVICES']==device['uuid']
    state=hardware(); assert device['uuid'] not in state['processes'],'Authorized GPU no longer idle'
    assert any(line.startswith(str(device['index'])+', '+device['uuid']) for line in state['gpus'].splitlines())
    atomic(OUT/f'worker{rank}/preflight.json',dict(at=now(),pid=os.getpid(),**state))
    samples,partition=read(OUT/'sample_manifest.json'),read(OUT/'partition.json')
    a=None; started=time.time()
    try:
        a=Adapter('llava'); atomic(OUT/f'worker{rank}/environment.json',a.environment)
        if not (OUT/f'worker{rank}/smoke.json').exists(): smoke(a,rank,samples,partition)
        else: assert read(OUT/f'worker{rank}/smoke.json')['sources']==source_hashes()
        barrier=time.time()
        while not all((OUT/f'worker{i}/smoke.json').exists() for i in (0,1)):
            check_stop(); assert time.time()-barrier<900,'Smoke barrier timed out'; time.sleep(1)
        quality(a,rank,samples,partition)
        performance(a,rank,samples,partition)
        a.verify()
        atomic(OUT/f'worker{rank}/completion.json',dict(status='COMPLETE',at=now(),
               elapsed_seconds=time.time()-started,sources=source_hashes()))
    except BaseException:
        atomic(OUT/f'worker{rank}/STOP.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
    finally:
        del a
        import gc
        gc.collect(); torch.cuda.empty_cache()
        atomic(OUT/f'worker{rank}/release.json',dict(at=now(),pid=os.getpid(),allocated=torch.cuda.memory_allocated(),
               reserved=torch.cuda.memory_reserved(),process_exit_releases_remaining=True))


if __name__=='__main__': main()
