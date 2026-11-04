"""One independent model per image-cluster shard; no concurrent editing of execution code."""
import argparse
import collections
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import os
import time
import traceback
from unittest.mock import patch

import torch

from .common import OUT, ROOT, atomic, read, digest, key, verify, score, hardware, selection
from .engine import generate, CONDITIONS, LAYERS
from ..physical_handoff_fastpath import engine as frozen
from ..physical_handoff_fastpath.ledger import Census
from ..p1_layerwise_accuracy.engine import Adapter, tensor_hash, footprint
from ..physical_visual_handoff.engine import equivalence
from ..physical_handoff_fastpath.engine import Layout


class CompleteCensus(Census):
    @contextmanager
    def context(self):
        original = torch.bmm
        def bmm(a,b,*args,**kw):
            out = original(a,b,*args,**kw)
            self.add('attention_qk_av_bmm',2*out.numel()*a.shape[-1],
                     (tuple(a.shape),tuple(b.shape),tuple(out.shape)))
            return out
        with super().context(), patch.object(torch,'bmm',bmm):
            yield self


def prepare(adapter, sample, spec):
    adapter.prepare(sample,{})
    assert adapter.input_hashes == spec['input_hashes']
    layout = Layout.from_adapter(adapter,sample,{})
    assert layout.cutoff == 20 and layout.length-1 not in layout.relay
    return layout


@contextmanager
def operators(adapter):
    calls, mlp, rows = collections.Counter(), collections.Counter(), []
    with ExitStack() as stack:
        for i,l in enumerate(adapter.layers):
            for label in ('q_proj','k_proj','v_proj','o_proj'):
                def hook(m,a,out,i=i,label=label):
                    calls[i,label]+=1
                    rows.append(dict(layer=i,op=label,shape=list(a[0].shape)))
                handle=getattr(l.self_attn,label).register_forward_hook(hook)
                stack.callback(handle.remove)
            def hook(m,a,out,i=i): mlp[i]+=1
            handle=l.mlp.register_forward_hook(hook)
            stack.callback(handle.remove)
        yield calls,mlp,rows


def smoke(adapter,rank,p):
    samples=p['smoke']
    specs=read(OUT/'input_specs.json')
    pool=read(OUT/'sample_manifest.json')
    # Add one frozen OCR transfer input, without appending OCR tokens.
    samples=samples+[next(s for s in pool.values() if s['dataset']=='textvqa300')]
    audits=[]
    for ix,s in enumerate(samples):
        if 'dataset' in s:
            spec=specs[s['dataset']+'/'+key(s)]
        else:
            adapter.prepare(s,{})
            spec=dict(input_hashes=adapter.input_hashes,
                      selection=selection(adapter.processor,s,adapter.inputs['input_ids'][0].tolist()))
        layout=prepare(adapter,s,spec)
        native=frozen.generate(adapter,layout,'Native Full',2,True,True,True)
        noop=frozen.generate(adapter,layout,'Native-Noop',2,True,True,True)
        assert len(native[2])==len(noop[2])==2
        assert all(torch.equal(x,y) for x,y in zip(native[2],noop[2]))
        del native,noop
        base=generate(adapter,layout,'A0',spec['selection'],2,True,True,True)
        old=frozen.generate(adapter,layout,'Fast-P2',2,True,True,True)
        assert all(torch.equal(x,y) for x,y in zip(base[2],old[2]))
        assert base[0]['token_ids']==old[0]['token_ids']
        del base,old
        for physical,reference in (('A1','A3'),('A2','A2-Reference'),('B3','B1')):
            with operators(adapter) as (calls,mlp,op_rows):
                actual=generate(adapter,layout,physical,spec['selection'],2,True,True,True)
            expected=generate(adapter,layout,reference,spec['selection'],2,True,True,True)
            # Native precision shape changes are checked against unchanged historical bounds.
            e=equivalence('llava',expected[1][:1],actual[1][:1],expected[3],actual[3])
            e.update(sample=key(s),physical=physical,reference=reference,
                     exact_first=torch.equal(expected[1][0],actual[1][0]),
                     same_tokens=actual[0]['token_ids']==expected[0]['token_ids'],
                     operator_rows=op_rows,MLP_calls=dict(mlp),audit=actual[0]['audit'])
            assert all(mlp[i]==2 for i in range(32))
            if physical=='A1':
                assert all(calls[i,op]==0 for i in LAYERS for op in ('q_proj','k_proj','v_proj','o_proj'))
                assert e['exact_first'] and e['same_tokens']
            if physical=='A2':
                assert all(calls[i,op]==1 for i in LAYERS for op in ('q_proj','k_proj','v_proj','o_proj'))
                for r in op_rows:
                    if r['layer'] in LAYERS and r['op'] in ('q_proj','o_proj'):
                        assert r['shape'][1]==layout.length-577
            audits.append(e)
            atomic(OUT/f'shard{rank}/smoke_partial.json',audits)
            assert e['passed'], f'Physical numerical gate failed: {physical}, {key(s)}'
            del actual,expected
        # Functional writer controls must also audit the actual masked probabilities.
        for name in ('B1','B2-20261103','B2-20261107','B2-20261111'):
            result=generate(adapter,layout,name,spec['selection'],2,True,True,True)
            assert result[0]['audit']
            del result
        adapter.verify()
        print('SMOKE',rank,ix+1,len(samples),key(s),'PASS',flush=True)
    atomic(OUT/f'shard{rank}/smoke.json',dict(status='PASS',numerical=audits,
        exact_native_noop=True,exact_frozen_A0=True,all_MLP_native=True,
        no_target_QKVO_A1=True,partial_queries_A2=True,source_probability_zero=True,
        hooks_clean=footprint(adapter.model)==adapter.hooks,protocol_sha256=digest(p)))


def scalar(value):
    return {k:v for k,v in value.items() if k not in ('audit','kv_management_ms','step_ms','process_memory_snapshot')}


def sealed_read(path,p,spec):
    value=read(path)
    payload=value['payload']
    assert digest(payload)==value['sha256']
    assert payload['protocol_sha256']==digest(p) and payload['input_spec_sha256']==digest(spec)
    return payload


def save(path,payload):
    assert not path.exists(), 'Checkpoint overwrite prohibited'
    atomic(path,dict(payload=payload,sha256=digest(payload)))


def execute(adapter,rank,p):
    assert all(read(OUT/f'shard{r}/smoke.json')['status']=='PASS' for r in range(len(p['devices'])))
    samples=read(OUT/'sample_manifest.json')
    specs=read(OUT/'input_specs.json')
    selected=[(k,s) for k,s in sorted(samples.items()) if p['assignment'][s['image_sha256']]==rank]
    started=time.time()
    for index,(sample_key,s) in enumerate(selected):
        spec=specs[sample_key]
        path=OUT/f'shard{rank}/checkpoints'/sample_key
        path=path.with_suffix('.json')
        if path.exists():
            saved=sealed_read(path,p,spec)
            assert saved['sample_sha256']==digest(s)
            continue
        layout=prepare(adapter,s,spec)
        results={}
        for name in CONDITIONS:
            # Quality and shape census are excluded from benchmark timing.
            census=CompleteCensus(adapter)
            with census.context():
                value,logits,_,_,_=generate(adapter,layout,name,spec['selection'],32,False,True)
            value['score']=score(s,value['text'])
            value['first_logits_sha256']=tensor_hash(logits[0])
            base_token=int(logits[0].argmax()) if name=='A0' else results['A0']['quality']['token_ids'][0]
            value['full_first_token_logp']=float(logits[0].log_softmax(-1)[0,base_token])
            value['repeated']=any(value['token_ids'][i:i+3]==value['token_ids'][j:j+3]
                for i in range(len(value['token_ids'])-2) for j in range(i+3,len(value['token_ids'])-2))
            results[name]=dict(quality=scalar(value),flops=census.rows(),measurements=[])
            del logits
        if s['dataset']=='pilot60':
            for name in CONDITIONS:
                for _ in range(p['warmups']): generate(adapter,layout,name,spec['selection'])
            for rep in range(p['repetitions']):
                shift=(rep+index)%len(CONDITIONS)
                for name in CONDITIONS[shift:]+CONDITIONS[:shift]:
                    value=generate(adapter,layout,name,spec['selection'])[0]
                    assert value['token_ids']==results[name]['quality']['token_ids'],'Nondeterministic greedy sequence'
                    results[name]['measurements'].append(scalar(value))
        adapter.verify()
        save(path,dict(protocol_sha256=digest(p),input_spec_sha256=digest(spec),sample_sha256=digest(s),
                       sample=sample_key,layout=asdict(layout),conditions=results))
        print('PROGRESS',rank,index+1,len(selected),sample_key,'seconds',round(time.time()-started,1),flush=True)
    fixed(adapter,rank,p)
    profile(adapter,rank,p)
    atomic(OUT/f'shard{rank}/completion.json',dict(status='COMPLETE',samples=len(selected),
        elapsed_seconds=time.time()-started,protocol_sha256=digest(p)))


def fixed(adapter,rank,p):
    pool=read(OUT/'sample_manifest.json')
    specs=read(OUT/'input_specs.json')
    selected=[(k,s) for k,s in sorted(pool.items()) if s['dataset']=='pilot60' and p['assignment'][s['image_sha256']]==rank]
    for sample_key,sample in selected:
        spec=specs[sample_key]
        layout=prepare(adapter,sample,spec)
        for length in p['fixed_predictions']:
            path=OUT/f'shard{rank}/fixed{length}'/(key(sample)+'.json')
            if path.exists():
                sealed_read(path,p,spec)
                continue
            results={}
            for name in CONDITIONS:
                census=CompleteCensus(adapter)
                with census.context():
                    value=generate(adapter,layout,name,spec['selection'],length,True)[0]
                results[name]=dict(quality=scalar(value),flops=census.rows(),measurements=[])
                for _ in range(5): generate(adapter,layout,name,spec['selection'],length,True)
            for rep in range(20):
                shift=rep%len(CONDITIONS)
                for name in CONDITIONS[shift:]+CONDITIONS[:shift]:
                    v=generate(adapter,layout,name,spec['selection'],length,True)[0]
                    assert v['token_ids']==results[name]['quality']['token_ids']
                    results[name]['measurements'].append(scalar(v))
            save(path,dict(protocol_sha256=digest(p),input_spec_sha256=digest(spec),sample=key(sample),
                           length=length,conditions=results))
            print('FIXED',rank,length,key(sample),'COMPLETE',flush=True)


def profile(adapter,rank,p):
    if rank!=0: return
    s=p['smoke'][0]
    adapter.prepare(s,{})
    spec=dict(input_hashes=adapter.input_hashes,selection=selection(adapter.processor,s,adapter.inputs['input_ids'][0].tolist()))
    layout=prepare(adapter,s,spec)
    for name in CONDITIONS:
        path=OUT/f'shard{rank}/profiler/{name}.json'
        if path.exists():
            sealed_read(path,p,spec)
            continue
        generate(adapter,layout,name,spec['selection'],16,True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True,profile_memory=True) as prof:
            result=generate(adapter,layout,name,spec['selection'],16,True,profile=True)[0]
        kernels=collections.defaultdict(lambda:dict(count=0,device_us=0.))
        for e in prof.events():
            if str(e.device_type).endswith('CUDA'):
                kernels[e.name]['count']+=1
                kernels[e.name]['device_us']+=e.device_time_total
        events=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                     self_device_us=e.self_device_time_total) for e in prof.key_averages()]
        save(path,dict(protocol_sha256=digest(p),input_spec_sha256=digest(spec),condition=name,
                       sample=key(s),events=events,cuda_activities=kernels,kv_management_ms=result['kv_management_ms']))
        print('PROFILE',name,'COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['smoke','run'])
    parser.add_argument('--rank',type=int,required=True)
    args=parser.parse_args()
    p=verify()
    gpu=p['devices'][args.rank]
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu['uuid']
    state=hardware()
    assert gpu['uuid'] not in state['processes'], 'GPU acquired by another task; stop'
    atomic(OUT/f'shard{args.rank}/{args.action}_preflight.json',state)
    a=None
    try:
        a=Adapter('llava')
        atomic(OUT/f'shard{args.rank}/environment.json',a.environment)
        if args.action=='smoke': smoke(a,args.rank,p)
        else: execute(a,args.rank,p)
    except BaseException:
        atomic(OUT/f'shard{args.rank}/{args.action}_failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        del a
        torch.cuda.empty_cache()


if __name__=='__main__': main()
