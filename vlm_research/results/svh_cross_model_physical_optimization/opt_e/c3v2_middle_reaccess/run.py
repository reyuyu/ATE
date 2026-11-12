"""Two resident models, paired image shards and per-request atomic checkpoints."""
import argparse
import gc
import os
import subprocess
import sys
import time
import traceback

import torch

from .common import *
from .engine import generate, prepare
from ..accuracy_autoresearch.engine import generate as inherited_generate
from ..accuracy_autoresearch.common import config
from ..ate_fulljoint_v2.engine import LAYERS
from ..fastp2_bypass_writer.common import hardware
from ..fastp2_bypass_writer.run import operators
from ..p1_layerwise_accuracy.engine import Adapter, tensor_hash
from ..physical_visual_handoff.engine import equivalence


def verify_freeze():
    f=read(OUT/'execution_freeze.json')
    assert f['sources']==sources(),'Inference source changed after freeze'
    for name in ('sample_manifest','partition','protocol'):
        assert f[name+'_sha256']==sha(OUT/(name+'.json'))
    return f


def check_stop():
    assert not list(OUT.glob('worker*/STOP.json')),'A worker failed; stop all task-owned GPU work'


def wait_for(name):
    start=time.time()
    while not all((OUT/f'worker{r}/{name}').exists() for r in range(2)):
        check_stop()
        assert time.time()-start<14400,'Peer stage timeout'
        time.sleep(1)


def compact(g):
    for row in g['kv']['layers']:
        positions=row.pop('original_positions',None)
        if positions is not None:
            row['original_positions_sha256']=digest(positions)
            row['original_positions_count']=len(positions)
    for row in g.get('visual_audit',[]):
        positions=row.pop('original_query_positions',[])
        row['original_query_positions_sha256']=digest(positions)
        row['query_count']=len(positions)
    return g


def smoke(a,rank,samples,p):
    rows=[]
    for key in p['smoke']:
        check_stop()
        s=samples[key]; layout,w,binding=prepare(a,s)
        native=generate(a,layout,'Native',3,fixed=True,capture='all')
        noop=generate(a,layout,'Native',3,fixed=True,capture='all',noop=True)
        assert len(native[2])==len(noop[2])==3
        assert all(torch.equal(x,y) for x,y in zip(native[2],noop[2]))
        inherited=inherited_generate(a,layout,config('native',head=1),w,3,True,False)
        assert torch.equal(native[1][0],inherited[1][0])
        assert native[0]['token_ids'][:len(inherited[0]['token_ids'])]==inherited[0]['token_ids']
        with torch.inference_mode():
            hf=a.model.generate(**a.inputs,max_new_tokens=32,do_sample=False,num_beams=1,use_cache=True,logits_to_keep=1)
        native_natural=generate(a,layout,'Native',32,capture='all')
        assert hf[0,layout.length:].tolist()==native_natural[0]['token_ids']
        first_native=tensor_hash(native[1][0]); del native,noop,inherited,hf,native_natural
        details=[]
        for label in LABELS[1:]:
            with operators(a) as (calls,mlp,shapes):
                actual=generate(a,layout,label,3,fixed=True,capture='all',audit=True)
            assert all(mlp[i]==3 for i in range(32))
            assert all(calls[i,op]==1 for i in LAYERS for op in ('q_proj','k_proj','v_proj','o_proj'))
            for row in shapes:
                i=row['layer']; n=layout.length if i<CUTOFF[label] else len(layout.text)
                if i in LAYERS and row['op'] in ('q_proj','o_proj'): n-=1
                assert row['shape'][1] in (1,n),(row,n)
            assert len(actual[3])==32 and len(actual[4])==3
            info=actual[0]['kv']
            assert info['retained_image_layers']==list(KEEP[label])
            assert info['historical_cache_copies']==info['append_concatenations']==0
            assert actual[0]['audit'] and all(r['blocked_probability_max']==0 for r in actual[0]['audit'])
            if label=='C3-V2':
                ref=inherited_generate(a,layout,CFG,w,3,True,True)
                assert all(torch.equal(x,y) for x,y in zip(actual[2][:len(ref[2])],ref[2]))
                assert actual[0]['token_ids'][:len(ref[0]['token_ids'])]==ref[0]['token_ids']
                ref_kind='Original frozen C3-V2, exact H1 returned logits'
                e=dict(passed=True,exact=True)
            else:
                ref=generate(a,layout,label,3,fixed=True,capture='all',audit=True,dense=True)
                e=equivalence('llava',ref[1],actual[1],ref[3],actual[3])
                assert e['passed'] and actual[0]['token_ids']==ref[0]['token_ids'],(key,label,e)
                assert len(actual[0]['visual_audit'])==84
                for event in actual[0]['visual_audit']:
                    assert event['answer_image_mass']>0 if event['layer'] in KEEP[label] else event['answer_image_mass']==0
                    if event['prefill'] and event['layer']>=20: assert event['relay_image_mass']==0
                ref_kind='Dense functional image-mask + zero-output reference; same original positions'
            for x,y in zip(actual[4],ref[4]):
                assert x.keys()==y.keys() and all(torch.equal(x[f],y[f]) for f in x)
            first_sha=tensor_hash(actual[1][0])
            reference_audit=dict(reference=ref_kind,equivalence=e,all_MLP_executed=True,
                projection_shapes=shapes,actual_KV=info,visual_probabilities=actual[0]['visual_audit'])
            compact(actual[0])
            del actual,ref
            natural=generate(a,layout,label,32,capture='all')
            if label=='C3-V2':
                old=load(HISTORY/'checkpoints'/key/'C3.json')
                assert old['config']==CFG and old['input_binding']==binding and old['sample_sha256']==digest(s)
                assert natural[0]['token_ids']==old['generation']['token_ids']
                assert first_sha==old['generation']['first_logits_sha256']
                natural_ref=inherited_generate(a,layout,CFG,w,32,True,False)
                assert all(torch.equal(x,y) for x,y in zip(natural[2],natural_ref[2]))
            else:
                natural_ref=generate(a,layout,label,32,capture='all',dense=True)
                e_natural=equivalence('llava',natural_ref[1],natural[1],[],[])
                assert e_natural['passed'] and natural[0]['token_ids']==natural_ref[0]['token_ids'],(key,label,e_natural)
                reference_audit['natural_sequence_equivalence']=e_natural
            reference_audit['natural_tokens']=natural[0]['token_ids']
            reference_audit['natural_reference_tokens']=natural_ref[0]['token_ids']
            details.append(dict(condition=label,first_logits_sha256=first_sha,**reference_audit))
            del natural,natural_ref
            a.verify()
        rows.append(dict(key=key,native_noop_strict=True,native_HF_tokens_exact=True,
            native_first_logits_sha256=first_native,conditions=details))
        print('SMOKE',rank,key,'PASS',flush=True)
    verify_freeze()
    atomic(OUT/f'worker{rank}/smoke.json',dict(status='PASS',at=now(),sources=sources(),checks=rows))


def quality(a,rank,samples,p):
    keys=sorted((k for k,s in samples.items() if p['image_shard'][s['image_sha256']]==rank),
        key=lambda k:(samples[k]['image_sha256'],k))
    start=time.time()
    for ix,key in enumerate(keys):
        check_stop(); s=samples[key]; layout,w,binding=prepare(a,s)
        for label in LABELS:
            path=quality_path(key,label)
            if path.exists():
                old=load(path)
                assert old['sample_sha256']==digest(s) and old['input_binding']==binding and old['sources']==sources()
                continue
            g,logits,_,_,_=generate(a,layout,label,32,capture='first')
            g['first_logits_sha256']=tensor_hash(logits[0])
            g['first_top1_token']=int(logits[0].argmax())
            g['first_top1_logp']=float(logits[0].log_softmax(-1).max())
            history=None
            if label=='C3-V2':
                path_old=HISTORY/'checkpoints'/key/'C3.json'; old=load(path_old)
                assert old['sample_sha256']==digest(s) and old['config']==CFG and old['input_binding']==binding
                assert old['generation']['token_ids']==g['token_ids'],'C3 historical tokens changed'
                assert old['generation']['first_logits_sha256']==g['first_logits_sha256'],'C3 H1 historical logits changed'
                history=dict(path=str(path_old),sha256=sha(path_old),exact_tokens=True,exact_first_logits=True)
            save(path,dict(key=key,condition=label,rank=rank,at=now(),sample_sha256=digest(s),
                input_binding=binding,sources=sources(),generation=compact(g),score=score(s,g['text']),history=history))
            del logits,g
            a.verify()
        if ix%10==0 or ix+1==len(keys):
            state=dict(at=now(),stage='quality',questions=ix+1,total=len(keys),elapsed_seconds=time.time()-start)
            atomic(OUT/f'worker{rank}/progress.json',state); print('PROGRESS',rank,state,flush=True)
    atomic(OUT/f'worker{rank}/quality_complete.json',dict(status='COMPLETE',at=now(),questions=len(keys)))


def performance(a,rank,samples,p):
    keys=[k for k in p['performance48'] if p['image_shard'][samples[k]['image_sha256']]==rank]
    start=time.time()
    for ix,key in enumerate(keys):
        check_stop(); s=samples[key]; layout,_,binding=prepare(a,s)
        for mode in ('natural_eos','fixed16'):
            for repeat in range(-3,5):
                offset=(ix+repeat+3)%4
                labels=LABELS[offset:]+LABELS[:offset]
                for label in labels:
                    path=performance_path(key,mode,repeat,label)
                    if path.exists():
                        old=load(path)
                        assert old['sources']==sources() and old['sample_sha256']==digest(s) and old['input_binding']==binding
                        continue
                    g,*_=generate(a,layout,label,16 if mode=='fixed16' else 32,fixed=mode=='fixed16')
                    if mode=='natural_eos':
                        expected=load(quality_path(key,label))['generation']['token_ids']
                        assert g['token_ids']==expected,'Performance and quality token sequences differ'
                    else: assert len(g['token_ids'])==16
                    save(path,dict(key=key,condition=label,rank=rank,at=now(),mode=mode,repeat=repeat,
                        warmup=repeat<0,sample_sha256=digest(s),sources=sources(),input_binding=binding,generation=compact(g)))
                    del g
            a.verify()
        state=dict(at=now(),stage='performance',questions=ix+1,total=len(keys),elapsed_seconds=time.time()-start)
        atomic(OUT/f'worker{rank}/progress.json',state); print('PERFORMANCE',rank,state,flush=True)
    atomic(OUT/f'worker{rank}/performance_complete.json',dict(status='COMPLETE',at=now(),questions=len(keys)))


def worker(rank):
    verify_freeze(); d=DEVICES[rank]
    assert os.environ['CUDA_VISIBLE_DEVICES']==d['uuid']
    hw=hardware(); assert d['uuid'] not in hw['processes'],'Selected GPU no longer idle'
    atomic(OUT/f'worker{rank}/preflight.json',dict(at=now(),pid=os.getpid(),**hw))
    a=None; start=time.time()
    try:
        samples=read(OUT/'sample_manifest.json'); p=read(OUT/'partition.json')
        a=Adapter('llava'); atomic(OUT/f'worker{rank}/environment.json',a.environment)
        if not (OUT/f'worker{rank}/smoke.json').exists(): smoke(a,rank,samples,p)
        else: assert read(OUT/f'worker{rank}/smoke.json')['sources']==sources()
        wait_for('smoke.json')
        left,right=[read(OUT/f'worker{i}/smoke.json')['checks'] for i in range(2)]
        for x,y in zip(left,right):
            assert x['key']==y['key'] and x['native_first_logits_sha256']==y['native_first_logits_sha256']
            assert [c['first_logits_sha256'] for c in x['conditions']]==[c['first_logits_sha256'] for c in y['conditions']]
            assert [c['natural_tokens'] for c in x['conditions']]==[c['natural_tokens'] for c in y['conditions']]
        quality(a,rank,samples,p)
        wait_for('quality_complete.json')
        performance(a,rank,samples,p)
        atomic(OUT/f'worker{rank}/completion.json',dict(status='COMPLETE',at=now(),elapsed_seconds=time.time()-start))
    except BaseException:
        atomic(OUT/f'worker{rank}/STOP.json',dict(at=now(),traceback=traceback.format_exc())); raise
    finally:
        del a; gc.collect(); torch.cuda.empty_cache()
        atomic(OUT/f'worker{rank}/release.json',dict(at=now(),pid=os.getpid(),allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),process_exit_releases_remaining=True))


def driver():
    verify_freeze(); h=hardware()
    assert all(d['uuid'] not in h['processes'] for d in DEVICES),'Selected devices must be idle'
    atomic(OUT/'driver_preflight.json',dict(at=now(),**h))
    processes=[]; streams=[]
    try:
        for rank,d in enumerate(DEVICES):
            path=OUT/f'worker{rank}/execution.log'; path.parent.mkdir(parents=True,exist_ok=True)
            stream=path.open('a'); streams.append(stream)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=d['uuid'],PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            processes.append(subprocess.Popen([sys.executable,'-u','-m','vlm_research.c3v2_middle_reaccess.run','--rank',str(rank)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT))
        atomic(OUT/'driver_pids.json',dict(at=now(),driver=os.getpid(),workers=[p.pid for p in processes]))
        codes=[p.wait() for p in processes]
        atomic(OUT/'driver_exit.json',dict(at=now(),returncodes=codes))
        assert codes==[0,0],codes
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            if p.poll() is None: p.wait(timeout=60)
        for stream in streams: stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--driver',action='store_true'); p.add_argument('--rank',type=int)
    args=p.parse_args()
    if args.driver: driver()
    else:
        assert args.rank in (0,1); worker(args.rank)
