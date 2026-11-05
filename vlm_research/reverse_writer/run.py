"""Two resident, image-disjoint workers with immutable inputs and atomic results."""
import argparse
import copy
import gc
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager

import torch

from .common import *
from .engine import prepare, generate
from ..accuracy_autoresearch.engine import generate as inherited_generate, ResearchWriter
from ..ate_fulljoint_v2.engine import LAYERS
from ..fastp2_bypass_writer.common import hardware
from ..fastp2_bypass_writer.run import operators
from ..p1_layerwise_accuracy.engine import Adapter, tensor_hash
from ..physical_visual_handoff.engine import equivalence


def save(path,payload):
    assert not path.exists(),f'Checkpoint overwrite: {path}'
    atomic(path,dict(payload=payload,sha256=digest(payload)))


def check_stop():
    assert not list(OUT.glob('worker*/STOP.json')), 'Worker validation failed; no further GPU work'
    limit=datetime.datetime.fromisoformat(read(OUT/'protocol.json')['stop_new_gpu_work_by']).timestamp()
    if time.time()>=limit: raise TimeoutError('Frozen GPU-work deadline reached')


def previous(key,label='C3-V2'):
    p=OLD/'checkpoints'/key/(label.replace(' ','_')+'.json')
    if not p.exists(): return None,None
    b=read(p); assert b['sha256']==digest(b['payload'])
    return b['payload'],p


@contextmanager
def first_logits(a):
    values=[]
    def observe(m,args,out):
        assert torch.isfinite(out.logits).all()
        if not values: values.append(out.logits[:,-1].detach().float().cpu().clone())
    h=a.model.register_forward_hook(observe)
    try: yield values
    finally: h.remove()


def smoke(a,rank,samples,partition):
    checks=[]
    for key in partition['smoke']:
        check_stop(); s=samples[key]; l,w,spec=prepare(a,s,'C3')
        native_cfg=config('native',head=1)
        native=inherited_generate(a,l,native_cfg,w,2,True,True)
        noop=inherited_generate(a,l,native_cfg,w,2,True,True,True)
        assert native[0]['token_ids']==noop[0]['token_ids']
        assert len(native[2])==len(noop[2])==2
        assert all(torch.equal(x,y) for x,y in zip(native[2],noop[2]))
        with torch.inference_mode():
            hf=a.model.generate(**a.inputs,max_new_tokens=2,do_sample=False,num_beams=1,use_cache=True,logits_to_keep=1)
        assert hf[0,l.length:].tolist()==native[0]['token_ids']
        nsha=tensor_hash(native[1][0]); del native,noop,hf
        details=[]
        full=ResearchWriter(a,l,CFG,list(l.relay),2)
        original_mask=full.prefill_early.clone(); del full
        for label in LABELS:
            l,w,spec=prepare(a,s,label)
            ctl=ResearchWriter(a,l,CFG,w,2)
            expected=original_mask.clone()
            for q in set(l.relay)-set(w): expected[0,0,q,list(l.image)]=-torch.inf
            assert torch.equal(ctl.prefill_early,expected)
            assert torch.equal(ctl.active_mask,expected[...,:-1,:])
            assert (expected==0).any(-1).all()
            del ctl,expected
            with operators(a) as (calls,mlp,shapes):
                actual=generate(a,l,w,2,True,True)
            ref=generate(a,l,w,2,True,True,True)
            e=equivalence('llava',ref[1],actual[1],ref[3],actual[3])
            assert e['passed'] and actual[0]['token_ids']==ref[0]['token_ids'],(key,label,e)
            assert all(mlp[i]==2 for i in range(32))
            for row in shapes:
                i=row['layer']; want=l.length if i<20 else len(l.text)
                if i in LAYERS and row['op'] in ('q_proj','o_proj'): want-=1
                assert row['shape'][1] in (1,want),(row,want)
            assert all(calls[i,op]==1 for i in LAYERS for op in ('q_proj','k_proj','v_proj','o_proj'))
            assert actual[0]['audit'] and all(r['blocked_probability_max']==0 for r in actual[0]['audit'])
            kv=actual[0]['kv']
            assert kv['historical_cache_copies']==kv['append_concatenations']==0
            assert all(r['used_tokens']==(0 if r['layer'] in LAYERS else len(l.text)+1) for r in kv['layers'])
            for x,y in zip(actual[4],ref[4]):
                assert x.keys()==y.keys() and all(torch.equal(x[f],y[f]) for f in x)
            old,p=previous(key)
            history=None
            if label=='C3' and old is not None:
                assert old['config']==CFG and old['input_binding']==spec
                assert tensor_hash(actual[1][0])==old['generation']['first_logits_sha256']
                assert actual[0]['token_ids']==old['generation']['token_ids'][:2]
                history=dict(path=str(p),sha256=sha(p),first_logits_exact=True,tokens_exact=True)
            details.append(dict(condition=label,first_logits_sha256=tensor_hash(actual[1][0]),
                token_ids=actual[0]['token_ids'],dense_reference=e,history=history,
                exact_mask_delta=True,blocked_probability_zero=True,all_MLP_executed=True,physical_shapes=True,
                KV_text_only=True,original_RoPE=True))
            del actual,ref
            a.verify()
        del original_mask
        checks.append(dict(sample=key,native_first_logits_sha256=nsha,native_noop_exact=True,official_tokens_exact=True,conditions=details))
        atomic(OUT/f'worker{rank}/smoke_partial.json',checks)
        print('SMOKE',rank,key,'PASS',flush=True)
    atomic(OUT/f'worker{rank}/smoke.json',dict(status='PASS',sources=source_hashes(),checks=checks))


def compact(g):
    g.pop('process_memory_snapshot',None)
    g['kv']={k:v for k,v in g['kv'].items() if k!='layers'}
    ids=g['token_ids']
    g['repeated']=any(ids[i:i+3]==ids[j:j+3] for i in range(len(ids)-2) for j in range(i+3,len(ids)-2))
    return g


def quality(a,rank,samples,partition):
    own=[k for k,s in samples.items() if partition['image_shard'][s['image_sha256']]==rank]
    def order(k):
        s=samples[k]; family='GQA' if s['dataset']=='GQA' else s['task']
        return (TASKS.index(family),partition['canonical_input'][k]!=k,k)
    keys=sorted(own,key=order); generated=0; reused=0; start=time.time()
    phases=[('core',keys,CORE),('explore120',[k for k in keys if k in partition['explore120']],tuple(c for c in LABELS if c not in CORE)),
            ('complete',keys,tuple(c for c in LABELS if c not in CORE))]
    for phase,phase_keys,labels in phases:
        for ix,k in enumerate(phase_keys):
            s=samples[k]
            for label in labels:
                check_stop()
                if record(k,label,s) is not None: continue
                l,w,spec=prepare(a,s,label)
                specp=OUT/'input_specs'/(k+'.json')
                base_spec=dict(spec,writers=list(l.relay))
                if specp.exists(): assert read(specp)==base_spec
                else: atomic(specp,base_spec)
                alias=partition['canonical_input'][k]
                source=record(alias,label,samples[alias]) if alias!=k else None
                provenance=None
                if source is not None:
                    assert source['input_binding']==spec
                    g=copy.deepcopy(source['generation']); reused+=1
                    provenance=dict(kind='identical_input_same_GPU',sample=alias,checkpoint_sha256=sha(checkpoint(alias,label)))
                elif label=='C3':
                    old,p=previous(k)
                    if old is not None:
                        assert old['config']==CFG and old['input_binding']==spec
                        assert old['sources']==read(OLD/'execution_freeze.json')['sources']
                        g=copy.deepcopy(old['generation']); reused+=1
                        provenance=dict(kind='audited_historical_C3',path=str(p),checkpoint_sha256=sha(p),
                            code_commit='801d1ccc8f418b7a5db35f778f84665a4beab8a5',logits_shape_parity_verified_in_smoke=True)
                if provenance is None:
                    with first_logits(a) as logits: g=generate(a,l,w)[0]
                    assert len(logits)==1
                    g['first_logits_sha256']=tensor_hash(logits[0])
                    g['first_top1_token']=int(logits[0].argmax())
                    assert g['token_ids'][0]==g['first_top1_token']
                    g['first_top1_logp']=float(logits[0].log_softmax(-1).max())
                    g=compact(g); generated+=1
                save(checkpoint(k,label),dict(sample=k,sample_sha256=digest(s),condition=label,config=CFG,
                    writers=w,sources=source_hashes(),input_binding=spec,generation=g,scoring=scored(s,g['text']),
                    reuse=provenance,at=now(),rank=rank))
                a.verify()
            if ix%10==0 or ix+1==len(phase_keys):
                state=dict(at=now(),phase=phase,questions=ix+1,total=len(phase_keys),generated=generated,reused=reused,elapsed_seconds=time.time()-start)
                atomic(OUT/f'worker{rank}/progress.json',state); print('PROGRESS',rank,state,flush=True)
        atomic(OUT/f'worker{rank}/{phase}_complete.json',dict(at=now(),status='COMPLETE',questions=len(phase_keys),conditions=list(labels)))
    atomic(OUT/f'worker{rank}/quality_complete.json',dict(status='COMPLETE',questions=len(keys),generated=generated,reused=reused,elapsed_seconds=time.time()-start))


def worker(rank):
    f=read(OUT/'execution_freeze.json')
    assert f['sources']==source_hashes()
    for name in ('sample_manifest','partition','protocol'):
        assert sha(OUT/(name+'.json'))==f[name+'_sha256']
    dev=DEVICES[rank]; assert os.environ['CUDA_VISIBLE_DEVICES']==dev['uuid']
    hw=hardware(); assert dev['uuid'] not in hw['processes'],'Authorized GPU is no longer idle'
    atomic(OUT/f'worker{rank}/preflight.json',dict(at=now(),pid=os.getpid(),**hw))
    a=None; start=time.time()
    try:
        samples=read(OUT/'sample_manifest.json'); partition=read(OUT/'partition.json')
        a=Adapter('llava'); atomic(OUT/f'worker{rank}/environment.json',a.environment)
        if not (OUT/f'worker{rank}/smoke.json').exists(): smoke(a,rank,samples,partition)
        else: assert read(OUT/f'worker{rank}/smoke.json')['sources']==source_hashes()
        begin=time.time()
        while not all((OUT/f'worker{i}/smoke.json').exists() for i in range(2)):
            check_stop(); assert time.time()-begin<900; time.sleep(1)
        left,right=[read(OUT/f'worker{i}/smoke.json')['checks'] for i in range(2)]
        assert left==right,'Cross-GPU Smoke logits/tokens/reference audits differ; stop and diagnose'
        atomic(OUT/'gpu_smoke_validation.json',dict(status='PASS',at=now(),cross_gpu_exact=True,samples=partition['smoke'],sources=source_hashes())) if rank==0 else None
        quality(a,rank,samples,partition)
        atomic(OUT/f'worker{rank}/completion.json',dict(status='COMPLETE',at=now(),elapsed_seconds=time.time()-start))
    except TimeoutError:
        atomic(OUT/f'worker{rank}/deadline_stop.json',dict(status='PARTIAL',at=now(),reason=traceback.format_exc()))
    except BaseException:
        atomic(OUT/f'worker{rank}/STOP.json',dict(at=now(),traceback=traceback.format_exc())); raise
    finally:
        del a; gc.collect(); torch.cuda.empty_cache()
        atomic(OUT/f'worker{rank}/release.json',dict(at=now(),pid=os.getpid(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),process_exit_releases_remaining=True))


def driver():
    state=hardware()
    assert all(d['uuid'] not in state['processes'] for d in DEVICES),'Both authorized devices must be free'
    atomic(OUT/'driver_preflight.json',dict(at=now(),**state))
    processes=[]; logs=[]
    try:
        for rank,d in enumerate(DEVICES):
            log=(OUT/f'worker{rank}/execution.log'); log.parent.mkdir(parents=True,exist_ok=True)
            stream=log.open('a'); logs.append(stream)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=d['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            p=subprocess.Popen([sys.executable,'-u','-m','vlm_research.reverse_writer.run','--rank',str(rank)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            processes.append(p)
        atomic(OUT/'driver_pids.json',dict(at=now(),driver=os.getpid(),workers=[p.pid for p in processes]))
        codes=[p.wait() for p in processes]
        atomic(OUT/'driver_exit.json',dict(at=now(),returncodes=codes))
        assert all(c==0 for c in codes),codes
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            if p.poll() is None: p.wait(timeout=60)
        for stream in logs: stream.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--driver',action='store_true'); parser.add_argument('--rank',type=int)
    args=parser.parse_args()
    if args.freeze: freeze()
    elif args.driver: driver()
    else:
        assert args.rank in (0,1); worker(args.rank)


if __name__=='__main__': main()
