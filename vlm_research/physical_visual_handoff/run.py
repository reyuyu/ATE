"""Small, checkpointed quality pilot. No availability polling and no automatic expansion beyond 240."""
import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch

from .common import (OUT, ROOT, HISTORY, BRANCH, MODELS, CONDITIONS, PAIRS, atomic, digest, read, key,
                     prepare, sources, write_csv, TOLERANCES)
from .engine import Adapter, run, equivalence


def worker(kind):
    folder=OUT/kind
    frozen=read(OUT/'protocol.json')
    assert frozen['sources']==sources()
    manifest=read(folder/'sample_manifest.json')
    specs=read(folder/'input_bindings.json')
    baseline=read(HISTORY/kind/'baselines.json')
    adapter=Adapter(kind)
    records=[]; audits=[]
    def execute(s,c,audit=False,capture=False):
        adapter.prepare(s,specs[key(s)])
        return run(adapter,s,specs[key(s)],c,audit=audit,capture=capture)
    # All CPU and same-shape native gates precede any shape-changing measurements.
    for s in manifest['smoke']:
        adapter.prepare(s,specs[key(s)])
        native, ns, _=run(adapter,s,specs[key(s)],'N0',native=True,capture=True)
        assert native['token_ids']==baseline[key(s)]['token_ids'], 'Historical Native tokens mismatch'
        noop, fs, _=execute(s,'N0',capture=True)
        assert noop['token_ids']==native['token_ids'] and len(ns)==len(fs)
        assert all(torch.equal(x,y) for x,y in zip(ns,fs)), 'Same-shape Native/No-op logits not exact'
        assert noop['all_valid_logits_sha256']==native['all_valid_logits_sha256'], 'All effective returned logits parity failed'
        for dense,physical in PAIRS:
            a,al,ah=execute(s,dense,audit=True,capture=True)
            b,bl,bh=execute(s,physical,audit=True,capture=True)
            proof=equivalence(kind,al,bl,ah,bh) if len(al)==len(bl) else dict(passed=False,length_mismatch=True)
            proof.update(question_id=s['question_id'],task=s['task'],dense=dense,physical=physical,
                         tokens_equal=a['token_ids']==b['token_ids'],mask_checks=len(b['mask_events']),
                         persistent_cache_bytes=b['cache']['live_kv_bytes'],
                         released_tensor_references=b['cache']['old_tensor_references_alive'])
            audits.append(proof)
            atomic(folder/'smoke_audit.json',audits)
            assert proof['passed'] and proof['tokens_equal'], f'Equivalence failed: {proof}'
            assert proof['released_tensor_references']==0
        adapter.verify()
        print('SMOKE',kind,key(s),'PASS',flush=True)
    atomic(folder/'smoke.json',dict(passed=True,cases=len(manifest['smoke']),sources=sources()))
    samples=manifest['samples']
    # GQA60 first; only already verified paths enter the remaining 180 images.
    order=sorted(samples,key=lambda s:(key(s) not in manifest['pilot60'],key(s)))
    for index,s in enumerate(order):
        adapter.prepare(s,specs[key(s)])
        sample_records={}
        for c in CONDITIONS:
            path=folder/'checkpoints'/key(s)/(c+'.json')
            if path.exists():
                record=read(path)
                assert record['binding']==digest(frozen) and record['sample_sha256']==digest(s)
            else:
                result,_,_=run(adapter,s,specs[key(s)],c,audit=False)
                record=dict(binding=digest(frozen),sample_sha256=digest(s),question_id=s['question_id'],task=s['task'],
                            image_sha256=s['image_sha256'],gold=s['answer'],generation=result)
                atomic(path,record)
            sample_records[c]=record
            records.append(record)
            for dense,physical in PAIRS:
                if c==physical:
                    assert sample_records[dense]['generation']['token_ids']==record['generation']['token_ids'], \
                        f'Quality equivalence gate failed {key(s)} {dense}/{physical}; do not run later stages'
        for dense,physical in PAIRS:
            assert sample_records[dense]['generation']['token_ids']==sample_records[physical]['generation']['token_ids'], \
                f'Quality equivalence gate failed {key(s)} {dense}/{physical}'
        adapter.verify()
        atomic(folder/'progress.json',dict(status='RUNNING',questions=index+1,total=240,labels=len(records),
               stage='GQA60' if index<60 else 'GQA240',time=time.time()))
        if (index+1)%10==0: print('QUALITY',kind,index+1,'/240',flush=True)
    atomic(folder/'quality_complete.json',dict(status='COMPLETE',questions=240,labels=len(records)))
    from .measure import benchmark
    benchmark(adapter,manifest,specs)
    from .analyze import analyze
    analyze(kind)


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','worker']);p.add_argument('--kind',choices=list(MODELS))
    args=p.parse_args()
    try:
        if args.stage=='prepare':
            prepare()
            atomic(OUT/'protocol.json',dict(seed=20261104,model=MODELS,sources=sources(),
                   conditions=CONDITIONS, cutoff=dict(llava=20,qwen=22), tolerances=TOLERANCES,
                   output_root=str(OUT),max_questions=240,pilot60=True,bootstrap=20000,
                   shape_changing_tokens_gate='Exact greedy tokens in smoke and quality; stop on unexplained divergence',
                   benchmark=dict(warmups=5,repetitions=20,fixed_decode_tokens=[16,64,128]),
                   historical_model_hashes={k:read(HISTORY/k/'protocol.json')['weights'] for k in MODELS},
                   source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        else:
            worker(args.kind)
    except BaseException:
        atomic(OUT/(args.kind or 'preflight')/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise


if __name__=='__main__':main()
