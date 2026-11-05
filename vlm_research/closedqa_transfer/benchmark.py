"""Resume fixed timing after correcting measurement-label serialization only.

The original frozen generator and quality checkpoints are left unchanged.
This entry point has its own source binding, not a silently edited worker.
"""
import argparse
import gc
import os
import time
import traceback
from unittest.mock import patch

import torch

from .common import *
from .engine import prepare,generate
from . import run as frozen
from ..p1_layerwise_accuracy.engine import Adapter
from ..fastp2_bypass_writer.run import CompleteCensus


def measurement(label,rep,g):
    row=dict(frozen.compact(g))
    row.update(condition=label,rep=rep)
    return row


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
                assert g['token_ids']==record(k,label,s)['generation']['token_ids']
                measurements.append(measurement(label,rep,g))
        for label in CONFIGS:
            l,w,_=prepare(a,s,label); census=CompleteCensus(a)
            with census.context(): g=generate(a,l,label,w)[0]
            assert g['token_ids']==record(k,label,s)['generation']['token_ids']
            census_rows.append(dict(condition=label,rows=census.rows(),KV=g['kv'],
                                    n_prompt=l.length,n_text=len(l.text)))
        frozen.save(path,dict(sample=k,sample_sha256=digest(s),sources=source_hashes(),rank=rank,
            measurements=measurements,flops=census_rows,at=now(),
            performance_entry_sha256=sha(Path(__file__))))
        a.verify(); print('PERFORMANCE',rank,ix+1,len(keys),k,flush=True)
    atomic(OUT/f'worker{rank}/performance_complete.json',dict(status='COMPLETE',questions=len(keys),
        timed_requests=len(keys)*25,warmup_requests=len(keys)*15,census_requests=len(keys)*5,
        performance_entry_sha256=sha(Path(__file__))))


def check_stop():
    assert not list(OUT.glob('worker*/RECOVERY_STOP.json')),'Resume worker failed'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,required=True); args=ap.parse_args()
    rank=args.rank; assert rank in (0,1)
    f=read(OUT/'execution_freeze.json'); assert f['sources']==source_hashes()
    p=read(OUT/'performance_entry_freeze.json'); assert p['sha256']==sha(Path(__file__))
    assert p['original_execution_freeze_sha256']==sha(OUT/'execution_freeze.json')
    for i in (0,1):
        if (OUT/f'worker{i}/STOP.json').exists():
            failure=read(OUT/f'worker{i}/STOP.json')['traceback']
            assert "multiple values for keyword argument 'condition'" in failure or 'A worker failed; preserving checkpoints' in failure,failure
    assert any("multiple values for keyword argument 'condition'" in read(v)['traceback'] for v in OUT.glob('worker*/STOP.json'))
    assert all(read(OUT/f'worker{i}/smoke.json')['status']=='PASS' for i in (0,1))
    state=hardware(); d=f['devices'][rank]
    assert os.environ['CUDA_VISIBLE_DEVICES']==d['uuid'] and d['uuid'] not in state['processes']
    atomic(OUT/f'worker{rank}/resume_preflight.json',dict(at=now(),pid=os.getpid(),**state))
    samples,partition=read(OUT/'sample_manifest.json'),read(OUT/'partition.json')
    a=None; started=time.time()
    try:
        a=Adapter('llava')
        # Same device, source, inputs and Head; verify resumed Native logits against stored binding.
        key=next(k for k,s in samples.items() if partition['image_shard'][s['image_sha256']]==rank and record(k,'Native Full',s))
        l,w,_=prepare(a,samples[key],'Native Full')
        result=generate(a,l,'Native Full',w,2,True)
        from ..p1_layerwise_accuracy.engine import tensor_hash
        assert tensor_hash(result[1][0])==record(key,'Native Full',samples[key])['generation']['first_logits_sha256']
        del result
        atomic(OUT/f'worker{rank}/resume_parity.json',dict(status='PASS',sample=key,first_logits_exact=True,
            sources=source_hashes(),performance_entry_sha256=sha(Path(__file__))))
        with patch.object(frozen,'check_stop',check_stop):
            if not (OUT/f'worker{rank}/quality_complete.json').exists(): frozen.quality(a,rank,samples,partition)
        performance(a,rank,samples,partition); a.verify()
        atomic(OUT/f'worker{rank}/completion.json',dict(status='COMPLETE',at=now(),
            elapsed_seconds=time.time()-started,sources=source_hashes(),performance_entry_sha256=sha(Path(__file__)),
            recovered_serialization_failure=True))
    except BaseException:
        atomic(OUT/f'worker{rank}/RECOVERY_STOP.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
    finally:
        del a; gc.collect(); torch.cuda.empty_cache()
        atomic(OUT/f'worker{rank}/resume_release.json',dict(at=now(),pid=os.getpid(),
            allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))


if __name__=='__main__': main()
