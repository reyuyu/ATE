import fcntl
import time

import numpy as np
import torch

from .common import OUT, CONDITIONS, atomic, key, write_csv
from .engine import run


def benchmark(adapter, manifest, specs):
    folder=OUT/adapter.kind
    sample=manifest['samples'][0]
    adapter.prepare(sample,specs[key(sample)])
    # Serialize timing and profiler workloads across the two model workers.
    with (OUT/'latency.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        rows=[]
        for length in (16,64,128):
            for c in CONDITIONS:
                results=[]
                for repeat in range(25):
                    r,_,_=run(adapter,sample,specs[key(sample)],c,max_tokens=length,fixed=True,native=c=='N0')
                    if repeat>=5: results.append(r)
                row=dict(model=adapter.kind,condition=c,forced_tokens=length,warmups=5,repetitions=20,
                         natural_accuracy=False,backend='eager',same_gpu=True,includes_vision=True)
                measurements={
                    'vision_ms':[r['vision_ms'] for r in results],
                    'language_prefill_ms':[r['language_prefill_ms'] for r in results],
                    'ttft_ms':[r['ttft_ms'] for r in results],
                    'decode_ms_per_token':[float(np.mean(r['decode_ms'])) for r in results],
                    'end_to_end_ms':[r['latency_ms'] for r in results],
                    'live_kv_bytes':[r['cache']['live_kv_bytes'] for r in results],
                    'peak_allocated_bytes':[r['peak_allocated'] for r in results],
                    'peak_reserved_bytes':[r['peak_reserved'] for r in results],
                }
                for name, values in measurements.items():
                    for suffix,q in (('median',.5),('p10',.1),('p90',.9)):
                        row[name+'_'+suffix]=float(np.quantile(values,q))
                rows.append(row)
                write_csv(folder/'latency_benchmark.csv',rows)
                print('BENCH',adapter.kind,c,length,flush=True)
        from torch.profiler import profile, ProfilerActivity
        profiler=[]
        for c in ('N0','D3','P4'):
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                r,_,_=run(adapter,sample,specs[key(sample)],c,max_tokens=16,fixed=True,audit=True,native=c=='N0')
            for e in prof.key_averages(group_by_input_shape=True):
                if any(k in e.key for k in ('mm','softmax','matmul','layer_norm')):
                    profiler.append(dict(model=adapter.kind,condition=c,operator=e.key,input_shapes=e.input_shapes,
                                         calls=e.count,cpu_us=e.cpu_time_total,
                                         device_us=getattr(e,'device_time_total',getattr(e,'cuda_time_total',0))))
            atomic(folder/('execution_'+c+'.json'),r['execution'])
        write_csv(folder/'gpu_profiler_summary.csv',profiler)
        # Release-after-layer is a separate engineering equivalence gate, not a new scientific condition.
        release=[]
        for c in ('P2','P3','P4'):
            a,al,ah=run(adapter,sample,specs[key(sample)],c,audit=True,capture=True)
            b,bl,bh=run(adapter,sample,specs[key(sample)],c,audit=True,capture=True,immediate=True)
            assert a['token_ids']==b['token_ids'] and all(torch.equal(x,y) for x,y in zip(al,bl))
            release.append(dict(condition=c,exact_logits=True,exact_tokens=True,
                postprefill_bytes=a['cache']['live_kv_bytes'],immediate_bytes=b['cache']['live_kv_bytes'],
                old_references_alive=b['cache']['old_tensor_references_alive']))
        atomic(folder/'immediate_release_validation.json',release)
