"""Measured single-request physical costs, separate from shared-prefix quality."""
from contextlib import ExitStack
from unittest.mock import patch
import time
from common import *


def execute(model,processor,pool,job,shard,bound):
    if job.get('phase')=='GATE':
        return physical_gate(model,processor,pool,job,shard,bound)
    import torch
    from engine import Request
    from vlm_research.qwen3vl_svh_sqa7908_repair.engine import generate as frozen_svh
    from vlm_research.qwen3vl_c3v2_physical.engine import generate as native
    store=Store(OUT/f'performance{shard}.sqlite')
    methods={'Native':None,'SVH':[],**job['conditions']}
    try:
        for row in sorted(pool,key=lambda r:(r['image_sha256'],r['key'])):
            if row.get('historical_blocked_status'):
                continue
            inputs,layout=bound(row)
            for mode in ['natural','fixed16','fixed32']:
                for repeat in range(-3,5):
                    names=list(methods)
                    shift=(repeat+3)%len(names)
                    names=names[shift:]+names[:shift]
                    for name in names:
                        key=f"{job['id']}|{row['key']}|{mode}|{repeat}|{name}"
                        if store.get(key) is not None:
                            continue
                        fixed=mode!='natural'
                        cap=int(mode[5:]) if fixed else row['max_new_tokens']
                        layers=methods[name]
                        vision=[]
                        original_vision=model.model.get_image_features
                        def watched(*args,**kw):
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record()
                            out=original_vision(*args,**kw)
                            b.record()
                            vision.append((a,b))
                            return out
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start=time.perf_counter()
                        with ExitStack() as stack:
                            stack.enter_context(patch.object(model.model,'get_image_features',watched))
                            if fixed and name in ['Native','SVH']:
                                stack.enter_context(patch.object(model.generation_config,'eos_token_id',[]))
                            if name=='Native':
                                result,_=native(model,processor,inputs,layout,'N',max_tokens=cap)
                                kv=result['audit']['kv']
                                prefill=result['ttft_ms']
                                decode=result['decode_ms']
                                builder=[]
                            elif name=='SVH':
                                result,_=frozen_svh(model,processor,inputs,layout,'C',max_tokens=cap)
                                kv=result['audit']['kv']
                                prefill=result['ttft_ms']
                                decode=result['decode_ms']
                                builder=[]
                            else:
                                with Request(model,processor,inputs,layout,collect=layers,prefill_layers=layers,cap=cap) as request:
                                    result,_=request.generate(layers,physical=True,cap=cap,fixed=fixed,consume=True)
                                    kv=result['kv']
                                    prefill=result['prefill_ms']
                                    decode=result['decode_ms']
                                    builder=result['temporary_sidecar']
                        torch.cuda.synchronize()
                        elapsed=(time.perf_counter()-start)*1000
                        assert not fixed or len(result['token_ids'])==cap
                        store.put(key,dict(key=row['key'],benchmark=row['benchmark'],analysis_task=row['analysis_task'],
                            image_sha256=row['image_sha256'],condition=name,layers=layers,mode=mode,repeat=repeat,warmup=repeat<0,
                            e2e_ms=elapsed,prefill_including_vision_ms=prefill,vision_ms=sum(a.elapsed_time(b) for a,b in vision),
                            decode_ms=decode,tpot_ms=decode/max(1,len(result['token_ids'])-1),
                            tokens=result['token_ids'],generated_tokens=len(result['token_ids']),text=result['text'],
                            kv_live_bytes=kv['live_bytes'],kv_capacity_bytes=kv['capacity_bytes'],
                            persistent_image_bytes=kv.get('persistent_image_bytes'),peak_allocated=torch.cuda.max_memory_allocated(),
                            peak_reserved=torch.cuda.max_memory_reserved(),sidecar=builder,physical=True,
                            gpu_uuid=UUIDS[shard],timing_boundary='Public single-request invocation, inputs ready; vision recomputed, cold independent KV',
                            hardware_tflops='NOT_MEASURED_NO_HARDWARE_COUNTERS'))
                print('PERF_DONE',shard,row['key'],mode,flush=True)
    finally:
        store.conn.close()


def physical_gate(model,processor,pool,job,shard,bound):
    import torch
    from engine import Request,ACTIVE
    cases=[]
    try:
        for row in pool:
            inputs,layout=bound(row)
            for name,layers in job['conditions'].items():
                with Request(model,processor,inputs,layout,collect=ACTIVE,prefill_layers=layers,audit=True,cap=4) as request:
                    reference,a=request.generate(layers,capture=True)
                with Request(model,processor,inputs,layout,collect=layers,prefill_layers=layers,audit=True,cap=4) as request:
                    physical,b=request.generate(layers,physical=True,capture=True)
                error=max(float((x-y).abs().max()) for x,y in zip(a,b))
                case=dict(key=row['key'],condition=name,layers=layers,max_logit_error=error,
                    vector_counts=[len(a),len(b)],same_tokens=reference['token_ids']==physical['token_ids'],
                    prefill_non_target_unchanged=reference['prefill_non_target_hashes']==physical['prefill_non_target_hashes'],
                    physical_kv=physical['kv'],historical_strict_gate='FAIL_RETAINED')
                cases.append(case)
                write(OUT/f'physical_gate{shard}.json',dict(status='IN_PROGRESS',cases=cases,gpu_uuid=UUIDS[shard]))
                assert len(a)==len(b) and error<=.02 and case['same_tokens'] and case['prefill_non_target_unchanged'],case
            del inputs
        write(OUT/f'physical_gate{shard}.json',dict(status='PASS_SELECTED_STORAGE_ONLY',cases=cases,
            historical_strict_gate='FAIL_RETAINED',gpu_uuid=UUIDS[shard],time=time.time()))
    except BaseException as exc:
        write(OUT/f'physical_gate{shard}.json',dict(status='FAIL',cases=cases,error=repr(exc),
            historical_strict_gate='FAIL_RETAINED',gpu_uuid=UUIDS[shard],time=time.time()))
        raise
